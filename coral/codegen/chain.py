# whole networks as ONE Edge TPU program, byte-identical to edgetpu_compiler (docs/isa/codegen_chain.md): chains of VALID k x k
# convolutions (fused ReLU clamps), 2x2 max pools and a final linear layer, activations kept in narrow memory between the layers.
#   caching, execution = gen_chain(layers)          layers: [Conv, Conv, Pool, ..., Linear | Conv]
#   blob = chain_blob(layers, weights)              io = chain_io(layers)
# `python -m coral.codegen.chain` runs the acceptance test (test/test_codegen_chain.py).
# Every convolution is coral/codegen/conv2d.py's tile mapping (each tile a block of the image); between layers nothing goes back to
# the host: a conv op writes its outputs straight into the block layout of its consumer (the next conv's input block, with the halo
# rows / columns left free, or the window block of a max pool), the halos move over the mesh, a max pool relays its window block to
# wide memory and pools it into the next consumer's block.
from __future__ import annotations
from dataclasses import dataclass, field
from coral.isa import cdiv, round_up, ttu, f32_bits, op as OP, wide_narrow as WN, ring_mesh as RM, scalar as SC, eltwise as EL
from coral.codegen import Emitter, Piece, caching_program, run_test, WIDE_TOP
from coral.codegen import conv2d as C2, conv as CC, fc as FC
from coral.codegen.conv2d import Conv2DGeom, MESH_N, MESH_S, MESH_W, MESH_E

# ***** the layer description *****
@dataclass
class Conv:
  """CONV_2D x[1,H,W,Cin] * w[Cout,kh,kw,Cin] -> [1,OH,OW,Cout], VALID; q = the op's quantization (conv2d.conv2d_quant)"""
  H: int
  W: int
  Cin: int
  Cout: int
  kh: int
  kw: int
  stride: int = 1
  q: dict = field(default_factory=dict)
  @property
  def g(self) -> Conv2DGeom: return Conv2DGeom(self.H, self.W, self.Cin, self.Cout, self.kh, self.kw, self.stride, "VALID")

@dataclass
class Pool:
  """MAX_POOL_2D k x k / stride, VALID, on the previous conv's output; q = (scale, zero point) of that output"""
  k: int
  stride: int
  q: tuple = (1.0, 0)

@dataclass
class Linear:
  """the final FULLY_CONNECTED y[N] = W[N,K] x[K] on the flattened (y, x, c) map of the layer before (batch 1)"""
  N: int
  K: int
  q: dict = field(default_factory=dict)

# ***** sync-counter state of every tile since the last reset17 *****
AV, NIN, EIN, SIN, WIN = 0, 3, 4, 5, 6        # tile sync counters (scalar.TILE_COUNTERS): op outputs, mesh inputs from N / E / S / W
IN_OF = {MESH_N: SIN, MESH_S: NIN, MESH_W: EIN, MESH_E: WIN}   # a move north arrives from the south neighbour, ...

class State:
  def __init__(self): self.reset()
  def reset(self):
    self.cnt = [[0] * 19 for _ in range(16)]
    self.par = 0                                  # parameter fills of the convs since the reset
    self.lastG = [1] * 16                         # 64-channel groups of the last op on each tile (it counts once per group)
  def av_done(self, t:int) -> int: return self.cnt[t][AV] - (self.lastG[t] - 1)   # the op counter once its first group is done

def rec(cid:int, count:int, levels:int) -> tuple[int, int]: return cid << 1, count << 2 | levels   # mesh sync record (id, value)

# ***** geometry helpers *****
def pool_geom(c:Conv, p:Pool) -> Conv2DGeom:
  """the window blocks of a max pool on conv c's output: the conv2d spans with kernel k, stride s on the OH x OW map"""
  g = c.g
  return Conv2DGeom(g.OH, g.OW, c.Cout, c.Cout, p.k, p.k, p.stride, "VALID")

def into_block(gc:Conv2DGeom, base:int, t:int, rows:int, cols:int) -> tuple:
  """(base, position levels, wide) of tile t's rows x cols outputs written into consumer gc's input block at narrow base (the own
  rows / columns of the consumer's spans; rows of a wider block -> 'wide')"""
  rs, cs = gc.rows[t // 4], gc.cols[t % 4]
  w = gc.C4 // 4
  b = base + (rs.d * cs.nb + cs.d) * gc.C4
  if cs.nb == cols or rows == 1: return b, [(w, cols * rows)], False
  return b, [(w, cols), (cs.nb * w, rows)], True

# ***** halo exchange with sync records *****
def halo(e:Emitter, g:Conv2DGeom, X:int, st:State, fill:int=0, inrec=None):
  """conv2d.halo_stage on the chain's blocks: every outbound half waits for the data it sends (records: the relay's IN counter, the
  op counter when an op wrote the block since the last reset, then the vertical IN counters of a horizontal move), values = the
  counts since the last reset17. inrec(t, op) -> True when tile t's inbound half also waits for its op (see the docs). A receive
  counts once per 16-word piece of a position (inc; a max pool's 64-channel groups), a record waits for the first piece"""
  cw, C4 = g.cw, g.C4
  inc = cdiv(cw, 16)
  def addr(r, c, wrow, wcol): return X + (wrow * g.cols[c].nb + wcol) * C4
  got = [[] for _ in range(16)]
  for op in (MESH_N, MESH_S, MESH_W, MESH_E):
    vertical, per, recv = op in (MESH_N, MESH_S), [], []
    for t in range(16):
      r, c = divmod(t, 4)
      rs, cs = g.rows[r], g.cols[c]
      if vertical and not any(s_.no and s_.w0 < cs.i0 + cs.ni and cs.i0 < s_.w0 + s_.nw for s_ in g.cols):
        per.append(None)                                       # a tile column whose own columns no window needs moves nothing
        continue
      if vertical:
        (no, ro, relay), (ni, ri, fl) = C2._moves(g.rows, r, g.H, op == MESH_N)
        def dims(n, cs=cs): return [(1, cw), (cw, cs.ni), (cs.nb * cw, n)]
        o_at, i_at = (lambda r=r, c=c, ro=ro, cs=cs: addr(r, c, ro, cs.d)), (lambda r=r, c=c, ri=ri, cs=cs: addr(r, c, ri, cs.d))
      elif not rs.no:
        per.append(None)
        continue
      else:
        (no, ro, relay), (ni, ri, fl) = C2._moves(g.cols, c, g.W, op == MESH_W)
        def dims(n, cs=cs, rs=rs): return [(1, cw), (cw, n), (cs.nb * cw, rs.nw)]
        o_at, i_at = (lambda r=r, c=c, ro=ro, rs=rs: addr(r, c, rs.wo, ro)), (lambda r=r, c=c, ri=ri, rs=rs: addr(r, c, rs.wo, ri))
      f, recs = {}, []
      if no:
        f |= C2._half("o_", o_at(), dims(no))
        lv = bin(f["o_sdims"]).count("1") - (cw > 16)
        if relay: recs.append(rec(IN_OF[op], st.cnt[t][IN_OF[op]] + 1, lv))    # (an IN counter: the first of inc units)
        if st.cnt[t][AV] and (rs.ni if vertical else cs.ni): recs.append(rec(AV, st.av_done(t), lv))   # own data (written by an op)
        if not vertical: recs += [rec(k, st.cnt[t][k] - (inc - 1), lv) for k in sorted(got[t])]
      if ni:
        f |= C2._half("i_", i_at(), dims(ni)) | (dict(fill_en=1, fill=fill) if fl else {})
        recv.append(t)
        if inrec is not None and inrec(t, op) and st.cnt[t][AV]:
          recs += [rec(AV, st.av_done(t), bin(f["i_sdims"]).count("1") - (cw > 16)), (1, None)]
      for k, (rid, val) in enumerate(recs):
        if k == 5:                                    # the end marker of a full list spills into a sixth slot
          f["rsv685"] = rid
          break
        f |= {f"s{k}_id": rid} if val is None else {f"s{k}_id": rid, f"s{k}_val": val, f"s{k}_en_a": 1, f"s{k}_en_b": 1}
      per.append(f or None)
    for m, f in C2._group(per): e.tile(RM.encode_mesh(opcode=op, tile_mask=m, seq=e.seq, **f))
    for t in recv:
      st.cnt[t][IN_OF[op]] += inc
      if vertical and IN_OF[op] not in got[t]: got[t].append(IN_OF[op])

# ***** 42-bit sync records of the tile DMAs (narrowToWide from bit 543, wideToNarrow from bit 611; eltwise.md 8.4) *****
def with_records(words:list[int], lo:int, records:list[tuple]) -> list[int]:
  """OR records (en0, f1, f2, counter id, level, value) into an instruction whose sync area (starting at bit lo) is empty. The value is
  20 bits at +12: hi << 16 | count (the low bit is wide_narrow.py's en1, the rest its val)"""
  v = sum(w << (128 * j) for j, w in enumerate(words))
  for k, (en0, f1, f2, cid, lvl, val) in enumerate(records):
    v |= (en0 | f1 << 1 | f2 << 2 | cid << 4 | lvl << 9 | val << 12) << (lo + 42 * k)
  return [(v >> (128 * j)) & ((1 << 128) - 1) for j in range(len(words))]

NO_SYNC = dict(sync_en0=0, sync_f1=0, sync_f2=0, sync_id=0, sync_wait_lvl=0, sync_en1=0, sync_val=0, sync_en2=0, sync_f45=0)

# ***** convolution stage *****
def conv_stage(e:Emitter, c:Conv, X:int, Y:int, out, wl:dict, pieces:list[Piece], first:int, st:State) -> int:
  """fences, ringConsumer1 + bias load, one conv op per distinct tile (out(t) -> (base, levels, wide): where tile t's outputs go),
  the parameter broadcast; -> rows sent. Without a reset17 since the convs before, the sync values count on: R = their parameter
  fills (ringConsumer1, bias load, op progress count), AV = the ops completed on the tile (bias load, op)"""
  g = c.g
  tiles = sum(1 << t for t in range(16) if g.out_tile(t))
  R = st.par
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_drain)
  f = RM.decode_ringConsumer(C2.rc_param(g, e.seq, wl, tiles))
  e.tile(RM.encode_ringConsumer(**(f | dict(s_val=f["s_val"] + R))))
  bias = WN.decode_wide_to_narrow(CC.bias_load(0, 0, wl["bias"], g.b, g.fifo, g.G, g.cg))
  def bias_fields(t):
    r0 = (g.b << 16) | (1 + R)                      # the first fill (after the R before), b fills per group
    r1 = 0x10000 | ((st.cnt[t][AV] - 1) & 0xffff)   # the op counter - 1
    return bias | dict(sync_val=(r0 >> 1) & 0x7ffff, rsv643=r0 >> 20, sync_en1=r0 & 1, sync_dec=((r1 & 0x1ffff) ^ 0x10000) - 0x10000,
                       rsv682=r1 >> 17)
  for m, fb in C2._group([bias_fields(t) if tiles >> t & 1 else None for t in range(16)]):
    e.tile(WN.encode_wide_to_narrow(**(fb | dict(tile_mask=m, seq=e.seq))))
  def op_fields(t):
    fo = C2._conv_fields(g, t // 4, t % 4, X, Y, wl, c.q, out(t))
    n = 1 + R                                       # progress count: sync0 (count // 4), cfg1 bits 8-9 (count % 4)
    return fo | dict(sync0=0x4000 | n >> 2, cfg1=(fo["cfg1"] & ~0x300) | (n & 3) << 8, sync2=0xC6 + 0x80 * st.cnt[t][AV])
  for m, fo in C2._group([op_fields(t) if tiles >> t & 1 else None for t in range(16)]):
    e.tile(OP.encode_op(**(fo | dict(tile_mask=m, seq=e.seq))))
  for tile, off, n in pieces:
    e.tile(CC.rprod_param(e.seq, tile, off, n, first, dest=tiles))
    first += n
  for t in range(16):
    if tiles >> t & 1:                                             # a conv op counts once per 64-output group
      st.cnt[t][AV] += g.G
      st.lastG[t] = g.G
  st.par = R + g.b * g.G
  return first

# ***** max pool stage *****
def _pool_out(gp:Conv2DGeom, t:int, base:int, gc:Conv2DGeom|None) -> tuple[int, int, int, int]:
  """(out_base, position stride, row stride, block stride) in words of tile t's pooled outputs in the consumer's block"""
  rs, cs = gp.rows[t // 4], gp.cols[t % 4]
  w = gp.N4 // 4
  if gc is None: return base, w, cs.no * w, rs.no * cs.no * w
  crs, ccs = gc.rows[t // 4], gc.cols[t % 4]
  return base + (crs.d * ccs.nb + ccs.d) * gc.C4, w, ccs.nb * w, crs.nb * ccs.nb * w

def pool_stage(e:Emitter, c:Conv, p:Pool, P:int, win:int, out_base:int, gc:Conv2DGeom|None, st:State, records:bool):
  """gather the window blocks (the halo exchange of the pool's spans), relay each tile's window to wide memory (transposing
  narrowToWide), pool it into the consumer's block. records: the relays wait on the gather counters themselves (C a multiple of 64,
  observed) instead of a drain"""
  gp = pool_geom(c, p)
  G = cdiv(c.Cout, 64)
  halo(e, gp, P, st, inrec=inrec_after(c))
  tiles = [t for t in range(16) if gp.out_tile(t)]
  C, w = c.Cout, gp.N4 // 4
  relays = []
  for t in tiles:
    rs, cs = gp.rows[t // 4], gp.cols[t % 4]
    relays.append(EL.maxpool_relay_n2w_fields(0, 0, P + (rs.wo * cs.nb + cs.wo) * gp.C4, win, w, cs.nb * w, cs.nw, rs.nw, C))
  if not records: e.sync(SC.sync_drain)
  def relay(t):
    f = relays[tiles.index(t)]
    if not records: return f
    def cnt(k): return 0x10000 + ((st.cnt[t][k] - (G - 1)) & 0xffff)   # (the gather counted G per receive)
    recs = ((1, 1, 0, AV, 3, 0x10000 | st.av_done(t)),) + tuple((0, 0, 1, k, 3, cnt(k)) for k in (EIN, WIN, SIN, NIN)) + ((0, 0, 1, 0, 0, 0),)
    return f | NO_SYNC | dict(recs=recs)
  for m, f in C2._group([relay(t) if t in tiles else None for t in range(16)]):
    recs = f.pop("recs", None)
    words = WN.encode_narrow_to_wide(**(f | dict(tile_mask=m, seq=e.seq)))
    e.tile(with_records(words, 543, list(recs)) if recs else words)
  e.sync(SC.sync_drain)
  zp, (s, _) = p.q[1], p.q
  cmin, cmax = EL.out_clamps(s, zp)
  per = []
  for t in range(16):
    if t not in tiles:
      per.append(None)
      continue
    rs, cs = gp.rows[t // 4], gp.cols[t % 4]
    ob, R0, R1, RB = _pool_out(gp, t, out_base, gc)
    Sy, blk = EL.maxpool_layout(p.k, p.k, p.stride, p.stride, rs.no, cs.no)
    f = EL.maxpool_op_fields(p.k, p.k, p.stride, p.stride, rs.no, cs.no, C, (R0, R1), 0, 0, ob, win, zp, cmin, cmax, Sy, blk)
    Wg = cdiv(min(C, 64), 4)
    f |= OP.ttu_fine("out", [4, 4 * R0, 4 * R1, 4 * Wg, 4 * RB], [Wg, cs.no, rs.no, cdiv(C, 64)])
    per.append(f | dict(mult_bits=f32_bits(EL.mul32(s, EL.recip32(s)))))
  for m, f in C2._group(per): e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))
  for t in tiles:                                                             # a pool op counts once per channel group
    st.cnt[t][AV] += cdiv(C, 64)
    st.lastG[t] = cdiv(C, 64)


# ***** the final FULLY_CONNECTED (batch 1): reshape the last map into the FC's input layout, then codegen/fc.py's stages *****
@dataclass(frozen=True)
class MapGeom:
  """the last map before the Linear: P x Q positions of C channels, tile row r holds rows [ir[r], ir[r] + pr[r]), tile column c the
  columns [jc[c], jc[c] + qc[c]), each tile its block densely at the map's narrow address"""
  P: int
  Q: int
  C: int
  ir: tuple
  pr: tuple
  jc: tuple
  qc: tuple
  @property
  def W(self) -> int: return self.C // 4                   # words per position
  def seg(self, r:int) -> int: return self.ir[r] * self.Q * self.C   # first byte of tile row r's rows in the flat vector

def map_geom(layers:list) -> MapGeom:
  last = layers[-2]
  g = pool_geom(layers[-3], last) if isinstance(last, Pool) else last.g
  C = layers[-3].Cout if isinstance(last, Pool) else last.Cout
  return MapGeom(g.OH, g.OW, C, tuple(s.o0 for s in g.rows), tuple(s.no for s in g.rows), tuple(s.o0 for s in g.cols),
                 tuple(s.no for s in g.cols))

def _fc_rows(m:MapGeom, fg) -> list[tuple[int, int, list, list]]:
  """per FC input row R (tiles 4R..4R+3 hold x[4bR, 4bR + 4b)): (first byte, end byte, heads, takes). The column-0 tile of map tile
  row R assembles a buffer of whole map rows: heads [(tile row r < R, its last n rows)] (when the FC row starts in the rows above),
  its own rows, takes [(tile row r > R, its first n rows)] (the rest up to the FC row's end)"""
  K, b, RB, out = m.P * m.Q * m.C, fg.b, m.Q * m.C, []
  for R in range(cdiv(fg.T_in, 4)):
    lo, hi = 4 * b * R, min(4 * b * (R + 1), K)
    own_end = m.seg(R) + m.pr[R] * RB
    heads, r = [], R - 1
    first = lo // RB                                   # the first map row the FC row needs
    while first < m.ir[R] - sum(n for _, n in heads):
      if r < 0: raise NotImplementedError("an FC input row starts before the map")
      n = min(m.pr[r], m.ir[R] - sum(n_ for _, n_ in heads) - first)
      if n: heads.insert(0, (r, n))
      r -= 1
    need, takes, r = cdiv(hi - own_end, RB) if hi > own_end else 0, [], R + 1
    while need:
      if r > 3: raise NotImplementedError("an FC input row reaches past the last map tile row")
      if m.pr[r]:
        takes.append((r, min(need, m.pr[r])))
        need -= takes[-1][1]
      r += 1
    out.append((lo, hi, heads, takes))
  return out

def _extra(row:tuple) -> int: return sum(n for _, n in row[3])     # map rows an FC row takes from the tile rows below it
def _head(row:tuple) -> int: return sum(n for _, n in row[2])      # map rows it takes from the tile rows above it

def _copy(tiles:int, src:int, dst:int, W:int, n:int, rows:int, in_s:list[int], out_s:list[int]) -> dict:
  """a reformat copy through the identity row (fused._local_copy): rows x n positions of W words; in_s / out_s = the strides of the
  in TTU's dims 2, 4, 6 and the out TTU's dims 1..3 (words)"""
  return dict(tile_mask=tiles, loop1=W - 1, loop2=n - 1, loop3=rows - 1, in_base=src, in_mode7=1, in_tflags=0x15,
              **ttu("in_", [1, 0, in_s[0], 0, in_s[1], 0, in_s[2]], [W, 1, n, 1, rows], 8), out_base=dst, out_mode7=3, out_tflags=3,
              **ttu("out_", [1] + out_s, [W, n, rows], 8), **ttu("par_", [1], [1, W, 1, n, rows], 8), par_mode7=1, psum_hmode=1,
              **ttu("psum_", [], [1, W, 1, n, rows], 8), psum_tflags=0x840, cfg2=6, dp_mode=4, reduce_mask=1, out_ch_last=3, out_ch=3,
              **OP.NO_REQUANT)

def _levels(W:int, n:int, rs:int, rows:int) -> list[tuple[int, int]]:
  return [(1, W)] + [(W, n)] * (n > 1) + [(rs, rows)] * (rows > 1)

def _gmesh(op:int, tiles:int, seq:int, p:str, addr:int, lv:list[tuple[int, int]], modes:bool=True) -> dict:
  f = {f"{p}addr": addr, f"{p}sdims": (1 << len(lv)) - 1, **ttu(p, [s for s, _ in lv], [c for _, c in lv], 4)}
  if modes: f |= dict(out_mode=(2 * len(lv) + 1) & 3, rsv475=(2 * len(lv) + 1) >> 2) if p == "o_" else dict(in_mode=2 * len(lv) + 1)
  return f

def _phase(e:Emitter, op:int, moves:list, R1:int, north:bool=False, fills:int=0):
  """a gather phase like codegen/fc.py's west phase: moves = [(k, sender tile, relay tiles, receiver tile, out fields, in fields, words)]
  per step k (ascending; north: descending, the farthest first); senders, relays (grouped by word count, codegen/fc.py's relay at R1),
  receivers, then a mesh fence per step on every tile but the step's relays. North (the FC reshape's moves over 2+ tile rows): the
  relays' IN record and the fence count the identity row's fill rounds (fills) instead of the +1, the OUT record has no +1, and
  after fills the fence covers all tiles (observed on two chains)"""
  cum, last = 0, 0xffff
  for k in sorted({m[0] for m in moves}, reverse=north):
    ms = [m for m in moves if m[0] == k]
    for msk, f in C2._group([next((m[4] for m in ms if m[1] == t), None) for t in range(16)]):
      e.tile(RM.encode_mesh(opcode=op, tile_mask=msk, seq=e.seq, **f))
    cls: dict[int, int] = {}                     # relay tiles by word count, in order of first appearance
    for m in ms:
      if m[2]: cls[m[6]] = cls.get(m[6], 0) | m[2]
    classes = list(cls.items())
    for w, rel in classes:
      words = FC.relay(op, rel, e.seq, R1, w, cum + 1)
      if north: words = RM.encode_mesh(**(RM.decode_mesh(words) | dict(s0_val=5 + 4 * (cum + fills), s1_val=-3 + 4 * cum)))
      e.tile(words)
    for msk, f in C2._group([next((m[5] for m in ms if m[3] == t), None) for t in range(16)]):
      e.tile(RM.encode_mesh(opcode=op, tile_mask=msk, seq=e.seq, **f))
    if classes:
      pk = cdiv(classes[0][0], 4)
      cum += pk
      last = 0xffff if north and fills else 0xffff & ~sum(rel for w, rel in classes if cdiv(w, 4) == pk)
    else: last = 0xffff
    e.sync(SC.sync_mesh, op, last, cum + fills if north else cum + 1)
  return last

def fc_tail(e:Emitter, layers:list, a:Alloc, first:int):
  """B = 1: the last map (P x Q x C, dense blocks at a.Y) becomes x[K] in the FC input layout (input tile t holds x[bt, bt+b) at X + bt):
  1. each map tile row gathers its rows on its column-0 tile (copy op + westward moves, A = row buffer)
  2. each column-0 tile receives the head of the next tile row's rows (northward)
  3. the column-0 tile of row R copies chunk 4R into place and sends chunks 4R+1.. east
  then codegen/fc.py's gather onto tile 0, its bias load, op and output block"""
  L = layers[-1]
  m = map_geom(layers)
  fg = FC.FCGeom(L.N, L.K)
  f = a.fc
  A, R1, X, Y, Rf = f["A"], f["R1"], f["X"], f["Y"], f["R"]
  if m.P * m.Q * m.C != L.K or m.C % 4: raise NotImplementedError("the Linear's K must be the map's P*Q*C, C a multiple of 4")
  rows = _fc_rows(m, fg)
  W, Qw = m.W, m.Q * m.W
  ident_n, ident_w = f["ident"]
  def prologue(n):     # the identity row again (its wide copy was overwritten): the narrowToWide waits for the n-th fill round
    ws = EL.ident_prologue(e.seq, 0xffff, ident_n, ident_w)
    fn = WN.decode_narrow_to_wide(ws[-1]) | dict(sync_en0=0, sync_f1=0, sync_en1=n & 1, sync_val=n >> 1)
    e.tile(*ws[:-1], WN.encode_narrow_to_wide(**fn))
    e.sync(SC.sync_drain)
  if f.get("prologue"): prologue(1)
  RB = m.Q * m.C                            # bytes of a map row
  head = [_head(rows[r]) if r < len(rows) else 0 for r in range(4)]
  bufrows = [head[r] + m.pr[r] + (_extra(rows[r]) if r < len(rows) else 0) for r in range(4)]
  own = [A + head[r] * RB for r in range(4)]  # where tile row r's own rows go in its buffer
  sbuf = [m.seg(r) - head[r] * RB for r in range(4)]   # the map byte at the buffer's start
  per = [_copy(0, a.Y, own[r], W, m.qc[0], m.pr[r], [W, m.qc[0] * W, m.pr[r] * m.qc[0] * W], [W, Qw, bufrows[r] * Qw]) | OP.wide("par", ident_w)
         if c == 0 and m.pr[r] and m.qc[0] else None for t in range(16) for r, c in [divmod(t, 4)]]
  for msk, fo in C2._group(per): e.tile(OP.encode_op(**(fo | dict(tile_mask=msk, seq=e.seq))))
  moves = []
  for r in range(4):
    for k in range(1, 4):
      if not (m.pr[r] and m.qc[k]): continue
      words = m.qc[k] * m.pr[r] * W
      moves.append((k, 4 * r + k, sum(1 << (4 * r + j) for j in range(1, k)), 4 * r,
                    _gmesh(0x16, 0, 0, "o_", a.Y, _levels(W, m.qc[k], m.qc[k] * W, m.pr[r])),
                    _gmesh(0x16, 0, 0, "i_", own[r] + m.jc[k] * m.C, _levels(W, m.qc[k], Qw, m.pr[r])), words))
  if moves: e.sync(SC.sync_drain, _phase(e, 0x16, moves, R1))
  e.sync(SC.sync_drain)
  # northward, per FC row from the last: rows from 2+ tile rows below first (the farthest first, relayed by the column-0 tiles
  # between, a mesh fence each), then the next tile row's; the relays' IN record and the fence count the identity row's fill rounds
  # (fills), the OUT record and the fence no +1, the fence covers the relays after fills (observed)
  for R in range(len(rows)):                # southward first: the tail of the tile rows above an FC row that starts there
    at = 0
    for r, n in rows[R][2]:
      if R - r > 1: raise NotImplementedError("an FC input row starts two or more tile rows above its own (not modelled)")
      lv = _levels(W, m.Q, Qw, n)
      e.tile(RM.encode_mesh(opcode=0x15, tile_mask=1 << (4 * r), seq=e.seq, **_gmesh(0x15, 0, 0, "o_", own[r] + (m.pr[r] - n) * RB, lv, False)))
      e.tile(RM.encode_mesh(opcode=0x15, tile_mask=1 << (4 * R), seq=e.seq, **_gmesh(0x15, 0, 0, "i_", A + at * RB, lv, False)))
      at += n
  fills, cum, got = int(bool(f.get("prologue"))), 0, [0] * 4
  for R in range(len(rows) - 1, -1, -1):
    at, mv = head[R] + m.pr[R], []
    for r, n in rows[R][3]:
      lv = _levels(W, m.Q, Qw, n)
      mv.append((r - R, r, _gmesh(0x17, 0, 0, "o_", own[r], lv, False), _gmesh(0x17, 0, 0, "i_", A + at * RB, lv, False), n * Qw))
      at += n
    for k, r, o, i, words in sorted(mv, key=lambda x: -x[0]):
      e.tile(RM.encode_mesh(opcode=0x17, tile_mask=1 << (4 * r), seq=e.seq, **o))
      if k > 1:
        rel = sum(1 << (4 * j) for j in range(R + 1, r))
        ws = RM.decode_mesh(FC.relay(0x17, rel, e.seq, R1, words, cum + 1)) | dict(s0_val=5 + 4 * (cum + fills), s1_val=-3 + 4 * cum)
        e.tile(RM.encode_mesh(**ws))
      e.tile(RM.encode_mesh(opcode=0x17, tile_mask=1 << (4 * R), seq=e.seq, **i))
      if k > 1:
        cum += cdiv(words, 4)
        got[R] += cdiv(words, 4)
        e.sync(SC.sync_mesh, 0x17, 0xffff if fills else 0xffff & ~rel, cum + fills)
      else: got[R] += 1
  e.sync(SC.sync_drain)
  if f.get("prologue"): prologue(1 + max(1, *got))       # (observed: 2 unless relayed packets arrived)
  K4w = cdiv(L.K, 4)
  for R, (lo, hi, _, _) in enumerate(rows):
    cw_ = min(fg.b, hi - lo) // 4
    e.tile(OP.encode_op(**(_copy(1 << (4 * R), A + lo - sbuf[R], X + lo, cw_, 1, 1, [(hi - lo) // 4] * 3, [K4w] * 3) | OP.wide("par", ident_w)
                           | dict(seq=e.seq))))
  moves = []
  for R, (lo, hi, _, _) in enumerate(rows):
    for j in range(1, 4):
      c0 = lo + j * fg.b
      if c0 >= hi: continue
      words = cdiv(min(fg.b, hi - c0), 4)
      lv = [(1, words)]
      moves.append((j, 4 * R, sum(1 << (4 * R + i) for i in range(1, j)), 4 * R + j, _gmesh(0x18, 0, 0, "o_", A + c0 - sbuf[R], lv),
                    _gmesh(0x18, 0, 0, "i_", X + c0, lv), words))
  last = _phase(e, 0x18, moves, R1)
  e.sync(SC.sync_drain, last)
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_reset17)
  FC.gather(e, fg, X, Rf, lambda t: 1 << t)
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_reset17)
  FC.ops_block(e, fg, FC.Place(), X, Y, a.ptiles[-1][1], dict(L.q), tail=False)
  e.sync(SC.broadcast_wait, first)
  e.sync(SC.signal_fence)
  FC.output_block(e, fg, FC.Place(), Y)
  e.sync(SC.epilogue)

# ***** memory allocation *****
@dataclass
class Alloc:
  """where everything goes. Narrow (bytes, every tile): stg / ident = the first conv's input staging and identity row, X[i] = conv i's
  input block, P[i] = the window block of the pool after conv i, Y = the last conv's dense outputs. Wide (64-byte units): wconv[i] =
  conv i's parameter FIFO / partial sums / bias, win[i] = the pool window after conv i. ptiles[i] = (tile, offset in 64-byte units)
  of conv i's parameters (the Linear's last)."""
  stg: int
  ident: int
  X: list
  P: dict
  Y: int
  wconv: list
  win: dict
  ptiles: list
  fc: dict = field(default_factory=dict)
  const: int|None = None           # wide address of the input stage's identity row (default: conv2d's)
  in_fifo: int|None = None         # wide address of the input stage's ring FIFO (default: conv2d's)

# ***** the default allocation: edgetpu_compiler's choices (observed rules, see docs/isa/codegen_chain.md) *****
# Time is counted in steps: 0 = the input stage, then one step per conv and one per max pool, then the FC's reshape (r) and its
# steps r+1.. . A buffer is live over [lo, hi]; two buffers may share memory when their intervals don't meet.
def _meets(a:tuple, b:tuple) -> bool: return a[2] <= b[3] and b[2] <= a[3]

def _relay_extent(gp:Conv2DGeom, C:int) -> int:
  """bytes of a pool's window block in narrow memory as the relays read it: the last quad of positions and the last channel group
  may reach past the block"""
  w, G = gp.N4 // 4, cdiv(C, 64)
  Wg = 16 if G > 1 else cdiv(C, 4)
  ext = 0
  for t in range(16):
    rs, cs = gp.rows[t // 4], gp.cols[t % 4]
    if not (rs.no and cs.no): continue
    last = (rs.wo * cs.nb + cs.wo) * w + 3 * w + Wg - 1 + (cdiv(cs.nw, 4) - 1) * 4 * w + (rs.nw - 1) * cs.nb * w + (G - 1) * 16
    ext = max(ext, 4 * (last + 1), gp.win_bytes(t // 4, t % 4))
  return ext

def window_units(c:Conv, p:Pool) -> int:
  """wide units the compiler reserves for a pool's window: the largest narrow block (rows x columns, not just the window) transposed
  (EL.maxpool_layout's 4 * rows * ceil(cols / 4)), per 64-channel group"""
  gp = pool_geom(c, p)
  return 4 * max(s.nb for s in gp.rows) * cdiv(max(s.nb for s in gp.cols), 4) * cdiv(c.Cout, 64)

def _narrow_buffers(layers:list) -> list[tuple]:
  """(name, bytes, lo, hi[, bytes the objective counts]): Stg / C (the input stage's staging slots and identity row), X<j> (conv j's
  input block), P<i> (the window block of the pool after layer i: its relays' reach, the block's own size in the objective), Y (the
  last map); with a Linear: A (the FC reshape's row buffer), R1 (its relay buffer) and the FC's x / y / relay buffers FX, FY, FR"""
  g0 = layers[0].g
  out = [("Stg", 2 * g0.slot, 0, 0), ("C", g0.Cs, 0, 0)]
  convs = [i for i, l in enumerate(layers) if isinstance(l, Conv)]
  step, cur = 0, ("X0", g0.Xs, 0)
  for j, i in enumerate(convs):
    c, g = layers[i], layers[i].g
    step += 1
    out.append((cur[0], cur[1], cur[2], step))
    nxt = layers[i + 1] if i + 1 < len(layers) else None
    if isinstance(nxt, Pool):
      gp = pool_geom(c, nxt)
      out.append((f"P{i}", _relay_extent(gp, c.Cout), step, step + 1, gp.Xs))
      step += 1
      after = layers[i + 2] if i + 2 < len(layers) else None
      cur = (f"X{j + 1}", after.g.Xs, step) if isinstance(after, Conv) else ("Y", max(gp.out_tile(t) for t in range(16)), step)
    elif isinstance(nxt, Conv): cur = (f"X{j + 1}", nxt.g.Xs, step)
    else: cur = ("Y", g.Ys, step)
  r = step + 1
  out.append((cur[0], cur[1], cur[2], r))
  if isinstance(layers[-1], Linear):
    L, m = layers[-1], map_geom(layers)
    rows = _fc_rows(m, FC.FCGeom(L.N, L.K))
    A = max((_head(rows[R]) + m.pr[R] + _extra(rows[R])) * m.Q * m.C for R in range(len(rows)))
    out += [("A", A, r, r), ("R1", 32, r, r), ("FX", round_up(L.K, 4), r, r + 2), ("FY", round_up(L.N, 4), r + 2, r + 3), ("FR", 32, r + 1, r + 1)]
  return out

def fc_narrow(N:int, K:int) -> tuple[int, int, int]:
  """(X, Y, R) of the final Linear in a chain: fc.narrow_layout's x-first case [x | y | R] whatever K (a standalone FC puts y first
  for K <= 188)"""
  if not (K <= 188 or round_up(N, 4) + 32 >= 256 * cdiv(K, 256) + 512): return FC.narrow_layout(N, K)
  n4, Rsz = round_up(N, 4), 32 if FC.FCGeom(N, K).T_in > 1 else 0
  base = round_up(K, 4)
  return (0, base + Rsz, base) if N <= 4 else (0, base, base + n4)

def _narrow_fixed(layers:list, bufs:list) -> dict:
  """the first conv's input block X0, staging slots Stg and identity row C: conv2d.narrow_layout's cases with Ys = the size of
  the first conv's output buffer; the Linear's x / y / relay: fc_narrow"""
  g0 = layers[0].g
  ys = next(b[1] for b in bufs if b[2] == 1 and b[0] not in ("Stg", "C", "X0"))
  Xs, R2, Cs = g0.Xs, 2 * g0.slot, g0.Cs
  if ys >= R2 + Cs and ys > Xs: fx = dict(Stg=0, C=R2, X0=ys)
  elif 2 * Xs >= g0.slot: fx = dict(X0=0, Stg=Xs, C=Xs + R2)
  elif ys > R2 or 4 * Xs < Cs: fx = dict(Stg=0, C=R2, X0=max(ys, R2 + Cs))
  else: fx = dict(Stg=0, X0=R2, C=R2 + Xs)
  if isinstance(layers[-1], Linear):
    X, Y, R = fc_narrow(layers[-1].N, layers[-1].K)
    fx |= dict(FX=X, FY=Y, FR=R)
  return fx

def _fits(bufs:list, asg:dict, i:int, a:int) -> bool:
  return all(not _meets(bufs[i], bufs[j]) or a + bufs[i][1] <= asg[j] or asg[j] + bufs[j][1] <= a for j in asg)

def _first_fit(bufs:list, asg:dict, i:int) -> int:
  """the lowest address where buffer i overlaps no placed buffer it is live with"""
  return next(a for a in sorted({0} | {asg[j] + bufs[j][1] for j in asg if _meets(bufs[i], bufs[j])}) if _fits(bufs, asg, i, a))

def _search(bufs:list, fixed:dict) -> dict:
  """the lowest peak, then the lowest sum of size x address (a pool window block counted with its own size, not its relays' reach),
  over first-fit placements in every order (branch and bound; the first optimum in buffer order)"""
  idx = {b[0]: i for i, b in enumerate(bufs)}
  base = {idx[n]: a for n, a in fixed.items() if n in idx}
  best: list = [None, None]
  def rec(rest:list, asg:dict, peak:int, s:int):
    if best[0] is not None and (peak, s) >= best[0]: return
    if not rest:
      best[0], best[1] = (peak, s), dict(asg)
      return
    for i in rest:
      asg[i] = a = _first_fit(bufs, asg, i)
      rec([j for j in rest if j != i], asg, max(peak, a + bufs[i][1]), s + bufs[i][4 if len(bufs[i]) > 4 else 1] * a)
      del asg[i]
  rec([i for i in range(len(bufs)) if i not in base], dict(base), max((base[i] + bufs[i][1] for i in base), default=0), 0)
  return {bufs[i][0]: a for i, a in best[1].items()}

def _greedy(bufs:list, fixed:dict) -> dict:
  """largest first, each at its first-fit address (equal sizes: the one that fits lowest first, then buffer order)"""
  idx = {b[0]: i for i, b in enumerate(bufs)}
  asg = {idx[n]: a for n, a in fixed.items() if n in idx}
  rest = [i for i in range(len(bufs)) if i not in asg]
  while rest:
    i = min(rest, key=lambda i: (-bufs[i][1], _first_fit(bufs, asg, i), i))
    asg[i] = _first_fit(bufs, asg, i)
    rest.remove(i)
  return {bufs[i][0]: a for i, a in asg.items()}

def narrow_alloc(layers:list) -> dict:
  """{buffer name: narrow byte address}; with a Linear also 'ident' (the identity row the FC reshape rebuilds, if it does). When the
  first conv's output buffer does not end up at address 0, the input stage takes conv2d's [X0 | Stg | C] case instead"""
  bufs = _narrow_buffers(layers)
  fixed = _narrow_fixed(layers, bufs)
  out = _narrow_alloc(layers, bufs, fixed)
  y1 = next(b[0] for b in bufs if b[2] == 1 and b[0] not in ("Stg", "C", "X0"))
  if out[y1] != 0 and fixed["X0"] != 0:
    g0 = layers[0].g
    out = _narrow_alloc(layers, bufs, fixed | dict(X0=0, Stg=g0.Xs, C=g0.Xs + 2 * g0.slot))
  return out

def _narrow_alloc(layers:list, bufs:list, fixed:dict) -> dict:
  if not isinstance(layers[-1], Linear): return _search(bufs, fixed)
  # FC chains: the other buffers largest first; then from the FC's y address on the reshape's row buffer A and the last map Y (A
  # first when an FC row takes rows from the tile rows below and none from above, else Y first; Y only where it is free, else after
  # A, else at its first fit), the relay buffer R1 right after them; the rebuilt identity row 32 bytes after R1
  L, m = layers[-1], map_geom(layers)
  idx = {b[0]: i for i, b in enumerate(bufs)}
  out = _greedy([b for b in bufs if b[0] not in ("A", "Y", "R1")], fixed)
  asg = {idx[n]: a for n, a in out.items()}
  fr = _fc_rows(m, FC.FCGeom(L.N, L.K))
  extra = any(_extra(r) for r in fr) and not any(_head(r) for r in fr)    # (observed)
  a, pending, deferred = fixed["FY"], ["A", "Y"] if extra else ["Y", "A"], False
  while pending:
    n = pending.pop(0)
    if n == "Y" and not _fits(bufs, asg, idx["Y"], a):
      if pending: pending.append("Y")
      else: deferred = True
      continue
    asg[idx[n]] = a
    a += bufs[idx[n]][1]
  asg[idx["R1"]] = a
  if deferred: asg[idx["Y"]] = _first_fit(bufs, asg, idx["Y"])
  out = {bufs[i][0]: v for i, v in asg.items()}
  out["ident"] = out["R1"] + 32
  return out

PARAM_TILES = (0, 1, 3, 7, 15, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14)   # the order the compiler hands out parameter tiles

def param_tiles(layers:list) -> list[tuple[int, int]]:
  """(tile, offset) of each Conv's (then the Linear's) cached parameters, one layer per tile in PARAM_TILES order, offset 0: the
  Linear first (tile 0: its op runs there), then convs with at most 4 outputs, then the convs by falling parameter rows (blocks:
  the bias row and one row per reduction word, per 64-output group), ties in layer order (observed)"""
  ps = [l for l in layers if isinstance(l, (Conv, Linear))]
  if len(ps) > 16: raise NotImplementedError("more than 16 layers with parameters (tiles shared)")
  tiles = list(PARAM_TILES)
  out: list = [None] * len(ps)
  if isinstance(ps[-1], Linear): out[-1] = (tiles.pop(0), 0)
  convs = sorted([k for k, l in enumerate(ps) if isinstance(l, Conv)], key=lambda k: (ps[k].Cout > 4, -ps[k].g.blocks, k))
  for k in convs: out[k] = (tiles.pop(0), 0)
  return out

def reuses_ident(layers:list) -> bool:
  """the FC reshape copies through the input stage's identity row when that is a single 256-byte row (it then stays live from the
  input stage to the reshape); otherwise the reshape rebuilds an identity row at WIDE_TOP - 4"""
  return isinstance(layers[-1], Linear) and cdiv(layers[0].g.Cs, 256) == 1

IDENT_TOP, IDENT_LOW = 128, 48   # a reused identity row goes first unless a window's key exceeds IDENT_TOP; else its key (wide_alloc)

def wide_alloc(layers:list) -> dict:
  """wide units (64 bytes) of the input stage's ring FIFO and identity row, every conv's regions and every pool's window. Greedy:
  buffers by falling key (their size; a pool window's key is 4x its size), ties in conv2d.regions' order (window, psum, bias,
  FIFO; identity row before ring FIFO), each at the highest address below WIDE_TOP clear of the buffers it is live with. A window
  is live during its conv and its pool. A reused identity row (reuses_ident) is live during the whole program: with 3+ conv / pool
  layers before the Linear it goes first, to the very top, unless a pool window's key exceeds IDENT_TOP; otherwise its key is
  IDENT_LOW (fitted on 14 chains; the compiler's actual criterion is unknown)."""
  g0 = layers[0].g
  convs = [i for i, l in enumerate(layers) if isinstance(l, Conv)]
  end = 2 * len(layers) + 4
  fifo, cs = 8 * g0.c_in, 4 * cdiv(g0.Cs, 256)
  reuse = reuses_ident(layers)
  bufs = [("in_fifo", fifo, 0, 0, fifo, 1)]
  step = 0
  for j, i in enumerate(convs):
    g = layers[i].g
    step += 1
    psum = 0 if g.pos_outer else 4 * g.ppt_max     # (allocated even when one position per tile leaves it unused)
    for n, s_, tie in (("psum", psum, 0), ("bias", 8 if g.fifo else 4, 1), ("par_fifo", 8 * g.a if g.fifo else 4 * g.a, 2)):
      if s_: bufs.append(((n, j), s_, step, step, s_, tie))
    if i + 1 < len(layers) and isinstance(layers[i + 1], Pool):
      w = window_units(layers[i], layers[i + 1])
      bufs.append((("win", i), w, step, step + 1, 4 * w, -1))
      step += 1
  def greedy(bufs:list) -> dict:
    addr: dict = {}
    for k in sorted(range(len(bufs)), key=lambda k: (-bufs[k][4], bufs[k][5], k)):
      s_ = bufs[k][1]
      live = [(addr[q], bufs[q][1]) for q in addr if _meets(bufs[k], bufs[q])]
      addr[k] = next(a for a in sorted({WIDE_TOP - s_} | {b - s_ for b, _ in live}, reverse=True) if all(a + s_ <= b or b + z <= a for b, z in live))
    return {bufs[k][0]: a for k, a in addr.items()}
  if reuse:
    top = len(layers) - 1 >= 3 and not any(b[0][0] == "win" and b[4] > IDENT_TOP for b in bufs)
    bufs.append(("const", cs, 0, end, float("inf") if top else IDENT_LOW, 0))
  else: bufs.append(("const", cs, 0, 0, cs, 0))
  named = greedy(bufs)
  wconv = [dict(par_fifo=named[("par_fifo", j)], bias=named[("bias", j)], psum=named.get(("psum", j), 0)) for j in range(len(convs))]
  win = {i: named[("win", i)] for i in convs if ("win", i) in named}
  return dict(in_fifo=named["in_fifo"], const=named["const"], wconv=wconv, win=win, reuse=reuse)

def default_alloc(layers:list) -> Alloc:
  """the compiler's allocation for the chain (observed rules; see the docs for what is known to differ)"""
  n, w = narrow_alloc(layers), wide_alloc(layers)
  convs = [i for i, l in enumerate(layers) if isinstance(l, Conv)]
  P = {i: n[f"P{i}"] for i in convs if f"P{i}" in n}
  fc = {}
  if isinstance(layers[-1], Linear):
    _, _, R = fc_narrow(layers[-1].N, layers[-1].K)
    fc = dict(A=n["A"], R1=n["R1"], X=n["FX"], Y=n["FY"], R=R, prologue=not w["reuse"],
              ident=(0, w["const"]) if w["reuse"] else (n["ident"], WIDE_TOP - 4))
  return Alloc(stg=n["Stg"], ident=n["C"], X=[n[f"X{j}"] for j in range(len(convs))], P=P, Y=n["Y"], wconv=w["wconv"], win=w["win"],
               ptiles=param_tiles(layers), fc=fc, const=w["const"], in_fifo=w["in_fifo"])

# ***** the program *****
def gathered_line(g:Conv2DGeom) -> bool:
  """the conv's outputs lie on a single tile row (column) while its input spans 2 or 3 of them (the halo gathers it there)"""
  return any(sum(1 for s in sp if s.no) == 1 and 1 < sum(1 for s in sp if s.ni) < 4 for sp in (g.rows, g.cols))

def resets_before(layers:list, i:int) -> int:
  """reset17 fences between conv i's halo and its stage: one after a pool (the convs after a pool start their counters from zero),
  one more when the halo gathered the input onto a single tile row or column (gathered_line; observed rules)"""
  return int(i > 0 and isinstance(layers[i - 1], Pool)) + int(i > 0 and gathered_line(layers[i].g))

def inrec_after(c:Conv):
  """the inbound halves of the moves north / west (data from the south / east neighbour) also wait for the op on tiles whose conv c
  computed fewer outputs than the busiest tile (observed rule; not after pools)"""
  g = c.g
  ppt = [g.out_tile(t) for t in range(16)]
  return lambda t, op: op in (MESH_N, MESH_W) and 0 < ppt[t] < max(ppt)

def gen_execution(layers:list, a:Alloc) -> bytes:
  convs = [i for i, l in enumerate(layers) if isinstance(l, Conv)]
  c0 = layers[0]
  g0 = c0.g
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  wl0 = C2.wide_layout(g0) | {k: v for k, v in (("const", a.const), ("in_fifo", a.in_fifo)) if v is not None}
  C2.input_stage(e, g0, dict(Stg=a.stg, X=a.X[0], C=a.ident), wl0)
  e.sync(SC.sync_reset17)
  st, first = State(), 0
  for j, i in enumerate(convs):
    c, g = layers[i], layers[i].g
    nxt = layers[i + 1] if i + 1 < len(layers) else None
    prev = layers[i - 1] if i else None
    halo(e, g, a.X[j], st, inrec=inrec_after(prev) if isinstance(prev, Conv) else None)
    for _ in range(resets_before(layers, i)):
      e.sync(SC.sync_reset17)
      st.reset()
    if isinstance(nxt, Conv):
      def out(t, gn=nxt.g, X=a.X[j + 1]): return into_block(gn, X, t, g.rows[t // 4].no, g.cols[t % 4].no)
    elif isinstance(nxt, Pool):
      def out(t, gp=pool_geom(c, nxt), P=a.P[i]): return into_block(gp, P, t, g.rows[t // 4].no, g.cols[t % 4].no)
    else:
      def out(t): return None
    first = conv_stage(e, c, a.X[j], a.Y, out, a.wconv[j], [(a.ptiles[j][0], a.ptiles[j][1], g.blocks)], first, st)
    if isinstance(nxt, Pool):
      after = layers[i + 2] if i + 2 < len(layers) else None
      gc = after.g if isinstance(after, Conv) else None
      pool_stage(e, c, nxt, a.P[i], a.win[i], a.X[j + 1] if gc else a.Y, gc, st, records=c.Cout % 64 == 0)
  if isinstance(layers[-1], Linear):
    for _ in range(fc_resets(layers, a)): e.sync(SC.sync_reset17)
    fc_tail(e, layers, a, first)
    return e.program()
  gl = layers[convs[-1]].g
  C2.output_stage(e, gl, a.Y, first)
  return e.program()

def fc_resets(layers:list, a:Alloc) -> int:
  """reset17 fences before the FC reshape: one, one more after a pool, one more when the last map leaves a tile row or column empty
  (observed)"""
  m = map_geom(layers)
  return 1 + isinstance(layers[-2], Pool) + (0 in m.pr or 0 in m.qc)

def gen_caching(layers:list, a:Alloc) -> bytes:
  """PARAMETER_CACHING: per conv (then the Linear) its ringConsumer into its parameter tile, then its infeed of the blob piece"""
  parts, base = [], 0
  ps = [l for l in layers if isinstance(l, (Conv, Linear))]
  for l, (tile, off) in zip(ps, a.ptiles):
    if isinstance(l, Conv):
      g = l.g
      rc = dict(tile_mask=1 << tile, addr=2 + off, sdims=int(g.blocks > 1), **ttu("", [1], [g.blocks], 4))
      parts.append([(rc, base, g.param_bytes, 4 * g.cg // 64)])
      base += g.param_bytes
    else:
      fg = FC.FCGeom(l.N, l.K)
      assert tile == 0 and fg.T == 1, "the final linear layer: one compute tile, tile 0"
      parts.append(FC.caching_part(fg, FC.Place(), off, base))
      base += fg.param_bytes
  return caching_program(base, parts)

def param_bytes(layers:list) -> int:
  return sum(l.g.param_bytes if isinstance(l, Conv) else FC.FCGeom(l.N, l.K).param_bytes for l in layers if isinstance(l, (Conv, Linear)))

def chain_blob(layers:list, params:list) -> bytes:
  """the parameter blob: per conv conv2d.conv2d_blob, then the Linear's FULLY_CONNECTED rows (per 16 / 64 outputs an int32 bias row,
  then [K/4][outputs][4]); params = [(weights uint8, int32 bias or None)] per Conv ([Cout, kh, kw, Cin]) and Linear ([N, K], K in
  (y, x, c) order), the weight zero points come from the layers' quantization"""
  import numpy as np
  out = []
  for l, (w, b) in zip([l for l in layers if isinstance(l, (Conv, Linear))], params):
    if isinstance(l, Conv): out.append(C2.conv2d_blob(w, l.q["w_zp"], b, l.H, l.W, l.stride, "VALID"))
    else: out.append(C2.conv2d_blob(np.asarray(w, np.uint8).reshape(l.N, 1, 1, l.K), l.q["w_zp"], b))   # the same rows as a 1x1 conv
  return b"".join(out)

# ***** the public API *****
def check_chain(layers:list):
  """raises NotImplementedError for what the generator does not cover (see the docs for the reasons)"""
  if not layers or not isinstance(layers[0], Conv): raise NotImplementedError("the chain starts with a Conv")
  for i, l in enumerate(layers):
    prev = layers[i - 1] if i else None
    if isinstance(l, Conv):
      if l.stride != 1: raise NotImplementedError("strided convolutions in a chain")
      if l.Cout <= 4: raise NotImplementedError("convs with at most 4 outputs (the compiler's 4-output parameter groups are not modelled)")
      if l.kh % 2 == 0 or l.kw % 2 == 0: raise NotImplementedError("even kernel sizes (conv2d's mode rule for them is not modelled)")
      if max(l.kh, l.kw) > 5: raise NotImplementedError("kernels larger than 5x5 (conv2d's rules and blob order are fitted on k <= 5)")
      if prev is not None:
        g, src = l.g, prev.g if isinstance(prev, Conv) else pool_geom(layers[i - 2], prev)
        if (g.H, g.W, g.Cin) != (src.OH, src.OW, layers[i - 2].Cout if isinstance(prev, Pool) else prev.Cout):
          raise ValueError(f"layer {i}: input {g.H}x{g.W}x{g.Cin} does not match the previous layer's output")
    elif isinstance(l, Pool):
      if not isinstance(prev, Conv): raise NotImplementedError("a max pool right after a conv only")
      if (l.k, l.stride) != (2, 2): raise NotImplementedError("max pools 2x2 / stride 2 only")
      if i == len(layers) - 1: raise NotImplementedError("a chain ending with a max pool")
    elif isinstance(l, Linear):
      if i != len(layers) - 1 or i < 1: raise NotImplementedError("the Linear is the last layer, after a conv or a pool")
      if FC.FCGeom(l.N, l.K).T != 1: raise NotImplementedError("a final Linear with more than 64 outputs")
      m = map_geom(layers)
      if m.P * m.Q * m.C != l.K: raise ValueError(f"Linear K={l.K} != the last map's {m.P}x{m.Q}x{m.C}")
      _fc_rows(m, FC.FCGeom(l.N, l.K))
    else: raise TypeError(f"layer {i}: {type(l).__name__}")
  if len([l for l in layers if isinstance(l, (Conv, Linear))]) > 16: raise NotImplementedError("more than 16 layers with parameters")

def gen_chain(layers:list, alloc:Alloc|None=None) -> tuple[bytes, bytes]:
  """(PARAMETER_CACHING, EXECUTION_ONLY) of the whole chain as one program pair, byte-identical to edgetpu_compiler's for the same
  model (alloc: the memory placement, default default_alloc = the compiler's choices). Raises NotImplementedError outside the
  covered set (check_chain) or when the program would need more than one bitstream."""
  check_chain(layers)
  a = default_alloc(layers) if alloc is None else alloc
  caching, exe = gen_caching(layers, a), gen_execution(layers, a)
  if len(exe) // 16 > C2.MAX_WORDS: raise NotImplementedError("more than one bitstream of 16384 words (not modelled)")
  return caching, exe

def chain_io(layers:list, images:int=1) -> dict:
  """the host contract. Input: input_bytes = x[H][W][Cin] uint8 (NHWC; `images` images stacked: [images*H][W][Cin]), padded with
  zeros to a multiple of 8 bytes. Output: output_bytes; with a final Linear y[N] in the first N bytes (output_layout: 1 position on tile 0), else
  the last conv's [OH][OW][Cout] tiled like conv2d_io (output (y, x) channel c at byte tile_byte_offset[y_tile[y] + x_tile[x]] +
  y_local_y_offset[y] * x_local_row_size[x] + x_local_byte_offset[x] + c; coral.fused.relayout reads it); image_rows[b] = the
  output row holding image b's result when images > 1 (the Linear became a conv over each image's map). param_bytes: chain_blob's"""
  g0, L = layers[0].g, layers[-1]
  out = dict(input_bytes=g0.S, input_shape=(g0.H, g0.W, g0.Cin), param_bytes=param_bytes(layers))
  if isinstance(L, Linear):
    n4 = round_up(L.N, 4)                          # one 1 x 1 position of N channels on tile 0
    return out | dict(output_bytes=FC.FCGeom(L.N, L.K).out_bytes, output_shape=(L.N,),
                      output_layout=dict(y_tile=[0], x_tile=[0], tile_byte_offset=[0] + [n4] * 15, x_local_byte_offset=[0],
                                         y_local_y_offset=[0], x_local_row_size=[n4]))
  g = L.g
  io = C2.conv2d_io(g.H, g.W, g.Cin, g.Cout, g.kh, g.kw, g.stride, "VALID")
  out |= dict(output_bytes=io["output_bytes"], output_shape=(g.OH, g.OW, g.Cout), output_layout=io["output_layout"])
  if images > 1:
    pitch = g0.H // images
    for l in layers[:-1]:
      if isinstance(l, Pool): pitch //= l.stride
    out["image_rows"] = [b * pitch for b in range(images)]
  return out

def chain_layers(H:int, W:int, Cin:int, specs:list[dict], images:int=1) -> list:
  """the layer description from per-layer specs, for an H x W x Cin image (`images` of them stacked into one tall image, as
  tools/export.py does: VALID convs and 2x2 pools never mix two images' rows when H is a multiple of 2^pools):
    {"conv": dict(Cout=, k= (or kh=, kw=), x_q=(scale, zp), w_q=(scale, zp), y_q=(scale, zp), act=0 none / 1 relu)}
    {"pool": dict(k=2, stride=2)}                     (keeps the conv's quantization)
    {"linear": dict(N=, x_q=, w_q=, y_q=, act=0)}     (last; with images > 1 a conv over each image's last map)
  Scales are the TFLite tensors' float32 scales: the requant multiplier is f32(f32(s_x*s_w) * f32(1/s_y)) and the clamps come from
  them (conv2d.conv2d_quant), as edgetpu_compiler computes them."""
  out, h, w, c, yq = [], H * images, W, Cin, None
  for i, spec in enumerate(specs):
    (kind, p), = spec.items()
    if kind == "pool":
      out.append(Pool(p.get("k", 2), p.get("stride", 2), yq))
      h, w = (h - out[-1].k) // out[-1].stride + 1, (w - out[-1].k) // out[-1].stride + 1
      continue
    q = C2.conv2d_quant(p["x_q"], p["w_q"], p["y_q"], p.get("act", 0))
    yq = p["y_q"]
    if kind == "conv":
      kh, kw = p.get("kh", p.get("k")), p.get("kw", p.get("k"))
      out.append(Conv(h, w, c, p["Cout"], kh, kw, p.get("stride", 1), q))
      h, w, c = (h - kh) // out[-1].stride + 1, (w - kw) // out[-1].stride + 1, p["Cout"]
    elif kind == "linear":
      assert i == len(specs) - 1, "the linear layer comes last"
      if images == 1: out.append(Linear(p["N"], h * w * c, q))
      else:                                         # image b's map: rows [b*pitch, b*pitch + rows) of the tall map
        pitch = H
        for s_ in specs[:-1]:
          if "pool" in s_: pitch //= s_["pool"].get("stride", 2)
        out.append(Conv(h, w, c, p["N"], h - (images - 1) * pitch, w, 1, q))
    else: raise ValueError(f"unknown layer kind {kind}")
  return out

def chain_params(layers:list, weights:list) -> list:
  """chain_blob's params from NCHW-style weights: per Conv w[Cout, Cin, kh, kw] -> [Cout, kh, kw, Cin]; per Linear w[N, K] with K
  in the flattened NCHW (c, y, x) order of the map before it -> (y, x, c) order (or, after a tall stack, the conv over the map
  [N, rows, cols, C]); biases unchanged"""
  import numpy as np
  out, ps = [], [l for l in layers if isinstance(l, (Conv, Linear))]
  for k, (l, (w, b)) in enumerate(zip(ps, weights)):
    w = np.asarray(w, np.uint8)
    if isinstance(l, Linear) or (k == len(ps) - 1 and w.ndim == 2):
      m = map_geom(layers) if isinstance(l, Linear) else None
      P, Q, C = (m.P, m.Q, m.C) if m else (l.kh, l.kw, l.Cin)
      w = w.reshape(-1, C, P, Q).transpose(0, 2, 3, 1)
      out.append((w.reshape(w.shape[0], -1) if isinstance(l, Linear) else w, b))
    else: out.append((w.transpose(0, 2, 3, 1), b))
  return out

if __name__ == "__main__": run_test("test_codegen_chain")
