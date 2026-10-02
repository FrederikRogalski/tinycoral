# FULLY_CONNECTED code generation, y[N] = W[N,K] x[K] (uint8), byte-identical to edgetpu_compiler (docs/isa/codegen.md):
#   caching, execution = gen_fc(N, K)
#   caching, execution = gen_fc(N, K, param_offset=4096, tile_shift=4)
# `python -m coral.codegen.fc` runs the acceptance test (test/test_codegen.py: the 355-entry FC table and fresh compiles).
from __future__ import annotations
from dataclasses import dataclass
from coral.isa import cdiv, round_up, ttu, op as OP, wide_narrow as WN, ring_mesh as RM, scalar as SC
from coral.codegen import Emitter, caching_program, run_test, WIDE_TOP, WIDE_OUT_FIFO, REF_QUANT

WIDE_FWD_FIFO = 0x1f70     # 64-byte units: ring broadcast of x from the gather tile to the other compute tiles
# ring sync-record ids op<<5 | tile SyncCounter: NARROW_TO_WIDE/op1, WIDE_TO_NARROW/op1, WIDE_TO_NARROW/op0
R_N2W, R_W2N_1, R_W2N_0 = (1 << 5) | 13, (1 << 5) | 11, 11

@dataclass(frozen=True)
class FCGeom:
  """tile geometry of y[N] = W[N,K] x[K] as edgetpu_compiler lays it out. bias=False: no bias rows (the compiler's form for N=1
  with an all-zero bias)"""
  N: int
  K: int
  bias: bool = True
  @property
  def G(self) -> int: return cdiv(self.N, 64)                    # 64-output groups
  @property
  def P(self) -> int: return cdiv(self.G, 16)                    # groups ("passes") per compute tile; the last tile may have fewer
  @property
  def T(self) -> int: return cdiv(self.G, self.P)                # compute tiles
  def gt(self, t:int) -> int: return min(self.P, self.G - self.P * t)               # groups of compute tile t
  def n_out(self, t:int) -> int: return min(64 * self.P, self.N - 64 * self.P * t)  # outputs of compute tile t
  @property
  def S(self) -> int: return 8 * cdiv(self.K, 8)                 # input DMA bytes
  @property
  def out_bytes(self) -> int: return 8 * cdiv(self.N, 8)         # output DMA bytes
  @property
  def b(self) -> int: return 64 * cdiv(self.S, 1024)             # input bytes per input tile
  @property
  def T_in(self) -> int: return cdiv(self.S, self.b)             # input tiles (x arrives in pieces of b bytes)
  def piece(self, t:int) -> int: return min(self.b, self.K - self.b * t)            # bytes of x on input tile t
  @property
  def kw(self) -> int: return cdiv(self.K, 4)                    # 4-byte input words = reduction steps
  @property
  def group_out(self) -> int: return round_up(self.N, 16) if self.N <= 64 else 64  # outputs per parameter group
  @property
  def group_bytes(self) -> int: return 4 * self.group_out * self.bias + self.group_out * 4 * self.kw   # int32 bias + [K/4][g][4] weights
  @property
  def param_bytes(self) -> int: return self.G * self.group_bytes
  def bias_rows(self, t:int) -> int: return 4 * self.gt(t) * self.bias      # 64-byte units of bias in front of tile t's weights

def narrow_layout(N:int, K:int) -> tuple[int, int, int]:
  """edgetpu_compiler's narrow placement (fitted on ~1450 compiles) -> byte addresses (X, Y, R) of x, y and the 32-byte mesh relay
  buffer R (only when x is gathered from several input tiles). Sizes are rounded up to 4 bytes; with M = 256*ceil(K/256) + 512:
    x first  [x | y | R]          unless K <= 188 or N4 + 32 >= M      (N4 = N rounded up to 4)
    y first  [y | R | .. | x]     x at max(M, N4 + |R|)  when N4 < M
             [y | x | R]          when N4 >= M
  A one-word y (N <= 4) goes after R instead of before it."""
  M, Rsz, n4 = 256 * cdiv(K, 256) + 512, 32 if FCGeom(N, K).T_in > 1 else 0, round_up(N, 4)
  def yR(base): return (base + Rsz, base) if N <= 4 else (base, base + n4)   # (Y, R)
  if not (K <= 188 or n4 + 32 >= M): return (0, *yR(round_up(K, 4)))
  if n4 < M: return (max(M, n4 + Rsz), *yR(0))
  return n4, 0, n4 + round_up(K, 4)

# *** ring DMAs (ring_mesh.md sections 2-3) ***
def _rcons_input(g:FCGeom, j:int, tiles:int, seq:int) -> list[int]:
  """ringConsumer0 of input row group j (tiles 4j..4j+3), fed by one infeed that multicasts its 256-byte packets. c = packets of
  the whole input, p = packets of this group; the consumer skips the other groups' 4*(c-p) 64-byte lanes."""
  c = cdiv(g.S, 256)
  p = cdiv(min(4 * g.b, g.S - 4 * g.b * j), 256)
  return RM.encode_ringConsumer(tile_mask=tiles, seq=seq, addr=WIDE_TOP - 8 * c, **ttu("", [1], [1 if p == 1 else c], 4), sdims=int(c > 1),
                                cbuf=1, slots=c, grp=p - 1, gstride=4 * (c - p) + 1 if p > 1 else 0, mode=3, s_id=R_W2N_1, s_val=-64 * c,
                                s_cnt=c, s_en_b=1, s_y=1)

def _rcons_forward(g:FCGeom, tiles:int, seq:int, fifo:int=WIDE_FWD_FIFO) -> list[int]:
  """ringConsumer0 on the compute tiles that receive x from the gather tile (single-slot FIFO, c packets)"""
  c = cdiv(g.K, 256)
  f = dict(cbuf=1, mode=3, s_id=R_W2N_1, **ttu("", [1], [1], 4)) if c == 1 else dict(cbuf=0, mode=1, s_id=R_W2N_0, **ttu("", [0], [c], 4))
  return RM.encode_ringConsumer(tile_mask=tiles, seq=seq, addr=fifo, slots=1, s_val=-64, s_cnt=1, s_en_b=1, s_y=1, **f)

def _rprod_forward(g:FCGeom, tile:int, dest:int, seq:int, fifo:int=WIDE_FWD_FIFO) -> list[int]:
  """ringProducer on the gather tile: multicast x (c packets through the single-slot FIFO) to `dest`"""
  return RM.encode_ringProducer(tile_mask=tile, seq=seq, addr=fifo, sdims=1, cbuf=1, slots=1, mode=1, r0_id=R_N2W, r0_val=1, r0_en_a=1,
                                r0_en_b=1, dest=dest, **ttu("", [1, 0], [1, cdiv(g.K, 256)], 4))

def rprod_output(nbytes:int, tile:int, k:int, seq:int, fifo:int=WIDE_OUT_FIFO, smem:bool=False) -> list[int]:
  """ringProducer: one tile's outputs to the scalar core (outfeed); k = output ordinal in a RING_PRODUCER_A record (serializes the
  tiles; op 2, op 1 for a scalar-memory outfeed)"""
  return RM.encode_ringProducer(tile_mask=tile, seq=seq, addr=fifo, sdims=1, cbuf=1, slots=1, mode=3, pcfg=12 if smem else 20,
                                r0_id=((1 if smem else 2) << 5) | 17, r0_val=k, r0_en_a=1, r0_en_b=1, r1_id=R_N2W, r1_val=1, r1_en_a=1,
                                r1_en_b=1, dest=1 << RM.SCALAR_CORE, **ttu("", [1, 0], [1, cdiv(nbytes, 256)], 4))

def _rcons_caching(g:FCGeom, t:int, tile:int, off:int) -> dict:
  """caching: the ringConsumer that writes tile t's n = gt(t) parameter groups (kw 256-byte rows each) to wide memory. addr
  (2 + the 4n bias rows) and aux_addr0/1 (bias rows 0 .. 4(n-1)) are the relocation fields (+off); no aux without bias."""
  n, kq = g.gt(t), g.kw
  aux = dict(aux_en0=1, aux_addr0=off, aux_addr1=4 * (n - 1) + off, aux_en1=1) if g.bias else {}
  return dict(tile_mask=tile, addr=2 + g.bias_rows(t) + off, sdims=1 if n == 1 else 3, **aux,
              **(ttu("", [1, kq], [kq, n], 4) if n > 1 else ttu("", [1], [kq], 4)))

# *** mesh DMAs (ring_mesh.md section 4) ***
def _mesh(op:int, tiles:int, seq:int, out:tuple|None=None, inn:tuple|None=None, mode:int=3) -> list[int]:
  """meshBus op with an outbound half (send to the neighbour in op's direction) and/or an inbound half (receive from the opposite
  neighbour), each (narrow byte address, words)"""
  f = dict(opcode=op, tile_mask=tiles, seq=seq)
  for p, half, m in (("o_", out, "out_mode"), ("i_", inn, "in_mode")):
    if half: f |= {f"{p}addr": half[0], f"{p}sdims": int(half[1] > 1), m: mode, **ttu(p, [1], [half[1]], 4)}
  return RM.encode_mesh(**f)

def relay(op:int, tiles:int, seq:int, R:int, nwords:int, c:int) -> list[int]:
  """mesh relay: forward nwords from the neighbour behind to the neighbour ahead in 4-word packets through the 4-slot buffer at narrow
  R (both halves), sync records (IN, 5 + 4c), (OUT, -3 + 4c) for the phase count c. A partial last packet of r words (1..3) sets
  grp = r-1 and the 16-bit field at bit 249 (+205 inbound) to 4*(4-r)+1; a single word moves as a 1-word packet."""
  packets, r = cdiv(nwords, 4), nwords % 4
  rin, rout = (SC.TILE_COUNTERS.index(n) << 1 for n in SC.MESH_COUNTERS[op])
  f = dict(opcode=op, tile_mask=tiles, seq=seq, out_mode=3, in_mode=3, s0_id=rin, s0_val=5 + 4 * c, s0_en_a=1, s0_en_b=1, s1_id=rout,
           s1_val=-3 + 4 * c, s1_en_a=1, s1_en_b=1, s2_id=1)
  for p in ("o_", "i_"):
    f |= {f"{p}addr": R, f"{p}sdims": 1, f"{p}cbuf": 1, f"{p}slots": 4, **ttu(p, [1, 0], [1 if nwords == 1 else 4, packets], 4)}
  if r and nwords > 1: f.update(o_grp=r - 1, i_grp=r - 1, rsv243=(4 * (4 - r) + 1) << 6, rsv448=(4 * (4 - r) + 1) << 6)
  return RM.encode_mesh(**f)

def gather(e:Emitter, g:FCGeom, X:int, R:int, ibit):
  """bring the input pieces (input tile t holds x[b*t : b*t+piece(t)] at X+b*t) to input tile 0: west within each row of 4 tiles,
  then north along column 0. Group k of a phase moves the data of position k, positions 1..k-1 relay it. Every west group ends
  with a mesh fence, north groups once they have relays. The west fence counts are +1."""
  rows = [list(range(4 * r, min(4 * r + 4, g.T_in))) for r in range(cdiv(g.T_in, 4))]
  def mask(ts): return sum(ibit(t) for t in ts)
  def words(nbytes): return cdiv(nbytes, 4)
  e.sync(SC.sync_reset17)
  cum, last = 0, 0xffff
  for k in range(1, min(4, g.T_in)):
    rs = [r for r in rows if len(r) > k]
    for r in rs: e.tile(_mesh(0x16, mask([r[k]]), e.seq, out=(X + g.b * r[k], words(g.piece(r[k])))))
    if k >= 2:
      classes: list[tuple[int, list]] = []      # relays forwarding the same number of words share one instruction
      for r in rs:
        if classes and classes[-1][0] == words(g.piece(r[k])): classes[-1][1].append(r)
        else: classes.append((words(g.piece(r[k])), [r]))
      for w, crs in classes: e.tile(relay(0x16, mask([t for r in crs for t in r[1:k]]), e.seq, R, w, cum + 1))
    for r in rs: e.tile(_mesh(0x16, mask([r[0]]), e.seq, inn=(X + g.b * r[k], words(g.piece(r[k])))))
    if k >= 2:   # the fence excludes the relays that forward as many packets as the first relay instruction
      pk = cdiv(classes[0][0], 4)
      cum += pk
      last = 0xffff & ~mask([t for w, crs in classes if cdiv(w, 4) == pk for r in crs for t in r[1:k]])
    e.sync(SC.sync_mesh, 0x16, last, cum + 1)
  e.sync(SC.sync_drain, last)
  e.sync(SC.sync_drain)
  cum = 0                   # north: the row gathered on tile 4k moves to tile 0, relayed by tiles 4, 8, ..., 4(k-1)
  for k in range(1, len(rows)):
    n = words(sum(g.piece(t) for t in rows[k]))
    e.tile(_mesh(0x17, mask([4 * k]), e.seq, out=(X + g.b * 4 * k, n), mode=0))
    if k >= 2: e.tile(relay(0x17, mask([4 * j for j in range(1, k)]), e.seq, R, n, cum))
    e.tile(_mesh(0x17, mask([0]), e.seq, inn=(X + g.b * 4 * k, n), mode=0))
    if k >= 2:
      cum += cdiv(n, 4)
      last = 0xffff & ~mask([4 * j for j in range(1, k)])
      e.sync(SC.sync_mesh, 0x17, last, cum)
  if len(rows) >= 3: e.sync(SC.sync_drain, last)
  e.sync(SC.sync_reset_mesh)

# *** tile placement ***
def shift_ok(N:int, K:int, s:int, move_gather:bool=True) -> bool:
  """move_gather=True (ring_mesh.md section 6, pure translation): the program moves from tiles [0, n) to [s, s+n) iff no used tile
  passes tile 15 and every westward chain stays inside one mesh row (a row of w >= 2 input tiles needs s%4 + w <= 4); northward
  hops stay neighbours for any s. move_gather=False: only the compute tiles move (x is still gathered on [0, T_in) and broadcast
  over the ring), so any s + T <= 16 works."""
  g = FCGeom(N, K)
  if not move_gather: return 0 <= s <= 16 - g.T
  if not 0 <= s <= 16 - max(g.T, g.T_in): return False
  return all(s % 4 + min(4, g.T_in - 4 * r) <= 4 for r in range(cdiv(g.T_in, 4)) if g.T_in - 4 * r >= 2)

def legal_shifts(N:int, K:int, move_gather:bool=True) -> list[int]: return [s for s in range(16) if shift_ok(N, K, s, move_gather)]

@dataclass(frozen=True)
class Place:
  """physical tiles: compute tile t -> t + s; input/gather tile t -> t + s (move_gather) or t"""
  s: int = 0
  move_gather: bool = True
  def c(self, t:int) -> int: return 1 << (t + self.s)
  def i(self, t:int) -> int: return 1 << (t + (self.s if self.move_gather else 0))
  def cspan(self, a:int, b:int) -> int: return sum(self.c(t) for t in range(a, b))

# *** program blocks ***
def caching_part(g:FCGeom, pl:Place, off:int, base:int=0) -> list[tuple[dict, int, int, int]]:
  """per compute tile its caching ringConsumer and infeed (its groups, from blob byte `base`)"""
  return [(_rcons_caching(g, t, pl.c(t), off), base + t * g.P * g.group_bytes, g.gt(t) * g.group_bytes, g.group_out // 16)
          for t in range(g.T)]

def broadcast(e:Emitter, g:FCGeom, X:int, src:int, dest:int, fifo:int=WIDE_FWD_FIFO):
  """x (g.K bytes at narrow X on the tiles src) -> the tiles dest over the ring, through the wide FIFO"""
  e.tile(WN.w2n_ring_recv(g.K, dest, X // 4, e.seq, fifo))
  e.tile(_rcons_forward(g, dest, e.seq, fifo))
  e.tile(_rprod_forward(g, src, dest, e.seq, fifo))
  e.tile(WN.n2w_output(g.K, src, X // 4, fifo, e.seq))

def input_block(e:Emitter, g:FCGeom, pl:Place, X:int, R:int):
  """one input vector: host DMA, ring scatter to the input tiles, mesh gather to input tile 0, ring broadcast to the other compute
  tiles (scalar.md 8.3). X = narrow address of the vector on every tile."""
  e.sync(SC.input_head)
  e.sync(SC.sync_wn_fence)
  e.scalar(SC.input_dma(g.S))
  for t in range(g.T_in): e.tile(WN.w2n_input(g.K, t, X // 4, e.seq, pl.i(t)))
  for j in range(cdiv(g.T_in, 4)):
    ts = sum(pl.i(t) for t in range(4 * j, min(4 * j + 4, g.T_in)))
    e.tile(_rcons_input(g, j, ts, e.seq))
    e.scalar(SC.av_infeed(4 * j * g.b, min(4 * g.b, g.S - 4 * j * g.b), ts))
  e.sync(SC.sync_drain)
  e.sync(SC.sync_reset17)
  if g.T_in >= 2: gather(e, g, X, R, pl.i)
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_reset17)
  if dest := pl.cspan(0, g.T) & ~pl.i(0): broadcast(e, g, X, pl.i(0), dest)
  e.sync(SC.sync_reset17)

def ops_block(e:Emitter, g:FCGeom, pl:Place, X:int, Y:int, off:int, quant:dict, tail:bool=True):
  """compute fence, bias loads (one per distinct group count), the ops, reset; tail: + the completion fence"""
  e.sync(SC.signal_fence)
  if g.bias:
    for n in sorted({g.gt(t) for t in range(g.T)}, reverse=True):
      e.tile(WN.w2n_bias(sum(pl.c(t) for t in range(g.T) if g.gt(t) == n), off, n, e.seq))
  for t in range(g.T):
    f = OP.fc_tile_fields(g.N, g.K, t, e.seq, X, Y + 64 * g.P * t, g.bias_rows(t) + off, quant["w_zp"], quant["in_zp"], quant["mult"],
                          quant["out_zp"], quant.get("clamp_min"), quant.get("clamp_max")) | dict(tile_mask=pl.c(t))
    if not g.bias: f.update(cfg1=0, sync0=0, sync1=0, cfg4=0)   # the bias ("scaling") path is off
    e.tile(OP.encode_op(**f))
  e.sync(SC.sync_reset17)
  if tail: e.sync(SC.signal_fence)

def output_block(e:Emitter, g:FCGeom, pl:Place, Y:int):
  """one output vector: narrowToWide per compute tile, host DMA, (outfeed, ringProducer) per tile in ordinal order"""
  e.scalar(SC.output_dma_head())
  for t in range(g.T): e.tile(WN.n2w_output(g.n_out(t), pl.c(t), (Y + 64 * g.P * t) // 4, WIDE_OUT_FIFO, e.seq))
  e.scalar(SC.output_dma_tail(g.out_bytes))
  for t in range(g.T):
    e.scalar(SC.outfeed(8 * cdiv(g.n_out(t), 8)))
    e.tile(rprod_output(g.n_out(t), pl.c(t), t, e.seq))
  e.scalar(SC.output_wait(e.seq, g.T), seqs=2)

def exe_program(g:FCGeom, pl:Place, X:int, R:int, Y:int, stage) -> bytes:
  """EXECUTION_ONLY: prologue, the input block, stage(e) (the matmul, ending with the completion fence), the output block"""
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  input_block(e, g, pl, X, R)
  stage(e)
  output_block(e, g, pl, Y)
  e.sync(SC.epilogue)
  return e.program()

def gen_fc(N:int, K:int, param_offset:int=0, tile_shift:int=0, move_gather:bool=True, quant:dict|None=None,
           bias:bool=True) -> tuple[bytes, bytes]:
  """(PARAMETER_CACHING, EXECUTION_ONLY) for y[N] = W[N,K] x[K] (uint8), byte-identical to edgetpu_compiler for tile_shift=0.
  param_offset: byte offset of the parameter region in every tile's wide memory (a multiple of 256: on the device other offsets
  hang the caching program). tile_shift: place the compute tiles on [s, s+T); move_gather=True translates the whole program
  (test_codegen.translate_tiles), False keeps the input scatter/gather on [0, T_in) and broadcasts x to the shifted tiles.
  quant: overrides of REF_QUANT (w_zp, in_zp, out_zp, mult, clamp_min, clamp_max). bias=False: no bias rows or load (the
  compiler's form for N=1 with an all-zero bias)."""
  assert param_offset % 256 == 0 and param_offset >= 0, "param_offset must be a non-negative multiple of 256"
  g = FCGeom(N, K, bias)
  assert g.T <= 16 and g.T_in <= 16, f"N={N} K={K} needs more than 16 tiles"
  if K <= 4 and g.P > 1: raise NotImplementedError("K <= 4 with N > 1024: the compiler uses another op schedule (op.md)")
  if not shift_ok(N, K, tile_shift, move_gather):
    raise ValueError(f"tile_shift={tile_shift} is not legal for N={N} K={K} (move_gather={move_gather}); "
                     f"legal: {legal_shifts(N, K, move_gather)}")
  pl, off, (X, Y, R) = Place(tile_shift, move_gather), param_offset // 64, narrow_layout(N, K)
  exe = exe_program(g, pl, X, R, Y, lambda e: ops_block(e, g, pl, X, Y, off, {**REF_QUANT, **(quant or {})}))
  return caching_program(g.param_bytes, [caching_part(g, pl, off)]), exe

if __name__ == "__main__": run_test("test_codegen")
