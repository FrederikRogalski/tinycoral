# The attention core of a transformer layer (TinyStories-15M: 6 heads of 48 dims, P positions) as ONE Edge TPU program, and its
# pieces byte-identical to edgetpu_compiler (docs/isa/codegen_attention.md):
#   prog = gen_scores(P, quant)        s[h, p] = SUM_c MUL(q[h, c] broadcast over p, K[h, p, c])            (compiler oracle)
#   prog = gen_pv(P, quant)            o[h, c] = SUM_p MUL(p[h, p, c], V[h, p, c]), p expanded on the host   (compiler oracle)
#   eltops.gen_softmax(6, P, ...)      p = SOFTMAX over the positions (scalar core)                          (compiler oracle)
#   prog, io = gen_attention(P, quant) o = concat_h softmax(beta * q_h . K_h^T) . V_h: the three in ONE STAND_ALONE program, the
#                                      broadcasts as stride-0 TTU levels, the softmax between them on the scalar core
#   attention_ref(q, K, V, quant)      gen_attention's numpy bit model, built from the pieces' bit models
# `python -m coral.codegen.attention` runs the acceptance test (test/test_codegen_attention.py), offline.
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from coral.isa import cdiv, round_up, f32, f32_bits, ttu, op as OP, wide_narrow as WN, ring_mesh as RM, scalar as SC, eltwise as EL
from coral.isa import scalar_core as S
from coral.isa.scalar_core import encode_op3
from coral.codegen import Emitter, run_test, WIDE_TOP, WIDE_OUT_FIFO
from coral.codegen import conv2d as C2, fc as FC
from coral.codegen.fc import rprod_output
from coral.codegen.chain import _gmesh, _copy

NH, DH = 6, 48                     # heads, dims per head
DM = NH * DH                       # 288
HW = DH // 4                       # 12 words per (head, position)
ROWS = EL.tile_split(NH)           # heads per tile row: [2, 1, 2, 1]
HEAD0 = [sum(ROWS[:r]) for r in range(4)]

# ***** geometry: a [1, 6, P, 48] tensor over the 16 tiles (eltwise.tile_split: tile row r holds heads ROWS[r], column c positions) *****
@dataclass(frozen=True)
class Geom:
  P: int
  @property
  def cols(self) -> list[int]: return EL.tile_split(self.P)          # positions per tile column
  def p0(self, c:int) -> int: return sum(self.cols[:c])              # first position of tile column c
  @property
  def smax(self) -> int: return max(self.cols)
  @property
  def even(self) -> bool: return self.P % 4 == 0                     # all columns hold the same number of positions

def _group(per:list[dict|None]) -> list[tuple[int, dict]]:
  """tiles with identical fields share one instruction (conv2d._group); a 'shape' entry keeps tile shapes apart that encode the same
  (edgetpu_compiler emits one instruction per block shape) and is dropped"""
  return [(m, {k: v for k, v in f.items() if k != "shape"}) for m, f in C2._group(per)]

# ***** stages *****
def g4_of(W:int) -> C2.Conv2DGeom: return C2.Conv2DGeom(NH, W, DH, DH, 1, 1, 1, "VALID")   # a [1, 6, W, 48] tensor as an image

def _copies(e:Emitter, g4:C2.Conv2DGeom, Stg:int, X:int, const:int, last:bool):
  """conv2d.copy_ops for the tiles that hold positions"""
  per = [None if (not g4.cols[t % 4].ni or (not last and g4.chunks(t // 4)[0] < 2)) else C2._copy_op(g4, t // 4, t % 4, Stg, X, const, last)
         for t in range(16)]
  for m, f in C2._group(per): e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))

def input4d(e:Emitter, W:int, Stg:int, X:int, C:int, const:int, in_fifo:int, ident:bool):
  """a [1, 6, W, 48] input (host bytes head-major, one DMA) into each tile's block at narrow X (row stride cols(c)*48 bytes):
  conv2d's input path of a 6 x W x 48 image (staging slots at Stg, ring FIFO at wide in_fifo), restricted to the tile columns that
  hold positions; with ident also the identity row (narrow C -> wide const) first, as every edgetpu_compiler program's first input"""
  g4 = g4_of(W)
  cmask = sum(1 << c for c in range(4) if g4.cols[c].ni)
  e.sync(SC.input_head)
  if ident: e.tile(*EL.ident_prologue(e.seq, 0xffff, C, const, 0))
  e.sync(SC.sync_wn_fence)
  _copies(e, g4, Stg, X, const, False)
  e.scalar(SC.input_dma(g4.S))
  fenced = [C2._ring_passes(g4, r)[0] >= 3 for r in range(4)]
  if any(fenced): e.tile(SC.sync(e.seq, tiles=sum(cmask << (4 * r) for r in range(4) if fenced[r]), counters=SC.tc("WIDE_TO_NARROW"), units=0))
  for part in (False, True):
    per = [C2._w2n_input(g4, t // 4, t % 4, Stg, in_fifo) if fenced[t // 4] == part and g4.cols[t % 4].ni else None for t in range(16)]
    for m, f in C2._group(per): e.tile(WN.encode_wide_to_narrow(**(f | dict(tile_mask=m, seq=e.seq))))
  for r in range(4):
    rc = RM.decode_ringConsumer(C2._rc_input(g4, r, e.seq, in_fifo)) | dict(tile_mask=cmask << (4 * r))
    e.tile(RM.encode_ringConsumer(**rc))
    lo, n, _ = C2._row_stream(g4, r)
    a, b = lo // 8, cdiv(lo + n, 8)
    e.scalar(C2._av_infeed(8 * a, 8 * (b - a), cmask << (4 * r)))
  e.sync(SC.sync_drain)
  _copies(e, g4, Stg, X, const, True)

def mul_stage(e:Emitter, g:Geom, a_in, X_b:int, M:int, Y:int, fifo:int, const:int, q:dict, out_rows):
  """MUL of two [6, P, 48] tensors (eltwise.md 5): reset17 / N2W / PARAMETERS fences, step 1 per tile (operand a through the in TTU:
  a_in(t) = (in_base, byte strides, counts), innermost first, 48 x cols x rows steps; a stride-0 level broadcasts a), the FIFO feed of
  operand b (block at X_b, eltwise.mul_feed_n2w_fields), step 2 per tile into Y with the output row stride out_rows(t) (words)"""
  e.sync(SC.sync_reset17)
  e.sync(SC.sync, counters=SC.tc("NARROW_TO_WIDE"))
  e.sync(SC.sync, counters=SC.tc("PARAMETERS"))
  per = []
  for t in range(16):
    nh, s_ = ROWS[t // 4], g.cols[t % 4]
    if not s_:
      per.append(None)
      continue
    in_base, strides, counts = a_in(t)
    f = EL.mul_op1_fields(HW, s_, nh, (HW, HW * s_, HW * s_ * nh), 0, 0, in_base, M, fifo, q["a_zp"], q["b_zp"], q["mult"], q["out_zp"],
                          q["clamp_min"], q["clamp_max"])
    per.append(f | OP.ttu_fine("in", strides, counts))
  for m, f in _group(per): e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))
  per = [EL.mul_feed_n2w_fields(0, 0, X_b, fifo, HW, g.cols[t % 4], ROWS[t // 4]) | dict(shape=(ROWS[t // 4], g.cols[t % 4]))
         if g.cols[t % 4] else None for t in range(16)]
  for m, f in _group(per): e.tile(WN.encode_narrow_to_wide(**(f | dict(tile_mask=m, seq=e.seq))))
  e.sync(SC.sync_wn_fence)
  per = []
  for t in range(16):
    nh, s_ = ROWS[t // 4], g.cols[t % 4]
    if not s_:
      per.append(None)
      continue
    ro = out_rows(t)
    per.append(EL.mul_op2_fields(HW, s_, nh, (HW, ro, ro * nh), 0, 0, M, Y, const))
  for m, f in _group(per): e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))
  e.sync(SC.sync, counters=SC.tc("NARROW_TO_WIDE"))
  e.sync(SC.sync, counters=SC.tc("PARAMETERS"))
  e.sync(SC.sync_reset17)

def dense(g:Geom, X:int):
  """a_in of mul_stage for a block [rows][cols(c)][48] at X on every tile: the plain walk (edgetpu_compiler's MULs)"""
  return lambda t: (X, [1, DH, DH * g.cols[t % 4], DH * g.cols[t % 4] * ROWS[t // 4]], [DH, g.cols[t % 4], ROWS[t // 4]])

def possum_stage(e:Emitter, g:Geom, Y:int, O:int, wide:int, q:dict):
  """SUM over the positions on the column-0 tiles: the transposing relay of the gathered block into wide memory (4 positions per
  256-byte row, eltwise.maxpool_relay_n2w_fields), then the pooling-class reduction op (opcode 3, cfg0 0x127, dp_mode 4): a
  1 x P 'sum pool' per head row, 48 channels, requantized"""
  e.sync(SC.sync_drain)
  def relay(nh):
    f = EL.maxpool_relay_n2w_fields(0, 0, Y, wide, HW, HW * g.P, g.P, nh, DH)
    return f | dict(wide_lvl_mask=0) if nh * cdiv(g.P, 4) == 1 else f      # a one-row wide walk has no level
  per = [relay(ROWS[t // 4]) if t % 4 == 0 else None for t in range(16)]
  for m, f in _group(per): e.tile(WN.encode_narrow_to_wide(**(f | dict(tile_mask=m, seq=e.seq))))
  e.sync(SC.sync_drain)
  per = []
  for t in range(16):
    if t % 4:
      per.append(None)
      continue
    f = EL.maxpool_op_fields(1, g.P, 1, 1, ROWS[t // 4], 1, DH, (HW, HW), 0, 0, O, wide, 0, q["clamp_min"], q["clamp_max"])
    f.pop("opcode")
    f.pop("in_zp")
    f.pop("offset")
    per.append(f | dict(w_zp=q["in_zp"], out_zp=q["out_zp"], mult_bits=f32_bits(q["mult"])))
  for m, f in _group(per): e.tile(encode_op3(**(f | dict(tile_mask=m, seq=e.seq))))

def possum_any(e:Emitter, g:Geom, Y:int, O:int, wide:int|None, const:int, q:dict):
  """after the MUL: the gather onto the column-0 tiles and the SUM over the positions as edgetpu_compiler does it for every P: P = 1
  no gather, a requantize op (eltwise.requant_op_fields: the sum of one element) from Y to O; otherwise gather_west, one reset17
  (two for P < 4), possum_stage"""
  if g.P == 1:
    e.sync(SC.sync_reset17)
    e.sync(SC.sync_reset17)
    per = [EL.requant_op_fields(ROWS[t // 4], HW, HW, HW, 0, 0, Y, O, const, q["in_zp"], q["out_zp"], q["mult"], q["clamp_min"], q["clamp_max"], 0)
           if t % 4 == 0 else None for t in range(16)]
    for m, f in _group(per): e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))
    return
  gather_west(e, g, Y)
  for _ in range(1 if g.P >= 4 else 2): e.sync(SC.sync_reset17)
  possum_stage(e, g, Y, O, wide, q)

def output_col0(e:Emitter, O:int):
  """the [6, 48] result on the column-0 tiles (tile row r: ROWS[r] heads x 48 bytes at narrow O) to the host, one DMA of 288 bytes"""
  e.sync(SC.signal_fence)
  e.scalar(SC.output_dma_head())
  per = [dict(nb=ROWS[t // 4] * DH, dims=tuple([(1, HW)] + [(HW, ROWS[t // 4])] * (ROWS[t // 4] > 1))) if t % 4 == 0 else None for t in range(16)]
  for m, f in _group(per): e.tile(WN.n2w_output(f["nb"], m, O // 4, WIDE_OUT_FIFO, e.seq, list(f["dims"])))
  e.scalar(SC.output_dma_tail(DM))
  for k, t in enumerate((0, 4, 8, 12)):
    e.scalar(SC.outfeed(round_up(ROWS[t // 4] * DH, 8)))
    e.tile(rprod_output(ROWS[t // 4] * DH, 1 << t, k, e.seq))
  e.scalar(SC.output_wait(e.seq, 4), seqs=2)
  e.sync(SC.epilogue)

# ***** quantization (edgetpu_compiler's float32 formulas) *****
DEFAULT_PV = dict(p=(1/256, 0), v=(1/16, 128), pv=(1/64, 128), o=(1/16, 128))
def sum_quant(in_q:tuple, out_q:tuple) -> dict:
  lo, hi = EL.out_clamps(out_q[0], out_q[1])
  return dict(in_zp=in_q[1], out_zp=out_q[1], mult=EL.mul32(in_q[0], EL.recip32(out_q[0])), clamp_min=lo, clamp_max=hi)

# ***** p.V with p expanded on the host: MUL [1,6,P,48] x [1,6,P,48], then SUM over the positions (axis 2) *****
def pv_alloc(g:Geom) -> dict:
  """edgetpu_compiler's memory for the p.V program (fitted on P = 1..256). Narrow (bytes): staging Stg at 0 (two slots of 48P bytes),
  the identity row C behind them; the MUL intermediate M takes 384P + 4 bytes, then the operand blocks Xp, Xv (2 * smax * 48 bytes
  each); the packed products / gathered rows Y; the sums O. P % 4 == 0: [M | Y (96P) | Xp | Xv], O = 0; otherwise Y = 0,
  M = 48 (P + round_up(P, 4)), [M | Xp | Xv], O = M. Wide (64-byte units): the relay block (2 rows x ceil(P/4) quads x 4 units) and
  the input ring FIFO (8 per slot) from the top, the identity row right below the lower of the two (P <= 4: at the top), the MUL
  operand FIFO (32) below them. P = 1 has its own layout (no relay)."""
  P, blk = g.P, 2 * DH * g.smax
  slot = max(DH * P, 256)
  if P == 1:        # one position (requantize instead of the sum): v at 0 with its staging behind it, p behind the identity row
    return dict(Stg=0, Stg_v=96, C=512, M=96, Y=0, Xp=608, Xv=0, O=0, const=WIDE_TOP - 4, in_fifo=WIDE_TOP - 12, relay=None, mulfifo=WIDE_TOP - 36)
  if g.even:
    M, Y = 0, 384 * P + 4
    Xp, O = Y + 96 * P, 0
  else:
    Y, M = 0, DH * (P + round_up(P, 4))
    Xp, O = M + 384 * P + 4, M
  relay_u, fifo_u = 4 * 2 * cdiv(P, 4), 8 * cdiv(DH * P, 256)
  relay, in_fifo = WIDE_TOP - relay_u, WIDE_TOP - fifo_u
  const = WIDE_TOP - 4 if relay_u <= 8 and fifo_u <= 8 else min(relay, in_fifo) - 4
  if const == WIDE_TOP - 4: relay = in_fifo = const - 8
  return dict(Stg=0, Stg_v=0, C=2 * slot, M=M, Y=Y, Xp=Xp, Xv=Xp + blk, O=O, const=const, in_fifo=in_fifo, relay=relay,
              mulfifo=min(const, relay, in_fifo) - 32)

def gen_pv(P:int, quant:dict|None=None) -> bytes:
  """the STAND_ALONE program of o[1,6,48] = SUM_axis2(MUL(p[1,6,P,48], V[1,6,P,48])), byte-identical to edgetpu_compiler (P = 1..256).
  quant: p, v, pv (the product), o as (scale, zero point)."""
  q = DEFAULT_PV | (quant or {})
  g, a = Geom(P), pv_alloc(Geom(P))
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  input4d(e, P, a["Stg"], a["Xp"], a["C"], a["const"], a["in_fifo"], True)
  e.sync(SC.sync_reset17)
  input4d(e, P, a["Stg_v"], a["Xv"], a["C"], a["const"], a["in_fifo"], False)
  e.sync(SC.sync_reset17)
  mq = EL.mul_quant(q["p"], q["v"], q["pv"])
  mul_stage(e, g, dense(g, a["Xp"]), a["Xv"], a["M"], a["Y"], a["mulfifo"], a["const"], mq, lambda t: HW * (P - g.p0(t % 4)))
  possum_any(e, g, a["Y"], a["O"], a["relay"], a["const"], sum_quant(q["pv"], q["o"]))
  e.sync(SC.sync_reset17)
  output_col0(e, a["O"])
  return e.program()

# ***** scores = SUM over the channels (axis 3) of MUL(q [1,6,1,48] broadcast over the positions, K [1,6,P,48]) *****
ROW_FIFO = 0x2060           # the 4-row FIFO that streams the products to the reduction op (eltops' SUM feed)

def scores_alloc(g:Geom) -> dict:
  """edgetpu_compiler's memory for the scores program. Narrow (bytes): q's input staging (2 x 256) at 0, q (2 rows x 48 on the
  column-0 tiles) at Q = max(512, rep), the identity row behind q (Q = 512) or at 512, the replicated q block (rows x smax x 48)
  at 0, the K block right above it (X), K's staging above that, reused by the MUL intermediate; the products over K, the scores
  over the replicated q; a 32-byte mesh relay slot per q move (direct moves leave theirs unused) from 128 smax (+ 96 when
  smax >= 4), skipping q. Wide (64-byte units): the identity row at the top (8316), q's input FIFO below it, K's input FIFO
  stacked down from 8316, the MUL FIFO at 8284, the SUM FIFO at 8288."""
  P, rep = g.P, 2 * DH * g.smax
  Q = max(512, rep)
  relay0 = 128 * g.smax + (96 if g.smax >= 4 else 0)
  return dict(Stg_q=0, Q=Q, C=Q + 96 if Q == 512 else 512, rep=0, X=rep, Stg=2 * rep, M=2 * rep, Y=rep, S=0, relay0=relay0,
              const=WIDE_TOP - 4, in_fifo_q=WIDE_TOP - 12, in_fifo=WIDE_TOP - 4 - 8 * g4_of(P).c_in, mulfifo=WIDE_TOP - 36, sumfifo=ROW_FIFO)

def _relays(a:dict, n:int) -> list[int]:
  """the narrow addresses of n mesh relay buffers: 32-byte slots from relay0 that do not touch q's block [Q, Q + 96)"""
  out, x = [], a["relay0"]
  while len(out) < n:
    if not (x + 32 > a["Q"] and x < a["Q"] + 96): out.append(x)
    x += 32
  return out

def q_broadcast(e:Emitter, g:Geom, a:dict):
  """edgetpu_compiler's broadcast of q over the positions: on the column-0 tiles one copy of q per own position into the replicated
  block (row stride s0 * 48), then per destination column c and position j one eastward mesh move of q (relayed through columns
  1..c-1 in 16-word circular buffers, fc.relay) into the block slot j; every group ends with reset17s, the moves with a mesh fence"""
  def copies(j):
    per = []
    for t in range(16):
      if t % 4:
        per.append(None)
        continue
      nh, s0 = ROWS[t // 4], g.cols[0]
      per.append(chain_copy(a["Q"], DH * j, HW, 1, nh, [HW, HW, HW * nh], [HW, HW * s0, HW * s0 * nh]))
    for m, f in _group(per): e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))
  groups = [("copy", j) for j in range(g.cols[0])] + [("move", c, j) for c in (1, 2, 3) for j in range(g.cols[c])]
  rel = iter(_relays(a, sum(g.cols[1:])))
  for k, grp in enumerate(groups):
    if k: e.sync(SC.sync_reset17)
    if grp[0] == "copy": copies(grp[1])
    else:
      _, c, j = grp
      for r in (0, 1):
        nh = ROWS[r]
        e.tile(RM.encode_mesh(opcode=C2.MESH_E, tile_mask=0x0101 << (4 * r), seq=e.seq, **_mesh_half("o_", a["Q"], [(1, HW), (HW, nh)])))
      R = next(rel)
      if c >= 2:
        for r in (0, 1):
          e.tile(FC.relay(C2.MESH_E, sum(0x0101 << (4 * r + cc) for cc in range(1, c)), e.seq, R, HW * ROWS[r], 0))
      for r in (0, 1):
        nh = ROWS[r]
        e.tile(RM.encode_mesh(opcode=C2.MESH_E, tile_mask=0x0101 << (4 * r + c), seq=e.seq,
                              **_mesh_half("i_", DH * j, [(1, HW), (HW * g.cols[c], nh)])))
      fence = 0xffff & ~sum(0x0101 << cc for cc in range(1, c))
      e.sync(SC.sync, tiles=fence, counters=SC.tc("MESH_WEST_IN", "MESH_EAST_OUT"), count=1 if c == 1 else cdiv(HW * 2, 4), units=1 << 9)
      e.sync(SC.sync_drain, fence)
    e.sync(SC.sync_reset17)

def _mesh_half(p:str, addr:int, lv:list[tuple[int, int]]) -> dict:
  """one half of a mesh move (chain._gmesh) over the levels of count > 1"""
  return _gmesh(0, 0, 0, p, addr, [(s, n) for s, n in lv if n > 1] or lv[:1])

def chain_copy(src:int, dst:int, W:int, n:int, rows:int, in_s:list[int], out_s:list[int]) -> dict:
  """chain._copy (the reformat copy through the identity row at the top of wide memory) without a tile mask"""
  f = _copy(0, src, dst, W, n, rows, in_s, out_s)
  f.pop("tile_mask")
  return f | OP.wide("par", WIDE_TOP - 4)

def chansum_stage(e:Emitter, g:Geom, Y:int, S_:int, fifo:int, q:dict):
  """SUM over the 48 channels of every (head, position) (eltops' SUM, per tile over its positions): the packed products (12 words per
  position at Y) stream through the 4-row FIFO, one opcode-3 op accumulates them and writes one 4-byte word per score at S_"""
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_reset_par)
  per = [EL.mul_feed_n2w_fields(0, 0, Y, fifo, HW, g.cols[t % 4], ROWS[t // 4]) | dict(shape=(ROWS[t // 4], g.cols[t % 4]))
         if g.cols[t % 4] else None for t in range(16)]
  for m, f in _group(per): e.tile(WN.encode_narrow_to_wide(**(f | dict(tile_mask=m, seq=e.seq))))
  D = EL.mul_fifo_depth(HW)
  per = []
  for t in range(16):
    nh, s = ROWS[t // 4], g.cols[t % 4]
    if not s:
      per.append(None)
      continue
    per.append(dict(loop0=D - 1, loop1=cdiv(HW, D) - 1, loop2=s - 1, loop3=nh - 1,
                    **OP.ttu_fine("out", [4, 4, 4 * s, 4 * s * nh], [1, s, nh]), out_base=S_,
                    out_mode7=3, out_tflags=3, **OP.ttu_fine("par", [4], [D, cdiv(HW, D), s, nh]), **OP.wide("par", fifo), par_mode7=1,
                    par_tflags=0x40, par_fifo=D, **ttu("psum_", [], [HW, s, nh], 8), psum_tflags=0x840, cfg0=0x60, cfg1=0x12d, sync0=0x4000,
                    sync1=0x4000, cfg2=6, w_zp=q["in_zp"], dp_mode=1, reduce_mask=1, out_ch=3, out_ch_last=3, out_zp=q["out_zp"],
                    mult_bits=f32_bits(q["mult"]), clamp_min_bits=f32_bits(q["clamp_min"]), clamp_max_bits=f32_bits(q["clamp_max"])))
  for m, f in _group(per): e.tile(encode_op3(**(f | dict(tile_mask=m, seq=e.seq))))
  e.sync(SC.sync, counters=SC.tc("NARROW_TO_WIDE", "PARAMETERS"))

def output_blocks(e:Emitter, blocks:list[dict|None], Y:int):
  """per tile t a block of blocks[t]['nb'] bytes at narrow Y (narrow walk blocks[t]['dims'] in words) to the host, as
  conv2d.output_stage does it: one host DMA and (outfeed, ringProducer) per tile, or, when a block other than the last is not a
  multiple of 8 bytes, every tile's ringProducer into scalar-memory outfeeds packed into the 4096-word buffer that the scalar core
  flushes to the host (codegen/conv.py's path, its 0x23 / vector-slot words reproduced as observed). Then output_wait, epilogue."""
  from coral.codegen import conv as CC
  sizes = [b["nb"] if b else 0 for b in blocks]
  tiles = [t for t in range(16) if sizes[t]]
  smem = any(sizes[t] % 8 for t in tiles[:-1])
  e.sync(SC.sync_drain if smem else SC.signal_fence)
  e.scalar(SC.output_dma_head())
  for m, f in _group([dict(nb=b["nb"], dims=tuple(b["dims"]), shape=b["shape"]) if b else None for b in blocks]):
    e.tile(WN.n2w_output(f["nb"], m, Y // 4, WIDE_OUT_FIFO, e.seq, list(f["dims"])))
  if not smem:
    e.scalar(SC.output_dma_tail(sum(round_up(sizes[t], 8) for t in tiles)))
    for k, t in enumerate(tiles):
      e.scalar(SC.outfeed(round_up(sizes[t], 8)))
      e.tile(rprod_output(sizes[t], 1 << t, k, e.seq))
    n_wait = len(tiles)
  else:
    used, packets = 0, 0
    for t in tiles:
      e.tile(rprod_output(sizes[t], 1 << t, packets, e.seq, smem=True))
      packets += cdiv(sizes[t], 256)
      e.scalar(SC.scsync_smem_pre())
      n = sizes[t] // 4
      while n > CC.SMEM_WORDS - used:
        part = (CC.SMEM_WORDS - used) // 64 * 64
        if part:
          e.scalar(CC._smem_outfeed(used, part))
          used += part
          n -= part
        flush, left = used // 8 * 8, used % 8
        e.scalar(SC.scsync_smem_post() + CC._smem_flush(4 * flush) + SC.add64(4, 5, 4 * flush, 4, 5, 6))
        if left: e.scalar([SC.movi(7, flush), SC.movi(8, 0), 0x300008c0 | (left << 12), 0x139b800, 0x800, 0xc028300000800])
        used = left
      e.scalar(CC._smem_outfeed(used, n))
      used += n
    e.scalar(SC.scsync_smem_post() + C2._final_flush(used))
    n_wait = packets
  e.scalar(SC.output_wait(e.seq, n_wait), seqs=2)
  e.sync(SC.epilogue)

def output_tiles(e:Emitter, g:Geom, S_:int):
  """every tile's scores (rows x s words at narrow S_, 4 bytes per score) to the host"""
  def dims(nh, s): return [(st, n) for st, n in [(1, s), (s, nh)] if n > 1] or [(1, 1)]
  output_blocks(e, [dict(nb=4 * ROWS[t // 4] * g.cols[t % 4], dims=dims(ROWS[t // 4], g.cols[t % 4]), shape=(ROWS[t // 4], g.cols[t % 4]))
                    if g.cols[t % 4] else None for t in range(16)], S_)

DEFAULT_SCORES = dict(q=(1/16, 128), k=(1/16, 128), qk=(1/8, 128), s=(1/4, 128))
def gen_scores(P:int, quant:dict|None=None) -> bytes:
  """the STAND_ALONE program of s[1,6,P] = SUM_axis3(MUL(q[1,6,1,48], K[1,6,P,48])), byte-identical to edgetpu_compiler.
  quant: q, k, qk (the product), s as (scale, zero point). The output is 4 bytes per score, per tile (scores_io)."""
  q = DEFAULT_SCORES | (quant or {})
  g = Geom(P)
  a = scores_alloc(g)
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  if P == 1:     # one position: q's input is the replicated block itself (its staging right behind it), nothing to broadcast
    input4d(e, 1, a["rep"] + 2 * DH * g.smax, a["rep"], a["C"], a["const"], a["in_fifo_q"], True)
    e.sync(SC.sync_reset17)
  else:
    input4d(e, 1, a["Stg_q"], a["Q"], a["C"], a["const"], a["in_fifo_q"], True)
    for _ in range(4): e.sync(SC.sync_reset17)
    q_broadcast(e, g, a)
  input4d(e, P, a["Stg"], a["X"], a["C"], a["const"], a["in_fifo"], False)
  e.sync(SC.sync_reset17)
  mq = EL.mul_quant(q["q"], q["k"], q["qk"])
  mul_stage(e, g, dense(g, a["rep"]), a["X"], a["M"], a["Y"], a["mulfifo"], a["const"], mq, lambda t: HW * g.cols[t % 4])
  chansum_stage(e, g, a["Y"], a["S"], a["sumfifo"], sum_quant(q["qk"], q["s"]))
  e.sync(SC.sync_reset17)
  output_tiles(e, g, a["S"])
  return e.program()

# ***** the softmax on the scalar core, reading the scores as words *****
SMEM_TOP = 0x4000

def softmax_words(rows:int, n:int, src_w:int, M:int, dst_w:int, zp:int, s_in:float, beta:float, s_out:float, zp_out:int) -> list[int]:
  """eltops.softmax_scalar with word layouts: the scores are 32-bit words (the score in the low byte, as the tiles' SUM op writes
  them), row r's n words at word src_w + r*M; the probability p goes to word dst_w + r*M + j as 4 copies of its byte (p * 0x01010101),
  so that a tile op reading any byte lane of the word gets p (the p.V MUL broadcasts it with a stride-0 TTU level). The float32
  operations and their order are eltops.softmax_scalar's, so eltops.softmax_ref is its bit model. Registers: s4 / s5 row pointers
  (input word, output word), s8 max, s9 sum / reciprocal."""
  from coral.codegen.eltops import exp2_scale
  k = exp2_scale(beta)
  def deq(x): return [S.alu(S.SUB, x, x, imm=zp)] * (zp != 0) + [S.i2f(x, x), S.fimm(S.FMUL, x, x, s_in)]
  def score(i, x):     # s_x = low byte of word s_i; the loaded register is first used 5 bundles later, as in edgetpu_compiler's softmax
    return [S.ld(x, i)] + [S.NOP] * 4 + [S.alu(S.AND, x, x, imm=0xff)]
  w = [S.movi(4, src_w), S.movi(5, dst_w)]
  top = len(w)
  w += [S.mov(6, 4), S.mov(7, 5), S.movi(8, 0xff800000), S.mov(10, 6)]
  body = score(10, 9) + deq(9) + [S.alu(S.FMAX, 8, 8, reg=9), S.alu(S.ADD, 10, 10, imm=1)]
  w += [S.loop(n, len(body))] + body
  w += [S.movi(9, 0), S.mov(11, 6)]
  body = score(11, 10) + deq(10) + [S.alu(S.FSUB, 10, 10, reg=8), S.fimm(S.FMUL, 10, 10, k), S.sfu(S.EXP2, 10, 10), S.alu(S.ADD, 11, 11, imm=1),
                                     S.NOP, S.alu(S.FADD, 9, 9, reg=10)]
  w += [S.loop(n, len(body))] + body
  w += [S.mov(17, 6), S.mov(13, 7), S.sfu(S.RECIP, 9, 9)]
  body = score(17, 10) + deq(10) + [S.alu(S.FSUB, 10, 10, reg=8), S.fimm(S.FMUL, 10, 10, k), S.sfu(S.EXP2, 10, 10), S.alu(S.ADD, 17, 17, imm=1),
                                     S.NOP, S.alu(S.FMUL, 10, 10, reg=9), S.fimm(S.FMUL, 10, 10, f32(1 / f32(s_out)))]
  if zp_out: body += [S.f2i(10, 10), S.alu(S.ADD, 10, 10, imm=zp_out), S.i2f(10, 10)]
  body += [S.fimm(S.FMIN, 10, 10, 255.0), S.fimm(S.FMAX, 10, 10, 0.0), S.f2i(10, 10), S.alu(S.SHL, 11, 10, imm=8), S.alu(S.OR, 11, 11, reg=10),
           S.alu(S.SHL, 12, 11, imm=16), S.alu(S.OR, 11, 11, reg=12), S.st(13, 11), S.alu(S.ADD, 13, 13, imm=1)]
  w += [S.loop(n, len(body))] + body
  w += [S.alu(S.ADD, 4, 4, imm=M), S.alu(S.ADD, 5, 5, imm=M), S.alu(S.LT, 0, 4, imm=src_w + rows * M)]
  w += [S.branch(top - len(w))]
  return w

# ***** the composed program *****
def gather_west(e:Emitter, g:Geom, Y:int):
  """every tile row's packed products onto its column-0 tile (meshBus 0x16, westward): column c holds rows x [p0(c), P) positions
  at Y after the gather (row stride (P - p0(c)) * 48 bytes), sends all of it west when it holds any, receives [p0(c+1), P) behind its
  own positions; tiles that forward received positions wait on MESH_EAST_IN (conv2d's halo records). Empty columns (P < 4) relay."""
  per = []
  for t in range(16):
    r, c = divmod(t, 4)
    nh = ROWS[r]
    n_own = g.P - g.p0(c)
    n_in = g.P - g.p0(c + 1) if c < 3 else 0
    f = {}
    if c > 0 and n_own: f |= C2._half("o_", Y, [(1, HW), (HW, n_own), (HW * n_own, nh)])
    if n_in: f |= C2._half("i_", Y + DH * g.cols[c], [(1, HW), (HW, n_in), (HW * n_own, nh)])
    if c > 0 and n_own and n_in:
      f |= dict(s0_id=C2.REC_IN[C2.MESH_W], s0_val=4 + bin(f["o_sdims"]).count("1"), s0_en_a=1, s0_en_b=1)
    per.append(f or None)
  for m, f in _group(per): e.tile(RM.encode_mesh(opcode=C2.MESH_W, tile_mask=m, seq=e.seq, **f))

def q_to_columns(e:Emitter, g:Geom, Q:int, relays:list[int]):
  """q (ROWS[r] x 48 bytes at narrow Q on the column-0 tiles) to the same address on the tiles of every other column that holds
  positions: one eastward mesh move per destination column, relayed through the columns in between (fc.relay), with the mesh fence,
  drain and reset17s of the compiler's q moves (q_broadcast; for P = 4 it moves exactly this: one position per column)"""
  first = True
  for c in (1, 2, 3):
    if not g.cols[c]: continue
    if not first: e.sync(SC.sync_reset17)
    first = False
    for r in (0, 1):
      e.tile(RM.encode_mesh(opcode=C2.MESH_E, tile_mask=0x0101 << (4 * r), seq=e.seq, **_mesh_half("o_", Q, [(1, HW), (HW, ROWS[r])])))
    if c >= 2:
      for r in (0, 1):
        e.tile(FC.relay(C2.MESH_E, sum(0x0101 << (4 * r + cc) for cc in range(1, c)), e.seq, relays[c - 1], HW * ROWS[r], 0))
    for r in (0, 1):
      e.tile(RM.encode_mesh(opcode=C2.MESH_E, tile_mask=0x0101 << (4 * r + c), seq=e.seq, **_mesh_half("i_", Q, [(1, HW), (HW, ROWS[r])])))
    fence = 0xffff & ~sum(0x0101 << cc for cc in range(1, c))
    e.sync(SC.sync, tiles=fence, counters=SC.tc("MESH_WEST_IN", "MESH_EAST_OUT"), count=1 if c == 1 else cdiv(2 * HW, 4), units=1 << 9)
    e.sync(SC.sync_drain, fence)
    e.sync(SC.sync_reset17)

S_RING_OUTFEED_RESET = SC.scsync(smask=SC.scm("RING_OUTFEED"))

def tiles_to_smem(e:Emitter, g:Geom, S_:int, base_w:int, M:int, fifo:int):
  """every tile's scores (ROWS[r] x cols(c) words at narrow S_) into scalar memory, row h at word base_w + h*M: per tile a
  narrowToWide to the FIFO, a ringProducer to the scalar core (ordinal k) and a 2-D scalar-memory outfeed: the multi-row SOFTMAX's
  input path (eltops._sm_out) with one block per tile; then the RING_OUTFEED reset and the PRODUCER_A wait"""
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_drain)
  k = 0
  for t in range(16):
    r, c = divmod(t, 4)
    nh, s_ = ROWS[r], g.cols[c]
    if not s_: continue
    pk = cdiv(4 * nh * s_, 256)
    assert pk <= 2
    e.tile(WN.encode_narrow_to_wide(tile_mask=1 << t, seq=e.seq, narrow_addr=S_ // 4, narrow_lvl_mask=3, wide_addr=fifo, tail_lvl=2,
                                    sync_en1=0, sync_en2=0, **WN.ttu("n", [1, s_], [s_, nh], 6), **WN.ttu("w", [1], [pk], 4), wide_lvl_mask=1))
    e.tile(RM.encode_ringProducer(tile_mask=1 << t, seq=e.seq, addr=fifo, **ttu("", [1], [pk], 4), sdims=1, mode=1, pcfg=12 if pk > 1 else 4,
                                  r0_id=(int(pk > 1) << 5) | 17, r0_val=k, r0_en_a=1, r0_en_b=1, r1_id=13, r1_val=1, r1_en_a=1, r1_en_b=1,
                                  dest=1 << RM.SCALAR_CORE))
    e.scalar(SC.OUTFEED.encode(base=base_w + HEAD0[r] * M + g.p0(c), d0_stride=1, d0_limit=s_ - 1, d1_stride=(M - (s_ - 1)) & 0xffff,
                               d1_limit=nh - 1, d2_stride=-(s_ - 1 + (nh - 1) * M) & 0xffff, d3_stride=3, k173=1, rsv175=1, f222=0x3f))
    k += 1
  e.scalar(S_RING_OUTFEED_RESET)
  e.scalar(SC.output_wait(e.seq, k), seqs=2)

def smem_to_tiles(e:Emitter, tiles:int, X:int, src_w:int, row_words:int, rows:int, fifo:int, collapse:bool=True):
  """rows x row_words words of scalar memory from word src_w into narrow X of the tiles in `tiles` (dense): the path edgetpu_compiler
  uses to bring a SOFTMAX result back to tile 0 (a wideToNarrow and a ringConsumer on the receiver, then one scalar-memory infeed,
  bit 13); then a drain and a reset17. Use tiles = 1: with all 16 tiles in the bitmap it ran without hanging but delivered no data.
  Rows of one word: edgetpu_compiler writes the narrow walk as one level of `rows` words, with tail_en870 = 1 and tail_f871 = 0
  (meaning unknown); collapse=False keeps the general two-level walk and tail bits (gen_attention at P = 1, as it ran on the device)."""
  N = rows * row_words
  c = cdiv(4 * N, 256)
  if collapse and row_words == 1: nd = dict(**WN.ttu("n", [1], [rows], 6), narrow_lvl_mask=1, tail_en870=1, tail_f871=0)
  else: nd = dict(**WN.ttu("n", [1, row_words], [row_words, rows], 6), narrow_lvl_mask=3, tail_en870=0, tail_f871=1)
  if c == 1:
    e.tile(WN.encode_wide_to_narrow(tile_mask=tiles, seq=e.seq, wide_addr=fifo, **WN.ttu("w", [1], [1], 4), wide_circ=1, wide_rows=1,
                                    narrow_addr=X // 4, **nd, sync_wait_lvl=1, sync_id=14, sync_val=0x8000))
    e.tile(RM.encode_ringConsumer(tile_mask=tiles, seq=e.seq, addr=fifo, **ttu("", [1], [1], 4), cbuf=1, slots=1, mode=3, s_id=43, s_val=-64,
                                  s_cnt=1, s_en_b=1, s_y=1))
  else:
    e.tile(WN.encode_wide_to_narrow(tile_mask=tiles, seq=e.seq, wide_addr=fifo, **WN.ttu("w", [0], [c], 4), wide_rows=1, narrow_addr=X // 4,
                                    **nd, sync_id=14, sync_val=0x8000))
    e.tile(RM.encode_ringConsumer(tile_mask=tiles, seq=e.seq, addr=fifo, **ttu("", [0], [c], 4), slots=1, mode=1, s_id=11, s_val=-64, s_cnt=1,
                                  s_en_b=1, s_y=1))
  e.scalar(SC.INFEED.encode(rsv13=1, buf_off=src_w, d0_stride=1, d0_limit=N - 1, d1_stride=-(N - 1) & 0x1fffff, d2_stride=-(N - 1) & 0x1fffff,
                            k157=1, tiles=tiles, k434=15, f438=3))
  e.sync(SC.sync_drain)
  e.sync(SC.sync_reset17)

def narrow_sizes(g:Geom) -> dict:
  """the composed program's narrow buffers (bytes, every tile), in address order"""
  P, sm = g.P, g.smax
  return dict(Stg=2 * max(DH * P, 256), C=16, Q=2 * DH, R=3 * 64, XK=96 * sm, XV=96 * sm, Ms=384 * sm, Ys=96 * sm, S=8 * sm,
              Pn=24 * P, Mv=384 * sm, Yv=DH * (P + round_up(P, 4)), O=2 * DH)   # Yv: + the transposing relay's over-read of the last quad

def attention_alloc(g:Geom) -> dict:
  """our memory plan: no two buffers that are live at the same time share memory (the inputs share their staging and ring FIFO,
  the two MULs their FIFO). Narrow (bytes, every tile, narrow_sizes in order, 64-byte aligned): input staging, identity row, q,
  relay buffers, the K and V blocks, the scores' MUL intermediate, products and scores, the probabilities p (all 6 x P words on every
  tile), the p.V intermediate, products / gathered rows, the sums. Wide (64-byte units, from the top): identity row, output FIFO,
  q's and K / V's input FIFOs, MUL FIFO, SUM FIFO, the scores' scalar-memory FIFO, p's FIFOs (from scalar memory to tile 0, and of
  the broadcast to the other tiles), the p.V relay block. Scalar memory: the scores as words [6][P], then p as words [6][P]
  (4 copies of its byte each) at the top."""
  P = g.P
  a, at = {}, 0
  for name, size in narrow_sizes(g).items():
    a[name] = at
    at += round_up(size, 64)
  a["narrow_end"] = at
  wt = WIDE_TOP
  for name, size in [("const", 4), ("outfifo", 4), ("in_q", 8), ("in_kv", 8 * g4_of(P).c_in), ("mulfifo", 32), ("sumfifo", 16), ("smfifo", 8),
                     ("pfifo", 4), ("bfifo", 4), ("relay", 4 * 2 * cdiv(P, 4))]:
    wt -= size
    a[name] = wt
  a["wide_bottom"] = wt
  a["M"] = P
  a["p_smem_w"] = SMEM_TOP // 4 - 6 * P          # p words [6][P]
  a["s_smem_w"] = a["p_smem_w"] - 6 * P          # score words [6][P]
  assert a["outfifo"] == WIDE_OUT_FIFO and at <= 192 * 1024 and a["s_smem_w"] >= 0
  return a

DEFAULT_ATTN = dict(q=(1/16, 128), k=(1/16, 128), qk=(1/16, 128), s=(1/4, 128), p=(1/256, 0), v=(1/16, 128), pv=(1/256, 128), o=(1/16, 128),
                    beta=DH ** -0.5)

def gen_attention(P:int, quant:dict|None=None, debug_smem:bool=False) -> tuple[bytes, dict]:
  """ONE STAND_ALONE program: o[288] = concat_h softmax(beta * q_h . K_h^T) . V_h for 6 heads of 48 dims and P = 1..256 positions.
  inputs q [288], K and V [6][P][48] (head-major); quant: (scale, zero point) of q, k, qk (the q*K products), s (the scores), p (the
  probabilities), v, pv (the p*V products), o; beta (default 1/sqrt(48)). -> (program, host contract attention_io(P)).
  debug_smem: stop after the softmax and send the scalar memory's score and p words (12P words) to the host instead, with
  eltops.gen_softmax's ending (smem_to_host)."""
  if not 1 <= P <= 256: raise NotImplementedError("P must be 1..256 (the scalar memory holds 2 x 6 x 256 words)")
  q = DEFAULT_ATTN | (quant or {})
  g = Geom(P)
  a = attention_alloc(g)
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  # inputs: q onto the column-0 tiles (the compiler's scores input), then to the other columns; K and V as the compiler's p.V inputs
  input4d(e, 1, a["Stg"], a["Q"], a["C"], a["const"], a["in_q"], True)
  e.sync(SC.sync_reset17)
  q_to_columns(e, g, a["Q"], [a["R"], a["R"] + 64, a["R"] + 128])
  input4d(e, P, a["Stg"], a["XK"], a["C"], a["const"], a["in_kv"], False)
  e.sync(SC.sync_reset17)
  input4d(e, P, a["Stg"], a["XV"], a["C"], a["const"], a["in_kv"], False)
  e.sync(SC.sync_reset17)
  # scores: q broadcast over the positions (a stride-0 TTU level) times K, summed over the 48 channels
  mq = EL.mul_quant(q["q"], q["k"], q["qk"])
  def q_bcast(t): return (a["Q"], [1, 0, DH, DH * ROWS[t // 4]], [DH, g.cols[t % 4], ROWS[t // 4]])     # q_h for every position
  mul_stage(e, g, q_bcast, a["XK"], a["Ms"], a["Ys"], a["mulfifo"], a["const"], mq, lambda t: HW * g.cols[t % 4])
  chansum_stage(e, g, a["Ys"], a["S"], a["sumfifo"], sum_quant(q["qk"], q["s"]))
  # the softmax on the scalar core
  if debug_smem: e.scalar([S.movi(18, 0), S.movi(19, 0)])                   # the OUTPUT address (relocated), as gen_softmax
  tiles_to_smem(e, g, a["S"], a["s_smem_w"], a["M"], a["smfifo"])
  e.scalar(softmax_words(NH, P, a["s_smem_w"], a["M"], a["p_smem_w"], q["s"][1], f32(q["s"][0]), q["beta"], q["p"][0], q["p"][1]))
  if debug_smem:
    from coral.codegen.eltops import smem_to_host
    e.scalar(smem_to_host(a["s_smem_w"], 12 * P, 48 * P))
    e.sync(SC.epilogue)
    return e.program(), dict(inputs=attention_io(P)["inputs"], out_bytes=48 * P)
  # p to tile 0 exactly as edgetpu_compiler returns a SOFTMAX result (a scalar-memory infeed with a 16-tile bitmap delivered no p on the
  # device), then to the other tiles with FULLY_CONNECTED's ring broadcast of its input (fc.broadcast: tile 0 -> the compute tiles)
  smem_to_tiles(e, 1, a["Pn"], a["p_smem_w"], P, NH, a["pfifo"], collapse=False)    # P = 1: the walk that ran on the device
  if dest := sum(1 << t for t in range(1, 16) if g.cols[t % 4]):
    e.scalar(SC.scsync_nop())
    e.sync(SC.sync_reset17)
    FC.broadcast(e, FC.FCGeom(64, 24 * P), a["Pn"], 1, dest, a["bfifo"])
    e.sync(SC.sync_reset17)
  # p.V: p broadcast over the 48 channels (stride 0, the word of 4 copies) times V, gathered on the column-0 tiles, summed over positions
  mp = EL.mul_quant(q["p"], q["v"], q["pv"])
  def p_bcast(t):       # p of head h, position j: the word at Pn + 4 (P h + j), read 48 times
    r, c = divmod(t, 4)
    return a["Pn"] + 4 * (P * HEAD0[r] + g.p0(c)), [0, 4, 4 * P, 4 * P * ROWS[r]], [DH, g.cols[c], ROWS[r]]
  mul_stage(e, g, p_bcast, a["XV"], a["Mv"], a["Yv"], a["mulfifo"], a["const"], mp, lambda t: HW * (P - g.p0(t % 4)))
  possum_any(e, g, a["Yv"], a["O"], a["relay"], a["const"], sum_quant(q["pv"], q["o"]))
  e.sync(SC.sync_reset17)
  output_col0(e, a["O"])
  return e.program(), attention_io(P)

def attention_io(P:int) -> dict:
  """host contract of gen_attention: three input DMAs in this order, q [288] (288 bytes), K and V [6][P][48] uint8 head-major
  (K[h][p][c] = k_p[48h + c], 288P bytes each); one output DMA, o [288] (head h at 48h). Every size is a multiple of 8."""
  return dict(inputs=[("q", DM), ("k", DM * P), ("v", DM * P)], out_bytes=DM)

def attention_executable(prog:bytes, P:int, io:dict|None=None):
  """a coral.executable.Executable of the program for coral.runtime.run_executable: the input hints point into ONE host buffer
  q | K | V (attention_inputs), the output hint is o (io: another host contract, e.g. gen_attention(debug_smem=True)'s)"""
  from coral.executable import Executable, Bitstream, Hint
  io = io or attention_io(P)
  dmas = [(t, n) for t, n in S.dma_seqs(prog) if t in (SC.TAG_INPUT, SC.TAG_OUTPUT)]
  assert dmas == [(SC.TAG_INPUT, n) for _, n in io["inputs"]] + [(SC.TAG_OUTPUT, io["out_bytes"])], dmas
  hints, off = [Hint("instruction", "INFEED", chunk=0)], 0
  for name, n in io["inputs"]:
    hints.append(Hint("dma", "INFEED", "INPUT", name, off, n))
    off += n
  hints += [Hint("dma", "OUTFEED", "OUTPUT", "o", 0, io["out_bytes"]), Hint("interrupt", "OUTFEED", interrupt=0)]
  return Executable(None, "STAND_ALONE", 1, 0, [Bitstream(prog, [])], b"", hints, True, [], [], "beagle", 0, 0, 0, 0)

def attention_inputs(q:np.ndarray, K:np.ndarray, V:np.ndarray) -> bytes:
  """the host buffer of attention_executable: q [288], K and V as [P][288] (one row per position, the KV cache rows) -> q | K | V
  with K and V head-major [6][P][48]"""
  return np.asarray(q, np.uint8).tobytes() + heads(K).tobytes() + heads(V).tobytes()

# ***** numpy bit models (the device arithmetic as decoded; docs/isa/codegen_attention.md 6) *****
def _requant(acc:np.ndarray, mult:float, lo:float, hi:float, zp:int) -> np.ndarray:
  """a tile op's output stage: f32(acc) * f32 multiplier, clamped in float32 (bounds relative to the zero point), rounded half to
  even, plus the output zero point (NOTES.md: requantization)"""
  F = np.float32
  y = np.clip(np.asarray(acc).astype(F) * F(mult), F(lo), F(hi))
  return (np.rint(y).astype(np.int64) + zp).astype(np.uint8)

def mul_ref(a:np.ndarray, b:np.ndarray, a_q:tuple, b_q:tuple, y_q:tuple) -> np.ndarray:
  """MUL of two uint8 tensors (broadcasting as numpy): (a - za)(b - zb) exactly, then requantized with eltwise.mul_quant"""
  q = EL.mul_quant(a_q, b_q, y_q)
  acc = (np.asarray(a).astype(np.int64) - q["a_zp"]) * (np.asarray(b).astype(np.int64) - q["b_zp"])
  return _requant(acc, q["mult"], q["clamp_min"], q["clamp_max"], q["out_zp"])

def sum_ref(x:np.ndarray, axis:int, in_q:tuple, out_q:tuple) -> np.ndarray:
  """SUM over an axis (the opcode-3 reductions): sum(x - zx) exactly, requantized with sum_quant (eltops.reduce_ref's arithmetic)"""
  q = sum_quant(in_q, out_q)
  acc = (np.asarray(x).astype(np.int64) - q["in_zp"]).sum(axis)
  return _requant(acc, q["mult"], q["clamp_min"], q["clamp_max"], q["out_zp"])

def heads(X:np.ndarray) -> np.ndarray:
  """[P][288] rows (position-major, the KV cache) -> [6][P][48] head-major"""
  X = np.asarray(X, np.uint8)
  return np.ascontiguousarray(X.reshape(X.shape[0], NH, DH).transpose(1, 0, 2))

def scores_ref(q:np.ndarray, Kh:np.ndarray, quant:dict|None=None) -> np.ndarray:
  """gen_scores' bit model: q [6][48] (or [288]), Kh [6][P][48] -> s [6][P]: MUL (q broadcast over the positions), SUM over 48"""
  qq = DEFAULT_SCORES | (quant or {})
  prod = mul_ref(np.asarray(q, np.uint8).reshape(NH, 1, DH), Kh, qq["q"], qq["k"], qq["qk"])
  return sum_ref(prod, 2, qq["qk"], qq["s"])

def pv_ref(p_exp:np.ndarray, Vh:np.ndarray, quant:dict|None=None) -> np.ndarray:
  """gen_pv's bit model: p_exp, Vh [6][P][48] -> o [6][48]: MUL, SUM over the positions"""
  qq = DEFAULT_PV | (quant or {})
  return sum_ref(mul_ref(p_exp, Vh, qq["p"], qq["v"], qq["pv"]), 1, qq["pv"], qq["o"])

def attention_ref(q:np.ndarray, K:np.ndarray, V:np.ndarray, quant:dict|None=None, parts:bool=False):
  """gen_attention's bit model from the pieces' models: q [288], K and V [P][288] uint8 -> o [288] uint8.
  scores = sum_ref(mul_ref(q bcast, K)), p = eltops.softmax_ref(scores, beta), o = sum_ref(mul_ref(p bcast, V)).
  parts=True: also returns {'s': scores [6][P], 'p': p [6][P]}"""
  qq = DEFAULT_ATTN | (quant or {})
  from coral.codegen.eltops import softmax_ref
  Kh, Vh = heads(K), heads(V)
  s_ = sum_ref(mul_ref(np.asarray(q, np.uint8).reshape(NH, 1, DH), Kh, qq["q"], qq["k"], qq["qk"]), 2, qq["qk"], qq["s"])
  p_ = softmax_ref(s_, qq["s"], qq["p"], qq["beta"])
  o = sum_ref(mul_ref(p_[:, :, None], Vh, qq["p"], qq["v"], qq["pv"]), 1, qq["pv"], qq["o"]).reshape(DM)
  return (o, dict(s=s_, p=p_)) if parts else o

def scores_io(P:int) -> dict:
  """host contract of gen_scores: inputs q [6][48] (288 bytes) and K [6][P][48]; output: per tile (in tile order) its block of
  ROWS[r] x cols(c) scores, 4 bytes each (the score in the first byte), each block padded to 8 bytes, or packed back to back and
  the total padded to 8 when a block other than the last is not a multiple of 8 (scalar-memory path)"""
  g = Geom(P)
  blocks = [4 * ROWS[t // 4] * g.cols[t % 4] for t in range(16) if g.cols[t % 4]]
  smem = any(b % 8 for b in blocks[:-1])
  return dict(in_bytes=[DM, DM * P], out_bytes=round_up(sum(blocks), 8) if smem else sum(round_up(b, 8) for b in blocks), smem=smem)

def scores_unpack(buf:bytes, P:int) -> np.ndarray:
  """gen_scores' output bytes -> s [6][P] uint8"""
  g, io = Geom(P), scores_io(P)
  out, at = np.zeros((NH, P), np.uint8), 0
  b = np.frombuffer(buf, np.uint8)
  for t in range(16):
    r, c = divmod(t, 4)
    nh, s_ = ROWS[r], g.cols[c]
    if not s_: continue
    blk = b[at:at + 4 * nh * s_].reshape(nh, s_, 4)[:, :, 0]
    out[HEAD0[r]:HEAD0[r] + nh, g.p0(c):g.p0(c) + s_] = blk
    at += 4 * nh * s_ if io["smem"] else round_up(4 * nh * s_, 8)
  return out

if __name__ == "__main__": run_test("test_codegen_attention")
