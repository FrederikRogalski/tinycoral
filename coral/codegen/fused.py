# fused LLM programs, byte-identical to edgetpu_compiler (docs/isa/codegen_fused.md):
#   caching, execution = gen_ffn(Mp, quant=q)              y = w2((h1*sigmoid(h1)) * h3), h1 = w1 x, h3 = w3 x
#   caching, execution = gen_argmax(Mp, quant=q, placement=(10, 326912))   the classifier as a pooled conv
#   progs = gen_set(llm_specs(), Mp, plan(llm_specs(), Mp), quants)          the whole TinyStories-15M set, placed by us
# `python -m coral.codegen.fused` runs the acceptance test (test/test_codegen_fused.py).
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from coral.isa import cdiv, round_up, ttu, f32_bits, split, opcode
from coral.isa import op as OP, wide_narrow as WN, ring_mesh as RM, scalar as SC, eltwise as EL
from coral.codegen import Emitter, Piece, caching_program, piece_parts, run_test, WIDE_TOP, WIDE_OUT_FIFO, REF_QUANT
from coral.codegen import fc as FC, conv as CC
from coral.codegen.conv import ConvGeom, GRIDS
from coral.executable import Executable, Bitstream, Hint, Layer

IDENT = WIDE_TOP - 4            # the 4x4 identity row (constant row of the copy ops, weights of LOGISTIC and MUL step 2)

# *** quantization and parameter blobs ***
def conv_quant(x_q:tuple, w_q:tuple, y_q:tuple) -> dict:
  """conv / FC op fields of y = requant(x @ W.T): multiplier f32(f32(s_x * s_w) * f32(1 / s_y)); the clamps are the float32 round
  trip of the output range (eltwise.out_clamps: -zp and 255 - zp up to one ulp)"""
  lo, hi = EL.out_clamps(y_q[0], y_q[1])
  return dict(w_zp=w_q[1], in_zp=x_q[1], out_zp=y_q[1], mult=EL.mul32(EL.mul32(x_q[0], w_q[0]), EL.recip32(y_q[0])), clamp_min=lo,
              clamp_max=hi)

def ffn_ref_quant() -> dict:
  """a quantization of every FFN tensor ((scale, zero point)); g = LOGISTIC(h1) is fixed to (1/256, 0) by TFLite"""
  return dict(x=(1/32, 128), h1=(1/8, 128), h3=(1/8, 128), g=(1/256, 0), a=(1/16, 20), m=(1/16, 128), y=(1/4, 128),
              w1=(1/64, 128), w3=(1/64, 128), w2=(1/64, 128))

def conv_blob(wq:np.ndarray, w_zp:int, bias:np.ndarray|None=None, group:int=64) -> bytes:
  """one matmul's parameters as edgetpu_compiler lays them out (FULLY_CONNECTED and 1x1 conv alike): per group of `group` outputs an
  int32 bias row, then the weights as [K4/4][group][4] uint8. K is padded to a multiple of 4 with w_zp, the padded outputs are 0."""
  N, K = wq.shape
  Np, K4 = round_up(N, group), round_up(K, 4)
  w = np.zeros((Np, K4), np.uint8)
  w[:N, :] = w_zp
  w[:N, :K] = wq
  b = np.zeros(Np, "<i4")
  if bias is not None: b[:N] = bias
  return b"".join(b[g:g + group].tobytes() + np.ascontiguousarray(w[g:g + group].reshape(group, K4 // 4, 4).transpose(1, 0, 2)).tobytes()
                  for g in range(0, Np, group))

def ffn_params(W1q:np.ndarray, W3q:np.ndarray, W2q:np.ndarray, b1=None, b3=None, b2=None,
               zps:tuple[int, int, int]=(128, 128, 128)) -> bytes:
  """the FFN parameter blob: conv_blob(w1) + conv_blob(w3) + conv_blob(w2) in every form (W1q, W3q: [Hd, D], W2q: [D, Hd] uint8; b*:
  int32 biases, None = 0; zps: weight zero points). LOGISTIC's spline is an immediate of the 0x19 instruction."""
  return b"".join(conv_blob(w, z, b) for w, z, b in zip((W1q, W3q, W2q), zps, (b1, b3, b2)))

def blocks_of(N:int, K:int) -> int: return cdiv(N, 64) * (1 + round_up(K, 4) // 4)   # 256-byte rows: per 64 outputs bias + K4/4 weights

# *** elementwise stages, conv form (parts = [(0xffff, the tile block, 0)]) and FC form (parts = per tile its 64-byte slice) ***
Part = tuple[int, dict, int]    # (tile mask, eltwise.tile_geometry, byte offset of these tiles' slice)

def _logistic_stage(e:Emitter, parts:list[Part], src:int, dst:int, in_q:tuple, out_q:tuple, ident:int, ident_narrow:int|None=None):
  """dst = LOGISTIC(src): (the identity row first if ident_narrow is given), the NLU spline load, one op per part"""
  e.sync(SC.sync_reset17)
  if ident_narrow is not None: e.tile(*EL.ident_prologue(e.seq, 0xffff, ident_narrow, ident))
  e.sync(SC.sync_op_fence)
  e.tile(EL.encode_nlu(**EL.nlu_fields(sum(m for m, _, _ in parts), e.seq)))
  for m, tg, o in parts:
    e.tile(OP.encode_op(**EL.logistic_op_fields(tg["P"], tg["w"], tg["R"][0], tg["R"][0], m, e.seq, src + o, dst + o, ident,
                                                **EL.logistic_quant(in_q, out_q[1]), flat=tg["flat"])))
  e.sync(SC.sync_reset17)

def _mul_stage(e:Emitter, parts:list[Part], a:int, b:int, tmp:int, out:int, fifo:int, ident:int, q:dict):
  """out = a * b: step 1 (a x the FIFO of b, into the tile-local intermediate tmp), the FIFO feeds of b, step 2 (pack into out)"""
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_reset_n2w)
  e.sync(SC.sync_reset_par)
  def geo(tg): return (tg["w"], tg["cols"], tg["rows"])
  for m, tg, o in parts: e.tile(OP.encode_op(**EL.mul_op1_fields(*geo(tg), tg["R"], m, e.seq, a + o, tmp, fifo, **q)))
  for m, tg, o in parts: e.tile(WN.encode_narrow_to_wide(**EL.mul_feed_n2w_fields(e.seq, m, b + o, fifo, *geo(tg))))
  e.sync(SC.sync_wn_fence)
  for m, tg, o in parts: e.tile(OP.encode_op(**EL.mul_op2_fields(*geo(tg), tg["R"], m, e.seq, tmp, out + o, ident)))
  e.sync(SC.sync_reset_n2w)
  e.sync(SC.sync_reset_par)
  e.sync(SC.sync_reset17)

# *** FFN, conv form (Mp in 16..256) ***
# narrow memory (byte addresses, every tile) as edgetpu_compiler allocates it for D=288, Hd=768: Stg = input staging, C = identity row,
# X = x, h1, ga = g and then a (MUL writes its output over the FIFO operand), int1 / int2 = the MUL intermediates, hm = h3 and then m,
# y. The allocator's ordering is not understood (it changes with Mp), so the addresses are read from its programs.
FFN_NARROW = {16: dict(Stg=0, C=2304, X=4612, h1=768, ga=0, int1=1536, hm=768, int2=1536, y=0),
              32: dict(Stg=0, C=2304, X=3072, h1=1536, ga=0, int1=3648, hm=1536, int2=3072, y=0),
              64: dict(Stg=0, C=4608, X=6144, h1=3072, ga=0, int1=7296, hm=3072, int2=6144, y=0),
              128: dict(Stg=0, C=4608, X=6144, h1=8448, ga=0, int1=14592, hm=30724, int2=6144, y=0),
              256: dict(Stg=0, C=9216, X=12288, h1=16896, ga=0, int1=29184, hm=61444, int2=12288, y=0)}
FFN_ALONE_TILES = (1, 3, 0)     # edgetpu_compiler's tiles for w1, w3, w2 when the FFN is compiled alone (conv form)

@dataclass(frozen=True)
class FFNGeom:
  Mp: int
  D: int = 288
  Hd: int = 768
  @property
  def g1(self) -> ConvGeom: return ConvGeom(self.Mp, self.Hd, self.D)      # w1, w3: x[D] -> h[Hd]
  @property
  def g2(self) -> ConvGeom: return ConvGeom(self.Mp, self.D, self.Hd)      # w2: m[Hd] -> y[D]
  @property
  def blocks(self) -> tuple[int, int, int]: return (blocks_of(self.Hd, self.D),) * 2 + (blocks_of(self.D, self.Hd),)
  def wide(self) -> dict:
    """wide buffers (64-byte units): the identity row at the top, the input FIFO below it; the compute regions of w1 / w3 below the
    identity (still needed by the MUL packs), of w2 below the top (the identity is dead by then); the MUL operand FIFO (8 units per
    row) below the identity"""
    in_fifo, mul = IDENT - 8 * self.g1.c_in, IDENT - 8 * EL.mul_fifo_depth(cdiv(self.Hd, 4))
    r1, r2 = CC.regions(self.g1, IDENT), CC.regions(self.g2, WIDE_TOP)
    return dict(ident=IDENT, in_fifo=in_fifo, conv1=r1, conv2=r2, mul_fifo=mul, out_fifo=WIDE_OUT_FIFO,
                bottom=min(in_fifo, r1["bottom"], r2["bottom"], mul))

def ffn_param_limit(Mp:int, D:int=288, Hd:int=768) -> int:
  """lowest wide address (64-byte units) the FFN execution program uses on every tile: resident parameters must end below"""
  return min(FC_WIDE.values()) if Mp == 1 else FFNGeom(Mp, D, Hd).wide()["bottom"]

def _ffn_conv_exec(F:FFNGeom, pieces:list[list[Piece]], q:dict) -> bytes:
  g1, g2, nl, wl = F.g1, F.g2, FFN_NARROW[F.Mp], F.wide()
  parts = [(0xffff, EL.tile_geometry([1, *GRIDS[F.Mp], F.Hd], 0), 0)]
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  CC.input_stage(e, g1, nl["Stg"], nl["X"], nl["C"], wl["in_fifo"], wl["ident"])
  rows = CC.conv_stage(e, g1, nl["X"], nl["h1"], wl["conv1"], conv_quant(q["x"], q["w1"], q["h1"]), pieces[0])
  _logistic_stage(e, parts, nl["h1"], nl["ga"], q["h1"], q["g"], wl["ident"])
  _mul_stage(e, parts, nl["h1"], nl["ga"], nl["int1"], nl["ga"], wl["mul_fifo"], wl["ident"], EL.mul_quant(q["h1"], q["g"], q["a"]))
  rows = CC.conv_stage(e, g1, nl["X"], nl["hm"], wl["conv1"], conv_quant(q["x"], q["w3"], q["h3"]), pieces[1], rows)
  _mul_stage(e, parts, nl["ga"], nl["hm"], nl["int2"], nl["hm"], wl["mul_fifo"], wl["ident"], EL.mul_quant(q["a"], q["h3"], q["m"]))
  rows = CC.conv_stage(e, g2, nl["hm"], nl["y"], wl["conv2"], conv_quant(q["m"], q["w2"], q["y"]), pieces[2], rows)
  CC.output_stage(e, g2, nl["y"], rows)
  return e.program()

def ffn_pieces(Mp:int, D:int=288, Hd:int=768, param_tile=None, param_offset=0, param_limit_units:int|None=None) -> list[list[Piece]]:
  """the conv-form parameter pieces of w1, w3, w2. param_tile / param_offset: one entry per matmul (an int, or a tuple of tiles /
  byte offsets for a matmul split over tiles: every piece but the last fills its tile up to the limit); param_offset may also be one
  int for everything. Default: edgetpu_compiler's alone placement (tiles 1, 3, 0 at 0)."""
  offs = (param_offset,) * 3 if isinstance(param_offset, int) else tuple(param_offset)
  limit = ffn_param_limit(Mp, D, Hd) if param_limit_units is None else param_limit_units
  def units(o): return o // 64 if isinstance(o, int) else tuple(x // 64 for x in o)
  tiles = FFN_ALONE_TILES if param_tile is None else param_tile
  return [CC.split_pieces(n, t, units(o), limit) for n, t, o in zip(FFNGeom(Mp, D, Hd).blocks, tiles, offs)]

# *** FFN, FULLY_CONNECTED form (Mp = 1) ***
# w1 and w3 (768 outputs) run on tiles 0..11 (64 outputs each), so h1, g, a, h3 and m are 64-byte slices at base + 64t and LOGISTIC /
# MUL run per tile; x is broadcast to tiles 1..11 again before w3 (MUL's intermediate overwrites it); m is gathered on tile 0 over the
# mesh (the FC input gather with 12 input tiles) and broadcast to tiles 1..4 for w2. edgetpu_compiler's allocation for D=288, Hd=768:
FC_NARROW = dict(X=1536, h=768, g=0, int1=1824, int2=1536, ident=1824, R1=0, R2=288)   # bytes
FC_WIDE = dict(fwd1=0x1f70, fwd3=0x1f70 - 4, fwd2=0x1f70, ident=8060, mul_fifo=8288)  # 64-byte units
FC_ALONE_OFFSETS = (292 * 64, 0, 584 * 64)   # w1, w3, w2 when compiled alone (bytes)
# streamed FULLY_CONNECTED: all weights of a matmul on one parameter tile, streamed to the compute tiles every call (edgetpu_compiler's
# form for one-row matmuls that do not fit spread over the tiles)
STREAM_SLOTS = 32               # rows of the per-tile weight FIFO (ringBusConsumer1)
STREAM_FIFO = 8064              # weight FIFO, 32 rows x 4 units = 8064..8192
STREAM_BIAS = 8056              # bias rows (two 4-unit blocks); 8052 when the identity row (8060) is live
def _fills(kw:int) -> tuple[int, int]: return cdiv(kw, STREAM_SLOTS), kw - STREAM_SLOTS * (cdiv(kw, STREAM_SLOTS) - 1)   # FIFO fills, last fill

def _ffn_fc_exec(D:int, Hd:int, q:dict, stages) -> bytes:
  """stages[i](e, g, X, Y, quant, tail): matmul i (spread or streamed)"""
  g1, g2, nl, wl, pl = FC.FCGeom(Hd, D), FC.FCGeom(D, Hd), FC_NARROW, FC_WIDE, FC.Place()
  parts = [(1 << t, EL.tile_geometry([1, Hd], t), 64 * t) for t in range(g1.T)]
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  FC.input_block(e, g1, pl, nl["X"], nl["R1"])
  stages[0](e, g1, nl["X"], nl["h"], conv_quant(q["x"], q["w1"], q["h1"]), False)
  _logistic_stage(e, parts, nl["h"], nl["g"], q["h1"], q["g"], wl["ident"], nl["ident"])
  _mul_stage(e, parts, nl["h"], nl["g"], nl["int1"], nl["g"], wl["mul_fifo"], wl["ident"], EL.mul_quant(q["h1"], q["g"], q["a"]))
  e.sync(SC.sync_reset17)
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_reset17)
  FC.broadcast(e, g1, nl["X"], 1, (1 << g1.T) - 2, wl["fwd3"])
  e.sync(SC.sync_reset17)
  stages[1](e, g1, nl["X"], nl["h"], conv_quant(q["x"], q["w3"], q["h3"]), False)
  _mul_stage(e, parts, nl["g"], nl["h"], nl["int2"], nl["h"], wl["mul_fifo"], wl["ident"], EL.mul_quant(q["a"], q["h3"], q["m"]))
  e.sync(SC.sync_reset17)
  FC.gather(e, g2, nl["h"], nl["R2"], pl.i)
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_reset17)
  FC.broadcast(e, g2, nl["h"], 1, (1 << g2.T) - 2, wl["fwd2"])
  e.sync(SC.sync_reset17)
  stages[2](e, g2, nl["h"], nl["g"], conv_quant(q["m"], q["w2"], q["y"]), True)
  FC.output_block(e, g2, pl, nl["g"])
  e.sync(SC.epilogue)
  return e.program()

def _stream_op(g:FC.FCGeom, t:int, seq:int, X:int, Y:int, q:dict) -> dict:
  """the FC op of tile t reading its weights from the FIFO: the K loop runs in b fills of 32 rows (the last one r rows)"""
  kw, D, (b, r) = g.kw, STREAM_SLOTS, _fills(g.kw)
  if r not in (D, 8): raise NotImplementedError(f"streamed FC with a partial last FIFO fill of {r} rows (verified: 32 and 8)")
  assert g.gt(t) == 1, "streamed FC: one 64-output group per compute tile"
  f = OP.fc_tile_fields(g.N, g.K, t, seq, X, Y + 64 * g.P * t, 0, q["w_zp"], q["in_zp"], q["mult"], q["out_zp"], q.get("clamp_min"),
                        q.get("clamp_max"))
  f |= dict(loop0=D - 1, loop2=b - 1, loop3=0, **ttu("in_", [1, kw, D], [D, 1, b], 8), **ttu("par_", [1, 0, 0], [D, 1, b], 8),
            **ttu("psum_", [0, 1, 0], [D, 1, b], 8), **OP.wide("par", STREAM_FIFO), par_tflags=0x80, par_fifo=D, cfg1=0x14F, sync2=0xB6,
            sync3=0x80, sync4=0x80, psum_hmode=0, psum_tflags=0)
  if r != D:        # partial last fill: loop0 / in / par last-round records (observed for r = 8 of 32 rows)
    f |= dict(rsv174=0x8000 | (r - 1), in_twait=0x2000, rsv536=(D - r) << 22 | 1 << 18 | 3, par_fifo=D | 0x2000,
              rsv1230=(D - r) << 18 | 1 << 14 | 3, rsv1523=(r - 1) << 9 | 1 << 24)
  return f

def _stream_stage(e:Emitter, g:FC.FCGeom, X:int, Y:int, ptile:int, poff:int, q:dict, bias:int=STREAM_BIAS, tail:bool=False):
  """compute fence, per compute tile a weight FIFO consumer, the bias load, the ops, then the parameter tile streams group t (bias
  row + K4/4 weight rows) to tile t; wait for all of them (+ the completion fence)"""
  D, per, (b, r) = STREAM_SLOTS, 1 + g.kw, _fills(g.kw)
  e.sync(SC.signal_fence)
  e.sync(SC.sync_drain)
  for t in range(g.T):   # the K4/4 rows of a group stream through the 32-row FIFO; a partial last fill as in the input ring consumers
    e.tile(RM.encode_ringConsumer(opcode=0x12, tile_mask=1 << t, seq=e.seq, addr=STREAM_FIFO, aux_en0=1, aux_en1=1, s_en_a=1, s_en_b=1,
                                  s_y=1, sdims=1, cbuf=1, mode=3, s_id=(1 << 5) | 1, slots=D, aux_addr0=bias + 2, aux_addr1=bias + 4, s_val=-1,
                                  **ttu("", [1, 0, 0, 0], [D, b, 1, 1], 4), **(dict(grp=r - 1, gstride=4 * (D - r) + 1) if r != D else {})))
  e.tile(CC.bias_load(e.seq, (1 << g.T) - 1, bias, b))
  for t in range(g.T): e.tile(OP.encode_op(**_stream_op(g, t, e.seq, X, Y, q)))
  for t in range(g.T): e.tile(CC.rprod_param(e.seq, ptile, poff + 4 * per * t, per, per * t, dest=1 << t))
  e.sync(SC.sync_reset17)
  e.sync(SC.broadcast_wait, per * g.G)
  if tail: e.sync(SC.signal_fence)

def gen_fc_streamed(N:int, K:int, param_tile:int, param_offset:int, quant:dict|None=None) -> tuple[bytes, bytes]:
  """(PARAMETER_CACHING, EXECUTION_ONLY) of y[N] = W[N,K] x[K] with all parameters on one tile (param_offset bytes into its wide
  memory), streamed to the compute tiles 0..T-1 on every call; input and output as gen_fc. Verified for the TinyStories shapes (K = 288)."""
  assert param_offset % 256 == 0
  g, q = FC.FCGeom(N, K), {**REF_QUANT, **(quant or {})}
  assert g.P == 1 and g.T <= 15, "streamed FC: N <= 960 (one group per tile, parameter tile not counted)"
  X, Y, R = FC.narrow_layout(N, K)
  exe = FC.exe_program(g, FC.Place(), X, R, Y, lambda e: _stream_stage(e, g, X, Y, param_tile, param_offset // 64, q, tail=True))
  return caching_program(256 * blocks_of(N, K), piece_parts([[(param_tile, param_offset // 64, blocks_of(N, K))]])), exe

def _ffn_fc(D:int, Hd:int, q:dict, param_tile=None, param_offset=None, pieces=None) -> tuple[bytes, bytes]:
  gs = [FC.FCGeom(Hd, D), FC.FCGeom(Hd, D), FC.FCGeom(D, Hd)]
  if param_tile is None and pieces is None:   # spread: matmul i on its tiles 0..T-1 at one offset (bytes, a multiple of 256)
    offs = [o // 64 for o in (FC_ALONE_OFFSETS if param_offset is None else param_offset)]
    assert len(offs) == 3 and all(o % 4 == 0 for o in offs), "param_offset: three multiples of 256 bytes (w1, w3, w2)"
    parts = [FC.caching_part(g, FC.Place(), o, sum(x.param_bytes for x in gs[:i])) for i, (g, o) in enumerate(zip(gs, offs))]
    stages = [lambda e, g, X, Y, qq, tail, o=o: FC.ops_block(e, g, FC.Place(), X, Y, o, qq, tail) for o in offs]
    return caching_program(sum(g.param_bytes for g in gs), parts), _ffn_fc_exec(D, Hd, q, stages)
  blocks = [blocks_of(g.N, g.K) for g in gs]   # streamed: each matmul on one tile
  if pieces is None:
    offs = (param_offset or 0,) * 3 if isinstance(param_offset or 0, int) else tuple(param_offset)
    pieces = [[(t, o // 64, n)] for t, o, n in zip(param_tile, offs, blocks)]
  assert all(len(ps) == 1 for ps in pieces), "streamed FC form: one parameter tile per matmul"
  assert [ps[0][2] for ps in pieces] == blocks and all(ps[0][1] % 4 == 0 for ps in pieces)
  stages = [lambda e, g, X, Y, qq, tail, t=ps[0][0], o=ps[0][1], bias=bias: _stream_stage(e, g, X, Y, t, o, qq, bias, tail)
            for ps, bias in zip(pieces, (STREAM_BIAS, STREAM_BIAS - 4, STREAM_BIAS))]
  return caching_program(256 * sum(blocks), piece_parts(pieces)), _ffn_fc_exec(D, Hd, q, stages)

def gen_ffn(Mp:int, D:int=288, Hd:int=768, quant:dict|None=None, param_tile=None, param_offset=None,
            param_limit_units:int|None=None, pieces:list[list[Piece]]|None=None) -> tuple[bytes, bytes]:
  """(PARAMETER_CACHING, EXECUTION_ONLY) of the FFN y = w2((h1 * sigmoid(h1)) * h3), h1 = w1 x, h3 = w3 x for Mp rows
  (coral.fused.ffn_block), byte-identical to edgetpu_compiler for D=288, Hd=768 and Mp in {1, 16, 32, 64, 128, 256}.
  quant: (scale, zero point) of x, h1, h3, g, a, m, y and of the weights w1, w3, w2 (ffn_ref_quant() has the keys). The parameter
  blob is ffn_params(...) in every form. Placement:
    Mp >= 16 (1x1 convs on all 16 tiles, weights broadcast from their tiles every call): pieces = [[(tile, offset in 64-byte units,
      rows), ...] for w1, w3, w2], or param_tile / param_offset (bytes) / param_limit_units as for ffn_pieces(); default tiles (1, 3, 0).
    Mp = 1, spread (FULLY_CONNECTED, w1 / w3 on tiles 0..11, w2 on tiles 0..4): param_offset = (w1, w3, w2) in bytes, default
      (18688, 0, 37376) as compiled alone.
    Mp = 1, streamed (each matmul on one tile, streamed to the compute tiles every call): param_tile = (t1, t3, t2) with param_offset
      (bytes, one or three), or pieces = [[(tile, offset units, rows)]] * 3."""
  q = {**ffn_ref_quant(), **(quant or {})}
  if (D, Hd) != (288, 768): raise NotImplementedError("the narrow / wide allocation is only known for D=288, Hd=768")
  if Mp == 1: return _ffn_fc(D, Hd, q, param_tile, param_offset, pieces)
  assert Mp in GRIDS, f"Mp must be 1 or one of {list(GRIDS)}"
  F = FFNGeom(Mp, D, Hd)
  if pieces is None: pieces = ffn_pieces(Mp, D, Hd, param_tile, param_offset or 0, param_limit_units)
  assert [sum(n for _, _, n in ps) for ps in pieces] == list(F.blocks), "pieces must cover the three blobs"
  return caching_program(256 * sum(F.blocks), piece_parts(pieces)), _ffn_conv_exec(F, pieces, q)

# *** the classifier as a pooled conv ("argmax") ***
# coral.fused.argmax_block: CONV_2D of the vocabulary image [1, 160, 200, 288] (streamed in) with the Mp token activations as its
# weights [Mp, 1, 1, 288] (cached every call), then MAX_POOL 8x8 / 8 -> [1, 20, 25, Mp]. edgetpu_compiler unrolls it: 10 chunks of
# 16 image rows (one 921600-byte input DMA, one 50*Mp-byte output DMA each), 13 steps per chunk (4 pooling windows of 8x8 = 16
# image columns; the last step 1 window of 8 columns), then an output phase that moves the 50 pooled results into the output blocks.
# ~7400 instructions in 3 bitstreams of <= 16384 words.
AM_V, AM_K, AM_GRID, AM_POOL = 32000, 288, (160, 200), 8
AM_ROWS = 16                             # image rows per chunk (two rows of pooling windows)
AM_CHUNKS, AM_STEPS = AM_GRID[0] // AM_ROWS, 13
AM_IN_BYTES = AM_ROWS * AM_GRID[1] * AM_K
AM_STASH = IDENT - 900                   # wide copy of the tile's chunk block (225 rows) right below the identity row
AM_MAX_WORDS = 16384                     # instruction words per bitstream (start ... end)
OUT_COLS = (7, 6, 6, 6)                  # pooled columns per output tile column (tile_split(25))
OUT_TILES = (0, 1, 2, 3, 8, 9, 10, 11)   # tiles that hold the output: pooled rows 0 / 1 of the chunk on tile rows 0 / 2
# edgetpu_compiler's narrow allocation of the steps is fixed by two numbers per Mp (read from its programs): k = steps whose pooled
# result stays at the bottom of their conv-input block (which then stops moving up), b = pooled results placed at address 0 below
# the chunk block (which then moves up)
AM_ALLOC = {16: (5, 0), 32: (7, 0), 64: (9, 0), 128: (10, 2), 256: (12, 6)}
IMG = ConvGeom(AM_ROWS * AM_GRID[1], 0, AM_K, (AM_ROWS, AM_GRID[1]))   # the input path of a chunk: tile (r, c) gets rows 4r.., columns 50c..
M_W, M_E = 0x16, 0x18

def _winconv(Mp:int, s:int) -> ConvGeom:
  """the conv of step s: 16 positions per tile (8 in the last step) x K=288 against the Mp token weights in groups of min(Mp, 64)"""
  return ConvGeom(256 if s < 12 else 128, Mp, AM_K)

def am_alloc(Mp:int) -> tuple[list[dict], int, list[int]]:
  """narrow addresses (bytes, every tile): per step X (the chunk block, restored from the wide stash), X2 (the conv input), R (the
  input-move relay buffer), Y (conv output = the gathered window), P (the pooled result); O (the output blocks); the relay buffer of
  each output-phase move (output step s: O + 7*Mp + 32*s)"""
  p, (k, b) = Mp, AM_ALLOC[Mp]
  X2 = [57600 + p * min(s, k) for s in range(AM_STEPS)]
  steps = []
  for s in range(AM_STEPS):
    X, Y = p * min(s, b), p * min(s + 1, b) if b else 0
    if s < b: P = p * s
    elif s < k and s <= 11: P = X2[s]
    elif s <= 10: P = 62208 + p * s
    elif s == 11: P = X2[k] + 2304
    else: P = Y + 64 * p
    R = 62208 + p * s if s < 12 else X2[12] + 2304 + (p if k <= 11 else 0)
    steps.append(dict(X=X, X2=X2[s], R=R, Y=Y, P=P))
  return steps, p * b, [p * b + 7 * p + 32 * s for s in range(AM_STEPS)]

def am_wide(Mp:int, s:int) -> dict:
  """wide buffers (64-byte units) of step s: the pooling window right below the stash (the identity row in the last step, when the
  stash is no longer needed), the conv's FIFO / partial sums / bias stacked below it"""
  wc = _winconv(Mp, s)
  win = (AM_STASH if s < 12 else IDENT) - 64 * wc.G
  return dict(win=win, **CC.regions(wc, win))

def am_param_limit(Mp:int) -> int: return IDENT - 8 * IMG.c_in   # the input ring FIFO: resident parameters must end below it

def _step_cols(s:int) -> int: return 4 if s < 12 else 2     # image columns per tile column in step s

def am_transfers(s:int) -> list[tuple[int, int, int, int, int]]:
  """the moves that bring step s's 16 (8) image columns onto the tile columns: (dst column, src column, src local column, positions,
  dst position offset); image column j sits on tile column j // 50"""
  wpt, out = _step_cols(s), []
  for dc in range(4):
    for j in range(16 * s + wpt * dc, min(16 * s + wpt * dc + wpt, AM_GRID[1])):
      sc, lc = j // 50, j % 50
      if out and out[-1][0] == dc and out[-1][1] == sc: out[-1][3] += 1
      else: out.append([dc, sc, lc, 1, j - 16 * s - wpt * dc])
  return [tuple(t) for t in out]

def _local_copy(tiles:int, seq:int, src:int, dst:int, W:int, n:int, rows:int, in_rs:int, out_rs:int) -> dict:
  """copy rows x n positions of W words (row strides in_rs / out_rs words) through the identity row: the in TTU walks words, positions
  and rows in dims 0, 2, 4, each followed by a count-1 rewind dim"""
  return dict(tile_mask=tiles, seq=seq, loop1=W - 1, loop2=n - 1, loop3=rows - 1, in_base=src, in_mode7=1, in_tflags=0x15,
              **ttu("in_", [1, 0, W, 0, in_rs, 0, rows * in_rs], [W, 1, n, 1, rows], 8), out_base=dst, out_mode7=3, out_tflags=3,
              **ttu("out_", [1, W, out_rs, rows * out_rs], [W, n, rows], 8), **OP.wide("par", IDENT), **ttu("par_", [1], [1, W, 1, n, rows], 8),
              par_mode7=1, psum_hmode=1, **ttu("psum_", [], [1, W, 1, n, rows], 8), psum_tflags=0x840, cfg2=6, dp_mode=4, reduce_mask=1,
              out_ch_last=3, out_ch=3, **OP.NO_REQUANT)

def _input_moves(e:Emitter, s:int, X:int, X2:int, R:int):
  """step s's conv input: the mesh moves (west, then east; a group per transfer, each with a fence), then the local copies. West
  transfers in (src, dst) order, east in reverse; tile column c' gets image columns 16s + wpt*c' .. as 4 rows x wpt positions at X2;
  the chunk block X has rows of 50 positions"""
  wpt, tr = _step_cols(s), am_transfers(s)
  for op, moves in ((M_W, sorted([t for t in tr if t[1] > t[0]], key=lambda t: (t[1], t[0]))),
                    (M_E, sorted([t for t in tr if t[1] < t[0]], key=lambda t: (-t[1], -t[0])))):
    c = 0
    for dc, sc, lc, n, doff in moves:
      e.tile(RM.encode_mesh(opcode=op, tile_mask=0x1111 << sc, seq=e.seq, o_addr=X + 288 * lc, o_sdims=7, out_mode=3, rsv475=1,
                            **ttu("o_", [1, 72, 3600], [72, n, 4], 4)))
      relays = sum(0x1111 << k for k in range(min(sc, dc) + 1, max(sc, dc)))
      if relays:
        e.tile(FC.relay(op, relays, e.seq, R, 4 * 72 * n, c))
        c += 72 * n
      else: c += 1
      e.tile(RM.encode_mesh(opcode=op, tile_mask=0x1111 << dc, seq=e.seq, i_addr=X2 + 288 * doff, i_sdims=7, in_mode=7,
                            **ttu("i_", [1, 72, 72 * wpt], [72, n, 4], 4)))
      e.sync(SC.sync_mesh, op, 0xffff & ~relays, c)
  for dc, sc, lc, n, doff in sorted([t for t in tr if t[1] == t[0]], key=lambda t: t[1]):
    e.tile(OP.encode_op(**_local_copy(0x1111 << sc, e.seq, X + 288 * lc, X2 + 288 * doff, 72, n, 4, 3600, 72 * wpt)))

def _wdims(prefix:str, dims:list[tuple[int, int]]) -> dict:
  """mesh TTU from [(stride, count)] (count-1 dims dropped), sdims = mask of the used levels"""
  dims = [(s, c) for s, c in dims if c > 1]
  return {**ttu(prefix, [s for s, _ in dims], [c for _, c in dims], 4), f"{prefix}sdims": (1 << len(dims)) - 1}

def _layout(wc:ConvGeom, row:int, cols:int, rows:int) -> list[tuple[int, int]]:
  """[(stride in words, count)] of rows x cols positions in window rows of `row` positions; channel groups outermost"""
  return [(1, wc.cg // 4), (wc.N4 // 4, cols), (row * wc.N4 // 4, rows), (wc.cg // 4, wc.G)]

def _merge(dims:list[tuple[int, int]]) -> list[tuple[int, int]]:
  """merge contiguous position dims (the word dim and the channel-group dim stay)"""
  out = [dims[0]]
  for s, c in dims[1:-1]:
    if c == 1: continue
    if len(out) > 1 and out[-1][0] * out[-1][1] == s: out[-1] = (out[-1][0], out[-1][1] * c)
    else: out.append((s, c))
  return out + [dims[-1]]

def _am_conv(wc:ConvGeom, seq:int, tiles:int, X2:int, Y:int, wl:dict, q:dict, dims:list[tuple[int, int]], wide:bool) -> list[int]:
  """the step's conv op writing the window layout dims (count-1 position dims dropped, the group dim kept)"""
  od = [(s, c) for s, c in dims[:-1] if c > 1] + [dims[-1]]
  return CC.conv_op(wc, seq, X2, Y, wl, q, tiles, ([s for s, _ in od], [c for _, c in od]), wide)

MESH_REC_SEND, MESH_REC_FWD, MESH_REC_RECV_FWD = [(0, 7)], [(0, 7), (10, 7)], [(8, 7), (0, 7), (10, 7)]   # sync records (id, value)
def _recs(recs) -> dict:
  return {k: v for i, (rid, val) in enumerate(recs) for k, v in ((f"s{i}_id", rid), (f"s{i}_val", val), (f"s{i}_en_a", 1), (f"s{i}_en_b", 1))}
def _half(p:str, addr:int, dims:list[tuple[int, int]]) -> dict:
  """one half of a mesh move in the window layout dims; an inbound half ('i_') writes with in_mode 7"""
  return {f"{p}addr": addr, **_wdims(p, dims), **(dict(in_mode=7) if p == "i_" else {})}

def _gather(e:Emitter, wc:ConvGeom, Y:int, last:bool):
  """put the 64 conv outputs of every 8x8 window on its pool tile as 8 rows of 8 positions at Y. Regular steps: windows on tile
  quadrants, pool tiles 0, 2, 8, 10; the south tiles move north (even columns into the window's lower rows, odd ones behind the odd
  tile's block), then the odd columns move west into the right halves. Last step (2 image columns per tile): windows on tile rows
  0-1 / 2-3, pool tiles 0, 8; north moves, then a westward chain 3 -> 2 -> 1 -> 0 (column c's rows are 2*(4-c) positions wide)"""
  M, row, lay = wc.N, lambda c: 2 * (4 - c), lambda *a: _layout(wc, *a)
  if not last:   # (opcode, tiles, half, address, layout, sync records)
    for op, tiles, p, off, dims, recs in ((0x17, 0x0505, "i_", 32 * M, lay(8, 4, 4), []), (0x17, 0x0a0a, "i_", 16 * M, lay(4, 4, 4), []),
                                          (0x17, 0x5050, "o_", 0, lay(8, 4, 4), MESH_REC_SEND), (0x17, 0xa0a0, "o_", 0, lay(4, 4, 4), MESH_REC_SEND),
                                          (0x16, 0x0505, "i_", 4 * M, lay(8, 4, 8), []), (0x16, 0x0a0a, "o_", 0, lay(4, 4, 8), MESH_REC_FWD)):
      e.tile(RM.encode_mesh(opcode=op, tile_mask=tiles, seq=e.seq, **_half(p, Y + off, dims), **_recs(recs)))
    return
  for c in range(4): e.tile(RM.encode_mesh(opcode=0x17, tile_mask=0x0101 << c, seq=e.seq, **_half("i_", Y + 4 * row(c) * M, lay(row(c), 2, 4))))
  for c in range(4):
    e.tile(RM.encode_mesh(opcode=0x17, tile_mask=0x1010 << c, seq=e.seq, **_half("o_", Y, lay(row(c), 2, 4)), **_recs(MESH_REC_SEND)))
  for c in range(4):
    halves = (_half("o_", Y, lay(row(c), row(c), 8)) if c > 0 else {}) | (_half("i_", Y + 2 * M, lay(row(c), row(c + 1), 8)) if c < 3 else {})
    recs = MESH_REC_FWD if c == 3 else MESH_REC_RECV_FWD if c > 0 else []
    e.tile(RM.encode_mesh(opcode=0x16, tile_mask=0x0101 << c, seq=e.seq, **halves, **_recs(recs)))

def _with_records(words:list[int], lo:int, records:list[dict]) -> list[int]:
  """OR 42-bit sync records (en0, f1, f2, -, id:5, level:3, en1, value:19; eltwise.md 8.4) into an instruction at bit lo"""
  v = sum(w << (128 * j) for j, w in enumerate(words))
  for k, r in enumerate(records):
    v |= (r.get("en0", 0) | r.get("f1", 0) << 1 | r.get("f2", 0) << 2 | r.get("id", 0) << 4 | r.get("lvl", 0) << 9 |
          r.get("en1", 0) << 12 | r.get("val", 0) << 13) << (lo + 42 * k)
  return [(v >> (128 * j)) & ((1 << 128) - 1) for j in range(len(words))]

def _relay_n2w(wc:ConvGeom, seq:int, tiles:int, Y:int, win:int) -> list[int]:
  """eltwise's transposing MAX_POOL relay of the 8x8 window; for Mp >= 64 it waits itself on the gather's mesh counters (no drain
  before it): records (id 0), MESH_EAST_IN, MESH_WEST_IN, MESH_SOUTH_IN, MESH_NORTH_IN at level 4, the WEST/NORTH values 0x8000 /
  0xFFFF / 0xFFFE for 1 / 2 / 4 channel groups"""
  f = EL.maxpool_relay_n2w_fields(seq, tiles, Y, win, wc.N4 // 4, 2 * wc.N4, 8, 8, wc.N)
  if wc.N < 64: return WN.encode_narrow_to_wide(**f)
  gv, ge = {1: (0x8000, 0), 2: (0xFFFF, 1), 4: (0xFFFE, 1)}[wc.G]
  recs = [dict(en0=1, f1=1, lvl=4, en1=1, val=0x8000), dict(f2=1, id=4, lvl=4, en1=1, val=0x8000),
          dict(f2=1, id=6, lvl=4, en1=ge, val=gv), dict(f2=1, id=5, lvl=4, en1=1, val=0x8000), dict(f2=1, id=3, lvl=4, en1=ge, val=gv), dict(f2=1)]
  return _with_records(WN.encode_narrow_to_wide(**(f | dict(sync_en0=0, sync_f1=0, sync_en1=0, sync_en2=0))), 543, recs)

def _am_step(e:Emitter, Mp:int, s:int, A:dict, q:dict, pieces:list[Piece], pool_q:dict):
  last, wc, wl = s == 12, _winconv(Mp, s), am_wide(Mp, s)
  _input_moves(e, s, A["X"], A["X2"], A["R"])
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_drain)
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_drain)
  e.tile(CC.rc_param(wc, e.seq, wl))
  e.tile(CC.bias_load(e.seq, 0xffff, wl["bias"], wc.b, wc.fifo, wc.G, wc.cg))
  if not last:
    e.tile(_am_conv(wc, e.seq, 0x5555, A["X2"], A["Y"], wl, q, _layout(wc, 8, 4, 4), True))
    e.tile(_am_conv(wc, e.seq, 0xaaaa, A["X2"], A["Y"], wl, q, _merge(_layout(wc, 4, 4, 4)), False))
  else:
    for c in range(4):
      e.tile(_am_conv(wc, e.seq, 0x1111 << c, A["X2"], A["Y"], wl, q, _merge(_layout(wc, 2 * (4 - c), 2, 4)), c < 3))
  first = 0
  for tile, off, n in pieces:
    e.tile(CC.rprod_param(e.seq, tile, off, n, first))
    first += n
  _gather(e, wc, A["Y"], last)
  pt = 0x0101 if last else 0x0505
  if Mp < 64: e.sync(SC.sync_drain)
  e.tile(_relay_n2w(wc, e.seq, pt, A["Y"], wl["win"]))
  e.sync(SC.sync_drain)
  f = EL.maxpool_op_fields(8, 8, 8, 8, 1, 1, Mp, (Mp // 4, Mp // 4), pt, e.seq, A["P"], wl["win"],
                           **{k: v for k, v in pool_q.items() if k != "mult"})
  e.tile(OP.encode_op(**(f | dict(mult_bits=f32_bits(pool_q["mult"])))))
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_reset17)
  e.sync(SC.broadcast_wait, first)

def _out_dst(p:int) -> tuple[int, int]:
  """(output tile column, position in its block) of pooled column p"""
  c = 0
  while p >= OUT_COLS[c]:
    p -= OUT_COLS[c]
    c += 1
  return c, p

def _output_phase(e:Emitter, Mp:int, A:list[dict], O:int, relays:list[int]):
  """per step s: move the pooled results of tile columns 0 (pooled column 2s) and 2 (2s+1) of both tile rows 0 / 2 into the output
  blocks (west, then east moves with fences, then local copies; a drain when the step ends with a move)"""
  W = Mp // 4
  for s in range(AM_STEPS):
    e.sync(SC.sync_reset17)
    e.sync(SC.sync_reset17)
    moves = [(0, *_out_dst(2 * s))] + ([(2, *_out_dst(2 * s + 1))] if s < 12 else [])
    for op, group in ((M_W, [m for m in moves if m[0] > m[1]]), (M_E, sorted([m for m in moves if m[0] < m[1]], key=lambda m: -m[0]))):
      c = 0
      for sc, dc, idx in group:
        e.tile(RM.encode_mesh(opcode=op, tile_mask=0x0101 << sc, seq=e.seq, o_addr=A[s]["P"], out_mode=3, **_wdims("o_", [(1, W)])))
        relay = sum(0x0101 << k for k in range(min(sc, dc) + 1, max(sc, dc)))
        if relay:
          e.tile(FC.relay(op, relay, e.seq, relays[s], W, c))
          c += Mp // 16
        else: c += 1
        e.tile(RM.encode_mesh(opcode=op, tile_mask=0x0101 << dc, seq=e.seq, i_addr=O + Mp * idx, in_mode=3, **_wdims("i_", [(1, W)])))
        last = 0xffff & ~relay
        e.sync(SC.sync_mesh, op, last, c)
    loc = [m for m in moves if m[0] == m[1]]
    for sc, dc, idx in loc:
      e.tile(OP.encode_op(**_local_copy(0x0101 << sc, e.seq, A[s]["P"], O + Mp * idx, W, 1, 1, W, W * OUT_COLS[dc])))
    if not loc: e.sync(SC.sync_drain, last)

def _am_output_dma(e:Emitter, Mp:int, O:int, chunk:int, fifo:int):
  """narrowToWide of the output blocks (7 / 6 positions), one host DMA of 50*Mp bytes at offset 50*Mp*chunk, then (outfeed,
  ringProducer) per output tile (output FIFO 8308 while the identity row is still needed, 8312 in the last chunk)"""
  W = Mp // 4
  e.sync(SC.sync_reset17)
  e.sync(SC.signal_fence)
  e.scalar(SC.output_dma_head(50 * Mp * chunk))
  for tiles, n in ((0x0101, 7), (0x0e0e, 6)): e.tile(WN.n2w_output(n * Mp, tiles, O // 4, fifo, e.seq, [(1, W), (W, n)]))
  e.scalar(SC.output_dma_tail(50 * Mp))
  for k, t in enumerate(OUT_TILES):
    e.scalar(SC.outfeed(round_up(OUT_COLS[t % 4] * Mp, 8)))
    e.tile(FC.rprod_output(OUT_COLS[t % 4] * Mp, 1 << t, k, e.seq, fifo))
  e.scalar(SC.output_wait(e.seq, 8), seqs=2)

def _am_words(Mp:int, q:dict, pieces:list[Piece], pool_q:dict) -> list[int]:
  """the whole execution program as one instruction stream (no start/end): prologue, then per chunk the conv input path of a
  16 x 200 x 288 image, the stash of the chunk block, 13 steps, the output phase and the output DMA"""
  g, (A, O, relays) = IMG, am_alloc(Mp)
  stash = dict(tile_mask=0xffff, wide_lvl_mask=1, narrow_lvl_mask=7, sync_en0=0, sync_en1=0, sync_en2=0, **WN.ttu("w", [1], [225], 4),
               **WN.ttu("n", [1, 72, 3600], [72, 50, 4], 6))   # the chunk block <-> wide memory at AM_STASH
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  for ch in range(AM_CHUNKS):
    CC.input_stage(e, g, g.R, 0, 3 * g.R, IDENT - 8 * g.c_in, IDENT, const_row=ch == 0, host_off=g.S * ch)   # Stg, X, C
    e.sync(SC.sync_reset17)
    e.sync(SC.sync_reset17)
    e.tile(WN.encode_narrow_to_wide(seq=e.seq, narrow_addr=0, wide_addr=AM_STASH, tail_lvl=3, **stash))
    e.sync(SC.sync_reset17)
    for s in range(AM_STEPS):
      if s > 0:
        e.sync(SC.sync_reset17)
        e.sync(SC.sync_drain)
        e.tile(WN.encode_wide_to_narrow(seq=e.seq, wide_addr=AM_STASH, narrow_addr=A[s]["X"] // 4, tail_f871=1, **stash))
        e.sync(SC.sync_drain)
        e.sync(SC.sync_reset17)
        e.sync(SC.sync_reset17)
      _am_step(e, Mp, s, A[s], q, pieces, pool_q)
    _output_phase(e, Mp, A, O, relays)
    _am_output_dma(e, Mp, O, ch, WIDE_OUT_FIFO if ch == AM_CHUNKS - 1 else WIDE_OUT_FIFO - 4)
    if ch < AM_CHUNKS - 1: e.sync(SC.sync_reset17)
  e.sync(SC.epilogue)
  return e.words

def split_bitstreams(words:list[int], max_words:int=AM_MAX_WORDS) -> list[bytes]:
  """cut an instruction stream (no start/end) into bitstreams of at most max_words words as edgetpu_compiler does: groups that each
  start at a tile sync (0x1a), packed greedily; a cut bitstream ends with halt (code 1), 4 nops and end, the next one starts with
  start (the sequence numbers simply continue)"""
  groups: list[list[int]] = []
  for _, ws in split(b"".join(w.to_bytes(16, "little") for w in words)):
    if not groups or opcode(ws[0]) == 0x1a: groups.append([])
    groups[-1] += ws
  trailer, out, cur = SC.halt(1) + SC.nop() * 4, [], []
  for gw in groups:
    if cur and 1 + len(cur) + len(gw) + len(trailer) + 1 > max_words:
      out.append(cur + trailer)
      cur = []
    cur = cur + gw
  return [SC.program(body) for body in out + [cur]]

def argmax_quant(vocab_q:tuple, tokens_q:tuple, logits_q:tuple) -> tuple[dict, dict]:
  """(conv op fields, MAX_POOL fields): the vocabulary is the conv input, the token activations its weights. The MAX_POOL multiplier
  is the requantize one with equal scales, f32(s * f32(1/s)): 1.0 or 0.99999994"""
  return conv_quant(vocab_q, tokens_q, logits_q), EL.maxpool_quant(*logits_q) | dict(mult=EL.mul32(logits_q[0], EL.recip32(logits_q[0])))

def argmax_ref_quant() -> dict: return dict(vocab=(1/64, 128), tokens=(1/32, 128), logits=(1/4, 128))

def argmax_rows(Mp:int) -> tuple[int, int]:
  """(parameter rows, bytes per row) of the token blob: per channel group (min(Mp, 64) tokens) a bias row and K/4 rows"""
  wc = _winconv(Mp, 0)
  return wc.blocks, 4 * wc.cg

def argmax_params(tokens:np.ndarray, Mp:int, zp:int, bias:np.ndarray|None=None) -> bytes:
  """the argmax program's parameters = the token activations as conv weights: conv_blob with groups of min(Mp, 64) tokens; rows
  M..Mp-1 are padded with the token zero point"""
  M, K = tokens.shape
  t = np.full((Mp, K), zp, np.uint8)
  t[:M] = tokens
  return conv_blob(t, zp, None if bias is None else np.pad(bias, (0, Mp - M)), group=min(Mp, 64))

def argmax_pieces(Mp:int, placement=(0, 0)) -> list[Piece]:
  """placement = (tile, byte offset) of the token blob, or explicit pieces [(tile, offset in 64-byte units, rows)]"""
  rows, _ = argmax_rows(Mp)
  if isinstance(placement, tuple) and len(placement) == 2 and isinstance(placement[0], int):
    assert placement[1] % 256 == 0, "the parameter offset must be a multiple of 256 bytes"
    return [(placement[0], placement[1] // 64, rows)]
  assert sum(n for _, _, n in placement) == rows
  return list(placement)

def gen_argmax(Mp:int, V:int=AM_V, K:int=AM_K, grid:tuple[int, int]=AM_GRID, pool:int=AM_POOL, quant:dict|None=None,
               placement=(0, 0)) -> tuple[bytes, list[bytes]]:
  """(PARAMETER_CACHING, [EXECUTION_ONLY bitstreams]) of coral.fused.argmax_block for Mp token rows, byte-identical to edgetpu_compiler
  for Mp in {16, 32, 64, 128, 256}. quant: (scale, zero point) of vocab, tokens, logits (argmax_ref_quant() has the keys). placement:
  (tile, byte offset) of the token blob (argmax_params) in wide memory. The runtime contract is in argmax_hints / argmax_io."""
  if (V, K, tuple(grid), pool) != (AM_V, AM_K, AM_GRID, AM_POOL):
    raise NotImplementedError("gen_argmax is fitted to V=32000, K=288, grid=(160, 200), pool=8 (TinyStories-15M)")
  if Mp not in AM_ALLOC: raise NotImplementedError(f"Mp must be one of {list(AM_ALLOC)}")
  q = {**argmax_ref_quant(), **(quant or {})}
  cq, pq = argmax_quant(q["vocab"], q["tokens"], q["logits"])
  pieces, (rows, rb) = argmax_pieces(Mp, placement), argmax_rows(Mp)
  return caching_program(rb * rows, piece_parts([pieces], rb)), split_bitstreams(_am_words(Mp, cq, pieces, pq))

# *** runtime contract: hints, sizes, output layouts, Executables ***
def program_hints(bitstreams:list[bytes], in_name:str="x", out_name:str="y") -> list[Hint]:
  """the DMA hints of an execution program as edgetpu_compiler writes them: instruction chunk k when execution enters bitstream k,
  then every host DMA in issue order (inputs and outputs at running offsets), the completion interrupt last"""
  hints, off = [], {SC.TAG_INPUT: 0, SC.TAG_OUTPUT: 0}
  for k, bs in enumerate(bitstreams):
    hints.append(Hint("instruction", "INFEED", chunk=k))
    for tag, n in SC.dma_seqs(bs):
      if tag not in off: continue
      hints.append(Hint("dma", "INFEED", "INPUT", in_name, off[tag], n) if tag == SC.TAG_INPUT else
                   Hint("dma", "OUTFEED", "OUTPUT", out_name, off[tag], n))
      off[tag] += n
  return hints + [Hint("interrupt", "OUTFEED", interrupt=0)]

def caching_hints(param_bytes:int) -> list[Hint]:
  return [Hint("instruction", "INFEED", chunk=0), Hint("dma", "INFEED", "PARAMETER", "", 0, param_bytes),
          Hint("interrupt", "OUTFEED", interrupt=0)]

def argmax_hints(Mp:int, bitstreams:list[bytes]|None=None) -> list[Hint]:
  """per chunk c the vocabulary rows 16c..16c+15 (921600 bytes at 921600c) and its output (50*Mp bytes at 50*Mp*c), instruction
  chunks 1 and 2 where their bitstreams begin"""
  return program_hints(gen_argmax(Mp)[1] if bitstreams is None else bitstreams, "vocab", "blockmax")

def argmax_output_layout(Mp:int) -> dict:
  """the executable's output_layout of the [20, 25, Mp] block maxima: pooled position (y, x) at tile_byte_offset[y_tile[y] + x_tile[x]]
  + x_local_byte_offset[x] (one row per tile block). Chunk c (pooled rows 2c, 2c+1) is the c-th output DMA: tiles 0, 1, 2, 3 (row 2c:
  7, 6, 6, 6 positions of Mp bytes), then tiles 8, 9, 10, 11 (row 2c+1)."""
  tbo, off = [], 0
  for c in range(AM_CHUNKS):
    for t in range(16):
      tbo.append(off)
      if t in OUT_TILES: off += OUT_COLS[t % 4] * Mp
  xt = [c for c, n in enumerate(OUT_COLS) for _ in range(n)]
  return dict(y_tile=[8 * y for y in range(2 * AM_CHUNKS)], x_tile=xt, tile_byte_offset=tbo,
              x_local_byte_offset=[Mp * (x - sum(OUT_COLS[:xt[x]])) for x in range(len(xt))], y_local_y_offset=[0] * (2 * AM_CHUNKS),
              x_local_row_size=[Mp * OUT_COLS[c] for c in xt])

def argmax_io(Mp:int) -> dict:
  """host-side sizes: the input is the quantized vocabulary as [160][200][288] uint8 (V*K bytes, 10 DMAs of 921600), the output 10
  DMAs of 50*Mp bytes laid out by argmax_output_layout"""
  rows, rb = argmax_rows(Mp)
  return dict(input_bytes=AM_V * AM_K, input_chunk=AM_IN_BYTES, output_bytes=AM_CHUNKS * 50 * Mp, output_chunk=50 * Mp,
              param_bytes=rows * rb, output_layout=argmax_output_layout(Mp))

def executables(caching:bytes, execution:bytes|list[bytes], param_bytes:int, inputs:list[Layer], outputs:list[Layer],
                in_name:str="x", out_name:str="y") -> dict[str, Executable]:
  """coral.executable.Executables of generated programs with the hints coral.runtime.run_executable follows (the parameter blob is
  passed to run_executable(..., parameters=blob))"""
  bss = [execution] if isinstance(execution, bytes) else list(execution)
  def mk(kind, bs, hints, params, ins, outs): return Executable(None, kind, 1, 0, [Bitstream(b, []) for b in bs], params, hints, True, ins, outs,
                                                            "beagle", 0, 0, 0, 0)
  return {"PARAMETER_CACHING": mk("PARAMETER_CACHING", [caching], caching_hints(param_bytes), bytes(param_bytes), [], []),
          "EXECUTION_ONLY": mk("EXECUTION_ONLY", bss, program_hints(bss, in_name, out_name), b"", inputs, outputs)}

def gen_conv_pieces(Mp:int, N:int, K:int, pieces:list[Piece], quant:dict|None=None) -> tuple[bytes, bytes]:
  """gen_conv1x1 with every parameter piece at its own (tile, offset): one caching ringConsumer + infeed and one broadcast ringProducer
  per piece (the form edgetpu_compiler emits for split FFN convs)"""
  g = ConvGeom(Mp, N, K)
  assert sum(n for _, _, n in pieces) == g.blocks, "pieces must cover the blob"
  return CC.conv_programs(g, pieces, quant)

# *** the whole set: placement plan and generation ***
@dataclass(frozen=True)
class BlockSpec:
  """one program of the set: kind 'conv' (y = x @ W.T, W [N, K]), 'ffn' (K = D, N = Hd), 'argmax' (N = vocabulary)"""
  name: str
  kind: str
  N: int
  K: int

def block_specs(blocks) -> list[BlockSpec]:
  """BlockSpecs from coral.fused Blocks (or BlockSpecs / (name, kind, N, K) tuples)"""
  return [b if isinstance(b, BlockSpec) else BlockSpec(*b) if isinstance(b, tuple) else
          BlockSpec(b.name, b.kind, 768 if b.kind == "ffn" else b.N, b.K) for b in blocks]

def llm_specs(n_layers:int=6, argmax:bool=True) -> list[BlockSpec]:
  """the TinyStories-15M set as coral.fused registers it: per layer qkv, wo, ffn; then the classifier"""
  layer = (("qkv", "conv", 864), ("wo", "conv", 288), ("ffn", "ffn", 768))
  return [BlockSpec(f"L{l}.{n}", k, N, 288) for l in range(n_layers) for n, k, N in layer] + \
         ([BlockSpec("cls", "argmax", AM_V, AM_K)] if argmax else [])

def _fc_geoms(b:BlockSpec) -> list[FC.FCGeom]:   # the FULLY_CONNECTED matmuls of a block
  return [FC.FCGeom(b.N, b.K)] if b.kind == "conv" else [FC.FCGeom(b.N, b.K)] * 2 + [FC.FCGeom(b.K, b.N)]

def _blobs(b:BlockSpec, Mp:int) -> list[int]:
  """256-byte parameter rows of each matmul of a block (an argmax row of < 256 bytes still takes one wide row)"""
  if b.kind == "conv": return [blocks_of(b.N, b.K)]
  return list(FFNGeom(max(Mp, 16), b.K, b.N).blocks) if b.kind == "ffn" else [argmax_rows(Mp)[0]]

def program_limit(b:BlockSpec, Mp:int) -> int:
  """lowest wide address (64-byte units) of the block's execution program buffers"""
  if b.kind == "argmax": return am_param_limit(Mp)
  if b.kind == "ffn": return ffn_param_limit(Mp, b.K, b.N)
  return min(FC.WIDE_FWD_FIFO, STREAM_BIAS) if Mp == 1 else CC.param_limit(ConvGeom(Mp, b.N, b.K))   # Mp = 1: spread or streamed FC

def set_limit(specs:list[BlockSpec], Mp:int) -> int:
  """every resident region ends at or below the lowest buffer of every program (codegen_conv.md 9)"""
  return min(program_limit(b, Mp) for b in specs if Mp > 1 or b.kind != "argmax")

def plan(blocks, Mp:int, limit:int|None=None) -> dict[str, dict]:
  """our own placement of a whole set for Mp rows -> {name: {"form": .., "pieces" / "offsets": ..}}, offsets in bytes:
  Mp >= 16: the argmax token rows first at the bottom of tile 0 (room for Mp = 256 is reserved, so the layers get the same places for
    every Mp >= 16), then every layer blob in order (FFN = 3 blobs) laid out consecutively over the tiles' [0, limit) ranges and cut
    where a tile is full ('pieces' = [(tile, byte offset, rows)]; FFN: one list per matmul).
  Mp = 1: FULLY_CONNECTED forms. A matmul is spread over its compute tiles at one offset ('spread', offset per matmul) while it fits,
    else streamed with all its parameters on the emptiest tile ('streamed', [(tile, offset, rows)]); an FFN is spread or streamed as
    a whole. The argmax block has no batch-1 form and is skipped.
  Every region is 256-byte aligned and ends at or below limit (default: set_limit = the lowest program buffer)."""
  specs = block_specs(blocks)
  L = set_limit(specs, Mp) if limit is None else limit
  L -= L % 4
  out: dict[str, dict] = {}
  if Mp > 1:
    t, o = 0, 0
    for b in [b for b in specs if b.kind == "argmax"] + [b for b in specs if b.kind != "argmax"]:
      ps_all = []
      for n in _blobs(b, Mp):
        if b.kind == "argmax":      # reserve the largest token blob (Mp = 256) at tile 0 (no re-caching when Mp changes)
          need = max(argmax_rows(m)[0] for m in AM_ALLOC)
          assert (t, o) == (0, 0) and 4 * need <= L
          ps_all.append([(0, 0, n)])
          o = 4 * need
          continue
        ps = []
        while n:
          if t > 15: raise ValueError(f"the set does not fit 16 tiles below {L} units")
          k = min(n, (L - o) // 4)
          if k:
            ps.append((t, 64 * o, k))
            o += 4 * k
            n -= k
          if (L - o) // 4 == 0: t, o = t + 1, 0
        ps_all.append(ps)
      out[b.name] = dict(form="conv" if b.kind != "argmax" else "argmax", pieces=ps_all if b.kind == "ffn" else ps_all[0])
    return out
  fc = [b for b in specs if b.kind != "argmax"]
  for n_spread in range(len(fc), -1, -1):        # as many blocks spread as fit (the compiler's batch-1 set: the first layers)
    hw = [0] * 16
    offs: dict[str, list] = {b.name: [None] * len(_fc_geoms(b)) for b in fc[:n_spread]}
    for b, i, g in sorted([(b, i, g) for b in fc[:n_spread] for i, g in enumerate(_fc_geoms(b))], key=lambda x: -x[2].T):  # wide first
      o = max(hw[:g.T])
      offs[b.name][i] = 64 * o
      for tt in range(g.T): hw[tt] = o + 4 * g.P * (1 + g.kw)
    if max(hw) > L: continue
    res = {name: dict(form="spread", offsets=o) for name, o in offs.items()}
    pieces: dict[str, list] = {b.name: [None] * len(_fc_geoms(b)) for b in fc[n_spread:]}
    blobs = [(b, i, 4 * blocks_of(g.N, g.K), blocks_of(g.N, g.K)) for b in fc[n_spread:] for i, g in enumerate(_fc_geoms(b))]
    for b, i, need, n in sorted(blobs, key=lambda x: -x[2]):   # best fit decreasing, one tile per matmul
      cand = sorted((L - hw[tt] - need, tt) for tt in range(16) if hw[tt] + need <= L)
      if not cand: break
      tt = cand[0][1]
      pieces[b.name][i] = (tt, 64 * hw[tt], n)
      hw[tt] += need
    else:
      res.update({name: dict(form="streamed", pieces=ps) for name, ps in pieces.items()})
      return {b.name: res[b.name] for b in fc}
  raise ValueError(f"the batch-1 set does not fit below {L} units")

def _units(pieces:list[tuple[int, int, int]]) -> list[Piece]: return [(t, o // 64, n) for t, o, n in pieces]

def gen_block(b:BlockSpec, Mp:int, place:dict, quant:dict|None=None) -> tuple[bytes, list[bytes]]:
  """(caching, [execution bitstreams]) of one block of a set, placed by `place` (an entry of plan()). quant: conv blocks x / w / y,
  ffn blocks as gen_ffn, argmax blocks as gen_argmax"""
  if b.kind == "argmax": return gen_argmax(Mp, quant=quant, placement=_units(place["pieces"]))
  if b.kind == "ffn":
    if Mp > 1: pc, eo = gen_ffn(Mp, b.K, b.N, quant, pieces=[_units(ps) for ps in place["pieces"]])
    elif place["form"] == "spread": pc, eo = gen_ffn(1, b.K, b.N, quant, param_offset=tuple(place["offsets"]))
    else: pc, eo = gen_ffn(1, b.K, b.N, quant, pieces=[[p] for p in _units(place["pieces"])])
    return pc, [eo]
  q = None if quant is None else conv_quant(quant["x"], quant["w"], quant["y"])
  if Mp > 1: pc, eo = gen_conv_pieces(Mp, b.N, b.K, _units(place["pieces"]), q)
  elif place["form"] == "spread": pc, eo = FC.gen_fc(b.N, b.K, param_offset=place["offsets"][0], quant=q)
  else:
    (tt, o, _), = place["pieces"]
    pc, eo = gen_fc_streamed(b.N, b.K, tt, o, q)
  return pc, [eo]

def block_io(b:BlockSpec, Mp:int) -> dict:
  """host-side contract of one block: input / output bytes per call, the output layout, the parameter blob size"""
  if b.kind == "argmax": return argmax_io(Mp)
  N, K = (b.K, b.K) if b.kind == "ffn" else (b.N, b.K)
  pb = 256 * sum(_blobs(b, max(Mp, 16)))
  if Mp == 1: return dict(input_bytes=FC.FCGeom(N, K).S, output_bytes=FC.FCGeom(N, K).out_bytes, output_layout=None, param_bytes=pb)
  S, out = CC.io_sizes(Mp, N, K)
  return dict(input_bytes=S, output_bytes=out, output_layout=CC.output_layout(Mp, N), param_bytes=pb)

def gen_set(blocks, Mp:int, pl:dict[str, dict], quants:dict[str, dict]|None=None) -> dict[str, dict[str, Executable]]:
  """every program of the set, generated from the plan (no compiler): {name: {"PARAMETER_CACHING": Executable, "EXECUTION_ONLY":
  Executable}} with the hints and the output layout; run the caching program with run_executable(..., parameters=blob) (conv_blob /
  ffn_params / argmax_params)"""
  out = {}
  for b in block_specs(blocks):
    if Mp == 1 and b.kind == "argmax": continue
    q = (quants or {}).get(b.name)
    pc, eo = gen_block(b, Mp, pl[b.name], q)
    io = block_io(b, Mp)
    if b.kind == "argmax":
      q, names, ishape, oshape = q or argmax_ref_quant(), ("vocab", "blockmax"), (*AM_GRID, AM_K), (20, 25, Mp)
      iq, oq = q["vocab"], q["logits"]
    else:
      H, W = GRIDS[Mp] if Mp > 1 else (1, 1)
      names, ishape, oshape = ("x", "y"), (H, W, b.K), (H, W, b.K if b.kind == "ffn" else b.N)
      if b.kind == "ffn": iq, oq = (q or ffn_ref_quant())["x"], (q or ffn_ref_quant())["y"]
      else: iq, oq = (q or {}).get("x", (1/32, 128)), (q or {}).get("y", (1/4, 128))
    def layer(name, size, shape, q, is_out, lay=None): return Layer(name, size, *shape, q[1], q[0], "u8", is_out, [], 1, lay)
    out[b.name] = executables(pc, eo, io["param_bytes"], [layer(names[0], io["input_bytes"], ishape, iq, False)],
                              [layer(names[1], io["output_bytes"], oshape, oq, True, io["output_layout"])], *names)
  return out

def regions(b:BlockSpec, Mp:int, place:dict) -> list[tuple[int, int, int]]:
  """the wide-memory parameter regions [(tile, start unit, end unit)] a placed block keeps resident"""
  if Mp == 1 and place["form"] == "spread":
    return [(t, o // 64, o // 64 + 4 * g.P * (1 + g.kw)) for g, o in zip(_fc_geoms(b), place["offsets"]) for t in range(g.T)]
  flat = [p for q in place["pieces"] for p in q] if b.kind == "ffn" and Mp > 1 else place["pieces"]
  return [(t, o // 64, o // 64 + 4 * n) for t, o, n in flat]

def check_plan(blocks, Mp:int, pl:dict[str, dict], limit:int|None=None) -> list[str]:
  """problems of a placement: overlapping regions, regions above the limit, unaligned offsets (empty = fine)"""
  specs = [b for b in block_specs(blocks) if Mp > 1 or b.kind != "argmax"]
  L = set_limit(specs, Mp) if limit is None else limit
  regs = [(b.name, *r) for b in specs for r in regions(b, Mp, pl[b.name])]
  bad = [f"{n}: tile {t} [{a}, {z}) ends above {L}" for n, t, a, z in regs if z > L] + \
        [f"{n}: offset {a} not 256-byte aligned" for n, t, a, z in regs if a % 4]
  for i, (n1, t1, a1, z1) in enumerate(regs):
    for n2, t2, a2, z2 in regs[i + 1:]:
      if t1 == t2 and a1 < z2 and a2 < z1: bad.append(f"{n1} and {n2} overlap on tile {t1}: [{a1}, {z1}) [{a2}, {z2})")
  return bad

if __name__ == "__main__": run_test("test_codegen_fused")
