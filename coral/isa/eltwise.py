# elementwise and pooling tile ops and the instructions that feed them (docs/isa/eltwise.md): LOGISTIC (op 0x01 + the 0x19 NLU spline
# load), RELU / RELU6 / RELU_N1_TO_1 / QUANTIZE (requantize op), MUL (two ops), ADD (op with 16-bit weights), MAX_POOL_2D (op 0x02), and
# the narrowToWide / mesh-fill instructions that move data between chained ops. Every op builder returns a field dict for op.encode_op.
from __future__ import annotations
from fractions import Fraction
from coral.isa import Layout, cdiv, f32, f32_bits, ttu, wide_narrow as WN, ring_mesh as RM, scalar as SC
from coral.isa.op import ttu_fine, wide, NO_REQUANT

# *** opcode 0x19: NLU spline coefficient load, 13 words: 8 x 5 coefficients (f32), 7 breakpoints (f32), mode (u32), 2 reserved ***
NLU = Layout("nlu", (0x19,), 13, [("tile_mask", 12, 16), ("seq", 46, 16), *[(f"slot{k}", 62 + 32 * k, 32) for k in range(50)]])
encode_nlu, decode_nlu = NLU.encode, NLU.decode
# LOGISTIC: a piecewise quartic in the dequantized input, its output in units of 1/256 (TFLite fixes the output scale to 1/256).
# The same in every compiled program, independent of shape and quantization.
LOGISTIC_SLOTS = [
  0x4200549c, 0x415db7c6, 0x40111a70, 0x3e29f7bb, 0x3b95f260, 0x4304a187, 0x42ab6803, 0x41aee024, 0x40251eb8, 0x3df136b6,
  0x43077562, 0x42a46b63, 0x418371fd, 0x3f05681a, 0xbe04c80f, 0x4300006a, 0x427fe396, 0xbea8c7fe, 0xc0c9f57d, 0xbf94c0e4,
  0x430000b8, 0x427fb331, 0x3ef2682d, 0xc0cf35b5, 0x3f9cb2b2, 0x42ddc7c6, 0x42c56c6f, 0xc1d72011, 0x405be5c6, 0xbe2ff6d3,
  0x4343d750, 0x41efbc0b, 0xc0b6a9cd, 0x3efb0c86, 0xbc82c2d7, 0x4374cc57, 0x4085560c, 0xbf168355, 0x3d187e2a, 0xba69a584,
  0xc0b479b1, 0xc03dc92e, 0xbfaba1f5, 0x3db069f8, 0x3fe71fbc, 0x4099a343, 0x41078f0a, 0x00000003, 0x00000000, 0x00000000]
LOGISTIC_CLAMP = 0x41265904        # f32 10.396732: the op clamps x to [-c, c] before the NLU

def nlu_fields(tile_mask:int, seq:int, slots:list[int]=LOGISTIC_SLOTS) -> dict:
  """loads the spline into the NLU of every tile in tile_mask (the compiler uses the OR of the following LOGISTIC ops' masks)"""
  return dict(tile_mask=tile_mask, seq=seq, **{f"slot{k}": s for k, s in enumerate(slots)})

# *** tensor placement (compiler rules) ***
_EXTRA = {0: (), 1: (0,), 2: (0, 2), 3: (0, 1, 2)}
def tile_split(D:int) -> list[int]:
  """rows (columns) of a spatial extent D on tile row (column) 0..3: D//4 each, the remainder on 0 / 0,2 / 0,1,2"""
  return [D // 4 + (i in _EXTRA[D % 4]) for i in range(4)]
def tile_block(H:int, W:int, t:int) -> tuple[int, int]: return tile_split(H)[t // 4], tile_split(W)[t % 4]   # (rows, cols) on tile t
def hwc(shape:list[int]) -> tuple[int, int, int]:
  """(H, W, C) of a [1,n], [1,W,C] or [1,H,W,C] tensor"""
  return (1, 1, shape[1]) if len(shape) == 2 else (1, shape[1], shape[2]) if len(shape) == 3 else (shape[1], shape[2], shape[3])

def tile_geometry(shape:list[int], tile:int, local:bool=False) -> dict:
  """per-tile view of a tensor as the compiler lays it out: P = rows*cols positions, w = 4-byte words per position, R = (position, row,
  block) strides in words, flat = cfg1 bit 8 of the requantize / LOGISTIC op (1-D tensors and C % 4 != 0). A single-position tensor is
  1-D: chunks of 64*ceil(C/1024) bytes per tile, every stride the whole tensor's ceil(C/4) words, or the tile's own chunk (local=True:
  the tile-local layout an FC op broadcast to several tiles writes)."""
  H, W, C = hwc(shape)
  if H * W == 1:
    b = 64 * cdiv(C, 1024)
    w = cdiv(max(0, min(b, C - b * tile)), 4)
    return dict(P=1, rows=1, cols=1, w=w, R=(w if local else cdiv(C, 4),) * 3, flat=1)
  rows, cols = tile_block(H, W, tile)
  w = cdiv(C, 4)
  return dict(P=rows * cols, rows=rows, cols=cols, w=w, R=(w, cols * w, rows * cols * w), flat=int(C % 4 != 0))

# *** quantization: float32 arithmetic exactly as edgetpu_compiler ***
def mul32(a:float, b:float) -> float: return f32(f32(float(a)) * f32(float(b)))
def recip32(s:float) -> float: return f32(1.0 / f32(float(s)))
ACT_RANGE = {0: (None, None), 1: (0.0, None), 2: (-1.0, 1.0), 3: (0.0, 6.0)}   # fused activation: NONE RELU RELU_N1_TO_1 RELU6

def out_clamps(scale:float, zp:int, act:int=0) -> tuple[float, float]:
  """(clamp_min, clamp_max) relative to the output zero point: the real range [(0-zp)s, (255-zp)s] intersected with the activation,
  times f32(1/s); every step rounded to float32"""
  lo, hi = mul32(0 - zp, scale), mul32(255 - zp, scale)
  alo, ahi = ACT_RANGE[act]
  if alo is not None: lo = max(lo, f32(alo))
  if ahi is not None: hi = min(hi, f32(ahi))
  return mul32(lo, recip32(scale)), mul32(hi, recip32(scale))

def requant_quant(in_q:tuple, out_q:tuple, act:int=0) -> dict:
  """RELU (act 1), RELU6 (3), RELU_N1_TO_1 (2), QUANTIZE (0): y = clamp((x - in_zp) * mult) + out_zp"""
  cmin, cmax = out_clamps(out_q[0], out_q[1], act)
  return dict(in_zp=in_q[1], out_zp=out_q[1], mult=mul32(in_q[0], recip32(out_q[0])), clamp_min=cmin, clamp_max=cmax)

def logistic_quant(in_q:tuple, out_zp:int=0) -> dict:
  """x = clamp((q - in_zp) * in_scale, +-10.396732); y = spline(x) + out_zp"""
  return dict(in_zp=in_q[1], out_zp=out_zp, mult=f32(float(in_q[0])))

def mul_quant(a_q:tuple, b_q:tuple, y_q:tuple, act:int=0) -> dict:
  """y = clamp((a - a_zp)(b - b_zp) * mult) + y_zp; a is read through the in TTU (in_zp), b from the wide FIFO (w_zp)"""
  cmin, cmax = out_clamps(y_q[0], y_q[1], act)
  return dict(a_zp=a_q[1], b_zp=b_q[1], out_zp=y_q[1], mult=mul32(mul32(a_q[0], b_q[0]), recip32(y_q[0])), clamp_min=cmin, clamp_max=cmax)

def add_weights(s1:float, s2:float, limit:int=32767) -> tuple[int, int]:
  """integer weights (w1, w2), w2/w1 = the best rational approximation of s2/s1 with terms <= limit (verified on 40 scale ratios;
  limit is in [25319, 32768])"""
  r = Fraction(f32(float(s2))) / Fraction(f32(float(s1)))
  if r <= 1:
    fr = r.limit_denominator(limit)
    return fr.denominator, fr.numerator
  fr = (1 / r).limit_denominator(limit)
  return fr.numerator, fr.denominator

def add_quant(q1:tuple, q2:tuple, y_q:tuple, act:int=0) -> dict:
  """operand 1 at in_base, operand 2 at in_base + delta: acc = w1*q1 + w2*q2 + offset (16-bit MXU), y = clamp(acc * mult) + y_zp"""
  w1, w2 = add_weights(q1[0], q2[0])
  cmin, cmax = out_clamps(y_q[0], y_q[1], act)
  return dict(w1=w1, w2=w2, offset=-(w1 * q1[1] + w2 * q2[1]), out_zp=y_q[1], mult=mul32(f32(f32(float(q1[0])) / w1), recip32(y_q[0])),
              clamp_min=cmin, clamp_max=cmax)

def maxpool_quant(scale:float, zp:int) -> dict:
  cmin, cmax = out_clamps(scale, zp)
  return dict(zp=zp, clamp_min=cmin, clamp_max=cmax)

def _out(out_zp:int, mult:float, clamp_min:float, clamp_max:float) -> dict:
  return dict(out_zp=out_zp, mult_bits=f32_bits(mult), clamp_min_bits=f32_bits(clamp_min), clamp_max_bits=f32_bits(clamp_max))

# *** op builders. Strides in bytes (narrow) / 64-byte units (wide), R = (position, row, block) strides in words ***
def requant_op_fields(P:int, Cw:int, R_in:int, R_out:int, tile_mask:int, seq:int, in_base:int, out_base:int, ident_addr:int,
                      in_zp:int, out_zp:int, mult:float, clamp_min:float, clamp_max:float, flat:int=1) -> dict:
  """RELU-type requantize y = clamp((x - in_zp)*mult) + out_zp over P positions x Cw words per tile: the MXU multiplies each input word
  by the 4x4 identity at wide ident_addr. R_in / R_out: position strides in words."""
  single, loops = P * Cw == 1, [n for n in (P, Cw) if n > 1]
  return dict(tile_mask=tile_mask, seq=seq, loop0=P - 1, loop1=Cw - 1, **ttu_fine("in", [4, 4 * R_in, 4], [1, P, Cw]), in_base=in_base,
              in_mode7=3, in_tflags=1, **ttu_fine("out", [4, 4 * R_out, 4], [1, P, Cw]), out_base=out_base, out_mode7=3, out_tflags=1,
              **wide("par", ident_addr), **ttu("par_", [], loops, 8), **ttu("psum_", [], loops, 8), par_hmode=int(single), psum_hmode=int(single),
              psum_tflags=0 if single else 0x800, cfg0=0x5, cfg1=0x4D | flat << 8, sync0=0x4000, sync1=0x4000, cfg2=6, dp_mode=1, out_ch=3,
              out_ch_last=3, in_zp=in_zp, **_out(out_zp, mult, clamp_min, clamp_max))

def logistic_op_fields(P:int, Cw:int, R_in:int, R_out:int, tile_mask:int, seq:int, in_base:int, out_base:int, ident_addr:int,
                       in_zp:int, mult:float, out_zp:int=0, flat:int=1) -> dict:
  """LOGISTIC: the requantize op with mult = the input scale, both clamps = +-10.396732 and bit 1946 set; the preceding 0x19 loads
  the spline (nlu_fields)"""
  return requant_op_fields(P, Cw, R_in, R_out, tile_mask, seq, in_base, out_base, ident_addr, in_zp, out_zp, mult, 0.0, 0.0, flat) | \
         dict(rsv1946=1, clamp_min_bits=LOGISTIC_CLAMP | 1 << 31, clamp_max_bits=LOGISTIC_CLAMP)

def mul_fifo_depth(w:int) -> int: return min(4, cdiv(w, 2))   # wide FIFO rows holding operand b (one 4-byte word of b per 256-byte row)

def mul_op1_fields(w:int, cols:int, rows:int, R:tuple, tile_mask:int, seq:int, in_base:int, out_base:int, fifo_addr:int,
                   a_zp:int, b_zp:int, mult:float, out_zp:int, clamp_min:float, clamp_max:float) -> dict:
  """MUL step 1: every byte a_i (in TTU, one byte per step) times the FIFO row holding b's word, requantized, written as one 4-byte
  word per element (all lanes = the product) to the dense intermediate at out_base. w = words per position, (cols, rows) = tile
  block, R = strides of a, fifo_addr = wide address of the FIFO."""
  D = mul_fifo_depth(w)
  m4, P, r = 4 * w, cols * rows, w % D
  d = dict(tile_mask=tile_mask, seq=seq, loop0=4 * D - 1, loop1=cdiv(w, D) - 1, loop2=cols - 1, loop3=rows - 1,
           **ttu_fine("in", [1, 4 * R[0], 4 * R[1], 4 * R[2]], [m4, cols, rows]), in_base=in_base, in_mode7=3, in_tflags=3,
           **ttu_fine("out", [4, 4, 4 * m4, 4 * cols * m4, 4 * P * m4], [1, m4, cols, rows]), out_base=out_base, out_mode7=3, out_tflags=7,
           **ttu_fine("par", [1], [4 * D, cdiv(w, D), cols, rows]), **wide("par", fifo_addr), par_mode7=1, par_tflags=0x40,  # FIFO re-read
           par_fifo=D | (0x2000 if r else 0), **ttu("psum_", [], [1, m4, cols, rows], 8), psum_hmode=1, psum_mode7=1, psum_tflags=0x840,
           cfg0=0x60, cfg1=0x12D, sync0=0x4000, sync1=0x4000, cfg2=6, cfg3=7, dp_mode=4, out_ch=3, out_ch_last=3, in_zp=a_zp, w_zp=b_zp,
           **_out(out_zp, mult, clamp_min, clamp_max))
  if r: d.update(rsv174=0x4000 | (4 * r - 1), rsv1230=(1 << 13) | (2 * r - 1) | (D - r) << 18)   # partial last FIFO round
  return d

def mul_op2_fields(w:int, cols:int, rows:int, R_out:tuple, tile_mask:int, seq:int, in_base:int, out_base:int, ident_addr:int) -> dict:
  """MUL step 2 (dp_mode 2): packs the intermediate back to bytes, output byte k of word j = lane k of intermediate word 4j+k
  (4 steps against the identity rows, lane-select mode), no requantization"""
  m4, P = 4 * w, cols * rows
  return dict(tile_mask=tile_mask, seq=seq, loop0=3, loop1=w - 1, loop2=cols - 1, loop3=rows - 1,
              **ttu_fine("in", [4, 4 * m4, 4 * cols * m4, 4 * P * m4], [m4, cols, rows]), in_base=in_base, in_mode7=3, in_tflags=3,
              **ttu_fine("out", [4, 4 * R_out[0], 4 * R_out[1], 4 * R_out[2]], [w, cols, rows]), out_base=out_base, out_mode7=3, out_tflags=3,
              **ttu_fine("par", [1], [4, w, cols, rows]), **wide("par", ident_addr), par_mode7=1, par_tflags=0x40,
              **ttu("psum_", [], [4, w, cols, rows], 8), psum_tflags=0x840, cfg2=6, cfg3=7, dp_mode=2, out_ch=3, out_ch_last=3, **NO_REQUANT)

def add_op_fields(w:int, cols:int, rows:int, R_in:tuple, delta:int, R_out:tuple, tile_mask:int, seq:int, in_base:int, out_base:int,
                  wmat_addr:int, offset:int, mult:float, out_zp:int, clamp_min:float, clamp_max:float) -> dict:
  """ADD (dp_mode 5): per output word 4 accumulation steps, (operand 1 at in_base, operand 2 at in_base + 4*delta bytes) x (low, high
  16-bit half of the word), each against one row of the 4x4 16-bit weight matrix at wide wmat_addr (add_weight_fills)"""
  P = cols * rows
  return dict(tile_mask=tile_mask, seq=seq, loop0=1, loop1=1, loop2=cols - 1, loop3=rows - 1, loop5=w - 1,
              **ttu_fine("in", [2, 4 * delta, 4 * R_in[0], 4 * R_in[1], 4 * R_in[2], 4], [2, 2, cols, rows, 1, w]), in_base=in_base,
              in_mode7=3, in_tflags=0xF, **ttu_fine("out", [4, 4 * R_out[0], 4 * R_out[1], 4 * R_out[2], 4], [1, cols, rows, 1, w]),
              out_base=out_base, out_mode7=3, out_tflags=7, **ttu_fine("par", [4], [4, P * w]), **wide("par", wmat_addr), par_mode7=1,
              **ttu("psum_", [], [4, P * w], 8), psum_tflags=0x840, cfg0=0x9, cfg1=0x1CD, sync0=0x4000, sync1=0x4000, cfg2=0x16, dp_mode=5,
              reduce_mask=3, out_ch=3, out_ch_last=3, offset=offset, **_out(out_zp, mult, clamp_min, clamp_max))

def maxpool_layout(kh:int, kw:int, sh:int, sw:int, OH:int, OW:int) -> tuple[int, int]:
  """(Sy, blk) of the transposed window block in wide memory (64-byte units): a 256-byte row holds 4 input positions (one per byte
  lane, 64 channels per lane); an input row of Wb positions takes Sy = 4*ceil(Wb/4) lanes, the Hb x Wb block blk = Hb*Sy"""
  Hb, Wb = (OH - 1) * sh + kh, (OW - 1) * sw + kw
  return 4 * cdiv(Wb, 4), Hb * 4 * cdiv(Wb, 4)

def maxpool_op_fields(kh:int, kw:int, sh:int, sw:int, OH:int, OW:int, C:int, R_out:tuple, tile_mask:int, seq:int, out_base:int,
                      src_addr:int, zp:int, clamp_min:float, clamp_max:float, Sy:int|None=None, blk:int|None=None) -> dict:
  """MAX_POOL_2D (opcode 0x02): reads the window from wide memory (par TTU, one byte lane = one position x 64 channels per step, the
  layout maxpool_relay_n2w_fields writes) and max-reduces loops 0, 1 (kw x kh). OH x OW outputs per tile, C channels in groups of 64
  (loop4). R_out = (position, row) strides of the output in words."""
  G = cdiv(C, 64)
  cg = 4 * cdiv(min(64, C), 4)
  cl = 4 * cdiv(C - 64 * (G - 1), 4)
  Wg = cg // 4
  if Sy is None or blk is None: Sy, blk = maxpool_layout(kh, kw, sh, sw, OH, OW)
  d = dict(opcode=2, tile_mask=tile_mask, seq=seq, loop0=kw - 1, loop1=kh - 1, loop2=OW - 1, loop3=OH - 1, loop4=G - 1,
           **ttu_fine("par", [1, Sy, sw, sh * Sy, blk, G * blk], [kw, kh, OW, OH, G]), **wide("par", src_addr), par_mode7=3, par_tflags=0xF,
           **ttu_fine("out", [4, 4 * R_out[0], 4 * R_out[1], 4 * Wg, 4 * OH * R_out[1]], [Wg, OW, OH, G]), out_base=out_base, out_mode7=3,
           out_tflags=7, **ttu("psum_", [], [kw, kh, OW, OH, G], 8), psum_tflags=0x880, cfg0=0x127, cfg2=6, cfg3=7, dp_mode=4, reduce_mask=3,
           out_ch=cg - 1, out_ch_last=cl - 1, in_zp=zp, offset=-zp, **_out(zp, 1.0, clamp_min, clamp_max))
  if cl != cg: d.update(out_last_cnt=cl // 4 - 1, out_last_mode=3, out_last_skip=Wg - cl // 4)
  return d

# *** data-flow instructions around the ops (narrow addresses in bytes, wide in 64-byte units) ***
def ident_prologue(seq:int, tiles:int, narrow_byte:int, wide_addr:int, zero_words:int=0) -> list[list[int]]:
  """the 4x4 identity row (weights of the requantize / LOGISTIC / copy ops and MUL step 2): 4 inbound meshBus 0x17 fills of one word
  (1 << 8j at narrow_byte + 4j), optionally a zero row of zero_words words behind it (+ a drain), then the narrowToWide of it all to
  wide_addr (waiting on MESH_SOUTH_IN, or after the drain). Consecutive seqs from seq."""
  out = [RM.encode_mesh(opcode=0x17, tile_mask=tiles, seq=seq + j, i_addr=narrow_byte + 4 * j, i_inc0=1, fill_en=1, fill=1 << (8 * j),
                        in_mode=3 if j == 3 and not zero_words else 0) for j in range(4)]
  sync, n = dict(sync_f1=1, sync_id=5, sync_wait_lvl=1), 16 + 4 * zero_words
  if zero_words:
    out += [RM.encode_mesh(opcode=0x17, tile_mask=tiles, seq=seq + 4, i_addr=narrow_byte + 16, i_sdims=1, fill_en=1,
                           **ttu("i_", [1], [zero_words], 4)), SC.sync_drain(seq + 5)]
    sync = dict(sync_en0=0, sync_en1=0, sync_en2=0)
  return out + [WN.encode_narrow_to_wide(tile_mask=tiles, seq=seq + len(out), narrow_addr=narrow_byte // 4, narrow_lvl_mask=1,
                                         wide_addr=wide_addr, wide_lvl_mask=int(n > 256), tail_lvl=1, **sync, **WN.ttu("n", [1], [n // 4], 6),
                                         **WN.ttu("w", [1], [cdiv(n, 256)], 4))]

def mul_feed_n2w_fields(seq:int, tiles:int, narrow_byte:int, fifo_addr:int, w:int, cols:int=1, rows:int=1) -> dict:
  """narrowToWide that streams MUL operand b (w words per position, a dense cols x rows block at narrow_byte) into the D-row wide FIFO,
  one narrow word per wide row (sync PARAMETERS; MUL step 1 consumes the rows)"""
  D = mul_fifo_depth(w)
  r = w % D
  dims = [(1, w)] + [(w, cols)] * (cols > 1) + [(cols * w, rows)] * (rows > 1)
  single = w * cols * rows == 1                      # a 1-word walk has narrow_lvl_mask 0
  circ = int(D > 1 or w == 1)
  f = dict(tile_mask=tiles, seq=seq, narrow_addr=narrow_byte // 4, narrow_lvl_mask=0 if single else (1 << len(dims)) - 1, wide_addr=fifo_addr,
           wide_rows=D, sync_id=1, sync_val=0xFFFF, sync_f45=1, **WN.ttu("n", [s for s, _ in dims], [c for _, c in dims], 6),
           **(WN.ttu("w", [1, 0, 0], [D, cdiv(w, D), cols * rows], 4) if D > 1 else WN.ttu("w", [int(w == 1), 0], [w, cols * rows], 4)),
           wide_circ=circ, wide_lvl_mask=0 if single else circ, sync_f1=circ, sync_wait_lvl=circ, tail_lvl=int(single))
  if r: f["rsv502"] = (r - 1) << 6 | 1 << 22 | (D - r) << 24          # partial last FIFO round (as rsv174/1230 of step 1)
  return f

ADD_N2W_RSV589, ADD_N2W_RSV631 = 1099528405829, 4835777067986778041877318   # sync records MESH_SOUTH/WEST/EAST_IN (eltwise.md)
def add_weight_fills(seq:int, tiles:int, narrow_byte:int, w1:int, w2:int) -> list[list[int]]:
  """10 inbound mesh fills (round robin over meshBus 0x17, 0x18, 0x16, 0x15) that write the ADD weight matrix to narrow_byte .. +63:
  rows 0 / 1 = lo / hi half-word of operand 1 (16-bit weight at byte lanes 0 and 2), rows 2 / 3 the same for operand 2; seq .. seq+9"""
  plan = [(0, 0, w1), (4, 0, w1 << 16), (8, 3, 0), (24, 0, w1), (28, 0, w1 << 16), (32, 0, w2), (36, 0, w2 << 16), (40, 3, 0),
          (56, 0, w2), (60, 0, w2 << 16)]
  return [RM.encode_mesh(opcode=(0x17, 0x18, 0x16, 0x15)[j % 4], tile_mask=tiles, seq=seq + j, i_addr=narrow_byte + off, i_sdims=int(cnt > 0),
                         fill_en=1, fill=fill, in_mode=3, **ttu("i_", [1], [cnt + 1], 4)) for j, (off, cnt, fill) in enumerate(plan)]

def add_weight_n2w_fields(seq:int, tiles:int, narrow_byte:int, wide_addr:int) -> dict:
  """moves the 64-byte weight matrix to 4 wide rows (16 bytes each) at wide_addr; waits on the 4 MESH_*_IN counters"""
  return dict(tile_mask=tiles, seq=seq, narrow_addr=narrow_byte // 4, **WN.ttu("n", [1, 4], [4, 4], 6), narrow_lvl_mask=3, wide_addr=wide_addr,
              **WN.ttu("w", [1], [4], 4), wide_lvl_mask=1, sync_en1=0, sync_f1=1, sync_id=3, sync_val=0x8001, sync_wait_lvl=2,
              rsv589=ADD_N2W_RSV589, rsv631=ADD_N2W_RSV631, tail_lvl=1)

def maxpool_relay_n2w_fields(seq:int, tiles:int, narrow_byte:int, wide_addr:int, Rp:int, Rr:int, Wb:int, Hb:int, C:int) -> dict:
  """transposing narrowToWide (tail_f799=1): a Hb x Wb block of C-channel positions (position / row stride Rp / Rr words) to wide
  memory, every 4 positions x 4 channel words -> 4 columns x 4 byte lanes (lane = position, column = channel: maxpool_op_fields' layout).
  C > 64: one block of blk/4 rows per 64-channel group (16 channel words per group)."""
  G = cdiv(C, 64)
  Wg = 16 if G > 1 else cdiv(C, 4)
  quads = cdiv(Wb, 4)
  dims = [(Rp, 4)] + [(1, Wg)] * (Wg > 1) + [(4 * Rp, quads)] * (quads > 1) + [(Rr, Hb)] * (Hb > 1) + [(16, G)] * (G > 1)
  f = dict(tile_mask=tiles, seq=seq, narrow_addr=narrow_byte // 4, narrow_lvl_mask=(1 << len(dims)) - 1, wide_addr=wide_addr,
           wide_lvl_mask=3 if G > 1 else 1, sync_en1=0, sync_en2=0, sync_f1=1, tail_f799=1, tail_lvl=2 if Wg > 1 else 1,
           **WN.ttu("n", [s for s, _ in dims], [c for _, c in dims], 6), **WN.ttu("w", [1, Hb * quads if G > 1 else 0], [Hb * quads, G], 4))
  Wl = cdiv(C - 64 * (G - 1), 4)
  if G > 1 and Wl != 16: f["rsv282"] = (Wl - 1) << 20 | 3 << 36 | (16 - Wl) << 39   # partial last group (like op out_last_*)
  return f
