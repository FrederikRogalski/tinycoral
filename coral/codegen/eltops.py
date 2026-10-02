# SOFTMAX, L2_NORMALIZATION and SUM / MEAN over the channel axis, byte-identical to edgetpu_compiler (docs/isa/codegen_eltops.md):
#   prog = gen_softmax(rows, n, in_q, out_q, beta)     y[r] = softmax(beta * x[r]) over n positions: a float32 loop on the scalar core
#   prog = gen_l2norm(n, in_q)                          y = x / ||x||: two tile ops, the NLU's reciprocal-sqrt spline in between
#   prog = gen_reduce("sum" | "mean", n, in_q, out_q)   y = sum(x) or mean(x): one MXU reduction op (opcode 3)
# The compiler maps all three to STAND_ALONE executables without parameters, so every generator returns one program; *_io() give
# the host contract, executable() wraps a program for coral.runtime, softmax_ref / l2norm_ref / reduce_ref model the arithmetic,
# gen_scalar_probe() builds hardware probes of the scalar core's units (not compiler programs). `python -m coral.codegen.eltops`
# runs the acceptance test (test/test_codegen_eltops.py), offline.
from __future__ import annotations
import numpy as np
from coral.isa import cdiv, round_up, f32, f32_bits, bits_f32, ttu, scalar as SC, wide_narrow as WN, ring_mesh as RM, op as OP, eltwise as EL
from coral.isa import scalar_core as S
from coral.isa.scalar_core import encode_op3
from coral.codegen import Emitter, run_test, WIDE_OUT_FIFO
from coral.codegen import fc as FC
from coral.codegen.conv import _smem_outfeed

SMEM_TOP = 0x4000           # bytes of scalar memory (16 KiB): SOFTMAX keeps its tensor at the top
LOG2E = 1.4426950408889634
S_RING_OUTFEED_RESET = SC.scsync(smask=SC.scm("RING_OUTFEED"))   # before the PRODUCER_A wait of the scalar-memory outfeeds

def exp2_scale(beta:float) -> float:
  """the exp2 argument's factor: f32(f32(beta) * log2 e), beta first rounded to float32 as the TFLite model stores it (the plain
  f32(beta * log2 e) differs in the last bit for e.g. beta = 1/sqrt(48) and 1.3; both agree for powers of two)"""
  return f32(f32(beta) * LOG2E)

# *** SOFTMAX: the scalar-core program ***
# per row: max of the dequantized inputs; sum of exp2((x - max) * beta*log2(e)); its reciprocal; then
# y = clamp(f2i(exp2(..) * recip * 256) + zp_out, 0, 255), written back over the input bytes. Registers: s4 / s5 row pointers (in, out),
# s8 max, s9 sum / reciprocal, s18 / s19 the OUTPUT address (relocated MOVIs).
def softmax_scalar(rows:int, n:int, base:int, zp:int, s_in:float, beta:float, s_out:float, zp_out:int) -> list[int]:
  """the scalar-core softmax over the rows of n bytes from scalar-memory byte `base` (row stride round_up(n, 4)) to the end of
  scalar memory: three 0x23 loops of n iterations per row, the row loop a 0x22 branch on p0 = (row pointer < 0x4000)"""
  m, k = round_up(n, 4), exp2_scale(beta)
  def byte(i, x, a, sh, msk):   # s_x = byte s_i (a, sh, msk: scratch registers for the word address, shift and mask)
    return [S.alu(S.SHR, a, i, imm=2), S.ld(x, a), S.alu(S.AND, sh, i, imm=3), S.alu(S.SHL, sh, sh, imm=3), S.movi(msk, 0xff),
            S.alu(S.SHL, msk, msk, reg=sh)]
  def deq(x):                   # s_x = (s_x - zp) * s_in (no subtraction for zp 0)
    return [S.alu(S.SUB, x, x, imm=zp)] * (zp != 0) + [S.i2f(x, x), S.fimm(S.FMUL, x, x, s_in)]
  w = [S.movi(4, base), S.movi(5, base)]
  top = len(w)
  w += [S.mov(6, 4), S.mov(7, 5), S.movi(8, 0xff800000), S.mov(10, 6)]
  body = byte(10, 9, 11, 12, 13) + [S.alu(S.AND, 9, 9, reg=13), S.alu(S.SHR, 9, 9, reg=12)] + deq(9) + \
         [S.alu(S.FMAX, 8, 8, reg=9), S.alu(S.ADD, 10, 10, imm=1)]
  w += [S.loop(n, len(body))] + body
  w += [S.movi(9, 0), S.mov(11, 6)]
  body = byte(11, 10, 12, 13, 14) + [S.alu(S.AND, 10, 10, reg=14), S.alu(S.SHR, 10, 10, reg=13)] + deq(10) + \
         [S.alu(S.FSUB, 10, 10, reg=8), S.fimm(S.FMUL, 10, 10, k), S.sfu(S.EXP2, 10, 10), S.alu(S.ADD, 11, 11, imm=1), S.NOP,
          S.alu(S.FADD, 9, 9, reg=10)]
  w += [S.loop(n, len(body))] + body
  w += [S.mov(17, 6), S.mov(13, 7), S.sfu(S.RECIP, 9, 9)]
  body = byte(17, 10, 14, 15, 16) + [S.mov(11, 10), S.alu(S.AND, 10, 10, reg=16), S.alu(S.XOR, 11, 11, reg=10), S.alu(S.SHR, 10, 10, reg=15)] + \
         deq(10) + [S.alu(S.FSUB, 10, 10, reg=8), S.fimm(S.FMUL, 10, 10, k), S.sfu(S.EXP2, 10, 10), S.alu(S.ADD, 17, 17, imm=1),
                    S.alu(S.SHR, 14, 13, imm=2), S.alu(S.FMUL, 10, 10, reg=9), S.fimm(S.FMUL, 10, 10, f32(1 / f32(s_out)))]
  if zp_out: body += [S.f2i(10, 10), S.alu(S.ADD, 10, 10, imm=zp_out), S.i2f(10, 10)]
  body += [S.fimm(S.FMIN, 10, 10, 255.0), S.fimm(S.FMAX, 10, 10, 0.0), S.f2i(10, 10), S.alu(S.SHL, 10, 10, reg=15), S.alu(S.OR, 11, 11, reg=10),
           S.st(14, 11), S.alu(S.ADD, 13, 13, imm=1)]
  w += [S.loop(n, len(body))] + body
  w += [S.alu(S.ADD, 4, 4, imm=m), S.alu(S.ADD, 5, 5, imm=m), S.alu(S.LT, 0, 5, imm=SMEM_TOP)]
  w += [S.branch(top - len(w))]
  return w

def smem_to_host(base_word:int, data_words:int, nbytes:int) -> list[int]:
  """host DMA (tag 3) of nbytes from the OUTPUT address in s18 / s19, fed word by word from scalar memory (s6 walks it, s7 / s8 hold
  two words in flight); words past the data are pushed as zeros"""
  w = [S.mov(4, 18), S.mov(5, 19)] + SC.output_dma_tail(nbytes) + [S.movi(6, base_word)]
  pairs, odd = divmod(data_words, 2)
  if pairs:
    w += [S.ld(7, 6, inc=True), S.ld(8, 6, inc=True)]
    if pairs > 1: w += [S.loop(pairs - 1, 2), S.ld(7, 6, inc=True, v_op=8, vs_reg=7), S.ld(8, 6, inc=True, v_op=8, vs_reg=8)]
    w += [S.push(7), S.push(8)]
  if odd: w += [S.ld(7, 6, inc=True), S.NOP, S.push(7)]
  if (pad := nbytes // 4 - data_words): w += [S.movi(6, 0), S.loop(pad, 1), S.push(6)]
  return w

# *** SOFTMAX: data movement host -> tiles -> scalar memory ***
def w2n_input(nb:int, c:int, X:int, seq:int, tiles:int, top:int) -> list[int]:
  """wide_narrow.w2n_input with the input ring FIFO right below wide address `top` (64-byte units) instead of WIDE_TOP"""
  d = WN.decode_wide_to_narrow(WN.w2n_input(nb, c, X, seq, tiles))
  return WN.encode_wide_to_narrow(**(d | dict(wide_addr=d["wide_addr"] - (0x2080 - top))))

def rcons_input(g:FC.FCGeom, j:int, tiles:int, seq:int, top:int) -> list[int]:
  """fc._rcons_input with the FIFO below `top`"""
  d = RM.decode_ringConsumer(FC._rcons_input(g, j, tiles, seq))
  return RM.encode_ringConsumer(**(d | dict(addr=d["addr"] - (0x2080 - top))))

def sm_narrow_x(n:int) -> int:
  """narrow byte address of a single-row input on its tiles (edgetpu_compiler's allocation, fitted on n = 1..256, holds to 8192)"""
  return 0 if n > 188 or (n > 84 and n % 4) else 768

def _sm_single_row(e:Emitter, n:int, base:int):
  """one row: the FULLY_CONNECTED input path (64-byte pieces on tiles 0..T-1), then every tile's piece over the ring into scalar memory"""
  g = FC.FCGeom(n, n)
  X = sm_narrow_x(n)
  wl = _sm_wide(n, cdiv(g.S, 256), ident=False)
  e.sync(SC.input_head)
  e.sync(SC.sync_wn_fence)
  e.scalar(SC.input_dma(g.S))
  top = wl["in"] + 8 * cdiv(g.S, 256)
  for t in range(g.T_in): e.tile(w2n_input(n, t, X // 4, e.seq, 1 << t, top))
  for j in range(cdiv(g.T_in, 4)):
    ts = sum(1 << t for t in range(4 * j, min(4 * j + 4, g.T_in)))
    e.tile(rcons_input(g, j, ts, e.seq, top))
    e.scalar(SC.av_infeed(4 * j * g.b, min(4 * g.b, g.S - 4 * j * g.b), ts))
  e.sync(SC.sync_drain)
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_drain)
  e.scalar([S.movi(18, 0), S.movi(19, 0)])
  for t in range(g.T_in): e.tile(WN.n2w_output(g.piece(t), 1 << t, (X + g.b * t) // 4, wl["out"], e.seq))
  for t in range(g.T_in):
    e.tile(FC.rprod_output(g.piece(t), 1 << t, t, e.seq, wl["out"]))
    e.scalar(_smem_outfeed(base // 4 + g.b // 4 * t, cdiv(g.piece(t), 4)))
  return g.T_in

# several rows: every tile column t takes bytes [64t, 64t+64) of every row; the input streams through the tiles' ring FIFO into a
# one-row narrow staging ring, a copy op per tile packs its slice into a dense [rows][round_up(n, 4)] block (through the identity
# row), and each tile sends its slice into scalar memory
def _sm_wide(n:int, pk:int, ident:bool=True) -> dict:
  """the wide buffers (64-byte units): identity row (+ the zero row when n % 4 != 0), output FIFO (2 rows), input ring FIFO (pk
  packets), stacked down from 0x2080 in order of size, ties in that order"""
  bufs = sorted([("ident", 4 if n % 4 == 0 else 8, 0)] * ident + [("out", 8, 1), ("in", 8 * pk, 2)], key=lambda b: (-b[1], b[2]))
  top, out = 0x2080, {}
  for name, size, _ in bufs:
    top -= size
    out[name] = top
  return out

def _sm_copy_bulk(seq:int, W:int, X:int, ident:int) -> dict:
  """one tile: the W words that the wideToNarrow streams through narrow word X (stride 0 on both sides: a one-word handshake,
  in_tflags 0x40) go to 0.. through the identity (pack mode)"""
  return dict(tile_mask=1, seq=seq, loop1=W - 1, **OP.ttu_fine("in", [4, 0], [1, W]), in_base=X, in_mode7=1, in_tflags=0x40, rsv515=8,
              **OP.ttu_fine("out", [4, 4], [1, W]), out_mode7=3, **OP.wide("par", ident), par_hmode=1, **ttu("par_", [], [1, W], 8), par_mode7=1,
              psum_hmode=1, **ttu("psum_", [], [1, W], 8), psum_tflags=0x840, cfg0=3, cfg1=0x12b, sync0=0x4000, sync1=0x4000, cfg2=6, dp_mode=2,
              out_ch=3, out_ch_last=3, mult_bits=f32_bits(1.0), clamp_max_bits=f32_bits(2.0 ** 32))

def _sm_copy(seq:int, t:int, h:int, rows:int, n:int, m:int, X:int, ident:int, progress:tuple[int, int, int]) -> dict:
  """tile t: per row its h words of the staging ring (re-read every row) to out[r][16t ..] (row stride m bytes); h == 1: pack mode"""
  sync0, sync1, x = progress
  tw = dict(rsv515=2 * n & 127, in_twait=2 * n >> 7)
  if h == 1:
    return dict(tile_mask=1 << t, seq=seq, loop1=rows - 1, **OP.ttu_fine("in", [4, 0], [1, rows]), in_base=X, in_mode7=1, in_tflags=0x40, **tw,
                **OP.ttu_fine("out", [4, m], [1, rows]), out_base=64 * t, out_mode7=3, **OP.wide("par", ident), par_hmode=1,
                **ttu("par_", [], [1, rows], 8), par_mode7=1, psum_hmode=1, **ttu("psum_", [], [1, rows], 8), psum_tflags=0x840, cfg0=3,
                cfg1=0x12b, sync0=sync0, sync1=sync1, cfg2=6, dp_mode=2, out_ch=3, out_ch_last=3, mult_bits=f32_bits(1.0),
                clamp_max_bits=f32_bits(2.0 ** 32))
  return dict(tile_mask=1 << t, seq=seq, loop1=h - 1, loop2=rows - 1, **OP.ttu_fine("in", [4, 4, 0], [1, h, rows]), in_base=X, in_mode7=3,
              in_tflags=0x80, **tw, **OP.ttu_fine("out", [4, 4, m], [1, h, rows]), out_base=64 * t, out_mode7=3, out_tflags=1,
              **OP.wide("par", ident), par_hmode=1, **ttu("par_", [], [1, h, rows], 8), par_mode7=1, psum_hmode=1,
              **ttu("psum_", [], [1, h, rows], 8),
              psum_tflags=0x840, cfg0=3, cfg1=0x4b | x << 8, sync0=sync0, sync1=sync1, cfg2=6, dp_mode=3, out_ch=3, out_ch_last=3,
              mult_bits=f32_bits(1.0), clamp_max_bits=f32_bits(2.0 ** 32))

def _sm_w2n(seq:int, t:int, h:int, L:int, rows:int, n:int, X:int, inf:int, pk:int, f1:int) -> list[int]:
  """tile t receives the whole input stream into a one-row staging ring (L words) at narrow X; tiles t >= 1 skip the 16t words in
  front of their slice (the head chunk is cut to L - 16t words)"""
  v = h << 16 | (-h & 0xffff)                       # the sync decrement record: -h words (16 bits), then +h
  f = dict(tile_mask=1 << t, seq=seq, narrow_addr=X // 4, narrow_lvl_mask=1, mode=1, sync_id=14, sync_val=0x8000, sync_dec_lvl=1,
           sync_dec=(v & 0x1ffff) - (0x20000 if v & 0x10000 else 0), rsv682=v >> 17, sync_dec_mode=3, tail_f871=1, tail_en870=0, sync_f1=f1,
           rsv489=n // 2 & 31, size64=n // 2 >> 5, **WN.ttu("n", [1, 0], [L, rows], 6))
  f |= dict(wide_addr=inf, **_fifo(pk))
  if t:
    sk = 16 * t
    f.update(head_words_m1=L - sk - 1, head_en=1, head_tail16=t, skip_en0=1, rsv543=1, skip_base=X // 4, skip_m1=sk - 1, skip_end=X // 4 + sk - 1)
  return WN.encode_wide_to_narrow(**f)

def _fifo(pk:int) -> dict:
  """wide side of a ring FIFO walk: up to 2 packets in their own rows, more through one row refilled pk times"""
  return dict(**WN.ttu("w", [1], [pk], 4), wide_lvl_mask=1) if pk <= 2 else dict(wide_rows=1, **WN.ttu("w", [0], [pk], 4))

def _sm_w2n_bulk(seq:int, W:int, X:int, inf:int, pk:int) -> list[int]:
  return WN.encode_wide_to_narrow(tile_mask=1, seq=seq, wide_addr=inf, **_fifo(pk), narrow_addr=X // 4,
                                  **WN.ttu("n", [0], [W], 6), rsv489=2, sync_id=14, sync_val=0x8000, sync_dec=-1, sync_dec_mode=3, tail_en870=1)

def _sm_rcons(seq:int, t:int, inf:int, pk:int, aligned:bool=False, n:int=0) -> list[int]:
  """ringConsumer of tile t: pk packets into the FIFO (pk <= 2: all of them, else one slot refilled pk times with a WIDE_TO_NARROW
  record of -(64 + 16t) words; tile 0 of a stream of whole packets: the record (+h, -h) with h = 256 // n instead)"""
  if pk <= 2: return RM.encode_ringConsumer(tile_mask=1 << t, seq=seq, addr=inf, sdims=1, mode=1, **ttu("", [1], [pk], 4))
  f = dict(s_val=-64 - 16 * t, s_cnt=1)
  if aligned:
    h = 256 // n
    v = h << 16 | (-h & 0xffff)
    f = dict(s_val=(v & 0xffff) - (0x10000 if v & 0x8000 else 0), s_en_a=v >> 16 & 1, rsv363=v >> 17, s_cnt=0)
  return RM.encode_ringConsumer(tile_mask=1 << t, seq=seq, addr=inf, **ttu("", [0], [pk], 4), slots=1, mode=1, s_id=11, s_en_b=1, s_y=1, **f)

def _sm_out(e:Emitter, t:int, h:int, rows:int, m:int, T:int, outf:int, base_w:int):
  """tile t's slice of the packed block: narrowToWide (rows x h words, row stride m bytes) to the output FIFO, ringProducer to the
  scalar core, outfeed into scalar memory at the slice's place"""
  dims = [(1, h), (m // 4, rows)] if T > 1 else [(1, h * rows)]
  pk = cdiv(4 * h * rows, 256)
  sy = dict(sync_en1=0, sync_en2=0) if pk <= 2 else dict(sync_id=16, sync_val=0xffff, sync_f45=1)
  e.tile(WN.encode_narrow_to_wide(tile_mask=1 << t, seq=e.seq, narrow_addr=16 * t, narrow_lvl_mask=(1 << len(dims)) - 1, wide_addr=outf,
                                  tail_lvl=len(dims), **sy, **WN.ttu("n", [s for s, _ in dims], [c for _, c in dims], 6), **_fifo(pk)))
  rp = dict(**ttu("", [1], [pk], 4), sdims=1) if pk <= 2 else dict(**ttu("", [0], [pk], 4), slots=1)
  e.tile(RM.encode_ringProducer(tile_mask=1 << t, seq=e.seq, addr=outf, **rp, mode=1, pcfg=12 if pk > 1 else 4,
                                r0_id=(int(pk > 1) << 5) | 17, r0_val=t, r0_en_a=1, r0_en_b=1, r1_id=13, r1_val=1, r1_en_a=1, r1_en_b=1,
                                dest=1 << RM.SCALAR_CORE))
  if T == 1: e.scalar(SC.OUTFEED.encode(base=base_w, d0_stride=1, d0_limit=h * rows - 1, d1_stride=(1 - h * rows) & 0xffff,
                                        d2_stride=(1 - h * rows) & 0xffff, d3_stride=1, k173=3, f222=0x3f))
  else: e.scalar(SC.OUTFEED.encode(base=base_w + 16 * t, d0_stride=1, d0_limit=h - 1, d1_stride=(m // 4 - (h - 1)) & 0xffff, d1_limit=rows - 1,
                                   d2_stride=-(h - 1 + (rows - 1) * m // 4) & 0xffff, d3_stride=3, k173=1, rsv175=1, f222=0x3f))

def _sm_progress(n:int, pk:int, t:int) -> tuple[int, int, int]:
  """(sync0, sync1, cfg1 bits 8-9) of the copy ops: a progress count of 4n in 1/64 units (as codegen_conv's copy ops), or the
  'no wait' value 0x4000 / 0x4000 / 1 when the stream fits the FIFO (<= 2 packets) and on tile 0 of rows of 128 or 256 bytes"""
  if pk <= 2 or (t == 0 and n % 128 == 0): return 0x4000, 0x4000, 1
  t64 = 4 * n
  return (t64 % 64) // 16 << 14 | t64 // 64, 0x4000 + t64 // 64, (t64 % 64) // 16

def direct_input_dma(nb:int) -> list[int]:
  """the input DMA of the multi-row data path: host DMA, then an avDataPop without the staging-buffer walk (bit 126 set, no
  credit); 16 bytes: a two-unit walk instead"""
  u = nb // 8
  pop = SC.POP.encode(stream=1, d0_limit=u - 1, rsv125=2, k174=1, k178=1, b183=1, rows_neg=0x7fff, k199=1, k215=1) if u > 2 else \
        SC.POP.encode(stream=1, d0_stride=1, d0_limit=u - 1, d1_stride=-(u - 1) & 0xffff, d2_stride=-(u - 1) & 0xffff, k121=1, k174=1)
  return [S.movi(11, 0), S.movi(12, 0)] + SC.host_dma(SC.TAG_INPUT, nb) + pop

def direct_infeed(nb:int, tiles:int) -> list[int]:
  """its infeed: all nb bytes in one transfer to `tiles` (bit 162 set, no staging-buffer walk)"""
  u, f = nb // 8, dict(stream=1, d0_limit=nb // 8 - 1, k311=1, pop_wait=1, k336=1, k352=1, k392=1, tiles=tiles, k434=0xf, f438=3)
  if u > 2: return SC.INFEED.encode(**f, rsv161=2)
  return SC.INFEED.encode(**f, d0_stride=1, d1_stride=-(u - 1) & 0x1fffff, d2_stride=-(u - 1) & 0x1fffff, k157=1)

def _sm_copy_bytes(seq:int, t:int, b:int, T:int, rows:int, n:int, m:int, X:int, ident:int, progress:tuple[int, int, int], g:int=0) -> dict:
  """n % 4 != 0: tile t's b bytes of every row, read byte by byte (row stride n), written as ceil(b/4) words per row (row stride m;
  one tile: dense), the bytes past n from the zero row behind the identity. g = 0: from the whole input at narrow X (odd n);
  g = 2: from the one-chunk staging ring (n % 4 == 2: the stream comes in chunks of g rows, re-read per chunk)"""
  sync0, sync1, x = progress
  kw = cdiv(b, 4)
  c0 = 1 if b == 1 else 2 if b < 4 else 4       # byte steps per inner round
  c1 = 1 if b == 1 else 2 if b < 4 else kw
  last = (b - 1) % c0 + 1 if b > 1 else 1        # bytes in the last inner round
  mid = T > 1 and (b < 64) and b > 2             # the last tile of several: one extra count-1 level
  rl = [n, 0] if g else [n]                      # the rows: [n x g, 0 x rows/g] (staging ring) or [n x rows]
  rc = [g, rows // g] if g else [rows]
  ins, inc = [1, c0] + [n] * mid + rl, [c0, c1] + [1] * mid + rc
  if b == 1: ins, inc = [1, n] + rl, [1, 1] + rc
  if b == 2: ins, inc = [1, n] + rl, [2, 1] + rc
  if T == 1: outs, outc = [4, 4], [1, rows * kw]
  elif kw > 1: outs, outc = [4, 4, m], [1, kw, rows]
  else: outs, outc = [4, m], [1, rows]
  dense = kw > 1 and T > 1
  par = [1] if b == 1 else [1, 0] + [0] * mid
  loops = inc
  tw = 2 * g * n                                 # staging: twice the chunk bytes (as codegen_conv's copy ops)
  d = dict(tile_mask=1 << t, seq=seq, **{f"loop{k}": c - 1 for k, c in enumerate(loops)}, **OP.ttu_fine("in", ins, inc), in_base=X, in_mode7=3,
           in_tflags=(3 if mid else 0xc1) if g else 3 if mid else 1, **OP.ttu_fine("out", outs, outc), out_base=0 if T == 1 else 64 * t, out_mode7=3,
           out_tflags=int(dense), **OP.wide("par", ident), **OP.ttu_fine("par", par, loops), par_mode7=1,
           **(OP.ttu_fine("psum", [1], [1, 1, rows]) if b == 1 else ttu("psum_", [], loops, 8)), psum_tflags=0x840,
           cfg0=5 if dense and (not g or (n == 126 and t == 0)) else 3, cfg1=(0x8b if mid else 0x6b) | x << 8, sync0=sync0, sync1=sync1,
           cfg2=6, cfg3=7, dp_mode=3 if dense else 2, out_ch=3, out_ch_last=3, mult_bits=f32_bits(1.0), clamp_max_bits=f32_bits(2.0 ** 32))
  if b == 1: d["psum_hmode"] = 1
  partial = b > 2 and last != c0
  if partial:                                    # partial last inner round
    d.update(rsv174=0x4000 | (last - 1), rsv536=(c0 - last) << 20 | 2 << 16 | (last - 1) >> 1, par_fifo=((last - 1) & 1) << 13,
             rsv1230=(c0 - last) << 16 | 1 << 13 | (last - 1) >> 1, rsv1523=(last - 1) << 9 | 1 << 23)
  if g: d.update(rsv515=tw & 127 | int(mid), in_twait=tw >> 7 | (0x2000 if partial else 0))
  return d

def _sm_w2n_whole(seq:int, t:int, W:int, X:int, inf:int, pk:int) -> list[int]:
  """odd n: tile t receives the input stream (W words) into narrow X, skipping the 16t words in front of its slice"""
  f = dict(tile_mask=1 << t, seq=seq, narrow_addr=X // 4, narrow_lvl_mask=1, **WN.ttu("n", [1], [W - 16 * t], 6), wide_addr=inf, **_fifo(pk),
           sync_id=14, sync_val=0x8000, sync_f1=int(pk <= 2), tail_f871=1, tail_en870=0)
  if t: f.update(skip_en0=1, rsv543=1, skip_base=X // 4, skip_m1=16 * t - 1, skip_end=X // 4 + 16 * t - 1)
  return WN.encode_wide_to_narrow(**f)

def _sm_multi_row(e:Emitter, rows:int, n:int, base:int) -> int:
  """several rows (verified: 6 rows for n = 4..256; 3..32 rows at sample n, see the NotImplementedErrors): identity row, input DMA,
  per tile a copy op, its wideToNarrow and ringConsumer, one infeed; then per tile its slice into scalar memory. The stream is cut
  in rows (n % 4 == 0), pairs of rows (n % 4 == 2) or not at all (odd n: the whole input lands in narrow memory first)."""
  m, nb = round_up(n, 4), 8 * cdiv(rows * n, 8)
  if rows == 2: raise NotImplementedError("2 rows: edgetpu_compiler uses another copy form (not modelled)")
  if n > 256: raise NotImplementedError("several rows of more than 256 bytes: edgetpu_compiler streams them differently (not modelled)")
  if n < 4 or (n % 4 and rows * m < 48):
    raise NotImplementedError("tiny multi-row softmax (n < 4, or n % 4 != 0 and rows * round_up(n, 4) < 48): not modelled")
  if n % 4 == 2 and (rows % 2 or rows == 4): raise NotImplementedError("n % 4 == 2 with an odd number of rows or 4 rows: not modelled")
  if n % 2 and rows % 4 == 0: raise NotImplementedError("odd n with a multiple of 4 rows (streamed in groups of 4 rows): not modelled")
  pk, T = cdiv(nb, 256), cdiv(n, 64)
  wl = _sm_wide(n, pk)
  # narrow memory: the packed block at 0, then the input staging and the identity (+ zero row: 272 bytes), the bigger one first
  if n % 4 == 0: X, C = rows * m, rows * m + nb
  elif rows * m < 272:
    C = rows * m
    X = C + 272
  else:
    X = rows * m
    C = X + 8 * cdiv(rows * m, 8)
  e.sync(SC.input_head)
  e.tile(*EL.ident_prologue(e.seq, 0xffff, C, wl["ident"], 64 if n % 4 else 0))
  e.sync(SC.sync_wn_fence)
  e.scalar(direct_input_dma(nb))
  hs = [min(16, m // 4 - 16 * t) for t in range(T)]
  if n % 2:
    W = cdiv(rows * n, 4)
    for t in range(T):
      prog = (0x4000, 0x4000, 1) if pk <= 2 else ((W - 16 * t) % 4 << 14 | (W - 16 * t) // 4, 0x4000 + (W - 16 * t) // 4, (W - 16 * t) % 4)
      e.tile(OP.encode_op(**_sm_copy_bytes(e.seq, t, min(64, n - 64 * t), T, rows, n, m, X, wl["ident"], prog)))
    for t in range(T): e.tile(_sm_w2n_whole(e.seq, t, W, X, wl["in"], pk))
    for t in range(T): e.tile(_sm_rcons(e.seq, t, wl["in"], pk))
  elif n % 4 == 2:                               # rows stream in pairs (2n bytes = a whole number of words)
    for t in range(T):
      e.tile(OP.encode_op(**_sm_copy_bytes(e.seq, t, min(64, n - 64 * t), T, rows, n, m, X, wl["ident"], _sm_progress(2 * n, pk, t), g=2)))
    for t in range(T):   # (n = 126, tile 0: edgetpu_compiler's sync record is 2 words, its copy op cfg0 5; seen for 6 rows only)
      e.tile(_sm_w2n(e.seq, t, 2 if (n == 126 and t == 0) else 2 * hs[t], n // 2, rows // 2, 2 * n, X, wl["in"], pk, int(pk <= 2)))
    for t in range(T): e.tile(_sm_rcons(e.seq, t, wl["in"], pk))
  elif T == 1:
    e.tile(OP.encode_op(**_sm_copy_bulk(e.seq, rows * m // 4, X, wl["ident"])))
    e.tile(_sm_w2n_bulk(e.seq, rows * m // 4, X, wl["in"], pk))
    e.tile(_sm_rcons(e.seq, 0, wl["in"], pk))
  else:
    for t in range(T): e.tile(OP.encode_op(**_sm_copy(e.seq, t, hs[t], rows, n, m, X, wl["ident"], _sm_progress(n, pk, t))))
    for t in range(T): e.tile(_sm_w2n(e.seq, t, hs[t], n // 4, rows, n, X, wl["in"], pk, int(pk <= 2 or (t == 0 and n % 128 == 0))))
    for t in range(T): e.tile(_sm_rcons(e.seq, t, wl["in"], pk, t == 0 and n % 128 == 0, n))
  e.scalar(direct_infeed(nb, (1 << T) - 1))
  if T > 1 and pk >= 3 and n % 2 == 0:
    for t in range(1, T): e.sync(SC.sync, tiles=1 << t, counters=SC.tc("WIDE_TO_NARROW"), count=rows * n // 4, units=0x10)
    e.sync(SC.sync_drain, 1 << (T - 1))
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_drain)
  e.scalar([S.movi(18, 0), S.movi(19, 0)])
  for t in range(T): _sm_out(e, t, hs[t], rows, m, T, wl["out"], base // 4)
  return T

def softmax_io(rows:int, n:int) -> dict:
  """host contract: input rows x n uint8 row-major (padded to 8 bytes); output rows x round_up(n, 4) (row r at r*round_up(n, 4),
  the first n bytes valid), padded to 8 bytes"""
  m = round_up(n, 4) if rows > 1 else n
  return dict(in_bytes=8 * cdiv(rows * n, 8), out_bytes=8 * cdiv(rows * m, 8), out_row_stride=m)

def gen_softmax(rows:int, n:int, in_q:tuple=(1/16, 128), out_q:tuple=(1/256, 0), beta:float=1.0) -> bytes:
  """SOFTMAX over the last axis of a [rows, n] uint8 tensor (TFLite [1, rows, n] or [rows, n]): the STAND_ALONE program, byte-identical
  to edgetpu_compiler. in_q / out_q: (scale, zero point); TFLite fixes out_q = (1/256, 0)."""
  m = round_up(n, 4)
  assert rows * m <= SMEM_TOP, "the tensor must fit the 16 KiB scalar memory (edgetpu_compiler leaves bigger ones on the CPU)"
  base = SMEM_TOP - rows * m
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  T = _sm_single_row(e, n, base) if rows == 1 else _sm_multi_row(e, rows, n, base)
  e.scalar(S_RING_OUTFEED_RESET)
  e.scalar(SC.output_wait(e.seq, T), seqs=2)
  e.scalar(softmax_scalar(rows, n, base, in_q[1], f32(in_q[0]), beta, out_q[0], out_q[1]))
  io = softmax_io(rows, n)
  e.scalar(smem_to_host(base // 4, rows * m // 4, io["out_bytes"]))
  e.sync(SC.epilogue)
  return e.program()

# *** hardware probes of the scalar core's units (our own programs: the one-row softmax data path, another scalar loop) ***
PROBES = {"exp2": lambda d, x: S.sfu(S.EXP2, d, x), "recip": lambda d, x: S.sfu(S.RECIP, d, x), "f2i": lambda d, x: S.f2i(d, x),
          "i2f": lambda d, x: S.i2f(d, x), "fmul": lambda d, x, c: S.fimm(S.FMUL, d, x, c), "fadd": lambda d, x, c: S.fimm(S.FADD, d, x, c)}

def gen_scalar_probe(kind:str, words:int, c:float=0.0) -> bytes:
  """NOT FROM edgetpu_compiler, never run: the input (`words` float32 / int32 values, 4*words bytes) goes to scalar memory exactly as
  gen_softmax(1, 4*words) moves it, then every word w is replaced by f(w) and the block goes back to the host in place of the
  softmax output. kind: 'exp2' / 'recip' (special-function unit ops 3 / 2), 'f2i', 'i2f', 'fmul' / 'fadd' (by the immediate c).
  Four NOP bundles around the unit op: the pipeline latencies are not known (the compiler waits 2 bundles after exp2)."""
  n = 4 * words
  assert n <= 8192 and kind in PROBES
  base = SMEM_TOP - n
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  T = _sm_single_row(e, n, base)
  e.scalar(S_RING_OUTFEED_RESET)
  e.scalar(SC.output_wait(e.seq, T), seqs=2)
  op = PROBES[kind](10, 9, c) if kind in ("fmul", "fadd") else PROBES[kind](10, 9)
  body = [S.ld(9, 6)] + [S.NOP] * 4 + [op] + [S.NOP] * 4 + [S.st(6, 10), S.alu(S.ADD, 6, 6, imm=1)]
  e.scalar([S.movi(6, base // 4), S.loop(words, len(body))] + body)
  e.scalar(smem_to_host(base // 4, words, 8 * cdiv(n, 8)))
  e.sync(SC.epilogue)
  return e.program()

def executable(prog:bytes, in_bytes:int, out_bytes:int, in_name:str="x", out_name:str="y"):
  """a coral.executable.Executable of one generated program (STAND_ALONE, no parameters) that coral.runtime.run_executable can run;
  the hints are edgetpu_compiler's (instructions, input DMA, output DMA, interrupt)"""
  from coral.executable import Executable, Bitstream, Hint
  dmas = [(t, n) for t, n in S.dma_seqs(prog) if t in (SC.TAG_INPUT, SC.TAG_OUTPUT)]
  assert dmas == [(SC.TAG_INPUT, in_bytes), (SC.TAG_OUTPUT, out_bytes)], dmas
  hints = [Hint("instruction", "INFEED", chunk=0), Hint("dma", "INFEED", "INPUT", in_name, 0, in_bytes),
           Hint("dma", "OUTFEED", "OUTPUT", out_name, 0, out_bytes), Hint("interrupt", "OUTFEED", interrupt=0)]
  return Executable(None, "STAND_ALONE", 1, 0, [Bitstream(prog, [])], b"", hints, True, [], [], "beagle", 0, 0, 0, 0)


# *** L2_NORMALIZATION and SUM / MEAN over the channels of a single row: one tile computes, the input is gathered onto tile 0 ***
def _row_input(e:Emitter, n:int, X:int, R:int):
  """the FULLY_CONNECTED input path of n bytes (64*ceil(n/1024)-byte pieces on tiles 0..T-1 at narrow X) and the mesh gather onto
  tile 0 (R: the relay buffer)"""
  g = FC.FCGeom(n, n)
  e.sync(SC.input_head)
  e.sync(SC.sync_wn_fence)
  e.scalar(SC.input_dma(g.S))
  for t in range(g.T_in): e.tile(WN.w2n_input(n, t, X // 4, e.seq, 1 << t))
  for j in range(cdiv(g.T_in, 4)):
    ts = sum(1 << t for t in range(4 * j, min(4 * j + 4, g.T_in)))
    e.tile(FC._rcons_input(g, j, ts, e.seq))
    e.scalar(SC.av_infeed(4 * j * g.b, min(4 * g.b, g.S - 4 * j * g.b), ts))
  e.sync(SC.sync_drain)
  e.sync(SC.sync_reset17)
  if g.T_in >= 2: FC.gather(e, g, X, R, lambda t: 1 << t)

def _row_output(e:Emitter, nbytes:int, Y:int):
  """tile 0's nbytes at narrow Y to the host (one outfeed)"""
  e.sync(SC.sync_reset17)
  e.sync(SC.signal_fence)
  e.scalar(SC.output_dma_head())
  e.tile(WN.n2w_output(nbytes, 1, Y // 4, WIDE_OUT_FIFO, e.seq))
  e.scalar(SC.output_dma_tail(8 * cdiv(nbytes, 8)))
  e.scalar(SC.outfeed(8 * cdiv(nbytes, 8)))
  e.tile(FC.rprod_output(nbytes, 1, 0, e.seq))
  e.scalar(SC.output_wait(e.seq, 1), seqs=2)

ROW_FIFO = 0x2060           # the 4-row FIFO that streams x through wide memory (eltwise's MUL operand FIFO)
ROW_FIFO2 = 0x2058          # L2_NORMALIZATION: x again, one word per wide row, for the scaling op

def _feed(seq:int, X:int, w:int, fifo:int=ROW_FIFO, **kw) -> list[int]:
  """x (w words at narrow X on tile 0) into the 4-row FIFO, one word per wide row (eltwise.mul_feed_n2w_fields)"""
  return WN.encode_narrow_to_wide(**(EL.mul_feed_n2w_fields(seq, 1, X, fifo, w) | kw))

def _reduce_op_fields(w:int, seq:int, X:int|None, out:int, zp:int, q:dict, fifo:int=ROW_FIFO) -> dict:
  """the reduction op: every FIFO row (one word of x, through the par TTU) times the input word (L2: x again through the in TTU,
  opcode 1; SUM / MEAN: no in operand, opcode 3), accumulated over all w words; then requantized"""
  D = EL.mul_fifo_depth(w)
  r = w % D
  d = dict(tile_mask=1, seq=seq, loop0=D - 1, loop1=cdiv(w, D) - 1, **OP.ttu_fine("out", [4, 8, 8, 8], [2, 1, 1, 1]) if X is not None else
           OP.ttu_fine("out", [4, 4, 4, 4], [1, 1, 1, 1]), out_base=out, out_mode7=3, out_tflags=3,
           **OP.ttu_fine("par", [4], [D, cdiv(w, D)]), **OP.wide("par", fifo), par_mode7=1, par_tflags=0x40, par_fifo=D,
           **ttu("psum_", [], [w], 8), psum_tflags=0x840, cfg0=0x60, cfg1=0x12d, sync0=0x4000, sync1=0x4000, cfg2=6, w_zp=zp, dp_mode=1,
           reduce_mask=1, out_ch=3, out_ch_last=3, out_zp=q["out_zp"], mult_bits=f32_bits(q["mult"]), clamp_min_bits=f32_bits(q["clamp_min"]),
           clamp_max_bits=f32_bits(q["clamp_max"]))
  if X is not None: d |= dict(**OP.ttu_fine("in", [4, 4 * w, 4 * w], [w, 1, 1]), in_base=X, in_mode7=3, in_tflags=1, in_zp=zp)
  if r: d.update(rsv174=0x4000 | (r - 1), rsv1230=(1 << 13) | (r - 1) | (D - r) << 18)   # partial last FIFO round (r rows)
  if w == 1: d["psum_hmode"] = 1                     # a single step (as eltwise's single-step requantize ops)
  return d

def reduce_quant(kind:str, n:int, in_q:tuple, out_q:tuple) -> dict:
  """SUM: mult = s_in / s_out, the output range clamps; MEAN: mult = s_in / s_out / n and no clamps (edgetpu_compiler's float32
  formulas)"""
  if kind == "sum":
    lo, hi = EL.out_clamps(out_q[0], out_q[1])
    return dict(mult=EL.mul32(in_q[0], EL.recip32(out_q[0])), clamp_min=lo, clamp_max=hi, out_zp=out_q[1])
  return dict(mult=f32(EL.mul32(in_q[0], EL.recip32(out_q[0])) / n), clamp_min=float("-inf"), clamp_max=float("inf"), out_zp=out_q[1])

def reduce_io(n:int) -> dict:
  return dict(in_bytes=8 * cdiv(n, 8), out_bytes=8)

def gen_reduce(kind:str, n:int, in_q:tuple=(1/16, 128), out_q:tuple=(1/16, 128)) -> bytes:
  """SUM or MEAN of a [1, 1, n] uint8 tensor over its last axis (keep_dims), byte-identical to edgetpu_compiler for n % 4 == 0.
  The output is one byte (the host reads 8)."""
  assert kind in ("sum", "mean")
  if n % 4: raise NotImplementedError("n % 4 != 0: edgetpu_compiler adds a parameter blob (not modelled)")
  X, Y, R = FC.narrow_layout(1, n)
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  _row_input(e, n, X, R)
  e.sync(SC.sync_reset_par)
  e.tile(_feed(e.seq, X, n // 4))
  f = _reduce_op_fields(n // 4, e.seq, None, Y, in_q[1], reduce_quant(kind, n, in_q, out_q))
  e.tile(encode_op3(**f))
  e.sync(SC.sync, counters=SC.tc("NARROW_TO_WIDE", "PARAMETERS"))
  _row_output(e, 1, Y)
  e.sync(SC.epilogue)
  return e.program()

# L2_NORMALIZATION: v = max(sum((x - zp)^2) * s^2, s^2); r = NLU(v), the reciprocal-sqrt spline scaled to K / sqrt(v) with
# K = f32(65535 / f32(1/s)) (so r <= 65535 since v >= s^2), written as a 16-bit integer; y = clamp((x - zp) * r * mult, -128, 127) + 128
# with mult = f32(f32(s * f32(1/K)) * 128) (the requantize formula with the NLU output scale 1/K; TFLite fixes the output to (1/128, 128))
RSQRT_BREAKS = [0x3f96545a, 0x3fb9f21d, 0x3fe5fda8, 0x401006eb, 0x40332a53, 0x405bff99, 0x7f800000]   # 1.174 .. 3.437, inf
RSQRT_C = [   # the float32 spline of 1/sqrt(m), m in [1, 4): c0..c4 of segments 0..6; segment 7 (m >= 3.437) repeats segment 6
  0x401774aa, 0xc03aa03d, 0x401b19a9, 0xbf884aed, 0x3e4375ff, 0x4009d075, 0xc00c8cff, 0x3fc141fd, 0xbf0c6a0b, 0x3da66a68,
  0x3ff7d508, 0xbfcc5845, 0x3f632952, 0xbe856fdc, 0x3cffb721, 0x3fde2545, 0xbf9325ab, 0x3f0363d0, 0xbdf7eecc, 0x3c3ec282,
  0x3fc6d5cf, 0xbf530bc5, 0x3e96feed, 0xbd64530c, 0x3b8cc91c, 0x3fb2da7d, 0xbf199e29, 0x3e31e7c5, 0xbcd9c230, 0x3ad968d1,
  0x3fa395cf, 0xbeeb2d04, 0x3de41050, 0xbc69e220, 0x3a43bfa9]
RSQRT_C += RSQRT_C[-5:]

def rsqrt_scale(s_in:float) -> float:
  """K: the NLU output r = K / sqrt(v) as a 16-bit integer, K = f32(65535 / f32(1/s_in))"""
  return f32(65535 / EL.recip32(s_in))

def rsqrt_slots(s_in:float) -> list[int]:
  """the 0x19 NLU table of L2_NORMALIZATION: c = f32(C * K) for the float32 coefficients C of 1/sqrt on [1, 4) (every coefficient of
  473 compiled input scales reproduced), then the breakpoints and mode 3"""
  K = rsqrt_scale(s_in)
  return [f32_bits(f32(bits_f32(c) * K)) for c in RSQRT_C] + RSQRT_BREAKS + [3, 0, 0]

def l2norm_quant(s_in:float) -> dict:
  s = f32(s_in)
  return dict(sq=EL.mul32(s, s), mult=EL.mul32(EL.mul32(s, EL.recip32(rsqrt_scale(s))), 128.0))

def l2norm_io(n:int) -> dict:
  return dict(in_bytes=8 * cdiv(n, 8), out_bytes=8 * cdiv(n, 8))

def _l2_layout(n:int) -> tuple[int, int, int, int]:
  """(X, Y, R, Z): FULLY_CONNECTED's narrow layout of x and y, Z = the 16-bit reciprocal sqrt (after the relay buffer, or after y)"""
  X, Y, R = FC.narrow_layout(n, n)
  return X, Y, R, R + 32 if FC.FCGeom(n, n).T_in > 1 else Y + round_up(n, 4)

def l2norm_stage(e:Emitter, n:int, X:int, Y:int, Z:int, in_q:tuple, out_q:tuple=(1/128, 128), fifo:int=ROW_FIFO, fifo2:int=ROW_FIFO2):
  """the L2_NORMALIZATION ops on tile 0 (x at narrow X, gathered): the op fence, the NLU's rsqrt table, x into the 4-row FIFO at wide
  `fifo`, op 1 (sum of squares -> NLU -> the 16-bit r at narrow Z), x again into the one-row FIFO at wide `fifo2`, op 2 (y = (x - zp) r
  requantized, at narrow Y). out_q: the output quantization; TFLite's (1/128, 128) is the compiler's program, other scales only change
  op 2's multiplier, clamps and zero point (any output scale of the same requantize formula)"""
  assert n % 4 == 0
  s, zp, w = f32(in_q[0]), in_q[1], n // 4
  rounds, q = cdiv(w, EL.mul_fifo_depth(w)), l2norm_quant(s)
  if tuple(out_q) == (1/128, 128): mult, lo, hi, ozp = q["mult"], -128.0, 127.0, 128
  else:
    lo, hi = EL.out_clamps(out_q[0], out_q[1])
    mult, ozp = EL.mul32(EL.mul32(s, EL.recip32(rsqrt_scale(s))), EL.recip32(out_q[0])), out_q[1]
  e.sync(SC.sync_op_fence)
  e.tile(EL.encode_nlu(**EL.nlu_fields(1, e.seq, rsqrt_slots(s))))
  e.tile(_feed(e.seq, X, w, fifo, sync_en1=0, sync_val=0x8000))
  e.tile(OP.encode_op(**(_reduce_op_fields(w, e.seq, X, Z, zp, dict(mult=q["sq"], clamp_min=q["sq"], clamp_max=float("inf"), out_zp=0), fifo) |
                         dict(cfg2=7, rsv1946=7))))
  e.sync(SC.sync, counters=SC.tc("PARAMETERS"), count=rounds)
  single, v = w == 1, 0x10000 + rounds          # v: the 20-bit sync value at bit 12 of the record (en1 = its low bit)
  e.tile(WN.encode_narrow_to_wide(tile_mask=1, seq=e.seq, narrow_addr=X // 4, narrow_lvl_mask=int(not single), wide_addr=fifo2,
                                  wide_rows=1, wide_circ=int(single), sync_f1=int(single), sync_wait_lvl=int(single), sync_id=1,
                                  sync_en1=v & 1, sync_val=v >> 1, sync_f45=1, tail_f799=1, tail_lvl=int(single),
                                  **WN.ttu("n", [1], [w], 6), **WN.ttu("w", [int(single)], [w], 4)))
  e.tile(OP.encode_op(tile_mask=1, seq=e.seq, loop1=w - 1, **OP.ttu_fine("in", [2, 0, 8, 8], [1, w, 1, 1]), in_base=Z, in_mode7=1, in_tflags=3,
                      **OP.ttu_fine("out", [4, 4 * w, 4 * w], [w, 1, 1]), out_base=Y, out_mode7=3, out_tflags=1,
                      **OP.ttu_fine("par", [4, 0], [1, w]), **OP.wide("par", fifo2), par_mode7=1, par_tflags=0x40, par_fifo=1,
                      **OP.ttu_fine("psum", [4, 0], [1, w]), psum_mode7=1, psum_tflags=0x840, cfg0=0x67, cfg1=0x2d | ((rounds + 1) % 4) << 8,
                      sync0=0x4000 | (rounds + 1) // 4, sync1=0x4000, cfg2=0xe, cfg3=6, w_zp=zp, dp_mode=1, out_ch=3, out_ch_last=3, out_zp=ozp,
                      mult_bits=f32_bits(mult), clamp_min_bits=f32_bits(lo), clamp_max_bits=f32_bits(hi)))
  e.sync(SC.sync, counters=SC.tc("PARAMETERS"), count=2 * rounds)

def gen_l2norm(n:int, in_q:tuple=(1/16, 128)) -> bytes:
  """L2_NORMALIZATION of a [1, 1, n] uint8 tensor (output quantization (1/128, 128)), byte-identical to edgetpu_compiler for n % 4 == 0"""
  if n % 4: raise NotImplementedError("n % 4 != 0: edgetpu_compiler adds a parameter blob (not modelled)")
  X, Y, R, Z = _l2_layout(n)
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  _row_input(e, n, X, R)
  l2norm_stage(e, n, X, Y, Z, in_q)
  _row_output(e, n, Y)
  e.sync(SC.epilogue)
  return e.program()


# *** numpy models of the arithmetic (derived from the decoded programs; offline only, see docs/isa/codegen_eltops.md 7) ***
def _rne(x): return np.rint(x)                   # round half to even (the tile ops' requantization, measured on the device)
F2I = {"rne": np.rint, "trunc": np.trunc, "rna": lambda v: np.sign(v) * np.floor(np.abs(v) + 0.5)}

def softmax_ref(xq:np.ndarray, in_q:tuple=(1/16, 128), out_q:tuple=(1/256, 0), beta:float=1.0, exp2=None, recip=None,
                f2i:str="rne") -> np.ndarray:
  """bit model of the scalar-core program, per row of xq [rows, n] (uint8) -> uint8. Every step is a float32 operation of the
  program: x = f32(q - zp) * s; m = max x; e = EXP2((x - m) * exp2_scale(beta)); s = sequential f32 sum of e; r = RECIP(s);
  y = clamp(F2I(e * r * f32(1/s_out)) [+ zp_out], 0, 255). EXP2 / RECIP / F2I are the scalar core's special-function and convert
  units, not measured: defaults are correctly rounded float32 2^t and 1/x, and round-half-to-even conversion ('trunc' / 'rna' to
  try others)."""
  F = np.float32
  exp2 = exp2 or (lambda t: (2.0 ** t.astype(np.float64)).astype(F))
  recip = recip or (lambda v: F(1) / v)
  cv = F2I[f2i] if isinstance(f2i, str) else f2i
  xq = np.atleast_2d(np.asarray(xq))
  k, so = F(exp2_scale(beta)), F(f32(1 / f32(out_q[0])))
  x = (xq.astype(np.int64) - in_q[1]).astype(F) * F(in_q[0])
  mx = x.max(axis=1, keepdims=True)
  e = exp2((x - mx) * k).astype(F)
  ssum = np.add.accumulate(e, axis=1, dtype=F)[:, -1:]
  r = recip(ssum).astype(F)
  v = (e * r) * so
  if out_q[1]: v = (cv(v) + out_q[1]).astype(F)
  return cv(np.minimum(np.maximum(np.minimum(v, F(255)), F(0)), F(255))).astype(np.uint8)

def rsqrt_nlu(v:np.ndarray, s_in:float, rnd=np.rint) -> np.ndarray:
  """model of the NLU in L2_NORMALIZATION (rsv1946 = 7): v = m * 4^e with m in [1, 4); the segment of m (breakpoints), Horner
  c0 + m(c1 + m(c2 + m(c3 + m c4))) in float32, times 2^-e, rounded to an integer and clamped to 16 bits. Hypothesis: the
  range reduction and the evaluation order are inferred from the table's domain, not measured."""
  F = np.float32
  sl = rsqrt_slots(s_in)
  C = np.array(sl[:40], np.uint32).view(F).reshape(8, 5)
  br = np.array(sl[40:47], np.uint32).view(F)
  v = np.asarray(v, F)
  e = np.floor(np.log2(v.astype(np.float64)) / 2).astype(np.int64)
  m = (v.astype(np.float64) / 4.0 ** e).astype(F)
  seg = np.searchsorted(br, m, side="right")
  c = C[seg]
  p = c[..., 4]
  for j in (3, 2, 1, 0): p = (p * m + c[..., j]).astype(F)
  r = p.astype(np.float64) * 2.0 ** (-e)
  return np.clip(rnd(r), 0, 65535).astype(np.int64)

def l2norm_ref(xq:np.ndarray, in_q:tuple=(1/16, 128), rnd=np.rint) -> np.ndarray:
  """bit model of L2_NORMALIZATION over the last axis (uint8 -> uint8, output (1/128, 128)): acc = sum (x - zp)^2 exactly in int32;
  v = max(f32(acc) * f32(s^2), f32(s^2)); r = rsqrt_nlu(v); y = clamp(rne(f32((x - zp) * r) * mult), -128, 127) + 128"""
  F = np.float32
  q = l2norm_quant(in_q[0])
  xq = np.atleast_2d(np.asarray(xq))
  d = xq.astype(np.int64) - in_q[1]
  acc = (d * d).sum(axis=1, keepdims=True)
  v = np.maximum(acc.astype(F) * F(q["sq"]), F(q["sq"]))
  r = rsqrt_nlu(v, in_q[0], rnd)
  y = np.clip((d * r).astype(F) * F(q["mult"]), F(-128), F(127))
  return (_rne(y) + 128).astype(np.uint8)

def reduce_ref(kind:str, xq:np.ndarray, in_q:tuple=(1/16, 128), out_q:tuple=(1/16, 128)) -> np.ndarray:
  """bit model of SUM / MEAN over the last axis: acc = sum (x - zp) (int32); y = rne(clamp(f32(acc) * mult)) + zp_out (MEAN: no clamp;
  how the chip stores a result outside 0..255 is not known, the model saturates)"""
  F = np.float32
  q = reduce_quant(kind, np.asarray(xq).shape[-1], in_q, out_q)
  xq = np.atleast_2d(np.asarray(xq))
  acc = (xq.astype(np.int64) - in_q[1]).sum(axis=1)
  y = np.clip(acc.astype(F) * F(q["mult"]), F(q["clamp_min"]), F(q["clamp_max"]))
  return np.clip(_rne(y) + out_q[1], 0, 255).astype(np.uint8)

if __name__ == "__main__": run_test("test_codegen_eltops")
