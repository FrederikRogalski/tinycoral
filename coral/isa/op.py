# opcode 0x01 tile op (TensorOp) and 0x02 (pooling op), 17 words, reverse engineered from edgetpu_compiler output (docs/isa/op.md):
#   [0, 62)       header: predication, opcode, tile mask, dispatch sequence number
#   [62, 174)     main loop nest loop0..loop7 (iteration count - 1, innermost first)
#   [174, 557)    TTU in_   narrow-memory read (input activations)
#   [557, 957)    TTU out_  narrow-memory write (outputs after the non-linear unit), + a partial last group
#   [957, 1260)   TTU par_  wide-memory read (parameters)
#   [1260, 1563)  TTU psum_ wide-memory read (partial sums)
#   [1563, 2176)  datapath / output stage: flags, reduction mask, channel counts, zero points, f32 multiplier / clamps / offset
# A TTU increment is byte granular (narrow) or 64-byte granular (wide): {p}_inc{k} holds increment >> 2, its 2 low bits sit in the
# previous dim's {p}_mode{k-1} slot ({p}_hmode for dim 0); {p}_mode7 is a separate flag (docs/isa/eltwise.md).
from __future__ import annotations
from coral.isa import Layout, cdiv, round_up, f32_bits, ttu

def _ttu(p:str, lo:int, base:int, sel:bool, sw:int, cw:int) -> list[tuple]:
  """address (base bits, + a 14th bit for wide memory), 2-bit header mode, 8 dims of (inc s<sw>, cnt u<cw>, mode u2)"""
  out = [(f"{p}_base", lo, base)] + [(f"{p}_sel", lo + base, 1)] * sel + [(f"{p}_hmode", lo + base + sel, 2)]
  for k in range(8):
    r = lo + base + sel + 2 + (sw + cw + 2) * k
    out += [(f"{p}_inc{k}", r, sw, True), (f"{p}_cnt{k}", r + sw, cw), (f"{p}_mode{k}", r + sw + cw, 2)]
  return out

OP = Layout("op", (0x01, 0x02), 17, [
  ("tile_mask", 12, 16), ("seq", 46, 16), *[(f"loop{k}", 62 + 14 * k, 14) for k in range(8)],
  *_ttu("in", 191, 18, False, 17, 18), ("in_tflags", 507, 8), ("in_twait", 522, 14),
  *_ttu("out", 574, 18, False, 17, 18), ("out_tflags", 890, 3),
  ("out_last_inc", 901, 17, True), ("out_last_cnt", 918, 18), ("out_last_mode", 936, 2), ("out_last_skip", 941, 16),
  *_ttu("par", 957, 13, True, 13, 14), ("par_tflags", 1205, 8), ("par_fifo", 1216, 14),   # par_base: the parameter relocation field
  *_ttu("psum", 1260, 13, True, 13, 14), ("psum_tflags", 1508, 15),
  ("cfg0", 1563, 9), ("cfg1", 1578, 10), *[(f"sync{j}", 1588 + 16 * j, 16) for j in range(5)], ("cfg2", 1843, 5), ("cfg3", 1850, 4),
  ("w_zp", 1854, 8), ("in_zp", 1870, 8), ("dp_mode", 1923, 3), ("reduce_mask", 1926, 8), ("out_ch_last", 1934, 6), ("out_ch", 1940, 6),
  ("mult_bits", 1949, 32), ("clamp_max_bits", 1981, 32), ("clamp_min_bits", 2013, 32), ("offset", 2045, 32, True), ("out_zp", 2077, 8),
  ("cfg4", 2093, 1)])
encode_op, decode_op = OP.encode, OP.decode
NO_REQUANT = dict(mult_bits=f32_bits(1.0), clamp_max_bits=f32_bits(float("inf")), clamp_min_bits=f32_bits(float("-inf")))   # copies, packs

def wide(p:str, addr:int) -> dict[str, int]: return {f"{p}_base": addr & 0x1fff, f"{p}_sel": addr >> 13}   # 14-bit wide address
def ttu_fine(p:str, strides:list[int], counts:list[int]) -> dict[str, int]:
  """TTU p from strides in address units (narrow: bytes, wide: 64-byte units) and iteration counts, innermost first"""
  f, out = ttu(f"{p}_", strides, counts, 8), {}
  for k in range(8):
    inc = f[f"{p}_inc{k}"]
    out |= {f"{p}_inc{k}": inc >> 2, f"{p}_cnt{k}": f[f"{p}_cnt{k}"], f"{p}_hmode" if k == 0 else f"{p}_mode{k-1}": inc & 3}
  return out

def fc_tile_fields(N:int, K:int, t:int, seq:int, in_base:int, out_base:int, par_base:int, w_zp:int, in_zp:int, mult:float, out_zp:int,
                   clamp_min:float|None=None, clamp_max:float|None=None) -> dict:
  """FULLY_CONNECTED: the op of tile t for y[N] = W[N,K] x[K] (uint8), bit-exact with every compiler FC op with K <= 6144.
  in_base / out_base: narrow byte addresses of x / of the tile's first output y[64*P*t] (P = passes); par_base: wide address of the
  tile's weights (64-byte units, after its 256-byte bias rows). The clamps default to [-out_zp, 255-out_zp]."""
  kw, G = cdiv(K, 4), cdiv(N, 64)                      # 4-byte input words (reduction steps), 64-output groups
  P = cdiv(G, 16)                                      # passes per tile, of this tile
  gt = min(P, G - P * t)
  assert gt >= 1, "tile has no outputs"
  assert kw > 1 or P == 1, "K<=4 with N>1024 uses a different schedule (loop1=1, psum TTU active); not modelled"
  nt = min(64 * P, N - 64 * P * t)
  ch, last = min(64, round_up(nt, 4)), round_up(nt - 64 * (gt - 1), 4)   # channels per pass, in the last pass
  w0 = ch // 4
  d = dict(tile_mask=1 << t, seq=seq, loop0=kw - 1, loop3=gt - 1, in_base=in_base, in_mode7=3, in_tflags=1, out_base=out_base, out_mode7=3,
           out_tflags=1, par_base=par_base, par_mode7=1, par_tflags=3, psum_mode7=2, psum_hmode=int(kw == 1), psum_tflags=0x8C0 if gt > 1 else 0,
           **ttu("in_", [1, kw, kw, 0], [kw, 1, 1, gt], 8),           # x, re-read once per pass
           **ttu("out_", [1, cdiv(N, 4), w0], [w0, 1, gt], 8),        # row stride ceil(N/4) words, passes contiguous
           **ttu("par_", [1, 0, kw, kw], [kw, 1, 1, gt], 8),          # kw rows of 256 B per pass
           **ttu("psum_", [0, 1, 0, 0], [kw, 1, 1, gt], 8))           # not read, but the nest mirrors the main loop
  if last != ch: d.update(out_last_cnt=last // 4 - 1, out_last_mode=2, out_last_skip=w0 - last // 4)
  return d | dict(cfg0=0xA5, cfg1=0x16C, sync0=0x4000, sync1=0x4000, cfg2=6 if kw == 1 else 7, cfg4=1, dp_mode=3, reduce_mask=0b101,
                  out_ch=ch - 1, out_ch_last=last - 1, w_zp=w_zp, in_zp=in_zp, out_zp=out_zp, mult_bits=f32_bits(mult),
                  clamp_max_bits=f32_bits(255 - out_zp if clamp_max is None else clamp_max),
                  clamp_min_bits=f32_bits(-out_zp if clamp_min is None else clamp_min))
