# tile-local DMAs, 7 words each (docs/isa/wide_narrow.md):
#   0x14 wideToNarrow  wide memory / wide ring FIFO -> narrow memory (or -> the bias "scaling" store, mode 2)
#   0x13 narrowToWide  narrow memory -> wide memory / wide ring FIFO
#                                        0x14 wideToNarrow   0x13 narrowToWide
#   source address + TTU                 wide [60, 210)      narrow [62, 276)
#   destination address + TTU            narrow [267, 481)   wide [335, 485)
#   sync block (same relative layout)    [611, 699)          [543, 631)
# TTU units: narrow 4-byte words; wide one 256-byte row per step (wide addresses are in 64-byte units).
from __future__ import annotations
import coral.isa as isa
from coral.isa import Layout, cdiv

def _ttu(p:str, lo:int, levels:int) -> list[tuple]:
  return [f for d in range(levels) for f in ((f"{p}_inc{d}", lo + 33 * d, 17, True), (f"{p}_lim{d}", lo + 33 * d + 17, 16))]
def _sync(lo:int, w2n:bool) -> list[tuple]:
  """88-bit sync block; only wideToNarrow uses the decrement fields"""
  return [("sync_en0", lo, 1), ("sync_f1", lo + 1, 1), ("sync_f2", lo + 2, 1), ("sync_id", lo + 4, 5), ("sync_wait_lvl", lo + 9, 2),
          ("sync_en1", lo + 12, 1), ("sync_val", lo + 13, 19), ("sync_en2", lo + 44, 1), ("sync_f45", lo + 45, 1)] + \
         ([("sync_dec_lvl", lo + 51, 2), ("sync_dec", lo + 54, 17, True), ("sync_dec_mode", lo + 86, 2)] if w2n else [])

# the defaults: the bits set in every compiled instance
W2N = Layout("wideToNarrow", (0x14,), 7, [
  ("tile_mask", 12, 16), ("seq", 46, 14), ("wide_addr", 60, 14), *_ttu("w", 78, 4), ("wide_lvl_mask", 210, 4), ("wide_circ", 214, 1),
  ("wide_rows", 216, 11), ("narrow_addr", 267, 16), *_ttu("n", 283, 6), ("narrow_lvl_mask", 481, 6), ("mode", 487, 2),
  ("size64", 494, 13),                                   # bytes read from the wide side / 64 (min 4)
  ("head_words_m1", 507, 16), ("head_en", 523, 1), ("head_tail16", 530, 4), ("skip_en0", 542, 1), ("skip_en1", 544, 1),
  ("skip_base", 546, 16), ("skip_m1", 562, 16), ("skip_end", 578, 16), *_sync(611, True), ("tail_f867", 867, 1), ("tail_en870", 870, 1),
  ("tail_f871", 871, 1)], n_inc0=1, sync_en0=1, sync_en1=1, sync_en2=1, tail_en870=1)
N2W = Layout("narrowToWide", (0x13,), 7, [
  ("tile_mask", 12, 16), ("seq", 46, 14), ("narrow_addr", 62, 16), *_ttu("n", 78, 6), ("narrow_lvl_mask", 276, 6), ("wide_addr", 335, 14),
  *_ttu("w", 353, 4), ("wide_lvl_mask", 485, 4), ("wide_circ", 489, 1), ("wide_rows", 491, 11), *_sync(543, False), ("rsv631", 631, 168),
  ("tail_f799", 799, 1), ("tail_lvl", 801, 2)], n_inc0=1, sync_en0=1, sync_en1=1, sync_en2=1)
encode_wide_to_narrow, decode_wide_to_narrow = W2N.encode, W2N.decode
encode_narrow_to_wide, decode_narrow_to_wide = N2W.encode, N2W.decode

def ttu(p:str, strides:list[int], counts:list[int], levels:int) -> dict[str, int]: return isa.ttu(f"{p}_", strides, counts, levels, "lim")

# *** FULLY_CONNECTED / RELU recipes (edgetpu_compiler templates). Narrow addresses in 4-byte words, wide in 64-byte units ***
def w2n_input(S:int, c:int, X:int, seq:int=0, tiles:int|None=None) -> list[int]:
  """chunk c of an S-byte input (P = 64*ceil(S/1024) bytes per chunk, on tile c unless `tiles`) from the ring-consumer-0 FIFO
  (2*ceil(S/256) rows below wide address 0x2080) to narrow X + its offset"""
  P = 64 * cdiv(S, 1024)
  B = min(P, S - c * P)
  row, o = divmod((c % 4) * P, 256)
  f = dict(seq=seq, tile_mask=1 << c if tiles is None else tiles, wide_addr=0x2080 - 8 * cdiv(S, 256) + 4 * row, narrow_addr=X + c * P // 4,
           wide_lvl_mask=1, narrow_lvl_mask=1, mode=1, size64=max(4, cdiv(B, 64)), sync_id=14, sync_wait_lvl=2, sync_val=0x8000,
           sync_dec_lvl=2, sync_dec=-1, sync_dec_mode=3, **ttu("w", [1], [cdiv(o + B, 256)], 4), **ttu("n", [1], [cdiv(B, 4)], 6))
  if o: f.update(skip_en0=1, skip_en1=1, skip_m1=o // 4 - 1, skip_base=X, skip_end=X + o // 4 - 1)
  return encode_wide_to_narrow(**f)

def w2n_ring_recv(K:int, tile_mask:int, X:int, seq:int=0, fifo:int=0x1f70) -> list[int]:
  """receive the K-byte vector broadcast over the ring (through the wide FIFO) into narrow X"""
  R = cdiv(K, 256)
  return encode_wide_to_narrow(seq=seq, tile_mask=tile_mask, wide_addr=fifo, wide_rows=1, wide_circ=int(R == 1), narrow_addr=X,
                               narrow_lvl_mask=int(K > 4), sync_id=14, sync_wait_lvl=int(R == 1), sync_val=0x8000,
                               **ttu("w", [int(R == 1)], [R], 4), **ttu("n", [1], [cdiv(K, 4)], 6))

def w2n_bias(tile_mask:int, param_base:int=0, groups:int=1, seq:int=0) -> list[int]:
  """mode-2 load of the per-64-output bias block(s) at wide param_base (the relocation field)"""
  G = groups
  return encode_wide_to_narrow(seq=seq, tile_mask=tile_mask, wide_addr=param_base, narrow_addr=0x40, narrow_lvl_mask=3, mode=2,
                               size64=2, wide_lvl_mask=3 if G > 1 else 1, sync_f2=1, tail_f867=1, sync_id=0, sync_wait_lvl=2,
                               sync_val=0xffff, sync_f45=1, **ttu("w", [1, 1] if G > 1 else [1], [1, G] if G > 1 else [1], 4),
                               **ttu("n", [1, 64, 0], [32, 1, G], 6))

def n2w_output(nbytes:int, tile_mask:int, narrow_addr:int, wide_addr:int=0x2078, seq:int=0,
               dims:list[tuple[int, int]]|None=None) -> list[int]:
  """nbytes from narrow memory to a wide ring FIFO (0x2078: the output FIFO drained by ringProducer/outfeed; 0x1f70: FC ring send
  to other tiles); dims = the narrow walk [(stride, count)] in words, innermost first (default: contiguous)"""
  R, dims = cdiv(nbytes, 256), dims or [(1, cdiv(nbytes, 4))]
  return encode_narrow_to_wide(seq=seq, tile_mask=tile_mask, narrow_addr=narrow_addr, wide_addr=wide_addr, wide_rows=1,
                               narrow_lvl_mask=sum(1 << k for k, (_, c) in enumerate(dims) if c > 1), wide_circ=int(R == 1),
                               sync_f1=int(R == 1), sync_id=16, sync_wait_lvl=int(R == 1), sync_val=0xffff, sync_f45=1, tail_lvl=len(dims),
                               **ttu("n", [s for s, _ in dims], [c for _, c in dims], 6), **ttu("w", [int(R == 1)], [R], 4))
