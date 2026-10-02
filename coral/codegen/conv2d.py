# general CONV_2D (k x k kernels, stride 1 / 2, VALID / SAME) on the 4x4 tiles, byte-identical to edgetpu_compiler
# (docs/isa/codegen_conv2d.md). `python -m coral.codegen.conv2d` runs the acceptance test (test/test_codegen_conv2d.py).
# Every tile holds a block of the image (its rows / columns of eltwise.tile_split) and computes a block of the output.
# Execution program:
#   prologue, the identity row (+ zero row) -> wide memory
#   first copy ops, input DMA, input wideToNarrows, (ringConsumer, infeed) per tile row, last copy ops: each tile's own pixels
#   into its block, positions padded to words
#   halo exchange over the mesh: north, south (own columns), west, east (window rows); fills at the image edges (SAME)
#   ringConsumer1 (weights), bias wideToNarrow, the conv op per output block shape, ringProducer(s) broadcasting the cached weights
#   narrowToWide output per block shape, (outfeed, ringProducer) per tile or the scalar-memory path, PRODUCER_A/B waits, epilogue
from __future__ import annotations
from dataclasses import dataclass
from math import gcd
from functools import cached_property
from coral.isa import cdiv, round_up, ttu, f32_bits, op as OP, wide_narrow as WN, ring_mesh as RM, scalar as SC, eltwise as EL
from coral.codegen import Emitter, Piece, caching_program, piece_parts, run_test, WIDE_TOP, WIDE_OUT_FIFO
from coral.codegen.fc import rprod_output
from coral.codegen import conv as CC

NARROW_BYTES, MAX_WORDS = 192 * 1024, 16384     # narrow memory per tile; instruction words per bitstream

# ***** geometry *****
def fifo_rows(a:int) -> bool:
  """the weights of a fill of a rows stream through an a-slot ring FIFO (8 units per slot); 16..30 rows sit in a plain block"""
  return a <= 15 or a >= 31

@dataclass(frozen=True)
class Span:
  """one tile row (column): its own input rows [i0, i0+ni), its outputs [o0, o0+no), the input window [w0, w0+nw) they need
  (may reach outside the image: padding), and [lo, hi): the image rows it holds once the halos have arrived (its window inside the
  image, its own rows, and the rows it relays between its neighbours). The tile stores the block [b0, b0+nb) = window U [lo, hi)."""
  i0: int
  ni: int
  o0: int
  no: int
  w0: int
  nw: int
  lo: int = 0
  hi: int = 0
  @property
  def b0(self) -> int: return min(self.w0, self.lo) if self.no else self.lo
  @property
  def nb(self) -> int: return (max(self.w0 + self.nw, self.hi) if self.no else self.hi) - self.b0
  @property
  def d(self) -> int: return self.i0 - self.b0          # own rows start this far into the block
  @property
  def wo(self) -> int: return self.w0 - self.b0         # the window starts this far into the block

def axis(D:int, OD:int, k:int, s:int, pad:int) -> list[Span]:
  """the 4 tile spans of one image axis: inputs and outputs are both split with eltwise.tile_split. Halo rows a span needs from below
  (above) its own come from the next (previous) span, which relays them from the one after if it holds fewer: the chains hi / lo."""
  ins, outs, sp, i0, o0 = EL.tile_split(D), EL.tile_split(OD), [], 0, 0
  for p in range(4):
    sp.append(Span(i0, ins[p], o0, outs[p], o0 * s - pad, (outs[p] - 1) * s + k if outs[p] else 0))
    i0 += ins[p]
    o0 += outs[p]
  hi, lo = [0] * 4, [0] * 4
  for p in range(4):
    x = sp[p]
    hi[p] = max(x.i0 + x.ni, min(x.w0 + x.nw, D) if x.no else 0, hi[p - 1] if p else 0)
  for p in reversed(range(4)):
    x = sp[p]
    lo[p] = min(x.i0, max(x.w0, 0) if x.no else D, lo[p + 1] if p < 3 else D)
  return [Span(x.i0, x.ni, x.o0, x.no, x.w0, x.nw, lo[p], hi[p]) for p, x in enumerate(sp)]

@dataclass(frozen=True)
class Conv2DGeom:
  H: int
  W: int
  Cin: int
  Cout: int
  kh: int
  kw: int
  stride: int = 1
  padding: str = "VALID"
  @property
  def OH(self) -> int: return (self.H - self.kh) // self.stride + 1 if self.padding == "VALID" else cdiv(self.H, self.stride)
  @property
  def OW(self) -> int: return (self.W - self.kw) // self.stride + 1 if self.padding == "VALID" else cdiv(self.W, self.stride)
  @property
  def pad_top(self) -> int: return 0 if self.padding == "VALID" else max((self.OH - 1) * self.stride + self.kh - self.H, 0) // 2
  @property
  def pad_left(self) -> int: return 0 if self.padding == "VALID" else max((self.OW - 1) * self.stride + self.kw - self.W, 0) // 2
  @cached_property
  def rows(self) -> list[Span]: return axis(self.H, self.OH, self.kh, self.stride, self.pad_top)
  @cached_property
  def cols(self) -> list[Span]: return axis(self.W, self.OW, self.kw, self.stride, self.pad_left)
  @property
  def C4(self) -> int: return round_up(self.Cin, 4)     # bytes per window position (channels padded to a word)
  @property
  def cw(self) -> int: return self.C4 // 4              # words per window position
  @property
  def taps(self) -> int: return self.kh * self.kw
  @property
  def N4(self) -> int: return round_up(self.Cout, 4)
  @property
  def cg(self) -> int: return round_up(self.Cout, 16) if self.Cout <= 64 else 64   # outputs per parameter group
  @property
  def G(self) -> int: return cdiv(self.Cout, 64)        # parameter groups
  @property
  def R(self) -> int: return self.W * self.Cin          # bytes of one image row in the input stream
  @property
  def S(self) -> int: return 8 * cdiv(self.H * self.R, 8)   # input DMA bytes
  @property
  def odd(self) -> bool: return self.Cin % 4 != 0
  @property
  def c_in(self) -> int: return cdiv(self.R, 256)       # input ring FIFO slots
  def m(self, r:int) -> int:
    """image rows per input chunk of row group r: a whole number of words (4 / gcd(R, 4) rows), at most the group's rows"""
    return min(4 // gcd(self.R, 4), self.rows[r].ni)
  def L(self, r:int) -> int: return cdiv(self.m(r) * self.R, 4)   # words per input chunk
  @property
  def Lf(self) -> int: return 4 // gcd(self.R, 4) * self.R // 4   # words of a full chunk
  @property
  def slot(self) -> int: return max(4 * self.Lf, 256)   # narrow staging slot (two of them): one full chunk
  def chunks(self, r:int) -> tuple[int, int]:
    """(chunks, rows of the last chunk) of tile row r's own image rows"""
    n = cdiv(self.rows[r].ni, self.m(r))
    return n, self.rows[r].ni - self.m(r) * (n - 1)
  @property
  def Cs(self) -> int: return 16 + 256 * self.odd       # identity row (+ the zero row that pads the channels)
  def win_bytes(self, r:int, c:int) -> int: return self.rows[r].nb * self.cols[c].nb * self.C4
  def out_bytes(self, r:int, c:int) -> int: return self.rows[r].no * self.cols[c].no * self.N4
  @property
  def Xs(self) -> int: return max(self.win_bytes(t // 4, t % 4) for t in range(16))
  @property
  def Ys(self) -> int: return max(self.out_bytes(t // 4, t % 4) for t in range(16))
  @property
  def kwords(self) -> int: return self.taps * self.cw   # reduction steps (words) per output
  @property
  def ppt_max(self) -> int: return max(self.rows[r].no for r in range(4)) * max(self.cols[c].no for c in range(4))
  @property
  def kc_a(self) -> int:
    """weight rows per fill of the K-chunked op: one channel word of every tap; a 1x1 kernel takes codegen/conv.py's a, the largest
    divisor of the channel words <= min(15, cw/2)"""
    if self.taps > 1: return self.taps
    return max([d for d in range(1, min(15, self.cw // 2) + 1) if self.cw % d == 0] or [1])
  @property
  def pos_outer(self) -> bool:
    """position-outer (all K per weight fill, no partial sums) or K-chunked (partial sums per position in wide memory).
    edgetpu_compiler: K <= 30 rows: position-outer iff K <= positions + taps (codegen/conv.py's kw <= ppt + 1); K >= 31 rows (a FIFO
    of K slots): the mode with the smaller wide-memory footprint"""
    if self.cw == 1: return True
    kw, a = self.kwords, self.kc_a
    if kw <= 30: return kw <= self.ppt_max + self.taps
    return 8 * kw + 8 < 4 * self.ppt_max + (8 * a + 8 if fifo_rows(a) else 4 * a + 4)
  @property
  def a(self) -> int: return self.kwords if self.pos_outer else self.kc_a   # weight rows per fill
  @property
  def aw(self) -> int: return self.a // self.taps if not self.pos_outer else self.cw   # channel words per tap and fill
  @property
  def b(self) -> int: return self.kwords // self.a                          # fills per group
  @property
  def fifo(self) -> bool: return fifo_rows(self.a)
  @property
  def blocks(self) -> int: return self.G * (1 + self.kwords)
  @property
  def param_bytes(self) -> int: return self.G * (4 * self.cg + self.cg * 4 * self.kwords)
  def out_tile(self, t:int) -> int: return self.out_bytes(t // 4, t % 4)

# ***** memory layout *****
def narrow_layout(g:Conv2DGeom) -> dict:
  """byte addresses (every tile) of X = the conv window, Stg = the two staging slots of the ring input, Y = the outputs (Y reuses the
  staging: they are never live at the same time), C = the identity (+ zero) row. The candidate placements all need the same memory;
  edgetpu_compiler's choice, fitted on ~1500 compiles:
    Y bigger than the window and than 2 slots + C:   [Y | X], Stg at 0, C at 2 slots
    window at least half a slot:                     [X | Stg | C], Y over Stg
    smaller window, Y bigger than 2 slots or the window smaller than C/4:  [Stg | C | X] (X above Y), Y at 0
    otherwise                                        [Stg | X | C], Y at 0"""
  Xs, Ys, R2, Cs = g.Xs, g.Ys, 2 * g.slot, g.Cs
  if Ys >= R2 + Cs and Ys > Xs: return dict(Y=0, Stg=0, C=R2, X=Ys)
  if 2 * Xs >= g.slot: return dict(X=0, Stg=Xs, Y=Xs, C=Xs + R2)
  if Ys > R2 or 4 * Xs < Cs: return dict(Y=0, Stg=0, C=R2, X=max(Ys, R2 + Cs))
  return dict(Y=0, Stg=0, X=R2, C=R2 + Xs)

def regions(g:Conv2DGeom, top:int) -> dict:
  """the conv's compute buffers (64-byte units) stacked down from top: partial sums (4 per position, K-chunked only), bias (8, or 4
  without a FIFO), the weight FIFO (8 per slot) or the a-row weight block (4 per row), largest first, ties psum, bias, FIFO"""
  psum = 0 if g.pos_outer else 4 * g.ppt_max
  regs = [("psum", psum), ("bias", 8 if g.fifo else 4), ("par_fifo", 8 * g.a if g.fifo else 4 * g.a)]
  out = dict(psum=None)
  for name, size in sorted(regs, key=lambda r: -r[1]):
    if size:
      top -= size
      out[name] = top
  return out | dict(bottom=top)

def wide_layout(g:Conv2DGeom) -> dict:
  """64-byte units: the input phase's ring FIFO (8 per slot) and identity row (4 per 256 bytes) stacked down from WIDE_TOP, larger
  first (the identity row on a tie); the compute phase's regions also from WIDE_TOP"""
  fifo, cs = 8 * g.c_in, 4 * cdiv(g.Cs, 256)
  if fifo > cs:
    in_fifo = WIDE_TOP - fifo
    const = in_fifo - cs
  else:
    const = WIDE_TOP - cs
    in_fifo = const - fifo
  r = regions(g, WIDE_TOP)
  return r | dict(in_fifo=in_fifo, const=const, bottom=min(const, in_fifo, r["bottom"]))

def param_limit(g:Conv2DGeom) -> int: return wide_layout(g)["bottom"]

# ***** helpers *****
def _group(per_tile:list[dict|None]) -> list[tuple[int, dict]]:
  """tiles whose instructions are identical share one instruction (OR-ed tile mask), in the order of their lowest tile"""
  groups: dict[tuple, list] = {}
  for t, f in enumerate(per_tile):
    if f is None: continue
    key = tuple(sorted(f.items()))
    if key in groups: groups[key][0] |= 1 << t
    else: groups[key] = [1 << t, f]
  return [(m, f) for m, f in groups.values()]

def _dims(dims:list[tuple[int, int]]) -> list[tuple[int, int]]:
  """drop the (stride, count) levels of count 1"""
  return [(s, c) for s, c in dims if c > 1]

# ***** input path *****
# The ring delivers each row group's image rows as one byte stream; every tile's wideToNarrow cuts its part of it into chunks of
# m image rows (L words, m = 4 / gcd(R, 4)) that alternate between two staging slots; copy ops move the tile's own positions of each
# chunk into its window (words of C4 bytes). The first copy op handles all chunks but the last before the input DMA starts, the last
# one the last chunk.
def _stream_off(g:Conv2DGeom, r:int, c:int) -> int:
  """byte offset of tile (r, c)'s first position in its row group's ring stream (which starts at an 8-byte boundary)"""
  return g.rows[r].i0 * g.R % 8 + g.cols[c].i0 * g.Cin

def _copy_common(g:Conv2DGeom, r:int, c:int, Stg:int, X:int, const:int, last:bool) -> tuple:
  """(chunks, rows per chunk, in_base, out_base, sync fields) of a copy op. Progress: the words of the tile's wideToNarrow stream
  that must have arrived, as sync0 = (L % 4) << 14 | words // 4 and cfg1 bits 8-9 = words % 4"""
  rs, cs = g.rows[r], g.cols[c]
  n, rl = g.chunks(r)
  ph = _stream_off(g, r, c) % 4
  m, L = g.m(r), g.L(r)
  nch, mrows = (1, rl) if last else (n - 1, m)
  wrow = rs.d + (m * (n - 1) if last else 0)
  in_base = (Stg + ((n - 1) % 2) * g.slot if last else Stg) + ph
  out_base = X + (wrow * cs.nb + cs.d) * g.C4
  words = (n - 1) * L + cdiv((rl - 1) * g.R + ph + cs.ni * g.Cin, 4) if last else L
  sync = dict(sync0=((g.Lf % 4) << 14) | (words // 4), sync1=0x4000 + g.Lf // 4, rsv515=2 * g.slot & 127, in_twait=2 * g.slot >> 7)
  return nch, mrows, in_base, out_base, words % 4, sync

def _copy_op(g:Conv2DGeom, r:int, c:int, Stg:int, X:int, const:int, last:bool) -> dict:
  """Cin % 4 == 0 (one image row per chunk): the tile's positions, word by word, through the identity row"""
  cs, cw = g.cols[c], g.cw
  nch, _, in_base, out_base, x, sync = _copy_common(g, r, c, Stg, X, const, last)
  return dict(loop1=cw - 1, loop2=cs.ni - 1, loop3=nch - 1, in_base=in_base, in_mode7=3, in_tflags=0x80,
              **ttu("in_", [1, cw, 0], [cw, cs.ni, nch], 8), out_base=out_base, out_mode7=3, out_tflags=3,
              **ttu("out_", [1, 1, cw, cs.nb * cw], [1, cw, cs.ni, nch], 8), **sync, **OP.wide("par", const),
              **ttu("par_", [], [cw, 1, cs.ni, nch], 8), **ttu("psum_", [], [cw, 1, cs.ni, nch], 8), par_hmode=int(cw == 1),
              psum_hmode=int(cw == 1), psum_tflags=0x800, cfg0=7,
              cfg1=0x6B | x << 8, cfg2=6, dp_mode=4, out_ch=3, out_ch_last=3, **OP.NO_REQUANT)

def _narrow_copy_op(g:Conv2DGeom, r:int, c:int, Stg:int, X:int, const:int, last:bool) -> dict:
  """Cin < 4 (dp_mode 3): per position Cin bytes (from byte phase ph of the staging slot) into one window word, the missing channel
  bytes through the identity row"""
  rs, cs, C = g.rows[r], g.cols[c], g.Cin
  nch, mr, in_base, out_base, x, sync = _copy_common(g, r, c, Stg, X, const, last)
  ww = cs.nb
  ostr, ocnt = [1, 1, ww] + [g.m(r) * ww] * (nch > 1) + [rs.nb * ww], [1, cs.ni, mr] + [nch] * (nch > 1) + [1]
  return dict(loop0=C - 1, loop1=cs.ni - 1, loop2=mr - 1, loop3=nch - 1, in_base=in_base,
              **OP.ttu_fine("in", [1, C, g.R, 0], [C, cs.ni, mr, nch]), in_mode7=3, in_tflags=0xC1, out_base=out_base,
              **ttu("out_", ostr, ocnt, 8), out_mode7=3, out_tflags=7 if nch > 1 else 3, **sync, **OP.wide("par", const),
              **OP.ttu_fine("par", [1], [C, cs.ni, mr * nch]), par_mode7=1, **OP.ttu_fine("psum", [int(C == 1)], [C, cs.ni, mr * nch]),
              psum_tflags=0x840, cfg0=7, cfg1=0x6B | x << 8, cfg2=6, cfg3=7, dp_mode=3, reduce_mask=1, out_ch=3, out_ch_last=3,
              **OP.NO_REQUANT)

def _shift_copy_op(g:Conv2DGeom, r:int, c:int, Stg:int, X:int, const:int, last:bool) -> dict:
  """Cin > 4, Cin % 4 != 0 (cfg0 9): byte-granular, Cin bytes per position from byte phase ph, written word aligned (C4 bytes; the
  padding bytes come from the zero row behind the identity row)"""
  rs, cs, C, cw = g.rows[r], g.cols[c], g.Cin, g.cw
  nch, mr, in_base, out_base, x, sync = _copy_common(g, r, c, Stg, X, const, last)
  rw = cs.nb * cw
  rr, t = (C - 1) % 4, -C % 4                       # (Cin-1) % 4 and the padding bytes
  ostr, ocnt = [1, 1, cw, rw] + [g.m(r) * rw] * (nch > 1) + [rs.nb * rw], [1, cw, cs.ni, mr] + [nch] * (nch > 1) + [1]
  return dict(loop0=3, loop1=cw - 1, loop2=cs.ni - 1, loop3=mr - 1, loop4=nch - 1, rsv174=0x4000 | rr, in_base=in_base,
              **OP.ttu_fine("in", [1, C, g.R, 0], [C, cs.ni, mr, nch]), in_mode7=3, in_tflags=0xC1, out_base=out_base,
              **ttu("out_", ostr, ocnt, 8), out_mode7=3, out_tflags=7, rsv893=int(nch > 1), **sync, **OP.wide("par", const),
              **OP.ttu_fine("par", [1], [4, cw, cs.ni, mr * nch]), par_mode7=1, **ttu("psum_", [], [4, cw, cs.ni, mr * nch], 8),
              psum_tflags=0x840, par_fifo=(rr & 1) << 13, rsv1230=(rr >> 1) | (1 << 13) | (t << 16), rsv1523=(rr << 9) | (1 << 23),
              cfg0=9, cfg1=0x8B | x << 8, cfg2=6, cfg3=7, dp_mode=4, reduce_mask=1, out_ch=3, out_ch_last=3, **OP.NO_REQUANT)

def copy_ops(e:Emitter, g:Conv2DGeom, Stg:int, X:int, const:int, last:bool):
  fn = _copy_op if not g.odd else _narrow_copy_op if g.Cin < 4 else _shift_copy_op
  per = [None if (not last and g.chunks(t // 4)[0] < 2) else fn(g, t // 4, t % 4, Stg, X, const, last) for t in range(16)]
  for m, f in _group(per): e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))

def _row_stream(g:Conv2DGeom, r:int) -> tuple[int, int, int]:
  """(first byte, bytes, word shift) of row group r's input stream: its image rows; the infeed starts at an 8-byte boundary"""
  lo, n = g.rows[r].i0 * g.R, g.rows[r].ni * g.R
  return lo, n, (lo % 8) // 4

def _ring_passes(g:Conv2DGeom, r:int) -> tuple[int, int, int, int]:
  """(FIFO passes, packets, grp, gstride) of row group r's stream through the c_in-slot ring FIFO; a short last pass skips the
  missing packets (grp = its packets - 1, gstride = 4 * missing + 1)"""
  lo, n, _ = _row_stream(g, r)
  P = cdiv(8 * (cdiv(lo + n, 8) - lo // 8), 256)    # packets of the row group's infeed (it starts at an 8-byte boundary)
  refills = cdiv(P, g.c_in)
  p_last = P - g.c_in * (refills - 1)
  grp, gstride = (p_last - 1, 4 * (g.c_in - p_last) + 1) if p_last < g.c_in else (0, 0)
  return refills, P, grp, gstride

def _w2n_input(g:Conv2DGeom, r:int, c:int, Stg:int, in_fifo:int) -> dict:
  refills, P, grp, gstride = _ring_passes(g, r)
  sk = _stream_off(g, r, c) // 4
  L, rem, n = g.L(r), 64 * P - sk, g.chunks(r)[0]
  if rem <= L: L, nch, head = rem, 1, None
  elif cdiv(rem, L) > n + 1:                         # at most chunks + 1: the last one drains the rest of the packets
    nch = n + 1
    head = rem - (nch - 1) * L
    if head % L == 0: head -= L
  else:
    nch = cdiv(rem, L)
    head = rem - (nch - 1) * L
    if head == L: head = None
  f = dict(wide_addr=in_fifo, wide_lvl_mask=1, wide_circ=1, wide_rows=g.c_in, rsv227=(grp << 6) | (gstride << 22),
           narrow_addr=Stg // 4, narrow_lvl_mask=1, mode=1, sync_id=14, sync_wait_lvl=1, sync_val=0x8000, sync_dec_lvl=1, sync_dec=-1,
           sync_dec_mode=3, tail_f871=1, rsv489=max(4 * g.L(r), 256) // 2 & 31, size64=max(4 * g.L(r), 256) // 2 >> 5,
           **WN.ttu("w", [1, 0], [g.c_in, refills], 4), **WN.ttu("n", [1, 0], [L, nch], 6))
  if head is not None:
    t = (4 * (L - head)) & 0x3ffff
    f.update(head_words_m1=head - 1, head_en=1, rsv524=t & 63, head_tail16=(t >> 6) & 15, rsv534=t >> 10)
  if sk: f.update(skip_en0=1, skip_en1=1, skip_base=Stg // 4, skip_m1=sk - 1, skip_end=Stg // 4 + sk - 1)
  return f

def _rc_input(g:Conv2DGeom, r:int, seq:int, in_fifo:int) -> list[int]:
  refills, P, grp, gstride = _ring_passes(g, r)
  if g.c_in > 1: f = dict(sdims=1, cbuf=1, mode=3, s_id=(1 << 5) | 11, **ttu("", [1, 0], [g.c_in, refills], 4))
  elif refills == 1: f = dict(cbuf=1, mode=3, s_id=(1 << 5) | 11, **ttu("", [1], [1], 4))
  else: f = dict(mode=1, s_id=11, **ttu("", [0], [refills], 4))
  return RM.encode_ringConsumer(tile_mask=0xf << (4 * r), seq=seq, addr=in_fifo, slots=g.c_in, grp=grp, gstride=gstride,
                                s_val=-64 * g.c_in, s_cnt=g.c_in, s_en_b=1, s_y=1, **f)

def _av_infeed(offset:int, nbytes:int, tiles:int) -> list[int]:
  """scalar.av_infeed with the staging-buffer offset wrapped at 2^15 units (256 KiB) also for single-pass transfers, and a last pass
  that fills the whole 16384-unit buffer encoded as count_m1 = k203 = 0 (as the parameter infeeds do, codegen.md 4.5)"""
  f = SC.INFEED.decode(SC.av_infeed(offset, nbytes, tiles)) | dict(buf_off=offset // 8 % (1 << 15))
  if nbytes // 8 > 1 and nbytes // 8 % SC.AV_BUF == 0: f |= dict(count_m1=0, k203=0)
  return SC.INFEED.encode(**f)

def input_stage(e:Emitter, g:Conv2DGeom, nl:dict, wl:dict):
  Stg, X, C = nl["Stg"], nl["X"], nl["C"]
  e.sync(SC.input_head)
  e.tile(*EL.ident_prologue(e.seq, 0xffff, C, wl["const"], 64 * g.odd))
  e.sync(SC.sync_wn_fence)
  copy_ops(e, g, Stg, X, wl["const"], False)
  e.scalar(SC.input_dma(g.S))
  # row groups that refill their ring FIFO 3+ times get a WIDE_TO_NARROW fence; their wideToNarrows come after the others'
  fenced = [_ring_passes(g, r)[0] >= 3 for r in range(4)]
  if any(fenced): e.tile(SC.sync(e.seq, tiles=sum(0xf << (4 * r) for r in range(4) if fenced[r]), counters=SC.tc("WIDE_TO_NARROW"), units=0))
  for part in (False, True):
    per = [_w2n_input(g, t // 4, t % 4, Stg, wl["in_fifo"]) if fenced[t // 4] == part else None for t in range(16)]
    for m, f in _group(per): e.tile(WN.encode_wide_to_narrow(**(f | dict(tile_mask=m, seq=e.seq))))
  for r in range(4):
    e.tile(_rc_input(g, r, e.seq, wl["in_fifo"]))
    lo, n, _ = _row_stream(g, r)
    a, b = lo // 8, cdiv(lo + n, 8)
    e.scalar(_av_infeed(8 * a, 8 * (b - a), 0xf << (4 * r)))
  e.sync(SC.sync_drain)
  copy_ops(e, g, Stg, X, wl["const"], True)

# ***** halo exchange between neighbouring tiles *****
MESH_N, MESH_S, MESH_W, MESH_E = 0x17, 0x15, 0x16, 0x18

def _pad_after(sp:list[Span], p:int, D:int) -> int:
  s = sp[p]
  return max(0, s.w0 + s.nw - max(D, s.i0 + s.ni)) if s.no else 0
def _pad_before(sp:list[Span], p:int) -> int:
  s = sp[p]
  return max(0, min(0, s.i0) - s.w0) if s.no else 0

def _half(p:str, addr:int, dims:list[tuple[int, int]]) -> dict:
  """one half (o_ / i_) of a mesh move: [(1, words of a position), (positions), (rows)] with count-1 levels dropped. A mesh level moves
  at most 16 words: wider positions go in 16-word pieces over an outermost level (16, pieces); a short last piece of r words sets
  grp = r - 1 and the field at bit 249 (+205 inbound) to 4 * (16 - r) + the index of that level. in_mode = 2 * levels + 1, the
  pieces level not counted."""
  (_, cw), rest = dims[0], dims[1:]
  pieces = cdiv(cw, 16)
  lv = _dims([(1, min(cw, 16))] + rest)
  if pieces > 1: lv.append((16, pieces))
  f = {f"{p}addr": addr, f"{p}sdims": (1 << len(lv)) - 1, **ttu(p, [s for s, _ in lv], [c for _, c in lv], 4)}
  if pieces > 1 and cw % 16:
    f[f"{p}grp"] = cw % 16 - 1
    f["rsv243" if p == "o_" else "rsv448"] = (4 * (16 - cw % 16) + len(lv) - 1) << 6
  if p == "i_": f["in_mode"] = 2 * (len(lv) - (pieces > 1)) + 1
  return f

REC_IN = {MESH_N: SC.TILE_COUNTERS.index("MESH_SOUTH_IN") << 1, MESH_S: SC.TILE_COUNTERS.index("MESH_NORTH_IN") << 1,
          MESH_W: SC.TILE_COUNTERS.index("MESH_EAST_IN") << 1, MESH_E: SC.TILE_COUNTERS.index("MESH_WEST_IN") << 1}

def _moves(sp:list[Span], p:int, D:int, toward_prev:bool) -> tuple:
  """the moves of span p in one direction: (outbound count, its first block row, relay?), (inbound count, its first block row, fill?)
  toward_prev: data moves towards span p-1 (north / west), else towards p+1 (south / east). Missing neighbours (image edges) become
  fills of the padding."""
  s = sp[p]
  if toward_prev:     # p sends p-1 the rows [i0, hi(p-1)) and receives [i0+ni, hi(p)) from p+1
    n_out = max(0, sp[p - 1].hi - s.i0) if p > 0 else 0
    out = (n_out, s.d, n_out > s.ni)
    n_in = s.hi - (s.i0 + s.ni)
    if n_in and _pad_after(sp, p, D): raise NotImplementedError("a window needs both a neighbour's rows and padding on one side")
    inn = (n_in, s.i0 + s.ni - s.b0, False) if n_in else (_pad_after(sp, p, D), max(D, s.i0 + s.ni) - s.b0, True)
  else:               # p sends p+1 the rows [lo(p+1), i0+ni) and receives [lo(p), i0) from p-1
    n_out = max(0, s.i0 + s.ni - sp[p + 1].lo) if p < 3 else 0
    out = (n_out, s.d + s.ni - n_out, n_out > s.ni)
    n_in = s.i0 - s.lo
    if n_in and _pad_before(sp, p): raise NotImplementedError("a window needs both a neighbour's rows and padding on one side")
    inn = (n_in, s.lo - s.b0, False) if n_in else (_pad_before(sp, p), s.wo, True)
  return out, inn

def halo_stage(e:Emitter, g:Conv2DGeom, X:int, fill:int):
  """the window halos: first vertically (the tile's own columns: meshBus north, then south), then horizontally (whole window rows:
  west, then east). A tile's instruction holds its outbound half (rows its neighbour needs, possibly including rows it receives in
  the same phase: a relay, which waits on that IN counter) and its inbound half (rows from the opposite neighbour, or the padding
  filled with the input zero point at the image edges). Horizontal senders also wait on the vertical IN counters of their tile."""
  cw, C4 = g.cw, g.C4
  got = [[] for _ in range(16)]                      # vertical IN counters per tile (received or filled)
  def addr(r, c, wrow, wcol): return X + (wrow * g.cols[c].nb + wcol) * C4
  for op in (MESH_N, MESH_S, MESH_W, MESH_E):
    vertical, per = op in (MESH_N, MESH_S), []
    for t in range(16):
      r, c = divmod(t, 4)
      rs, cs = g.rows[r], g.cols[c]
      if vertical:
        (no, ro, relay), (ni, ri, fl) = _moves(g.rows, r, g.H, op == MESH_N)
        def dims(n): return [(1, cw), (cw, cs.ni), (cs.nb * cw, n)]
        o_at, i_at = (lambda: addr(r, c, ro, cs.d)), (lambda: addr(r, c, ri, cs.d))
      elif not rs.no:                                   # a tile row without outputs needs (and forwards) no columns
        per.append(None)
        continue
      else:
        (no, ro, relay), (ni, ri, fl) = _moves(g.cols, c, g.W, op == MESH_W)
        def dims(n): return [(1, cw), (cw, n), (cs.nb * cw, rs.nw)]       # the window rows
        o_at, i_at = (lambda: addr(r, c, rs.wo, ro)), (lambda: addr(r, c, rs.wo, ri))
      f = {}
      if no: f |= _half("o_", o_at(), dims(no))
      if ni:
        f |= _half("i_", i_at(), dims(ni)) | (dict(fill_en=1, fill=fill) if fl else {})
        if vertical: got[t].append(REC_IN[op])
      if no:
        recs = [REC_IN[op]] * relay + (sorted(got[t]) if not vertical else [])
        val = 4 + bin(f["o_sdims"]).count("1") - (cw > 16)      # 4 + the outbound levels (the 16-word pieces level not counted)
        for k, rid in enumerate(recs): f |= {f"s{k}_id": rid, f"s{k}_val": val, f"s{k}_en_a": 1, f"s{k}_en_b": 1}
      per.append(f or None)
    for m, f in _group(per): e.tile(RM.encode_mesh(opcode=op, tile_mask=m, seq=e.seq, **f))

# ***** compute *****
def _split_dims(subdims:list[tuple[int, int]]) -> list[tuple[int, int]]:
  """the TTU levels of one loop: contiguous levels merged, trailing count-1 levels dropped (one level is always kept)"""
  out = [subdims[0]]
  for s, c in subdims[1:]:
    ps, pc = out[-1]
    if s == ps * pc: out[-1] = (ps, pc * c)
    else: out.append((s, c))
  while len(out) > 1 and out[-1][1] == 1: out.pop()
  return out

def rc_param(g:Conv2DGeom, seq:int, wl:dict, tiles:int=0xffff) -> list[int]:
  """ringConsumer1 on the computing tiles: receives the broadcast parameter rows (per group: the bias row, then the weight rows) into the
  weight FIFO / block (a rows per fill, b fills per group); the bias rows go through the aux pair. A single-row fill uses the
  single-slot FIFO form."""
  bias, a, b, G = wl["bias"], g.a, g.b, g.G
  aux0 = bias + (1 if g.pos_outer else 2) - (a == 1)
  if a == 1:
    dims = ttu("", [1, 0], [1, G], 4) if g.pos_outer else ttu("", [0, 0, 0], [g.kwords, 1, G], 4)
    f = dict(slots=1, aux_addr0=aux0, aux_addr1=bias + 4, s_val=-1, mode=1, s_id=1, **dims)
  else:
    dims = ttu("", [1, 0, 0], [a, 1, G], 4) if g.pos_outer else ttu("", [1, 0, 0, 0], [a, b, 1, G], 4)
    f = dict(aux_addr0=aux0, aux_addr1=bias + 4 if g.fifo else bias, sdims=1, cbuf=1, mode=3, s_id=(1 << 5) | 1, **dims)
    if g.fifo: f |= dict(slots=a, s_val=-1)
  return RM.encode_ringConsumer(opcode=0x12, tile_mask=tiles, seq=seq, addr=wl["par_fifo"], aux_en0=1, aux_en1=1, s_en_a=1, s_en_b=1,
                                s_y=1, **f)

def _conv_fields(g:Conv2DGeom, r:int, c:int, X:int, Y:int, wl:dict, q:dict, out:tuple|None=None) -> dict:
  """the conv op of tile (r, c): main loops (words per tap and fill, taps, positions, fills, groups) or, position-outer, (channel words,
  taps, 1, positions, groups); the in TTU walks the window in the block at X (levels of one loop merged when contiguous, trailing
  count-1 levels dropped), the par TTU the weight FIFO / block, psum the partial sums (K-chunked); outputs are the tile's dense block
  [positions][N4] at Y in groups of min(64, N4) channels. out = (base, [(stride words, count)] position levels, wide): the outputs go
  into another layout instead (coral/codegen/chain.py: the next layer's block); wide = rows inside wider rows (cfg0 0xE7)"""
  rs, cs, cw, s = g.rows[r], g.cols[c], g.cw, g.stride
  rw = cs.nb * cw                                     # block row stride (words)
  taps = _split_dims([(cw, g.kw), (rw, g.kh)])
  pos = _split_dims([(s * cw, cs.no), (s * rw, rs.no)])
  ppt, G, aw, b = rs.no * cs.no, g.G, g.aw, g.b
  if g.pos_outer:   # all K per fill: channel words innermost, then the taps (the blob's [tap][channel word] order), positions outside
    ind = [(1, cw)] + taps + [(cw, 1)] + pos + [(0, G)]
    loops = dict(loop0=cw - 1, loop1=g.taps - 1, loop3=ppt - 1, loop4=G - 1)
    d = dict(**ttu("par_", [1, cw, 0, 0], [cw, g.taps, ppt, G], 8), psum_tflags=0x840 if ppt > 1 else 0x880 if G > 1 else 0,
             **OP.ttu_fine("psum", [int(g.kwords == 1)], [g.kwords, ppt, G]), cfg1=0x18F, cfg2=6 if g.kwords == 1 and ppt > 1 else 7,
             reduce_mask=0b111)
  else:             # fills of aw channel words of every tap, partial sums per position across the b fills
    ind = [(1, aw)] + taps + pos + [(aw, b), (0, G)]
    loops = dict(loop0=aw - 1, loop1=g.taps - 1, loop2=ppt - 1, loop3=b - 1, loop4=G - 1)
    d = dict(**ttu("par_", [1, aw, 0, 0, 0], [aw, g.taps, ppt, b, G], 8), **OP.wide("psum", wl["psum"] if ppt > 1 else 0), psum_mode7=2,
             psum_tflags=0x8C0 if ppt == 1 and G > 1 else 0, **OP.ttu_fine("psum", [0, 4], [g.a, ppt, b, G]), cfg1=0x16F,
             cfg2=7 if ppt == 1 else 2 if g.a == 1 else 3, reduce_mask=0b1011)
    if g.a == 1: d |= dict(psum_hmode=1)
  og = min(64, g.N4)                                  # outputs per op group (the blob's groups may be padded to 16)
  wg, last = og // 4, g.N4 - og * (G - 1)
  ob, od, wide = out if out is not None else (Y, [(g.N4 // 4, ppt)], False)
  od = [(1, wg)] + list(od) + [(wg, G)]
  d |= dict(**loops, in_base=X + (rs.wo * cs.nb + cs.wo) * g.C4, in_mode7=3, in_tflags=(1 << (len(ind) - 3)) - 1,
            **ttu("in_", [s_ for s_, _ in ind], [c_ for _, c_ in ind], 8), **OP.wide("par", wl["par_fifo"]), par_mode7=3,
            par_tflags=0xC0 if g.fifo else 0, par_fifo=g.a if g.fifo else 0, out_base=ob, out_mode7=3, out_tflags=3 if wide else 1,
            **ttu("out_", [s_ for s_, _ in od], [c_ for _, c_ in od], 8))
  if last < og: d |= dict(out_last_cnt=last // 4 - 1, out_last_mode=len(od) - 1, out_last_skip=wg - last // 4)   # mode: the group level
  return d | dict(cfg0=0xE7 if wide else 0xE5, sync0=0x4000, sync1=0x4000, sync2=0xC6, sync3=0x80, sync4=0x80, dp_mode=4, out_ch=og - 1,
                  out_ch_last=last - 1,
                  cfg4=1, w_zp=q["w_zp"], in_zp=q["in_zp"], out_zp=q["out_zp"], mult_bits=f32_bits(q["mult"]),
                  clamp_max_bits=f32_bits(q.get("clamp_max", 255 - q["out_zp"])), clamp_min_bits=f32_bits(q.get("clamp_min", -q["out_zp"])))

def conv_stage(e:Emitter, g:Conv2DGeom, X:int, Y:int, wl:dict, q:dict, pieces:list[Piece]) -> int:
  """fences, the weight consumer (ringConsumer1 on all tiles), the bias load, one conv op per output block shape, then one ringProducer
  per parameter piece broadcasting its cached rows; -> rows sent"""
  tiles = sum(1 << t for t in range(16) if g.out_tile(t))   # tiles with outputs
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_drain)
  e.tile(rc_param(g, e.seq, wl, tiles))
  e.tile(CC.bias_load(e.seq, tiles, wl["bias"], g.b, g.fifo, g.G, g.cg))
  for m, f in _group([_conv_fields(g, t // 4, t % 4, X, Y, wl, q) if tiles >> t & 1 else None for t in range(16)]):
    e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))
  first = 0
  for tile, off, n in pieces:
    e.tile(CC.rprod_param(e.seq, tile, off, n, first, dest=tiles))
    first += n
  return first

# ***** output *****
def smem_output(g:Conv2DGeom) -> bool:
  """a tile's output block that is not the last one is not a multiple of 8 bytes: the outputs go through the scalar core's memory
  (an odd last block is simply padded on the host path)"""
  sizes = [g.out_tile(t) for t in range(16) if g.out_tile(t)]
  return any(n % 8 for n in sizes[:-1])

def _final_flush(n:int) -> list[int]:
  """the last scalar-memory flush of n words: a host DMA of 4n bytes rounded up to 8, the copy of n // 2 eight-byte units, and for an
  odd n six more words that copy the last one (0x23 and vector-slot words, reproduced as observed)"""
  words = SC.output_dma_tail(round_up(4 * n, 8)) + [SC.movi(6, 0), 0x131f800, 0x1323800, 0x200008c0 | ((n // 2 - 1) << 12), 0xe00040131f800,
                                                     0x10000401323800, 0xe000400000800, 0x10000400000800]
  return words + ([0x131f800, 0x800, 0xe000400000800, SC.movi(6, 0), 0x100018c0, 0xc000400000800] if n % 2 else [])

def output_stage(e:Emitter, g:Conv2DGeom, Y:int, rows:int):
  """the output blocks: narrowToWide per block shape, then per tile (outfeed, ringProducer) to the host, or (Cout*positions not a
  multiple of 8 on some tile) every tile's ringProducer into scalar-memory outfeeds packed into a 4096-word buffer that the scalar
  core flushes to the host (codegen/conv.py's path, with per-tile sizes). Then the PRODUCER_A / PRODUCER_B waits and the epilogue."""
  smem, w = smem_output(g), g.N4 // 4
  sizes = [g.out_tile(t) for t in range(16)]
  e.sync(SC.sync_drain if smem else SC.signal_fence)
  e.scalar(SC.output_dma_head())
  per = []
  for t in range(16):
    rs, cs = g.rows[t // 4], g.cols[t % 4]
    if not sizes[t]:
      per.append(None)
      continue
    dims = [(1, w)] + [(w, cs.no)] * (cs.no > 1) + [(cs.no * w, rs.no)] * (rs.no > 1)
    per.append(dict(nb=sizes[t], dims=tuple(dims), shape=(rs.no, cs.no)))      # one instruction per output block shape
  for m, f in _group(per): e.tile(WN.n2w_output(f["nb"], m, Y // 4, WIDE_OUT_FIFO, e.seq, list(f["dims"])))
  tiles = [t for t in range(16) if sizes[t]]
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
    e.scalar(SC.scsync_smem_post() + _final_flush(used))
    n_wait = packets
  e.scalar(SC.output_wait(e.seq, n_wait), seqs=2)
  e.sync(SC.broadcast_wait, rows)
  e.sync(SC.epilogue)

# ***** programs *****
REF_QUANT2D = dict(w_zp=128, in_zp=128, out_zp=128, mult=EL.mul32(EL.mul32(1/128, 1/128), EL.recip32(1/16)),
                   clamp_min=EL.out_clamps(1/16, 128)[0], clamp_max=EL.out_clamps(1/16, 128)[1])

def conv2d_quant(x_q:tuple, w_q:tuple, y_q:tuple, act:int=0) -> dict:
  lo, hi = EL.out_clamps(y_q[0], y_q[1], act)
  return dict(w_zp=w_q[1], in_zp=x_q[1], out_zp=y_q[1], mult=EL.mul32(EL.mul32(x_q[0], w_q[0]), EL.recip32(y_q[0])), clamp_min=lo,
              clamp_max=hi)

def compiler_tiles(g:Conv2DGeom) -> tuple[int, ...]:
  """the parameter tiles of edgetpu_compiler's alone-compiled program: tile 0 if the blob fits below param_limit, else
  ceil(rows / capacity) pieces on codegen/conv.py's COMPILER_SPLIT_TILES"""
  n = cdiv(g.blocks, param_limit(g) // 4)
  if n not in CC.COMPILER_SPLIT_TILES: raise NotImplementedError(f"{n} parameter pieces: tile choice not observed")
  return CC.COMPILER_SPLIT_TILES[n]

def conv2d_programs(g:Conv2DGeom, pieces:list[Piece], quant:dict|None=None) -> tuple[bytes, bytes]:
  """(PARAMETER_CACHING, EXECUTION_ONLY) with the parameter blob in pieces [(tile, wide offset in 64-byte units, rows)]"""
  nl, wl = narrow_layout(g), wide_layout(g)
  q = {**REF_QUANT2D, **(quant or {})}
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  input_stage(e, g, nl, wl)
  e.sync(SC.sync_reset17)
  halo_stage(e, g, nl["X"], q["in_zp"] * 0x01010101)
  rows = conv_stage(e, g, nl["X"], nl["Y"], wl, q, pieces)
  output_stage(e, g, nl["Y"], rows)
  if len(e.words) + 2 > MAX_WORDS: raise NotImplementedError("more than one bitstream of 16384 words (not modelled)")
  return caching_program(g.param_bytes, piece_parts([pieces], 4 * g.cg)), e.program()

def gen_conv2d(H:int, W:int, Cin:int, Cout:int, kh:int, kw:int, stride:int=1, padding:str="VALID", param_tile:int|tuple[int, ...]=0,
               param_offset:int=0, quant:dict|None=None, param_limit_units:int|None=None) -> tuple[bytes, bytes]:
  """(PARAMETER_CACHING, EXECUTION_ONLY) of CONV_2D x[1,H,W,Cin] * w[Cout,kh,kw,Cin] -> y[1,OH,OW,Cout] (uint8), byte-identical to
  edgetpu_compiler. The parameters (conv2d_blob) live on tile param_tile at byte offset param_offset (a multiple of 256) of its wide
  memory and are broadcast to all tiles on every call. A tuple of tiles splits the blob as edgetpu_compiler splits big layers, every
  tile but the last filled from param_offset up to param_limit_units (64-byte units, default and maximum: this program's lowest wide
  buffer, param_limit()); compiler_tiles() gives the compiler's own choice. quant: the conv op's quantization fields (conv2d_quant;
  default: tools.tflite_gen.conv_model's). Raises NotImplementedError when the blob does not fit the tile(s)."""
  assert padding in ("VALID", "SAME"), "padding: VALID or SAME"
  tiles = (param_tile,) if isinstance(param_tile, int) else tuple(param_tile)
  assert all(0 <= t < 16 for t in tiles) and len(set(tiles)) == len(tiles), "param_tile: distinct tiles 0..15"
  assert param_offset >= 0 and param_offset % 256 == 0, "param_offset must be a non-negative multiple of 256"
  g, off = Conv2DGeom(H, W, Cin, Cout, kh, kw, stride, padding), param_offset // 64
  if g.kwords == 1 and g.G > 1 and any(g.rows[t // 4].no * g.cols[t % 4].no == 1 for t in range(16)):
    raise NotImplementedError("1x1 kernel, Cin <= 4, Cout > 64 with single-position tiles: edgetpu_compiler uses a paired-group op "
                              "(cfg0 0x127, dp_mode 5) that is not modelled")
  nl = narrow_layout(g)
  if max(nl["X"] + g.Xs, nl["Stg"] + 2 * g.slot, nl["C"] + g.Cs, nl["Y"] + g.Ys) > NARROW_BYTES:
    raise NotImplementedError("the blocks / outputs do not fit a tile's 192 KiB narrow memory (the compiler would tile the image)")
  limit = param_limit(g) if param_limit_units is None else min(param_limit_units, param_limit(g))
  try: pieces = CC.split_pieces(g.blocks, tiles, off, limit)
  except ValueError:
    raise NotImplementedError(f"{g.blocks} parameter rows at offset {param_offset} B do not fit {len(tiles)} tile(s) of {(limit - off) // 4} "
                              f"rows (edgetpu_compiler splits them over the tiles compiler_tiles() returns)") from None
  return conv2d_programs(g, pieces, quant)

# ***** host side *****
def conv2d_io(H:int, W:int, Cin:int, Cout:int, kh:int, kw:int, stride:int=1, padding:str="VALID") -> dict:
  """the host contract: input DMA bytes (x[H][W][Cin] uint8, padded to 8), output DMA bytes (16 tile blocks, each [rows][cols][N4]
  and padded to 8), the executable's output_layout (output (y, x) channel c is at byte tile_byte_offset[y_tile[y] + x_tile[x]] +
  y_local_y_offset[y] * x_local_row_size[x] + x_local_byte_offset[x] + c) and the parameter blob size"""
  g = Conv2DGeom(H, W, Cin, Cout, kh, kw, stride, padding)
  sizes = [g.out_tile(t) if smem_output(g) else round_up(g.out_tile(t), 8) for t in range(16)]
  ry = [(r, y) for r, sp in enumerate(g.rows) for y in range(sp.no)]
  cx = [(c, x) for c, sp in enumerate(g.cols) for x in range(sp.no)]
  last = max(t for t in range(16) if sizes[t])
  offs = [sum(sizes[:t]) if t <= last else sum(sizes[:last]) + g.out_tile(last) for t in range(16)]   # the last block's padding trails
  return dict(input_bytes=g.S, output_bytes=round_up(sum(sizes), 8), param_bytes=g.param_bytes,
              output_layout=dict(y_tile=[4 * r for r, _ in ry], x_tile=[c for c, _ in cx],
                                 tile_byte_offset=offs, x_local_byte_offset=[x * g.N4 for _, x in cx],
                                 y_local_y_offset=[y for _, y in ry], x_local_row_size=[g.cols[c].no * g.N4 for c, _ in cx]))

def conv2d_blob(w_u8, w_zp:int, bias_i32=None, H:int|None=None, W:int|None=None, stride:int=1, padding:str="VALID") -> bytes:
  """the parameter blob of conv weights w_u8 [Cout, kh, kw, Cin] (uint8, zero point w_zp) and int32 biases (None = 0), as
  edgetpu_compiler lays it out: per group of cg outputs (cg = Cout rounded up to 16 if Cout <= 64, else 64) an int32 bias row, then
  the weights as [K/4][cg][4] uint8 with K = the taps x the channels padded to 4 (with w_zp). Outputs Cout..N4-1 are all w_zp, the
  rest of the last group 0. K-chunked convs order K as [channel word][tap], position-outer convs as [tap][channel word]; which one
  depends on the image (H, W, stride, padding), needed only when the kernel has several taps and Cin > 4."""
  import numpy as np
  w = np.asarray(w_u8, np.uint8)
  Cout, kh, kw, Cin = w.shape
  C4 = round_up(Cin, 4)
  if kh * kw > 1 and C4 > 4:
    if H is None or W is None: raise ValueError("conv2d_blob: pass the image H, W (stride, padding): the weight order depends on them")
    tap_major = Conv2DGeom(H, W, Cin, Cout, kh, kw, stride, padding).pos_outer
  else: tap_major = False
  cg = round_up(Cout, 16) if Cout <= 64 else 64
  Np = round_up(Cout, cg)
  ww = np.zeros((Np, kh * kw, C4), np.uint8)
  ww[:round_up(Cout, 4)] = w_zp
  ww[:Cout, :, :Cin] = w.reshape(Cout, kh * kw, Cin)
  b = np.zeros(Np, "<i4")
  if bias_i32 is not None: b[:Cout] = np.asarray(bias_i32, np.int64)
  out = []
  for o in range(0, Np, cg):
    blk = ww[o:o + cg].reshape(cg, kh * kw, C4 // 4, 4)                 # [output][tap][channel word][4]
    out += [b[o:o + cg].tobytes(), np.ascontiguousarray(blk.transpose(1, 2, 0, 3) if tap_major else blk.transpose(2, 1, 0, 3)).tobytes()]
  return b"".join(out)

if __name__ == "__main__": run_test("test_codegen_conv2d")
