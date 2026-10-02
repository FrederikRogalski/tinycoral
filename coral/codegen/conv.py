# batched matmul y[Mp,N] = x[Mp,K] @ W[N,K].T as a 1x1 CONV_2D over a grid of Mp positions, byte-identical to edgetpu_compiler
# (docs/isa/codegen_conv.md). `python -m coral.codegen.conv` runs the acceptance test (test/test_codegen_conv.py).
# Execution program, one tile op per phase on all 16 tiles:
#   prologue, input fence, the constant (identity) row (+ a zero row when K % 4 != 0) -> wide memory
#   first copy op(s) (image rows 0..hp-2), input DMA, input wideToNarrow per tile column, (ringConsumer, infeed) per tile row
#   last copy op(s) (image row hp-1), fences
#   ringConsumer1 (weight FIFO), bias wideToNarrow, the conv op, ringProducer(s) broadcasting the cached weights
#   narrowToWide output, (outfeed, ringProducer) per tile (or the scalar-memory path), PRODUCER_A/B waits, epilogue
from __future__ import annotations
from dataclasses import dataclass
from coral.isa import cdiv, round_up, ttu, f32_bits, op as OP, wide_narrow as WN, ring_mesh as RM, scalar as SC, eltwise as EL
from coral.codegen import Emitter, Piece, caching_program, piece_parts, run_test, WIDE_TOP, WIDE_OUT_FIFO, REF_QUANT
from coral.codegen.fc import rprod_output

GRIDS = {16: (4, 4), 32: (8, 4), 64: (8, 8), 128: (16, 8), 256: (16, 16)}   # Mp -> (H, W)

@dataclass(frozen=True)
class ConvGeom:
  """1x1 conv over the H x W grid of Mp positions (GRIDS[Mp] unless `grid`) on the 4x4 tiles: tile (r, c) holds image rows
  [r*hp, (r+1)*hp) and columns [c*wp, (c+1)*wp); ring row group r receives image rows r*hp.. as one byte stream"""
  Mp: int
  N: int
  K: int
  grid: tuple[int, int]|None = None
  @property
  def H(self) -> int: return (self.grid or GRIDS[self.Mp])[0]
  @property
  def W(self) -> int: return (self.grid or GRIDS[self.Mp])[1]
  @property
  def hp(self) -> int: return self.H // 4            # image rows per tile
  @property
  def wp(self) -> int: return self.W // 4            # image columns per tile
  @property
  def ppt(self) -> int: return self.hp * self.wp     # positions per tile
  @property
  def K4(self) -> int: return round_up(self.K, 4)
  @property
  def kw(self) -> int: return self.K4 // 4           # 4-byte input words per position = reduction steps
  @property
  def N4(self) -> int: return round_up(self.N, 4)
  @property
  def cg(self) -> int: return min(64, self.N4)       # outputs per parameter group
  @property
  def G(self) -> int: return cdiv(self.N, 64)        # parameter groups
  @property
  def pos_outer(self) -> bool: return self.kw <= self.ppt + 1   # whole K per weight fill, positions outer, no partial sums
  @property
  def a(self) -> int:                                # weight rows per fill (inner K loop): the largest divisor of kw <= min(15, kw/2)
    return self.kw if self.pos_outer else max(d for d in range(1, min(15, self.kw // 2) + 1) if self.kw % d == 0)
  @property
  def b(self) -> int: return self.kw // self.a       # K chunks (outer K loop, partial sums)
  @property
  def fifo(self) -> bool: return self.a <= 15        # weights stream through an a-slot ring FIFO (else one kw-row block)
  @property
  def R(self) -> int: return self.W * self.K         # bytes of one image row (one ring-FIFO pass)
  @property
  def S(self) -> int: return 8 * cdiv(self.Mp * self.K, 8)   # input DMA bytes
  @property
  def S_row(self) -> int: return self.hp * self.R    # bytes of one ring row group's stream
  @property
  def c_in(self) -> int: return cdiv(self.R, 256)    # input ring FIFO slots (256-byte packets)
  @property
  def P(self) -> int: return cdiv(self.S_row, 256)   # packets per row group
  @property
  def refills(self) -> int: return cdiv(self.P, self.c_in)   # ring FIFO passes per row group
  @property
  def p_last(self) -> int: return self.P - self.c_in * (self.refills - 1)   # packets in the last pass
  @property
  def out_tile(self) -> int: return self.ppt * self.N4    # output bytes per tile (positions padded to 4 bytes)
  @property
  def odd(self) -> bool: return self.K % 4 != 0      # positions are not word aligned in the ring stream: byte-shifting copies
  @property
  def Cs(self) -> int: return 16 + 256 * self.odd    # constant row (+ a 256-byte zero row that pads K to K4)
  @property
  def slot(self) -> int: return max(self.R, 256)     # narrow staging: two slots of one image row each
  @property
  def blocks(self) -> int: return self.G * (1 + self.kw)  # 256-byte parameter rows: per group the bias and kw weight rows
  @property
  def param_bytes(self) -> int: return 256 * self.blocks

# *** memory layout ***
def narrow_layout(g:ConvGeom) -> dict:
  """edgetpu_compiler's narrow placement (byte addresses) of X (ppt*K4 input positions), Y (ppt*N4 outputs), Stg (two staging slots
  for the ring input; Y reuses them) and C (Cs bytes: constant + zero row). Y goes first when it is the bigger region (Mp >= 32:
  Ys >= 2*slot + Cs; Mp = 16, which has no first copy op: Ys > 2*slot)."""
  Xs, Ys, R2, Cs = g.ppt * g.K4, g.ppt * g.N4, 2 * g.slot, g.Cs
  if g.Mp == 16:
    if Ys > R2: return dict(Y=0, Stg=0, C=R2, X=max(Ys, R2 + Cs))
    return dict(Y=0, Stg=0, X=R2, C=R2 + Xs)
  if Ys >= R2 + Cs: return dict(Y=0, Stg=0, C=R2, X=Ys)
  return dict(X=0, Stg=Xs, Y=Xs, C=Xs + R2)

def regions(g:ConvGeom, top:int) -> dict:
  """the conv's compute buffers (64-byte units) stacked down from top, largest first, ties in the order partial sums (4 per position,
  reserved also when unused), bias (8), weight FIFO (8 per slot); position-outer with kw > 15: one kw-row block, the bias below"""
  if not g.fifo: return dict(par_fifo=top - 4 * g.kw, psum=None, bias=top - 4 * g.kw - 4, bottom=top - 4 * g.kw - 4)
  out = dict(psum=None)
  for name, size in sorted([("psum", 0 if g.pos_outer else 4 * g.ppt), ("bias", 8), ("par_fifo", 8 * g.a)], key=lambda r: -r[1]):
    if size:
      top -= size
      out[name] = top
  if g.ppt == 1: out["psum"] = None
  return out | dict(bottom=top)

def wide_layout(g:ConvGeom) -> dict:
  """the execution program's wide buffers (64-byte units): the input ring FIFO below WIDE_TOP, the constant row below it, the compute
  regions from WIDE_TOP; bottom = the lowest"""
  in_fifo, r = WIDE_TOP - 8 * g.c_in, regions(g, WIDE_TOP)
  const = in_fifo - 4 * cdiv(g.Cs, 256)
  return r | dict(in_fifo=in_fifo, const=const, bottom=min(const, r["bottom"]))

def param_limit(g:ConvGeom) -> int:
  """highest wide address (64-byte units, exclusive) the parameters may use: the execution program's lowest buffer"""
  return wide_layout(g)["bottom"]

# *** instruction builders ***
def _twait(g:ConvGeom) -> dict: return dict(rsv515=2 * g.R & 127, in_twait=2 * g.R >> 7)   # 2R in a 21-bit field at bit 515

def _copy_progress(g:ConvGeom, last:bool, ph:int=0) -> tuple[int, int]:
  """(sync0, cfg1 bits 8-9) of a copy op, a progress count in 1/64 units: T64 = 16 * (bytes of this tile's positions per image row)
  for the first copy op, 16 * (those bytes of the hp-1 earlier rows) + R + 4*ph rounded up to 16 for the last one (ph = start byte
  of the op's positions in its staging word). sync0 = (wp*K % 4) << 14 | T64 // 64, cfg1 bits 8-9 = (T64 % 64) // 16."""
  t64 = 16 * g.wp * g.K * (g.hp - 1) + round_up(g.R + 4 * ph, 16) if last else 16 * g.wp * g.K
  return (((g.wp * g.K) % 4) << 14) | (t64 // 64), (t64 % 64) // 16

def _copy_op(g:ConvGeom, seq:int, in_base:int, out_base:int, rows:int, const:int, last:bool) -> dict:
  """copy op: per refill, take this tile's wp positions (kw words each) from the staging buffer (re-read for every image row) and
  append them to X, through the constant row"""
  (sync0, x), kw, wp = _copy_progress(g, last), g.kw, g.wp
  return dict(tile_mask=0xffff, seq=seq, loop1=kw - 1, loop2=wp - 1, loop3=rows - 1, in_base=in_base, in_mode7=3, in_tflags=0x80,
              **ttu("in_", [1, kw, 0], [kw, wp, rows], 8), out_base=out_base, out_mode7=3, out_tflags=3,
              **ttu("out_", [1, 1, kw, wp * kw], [1, kw, wp, rows], 8), **_twait(g), **OP.wide("par", const),
              **ttu("par_", [], [kw, 1, wp, rows], 8), **ttu("psum_", [], [kw, 1, wp, rows], 8), psum_tflags=0x800, cfg0=7, cfg1=0x6B | x << 8,
              sync0=sync0, sync1=0x4000 + wp * kw, cfg2=6, dp_mode=4, out_ch=3, out_ch_last=3, **OP.NO_REQUANT)

def _shift_copy_op(g:ConvGeom, seq:int, mask:int, in_base:int, out_base:int, rows:int, const:int, last:bool) -> dict:
  """byte-granular copy (K % 4 != 0): per image row, take the wp positions of K bytes from byte address in_base of the staging slot
  and write them word aligned (K4 bytes, padded from the zero row) to X"""
  K, kw, wp, hp = g.K, g.kw, g.wp, g.hp
  sync0, x = _copy_progress(g, last, in_base % 4)
  r, t = (K - 1) % 4, -K % 4                         # (K-1) % 4 at bits 174/1229/1532, padding bytes at bit 1246
  ostr, ocnt = [1, 1, kw, wp * kw] + [wp * kw] * (rows > 1) + [hp * wp * kw], [1, kw, wp, 1] + [rows] * (rows > 1) + [1]
  return dict(tile_mask=mask, seq=seq, loop0=3, loop1=kw - 1, loop2=wp - 1, loop4=rows - 1, rsv174=0x4000 | r,
              **OP.ttu_fine("in", [1, K, g.R, 0], [K, wp, 1, rows]), in_mode7=3, in_base=in_base, in_tflags=0xC1, **_twait(g),
              out_base=out_base, out_mode7=3, out_tflags=7, rsv893=int(rows > 1), **ttu("out_", ostr, ocnt, 8),
              **OP.wide("par", const), **OP.ttu_fine("par", [1], [4, kw, wp, rows]), par_mode7=1, **ttu("psum_", [], [4, kw, wp, rows], 8),
              psum_tflags=0x840, par_fifo=(r & 1) << 13, rsv1230=(r >> 1) | (1 << 13) | (t << 16), rsv1523=(r << 9) | (1 << 23), cfg0=9,
              cfg1=0x8B | x << 8, sync0=sync0, sync1=0x4000 + (wp * K) // 4, cfg2=6, cfg3=7, dp_mode=4, reduce_mask=1, out_ch=3, out_ch_last=3,
              **OP.NO_REQUANT)

def _copy_ops(g:ConvGeom, seq0:int, Stg:int, X:int, const:int, last:bool) -> list[list[int]]:
  """the copy ops of one phase: first = image rows 0..hp-2 (before the input DMA), last = image row hp-1. K % 4 != 0: one
  byte-shifting op per distinct start byte (c*wp*K) % 4 of the tile columns c, in column order."""
  rows = 1 if last else g.hp - 1
  src = Stg + ((g.hp - 1) % 2) * g.slot if last else Stg
  dst = X + (g.hp - 1) * g.wp * g.K4 if last else X
  if not g.odd: return [OP.encode_op(**_copy_op(g, seq0, src, dst, rows, const, last))]
  groups: dict[int, int] = {}                        # start byte -> tile columns
  for c in range(4):
    ph = c * g.wp * g.K % 4
    groups[ph] = groups.get(ph, 0) | 0x1111 << c
  return [OP.encode_op(**_shift_copy_op(g, seq0 + i, mask, src + ph, dst, rows, const, last))
          for i, (ph, mask) in enumerate(groups.items())]

def _w2n_input(g:ConvGeom, col:int, seq:int, Stg:int, in_fifo:int, grp:int, gstride:int, rows:int=0xf, shift:int=0) -> list[int]:
  """column col of the row groups in `rows` (bitmask) reads its row group's ring FIFO as one stream from its own slice (sk words in),
  in chunks of R/4 words (one image row, rotated); the last chunk overhangs the stream end (head = words inside the P packets,
  tail = overhang); a single chunk is cut to the stream end. shift = words of the previous row group at the start of this group's
  infeed (its offset is rounded down to 8 bytes)."""
  sk = shift + col * g.wp * g.K // 4
  L, rem = g.R // 4, 64 * g.P - sk                   # 64*P = words of the P packets
  if rem <= L: L, n, head = rem, 1, None
  else:
    n = cdiv(rem, L)
    head = rem - (n - 1) * L
    if head == L: head = None
  mask = sum(0xf << (4 * r) for r in range(4) if rows >> r & 1) & (0x1111 << col)
  f = dict(seq=seq, tile_mask=mask, wide_addr=in_fifo, wide_lvl_mask=1, wide_circ=1, wide_rows=g.c_in, rsv227=(grp << 6) | (gstride << 22),
           narrow_addr=Stg // 4, narrow_lvl_mask=1, mode=1, sync_id=14, sync_wait_lvl=1, sync_val=0x8000, sync_dec_lvl=1, sync_dec=-1,
           sync_dec_mode=3, tail_f871=1, rsv489=g.R // 2 & 31, size64=g.R // 2 >> 5,
           **WN.ttu("w", [1, 0], [g.c_in, g.refills], 4), **WN.ttu("n", [1, 0], [L, n], 6))
  if head is not None:
    t = 4 * (L - head)
    f.update(head_words_m1=head - 1, head_en=1, rsv524=t & 63, head_tail16=(t >> 6) & 15, rsv534=t >> 10)
  if sk: f.update(skip_en0=1, skip_en1=1, skip_base=Stg // 4, skip_m1=sk - 1, skip_end=Stg // 4 + sk - 1)
  return WN.encode_wide_to_narrow(**f)

def _rc_input(g:ConvGeom, row:int, seq:int, in_fifo:int, grp:int, gstride:int) -> list[int]:
  """ringConsumer0 of row group `row`: c_in-slot FIFO refilled g.refills times; a single slot refilled several times takes FC's
  forward-consumer form"""
  if g.c_in > 1: f = dict(sdims=1, cbuf=1, mode=3, s_id=(1 << 5) | 11, **ttu("", [1, 0], [g.c_in, g.refills], 4))
  elif g.refills == 1: f = dict(cbuf=1, mode=3, s_id=(1 << 5) | 11, **ttu("", [1], [1], 4))
  else: f = dict(mode=1, s_id=11, **ttu("", [0], [g.refills], 4))
  return RM.encode_ringConsumer(tile_mask=0xf << (4 * row), seq=seq, addr=in_fifo, slots=g.c_in, grp=grp, gstride=gstride,
                                s_val=-64 * g.c_in, s_cnt=g.c_in, s_en_b=1, s_y=1, **f)

def rc_param(g:ConvGeom, seq:int, wl:dict) -> list[int]:
  """ringConsumer1 on all tiles: receives the broadcast parameters (per group: bias block, kw weight rows) into the weight FIFO
  (a slots, refilled b times per group); the bias blocks go through the aux pair"""
  bias = wl["bias"]
  if g.a == 1:      # single-slot FIFO refilled kw times per group (the forward-consumer form)
    f = dict(slots=1, aux_addr0=bias + 1, aux_addr1=bias + 4, s_val=-1, mode=1, s_id=1, **ttu("", [0, 0, 0], [g.kw, 1, g.G], 4))
  elif not g.pos_outer: f = dict(slots=g.a, aux_addr0=bias + 2, aux_addr1=bias + 4, s_val=-1, **ttu("", [1, 0, 0, 0], [g.a, g.b, 1, g.G], 4))
  elif g.fifo: f = dict(slots=g.a, aux_addr0=bias + 1, aux_addr1=bias + 4, s_val=-1, **ttu("", [1, 0, 0], [g.a, 1, g.G], 4))
  else: f = dict(aux_addr0=bias + 1, aux_addr1=bias, **ttu("", [1, 0, 0], [g.a, 1, g.G], 4))
  if g.a > 1: f |= dict(sdims=1, cbuf=1, mode=3, s_id=(1 << 5) | 1)
  return RM.encode_ringConsumer(opcode=0x12, tile_mask=0xffff, seq=seq, addr=wl["par_fifo"], aux_en0=1, aux_en1=1, s_en_a=1, s_en_b=1,
                                s_y=1, **f)

def bias_load(seq:int, tiles:int, bias:int, b:int, fifo:bool=True, G:int=1, cg:int=64) -> list[int]:
  """mode-2 load of G bias blocks (cg outputs each) at wide `bias`; b = weight FIFO fills per group, in a field at bit 639"""
  return WN.encode_wide_to_narrow(seq=seq, tile_mask=tiles, wide_addr=bias, wide_lvl_mask=1, wide_circ=int(fifo), wide_rows=int(fifo),
                                  narrow_addr=0x40, narrow_lvl_mask=3, mode=2, size64=2, sync_f2=1, sync_id=15, sync_wait_lvl=1,
                                  sync_val=(b << 15) & 0x7ffff, rsv643=b >> 4, sync_dec_lvl=2, sync_dec=-1, sync_dec_mode=3, tail_f867=1,
                                  **WN.ttu("w", [1, 0], [1, G], 4), **WN.ttu("n", [1, 64, 0], [cg // 2, 1, G], 6))

def conv_op(g:ConvGeom, seq:int, X:int, Y:int, wl:dict, q:dict, tiles:int=0xffff, out:tuple|None=None, wide:bool=False) -> list[int]:
  """the conv op. Position-outer: the weights of a group are loaded once, positions iterate outside the K loop, no partial sums;
  else K runs in b chunks of a rows with partial sums per position in wide memory. out = the output TTU (strides, counts) in words
  (default: the tile's block, groups of cg outputs); wide: the tile's rows sit inside wider window rows (cfg0 0xE7)"""
  a, b, kw, ppt, G = g.a, g.b, g.kw, g.ppt, g.G
  d = dict(tile_mask=tiles, seq=seq, loop4=G - 1, in_base=X, in_mode7=3, in_tflags=3, **OP.wide("par", wl["par_fifo"]), par_mode7=3)
  if g.pos_outer:
    d |= dict(loop0=kw - 1, loop3=ppt - 1, **ttu("in_", [1, kw, kw, kw, 0], [kw, 1, 1, ppt, G], 8), par_tflags=0xC0 if g.fifo else 0,
              par_fifo=kw if g.fifo else 0, **ttu("par_", [1, kw, 0, 0], [kw, 1, ppt, G], 8), psum_tflags=0x840,
              **ttu("psum_", [0, 0, 0], [kw, ppt, G], 8), cfg1=0x18F, cfg2=7, reduce_mask=0b111)
  else:
    d |= dict(loop0=a - 1, loop2=ppt - 1, loop3=b - 1, **ttu("in_", [1, kw, kw, a, 0], [a, 1, ppt, b, G], 8), par_tflags=0xC0, par_fifo=a,
              **ttu("par_", [1, a, 0, 0, 0], [a, 1, ppt, b, G], 8), **OP.wide("psum", wl["psum"] or 0), psum_mode7=2,
              psum_tflags=0x8C0 if ppt == 1 and G > 1 else 0, **ttu("psum_", [0, 1, 0, 0], [a, ppt, b, G], 8), cfg1=0x16F,
              cfg2=7 if ppt == 1 else 3 if a > 1 else 2, psum_hmode=int(a == 1), reduce_mask=0b1011)
  wg, last = g.cg // 4, g.N4 - g.cg * (G - 1)        # words per group, outputs of the last group (padded to 4)
  d |= dict(out_base=Y, out_mode7=3, out_tflags=3 if wide else 1, **ttu("out_", *(out or ([1, g.N4 // 4, wg], [wg, ppt, G])), 8))
  if last < g.cg: d |= dict(out_last_cnt=last // 4 - 1, out_last_mode=2, out_last_skip=wg - last // 4)
  return OP.encode_op(**d, cfg0=0xE7 if wide else 0xE5, sync0=0x4000, sync1=0x4000, sync2=0xC6, sync3=0x80, sync4=0x80, dp_mode=4,
                      out_ch=g.cg - 1, out_ch_last=last - 1, cfg4=1, w_zp=q["w_zp"], in_zp=q["in_zp"], out_zp=q["out_zp"],
                      mult_bits=f32_bits(q["mult"]), clamp_max_bits=f32_bits(q.get("clamp_max", 255 - q["out_zp"])),
                      clamp_min_bits=f32_bits(q.get("clamp_min", -q["out_zp"])))

def rprod_param(seq:int, tile:int, off:int, rows:int, first:int=0, dest:int=0xffff) -> list[int]:
  """parameter broadcast: the parameter tile streams its cached rows to ringBusConsumer1 of the tiles in dest; first = rows sent before"""
  return RM.encode_ringProducer(tile_mask=1 << tile, seq=seq, addr=off, sdims=1, pcfg=4, r0_id=(1 << 5) | 18, r0_val=first, r0_en_a=1,
                                r0_en_b=1, to_c1=1, dest=dest, **ttu("", [1], [rows], 4))

def _n2w_out(g:ConvGeom, seq:int, Y:int) -> list[int]:
  w = g.N4 // 4
  return WN.n2w_output(g.out_tile, 0xffff, Y // 4, WIDE_OUT_FIFO, seq, [(1, w)] + [(w, g.wp)] * (g.wp > 1) + [(g.wp * w, g.hp)] * (g.hp > 1))

# *** program stages ***
def input_stage(e:Emitter, g:ConvGeom, Stg:int, X:int, C:int, in_fifo:int, const:int, const_row:bool=True, host_off:int=0):
  """input head, the constant row (if const_row), first copy op(s), host input DMA (host_off bytes into the input), the input
  wideToNarrows and (ringConsumer, infeed) pairs, last copy op(s)"""
  e.sync(SC.input_head)
  if const_row: e.tile(*EL.ident_prologue(e.seq, 0xffff, C, const, 64 * g.odd))
  e.sync(SC.sync_wn_fence)
  if g.hp > 1: e.tile(*_copy_ops(g, e.seq, Stg, X, const, False))
  e.scalar(SC.input_dma(g.S, host_off))
  if g.Mp >= 128: e.sync(SC.sync_w2n)
  grp, gstride = (g.p_last - 1, 4 * (g.c_in - g.p_last) + 1) if g.p_last < g.c_in else (0, 0)
  shifts: dict[int, int] = {}                        # word shift of a row group's stream (its offset is rounded to 8 bytes) -> row groups
  for r in range(4):
    sh = r * g.S_row % 8 // 4
    shifts[sh] = shifts.get(sh, 0) | 1 << r
  for sh, rows in shifts.items():
    for col in range(4): e.tile(_w2n_input(g, col, e.seq, Stg, in_fifo, grp, gstride, rows, sh))
  for row in range(4):
    e.tile(_rc_input(g, row, e.seq, in_fifo, grp, gstride))
    lo, hi = (row * g.S_row) // 8, cdiv((row + 1) * g.S_row, 8)
    e.scalar(SC.av_infeed(8 * lo, 8 * (hi - lo), 0xf << (4 * row)))
  e.sync(SC.sync_drain)
  e.tile(*_copy_ops(g, e.seq, Stg, X, const, True))

def conv_stage(e:Emitter, g:ConvGeom, X:int, Y:int, wl:dict, q:dict, pieces:list[Piece], first:int=0) -> int:
  """fences, the weight FIFO consumer, the bias load, the conv op, one broadcast ringProducer per parameter piece; -> rows sent"""
  e.sync(SC.sync_reset17)
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_drain)
  e.tile(rc_param(g, e.seq, wl))
  e.tile(bias_load(e.seq, 0xffff, wl["bias"], g.b, g.fifo, g.G, g.cg))
  e.tile(conv_op(g, e.seq, X, Y, wl, q))
  for tile, off, n in pieces:
    e.tile(rprod_param(e.seq, tile, off, n, first))
    first += n
  return first

# scalar-memory staging of the outputs (the 0x23 and vector-slot words are not decoded; reproduced as observed)
SMEM_WORDS = 4096                # output staging buffer in scalar memory, 4-byte elements

def _smem_outfeed(base:int, n:int) -> list[int]:
  return SC.OUTFEED.encode(base=base, d0_stride=1, d0_limit=n - 1, d1_stride=(1 - n) & 0xffff, d2_stride=(1 - n) & 0xffff, d3_stride=1,
                           f222=0x3f)

def _smem_flush(nbytes:int) -> list[int]:
  """host DMA of the first nbytes of the staging buffer, then the scalar-memory -> DMA copy (0x23 carries nbytes/8 - 1)"""
  return SC.output_dma_tail(nbytes) + [SC.movi(6, 0), 0x131f800, 0x1323800, 0x200008c0 | ((nbytes // 8 - 1) << 12), 0xe00040131f800,
                                       0x10000401323800, 0xe000400000800, 0x10000400000800]

def output_stage(e:Emitter, g:ConvGeom, Y:int, rows:int):
  """the outputs, then wait for the PRODUCER_A outfeeds and the PRODUCER_B rows broadcast, epilogue. Outputs whose per-tile size is
  a multiple of 8 bytes go to the host directly: one outfeed per tile, (outfeed, ringProducer) in tile order. Otherwise (Mp=16,
  N4 % 8 == 4) every tile's ringProducer feeds a scalar-memory outfeed, packed one after the other into a 4096-word staging buffer
  that the scalar core sends to the host: a tile that does not fit puts as many whole 256-byte packets as fit, the buffer is flushed
  (a multiple of 8 words; the host pointer advances, the 1..7 left-over words move to the start) and the tile continues."""
  smem = g.out_tile % 8 != 0
  e.sync(SC.sync_drain if smem else SC.signal_fence)
  e.scalar(SC.output_dma_head())
  e.tile(_n2w_out(g, e.seq, Y))
  if not smem:
    e.scalar(SC.output_dma_tail(16 * round_up(g.out_tile, 8)))
    for t in range(16):
      e.scalar(SC.outfeed(round_up(g.out_tile, 8)))
      e.tile(rprod_output(g.out_tile, 1 << t, t, e.seq))
  else:
    per, used = g.out_tile // 4, 0
    for t in range(16):
      e.tile(rprod_output(g.out_tile, 1 << t, t * cdiv(g.out_tile, 256), e.seq, smem=True))
      e.scalar(SC.scsync_smem_pre())
      n = per
      while n > SMEM_WORDS - used:
        part = (SMEM_WORDS - used) // 64 * 64
        if part:
          e.scalar(_smem_outfeed(used, part))
          used += part
          n -= part
        flush, left = used // 8 * 8, used % 8
        e.scalar(SC.scsync_smem_post() + _smem_flush(4 * flush) + SC.add64(4, 5, 4 * flush, 4, 5, 6))
        if left: e.scalar([SC.movi(7, flush), SC.movi(8, 0), 0x300008c0 | (left << 12), 0x139b800, 0x800, 0xc028300000800])
        used = left
      e.scalar(_smem_outfeed(used, n))
      used += n
    e.scalar(SC.scsync_smem_post() + _smem_flush(4 * used))
  # PRODUCER_A counts one per output producer (host path) / per 256-byte packet (scalar-memory path)
  e.scalar(SC.output_wait(e.seq, 16 * (cdiv(g.out_tile, 256) if smem else 1)), seqs=2)
  e.sync(SC.broadcast_wait, rows)
  e.sync(SC.epilogue)

def conv_programs(g:ConvGeom, pieces:list[Piece], quant:dict|None=None) -> tuple[bytes, bytes]:
  """(PARAMETER_CACHING, EXECUTION_ONLY) with the parameter blob in pieces [(tile, wide offset in 64-byte units, rows)]"""
  nl, wl = narrow_layout(g), wide_layout(g)
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  input_stage(e, g, nl["Stg"], nl["X"], nl["C"], wl["in_fifo"], wl["const"])
  output_stage(e, g, nl["Y"], conv_stage(e, g, nl["X"], nl["Y"], wl, {**REF_QUANT, **(quant or {})}, pieces))
  return caching_program(g.param_bytes, piece_parts([pieces])), e.program()

def split_pieces(blocks:int, tiles:int|tuple[int, ...], offs:int|tuple[int, ...], limit:int) -> list[Piece]:
  """cut `blocks` 256-byte rows in order into pieces on `tiles` (offsets in 64-byte units, one per tile or shared): every piece but
  the last fills [offset, limit), the last takes the rest (edgetpu_compiler's split rule)"""
  tiles = (tiles,) if isinstance(tiles, int) else tuple(tiles)
  offs = (offs,) * len(tiles) if isinstance(offs, int) else tuple(offs)
  assert len(offs) == len(tiles) and all(o % 4 == 0 and o >= 0 for o in offs), "offsets: multiples of 4 units (256 B)"
  out, left = [], blocks
  for i, (t, o) in enumerate(zip(tiles, offs)):
    n = left if i == len(tiles) - 1 else min(left, (limit - o) // 4)
    if o + 4 * n > limit: raise ValueError(f"{blocks} blocks do not fit tiles {tiles} at {offs} below {limit}")
    if n > 0:
      out.append((t, o, n))
      left -= n
  return out

# *** host-side layout ***
def io_sizes(Mp:int, N:int, K:int) -> tuple[int, int]:
  """(input DMA bytes, output DMA bytes): the input is x[H][W][K] (padded to 8 bytes), the output 16 tiles of out_tile bytes"""
  g = ConvGeom(Mp, N, K)
  return g.S, 16 * g.out_tile

def output_layout(Mp:int, N:int) -> dict:
  """the executable's output_layout (coral.executable): position (y, x) of the H x W grid is at byte tile_byte_offset[y_tile[y] +
  x_tile[x]] + y_local_y_offset[y] * x_local_row_size[x] + x_local_byte_offset[x], N4 = N rounded up to 4 bytes per position"""
  g = ConvGeom(Mp, N, 256)
  return dict(y_tile=[4 * (y // g.hp) for y in range(g.H)], x_tile=[x // g.wp for x in range(g.W)],
              tile_byte_offset=[t * g.out_tile for t in range(16)], x_local_byte_offset=[(x % g.wp) * g.N4 for x in range(g.W)],
              y_local_y_offset=[y % g.hp for y in range(g.H)], x_local_row_size=[g.wp * g.N4] * g.W)

# the tiles edgetpu_compiler uses when a conv compiled alone does not fit one tile (the same in all ~900 multi-tile compiles seen)
COMPILER_SPLIT_TILES = {1: (0,), 2: (1, 3), 3: (3, 7, 15), 4: (7, 15, 14, 13), 5: (15, 14, 13, 12, 7)}

def compiler_tiles(Mp:int, N:int, K:int) -> tuple[int, ...]:
  """the parameter tiles of edgetpu_compiler's alone-compiled program: tile 0 if the blob fits below param_limit, else
  ceil(blocks / capacity) pieces on the tiles of COMPILER_SPLIT_TILES"""
  g = ConvGeom(Mp, N, K)
  n = cdiv(g.blocks, param_limit(g) // 4)
  if n not in COMPILER_SPLIT_TILES: raise NotImplementedError(f"{n} parameter pieces: tile choice not observed")
  return COMPILER_SPLIT_TILES[n]

def gen_conv1x1(Mp:int, N:int, K:int, param_tile:int|tuple[int, ...]=0, param_offset:int=0, quant:dict|None=None,
                param_limit_units:int|None=None) -> tuple[bytes, bytes]:
  """(PARAMETER_CACHING, EXECUTION_ONLY) for y[Mp,N] = x[Mp,K] @ W[N,K].T as a 1x1 CONV_2D over the GRIDS[Mp] grid, byte-identical
  to edgetpu_compiler. The parameters (per 64 outputs: int32 bias, then [K4/4][64][4] weights) live on tile param_tile at byte
  offset param_offset (a multiple of 256) of its wide memory; every call broadcasts them to all tiles. A tuple of tiles splits the
  blob as edgetpu_compiler splits big layers, every tile filled from param_offset up to param_limit_units (64-byte units, default
  and maximum: this program's lowest wide buffer, param_limit()). quant overrides REF_QUANT in the conv op.
  Raises NotImplementedError when the blob does not fit the tile(s) (compiler_tiles() gives the compiler's own split) and outside
  the modelled range (N < 61, more than 256 x 3840 input bytes, or image rows W*K shorter than one 256-byte ring packet)."""
  assert Mp in GRIDS, f"Mp must be one of {list(GRIDS)}"
  tiles = (param_tile,) if isinstance(param_tile, int) else tuple(param_tile)
  assert all(0 <= t < 16 for t in tiles) and len(set(tiles)) == len(tiles), "param_tile: distinct tiles 0..15"
  assert param_offset >= 0 and param_offset % 256 == 0, "param_offset must be a non-negative multiple of 256"
  g, off = ConvGeom(Mp, N, K), param_offset // 64
  if g.N4 < 64: raise NotImplementedError("N < 61: the compiler uses parameter groups of 16..48 outputs (not modelled)")
  if Mp * K > 256 * 3840: raise NotImplementedError("more than 256 x 3840 input bytes: the compiler switches to a mode that is not "
                                                    "modelled (at 256 x 4096 the programs differ; on the device they compute garbage)")
  if g.R < 256: raise NotImplementedError("W*K < 256: image rows shorter than one ring packet (not modelled)")
  limit = param_limit(g) if param_limit_units is None else min(param_limit_units, param_limit(g))
  try: pieces = split_pieces(g.blocks, tiles, off, limit)
  except ValueError:
    raise NotImplementedError(f"Mp={Mp} N={N} K={K}: {g.blocks} parameter blocks at offset {param_offset} B do not fit {len(tiles)} tile(s) "
                              f"of {(limit - off) // 4} blocks (edgetpu_compiler splits them over more tiles)") from None
  return conv_programs(g, pieces, quant)

if __name__ == "__main__": run_test("test_codegen_conv")
