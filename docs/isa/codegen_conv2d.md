# CONV_2D code generation (`coral/codegen/conv2d.py`)

`gen_conv2d(H, W, Cin, Cout, kh, kw, stride, padding)` builds the PARAMETER_CACHING and EXECUTION_ONLY programs of a uint8
CONV_2D `x[1,H,W,Cin] * w[Cout,kh,kw,Cin] -> y[1,OH,OW,Cout]` (k x k kernels, stride 1 or 2, VALID or SAME), byte-identical to
edgetpu_compiler for `tools.tflite_gen.conv_model(w, bias, H=H, W=W, stride=s, padding=...)` compiled alone. Like `gen_fc` and
`gen_conv1x1` it copies no compiler words: every instruction comes from the encoders in `coral/isa/` and the rules below, all fitted
by diffing compiled programs field by field. The exceptions are the undecoded scalar-side words of the scalar-memory output path
(0x23 and vector-slot ALU words, section 9), reproduced as observed, as in `codegen/conv.py`.

```
python -m coral.codegen.conv2d               # acceptance test (= python test/test_codegen_conv2d.py), offline
python -m coral.codegen.conv2d --n 150 --seed 1
python -m coral.codegen.conv2d --classes k3s1v,k5s2s
```

| API | |
|---|---|
| `gen_conv2d(H, W, Cin, Cout, kh, kw, stride=1, padding="VALID", param_tile=0, param_offset=0, quant=None)` | `(caching, execution)` |
| `gen_conv2d(..., param_tile=(t0, t1, ..), param_limit_units=L)` | a layer split over tiles (section 8); `compiler_tiles(g)` is the compiler's choice |
| `conv2d_io(H, W, Cin, Cout, kh, kw, stride, padding)` | `dict(input_bytes, output_bytes, output_layout, param_bytes)`: the host contract |
| `conv2d_blob(w_u8[Cout,kh,kw,Cin], w_zp, bias_i32=None, H, W, stride, padding)` | the parameter blob (section 10) |
| `conv2d_quant(x_q, w_q, y_q, act=0)` | the conv op's quantization fields for `quant=` (default: `conv_model`'s (1/128,128), (1/128,128), (1/16,128)) |
| `Conv2DGeom`, `narrow_layout`, `wide_layout`, `param_limit` | the geometry and memory rules below |

## 1. Verification

All offline: every comparison is byte-exact on both programs, the parameter blob (random weights, random int32 biases where noted),
the DMA sizes of the hints and the executable's `output_layout`.

| check | result |
|---|---|
| the 4 MNIST CNN convolutions (28x28x1 5x5 -> 24x24x32, 24x24x32 5x5 -> 20x20x32, 10x10x32 3x3 -> 8x8x64, 8x8x64 3x3 -> 6x6x64), random weights and biases | 4/4 |
| 12 shape classes (1x1/3x3/5x5 x stride 1/2 x VALID/SAME), H, W in 8..64, Cin in 1..128, Cout in 8..256: default run (seed 0, 30 per class) | 353/353, + 6/6 layers split over tiles (`compiler_tiles`), 1 excluded |
| same, seed 1, 150 per class | 1751/1751, + 47/47 split, 2 excluded |
| same, seed 2, 100 per class | 1170/1170, + 26/26 split, 4 excluded |
| other quantizations (zero points 0..250, scales, fused RELU / RELU6), via `conv2d_quant` | 5/5 |
| 20 convs co-compiled in one compiler call (tiles 0..15, then offsets 512 / 768 B): regenerated from the compiler's (tile, offset) | 20/20 |
| extremes: 64x64x128 -> 256, 8x8x1 -> 8, 64x8, 8x64, 63x61, 9x9x128 for every kernel / stride / padding | 64/64 (8 split) |
| not required, sampled: 3x5, 5x3, 1x3, 3x1, 2x2, 4x4, 7x7 kernels | 7/7 |
| layout and mode rules on their own (`narrow_layout`, `pos_outer` / FIFO choice) over extra sweeps | 1500/1500, 1600/1600 |

On the device (`test/test_hw_conv2d.py`): the four MNIST convolutions and 7 random shapes ran bit-exact on the first try. Tall stacks
of images per call, outputs through scalar memory (48 random shapes) and weights split over several tiles (CIFAR-10) followed
(`bench/RESULTS.md`).

**Excluded** (raise `NotImplementedError`): 1x1 kernels with Cin <= 4 and Cout > 64 where some tile has a single output position
(the compiler uses a different op class there, cfg0 0x127 / dp_mode 5, processing the channel groups in pairs; not decoded);
blobs that do not fit `param_tile` (pass `compiler_tiles(...)` to split as the compiler does). No shape of the tested range was
streamed (STAND_ALONE); the test counts such shapes as excluded.

## 2. Geometry (`Conv2DGeom`, `Span`, `axis`)

- Inputs and outputs are both split over the 4 tile rows / columns with `eltwise.tile_split`: tile (r, c) holds image rows
  `[i0, i0+ni)` and computes output rows `[o0, o0+no)` (likewise columns). SAME: `pad = max((O-1)s + k - D, 0) // 2` before.
- Its window is `[w0, w0+nw)` with `w0 = o0*s - pad`, `nw = (no-1)*s + k` (none if no = 0).
- Halo chains: `hi[p] = max(i0+ni, min(w0+nw, D) if no, hi[p-1])`, `lo[p] = min(i0, max(w0, 0) if no, lo[p+1])`. A span sends its
  neighbour the rows that neighbour needs, relaying rows of the span beyond when it holds fewer (multi-hop, e.g. 5x5 kernels on
  2-row tiles).
- **Block.** Every tile stores the block `[b0, b0+nb)` = its window U `[lo, hi)` (own rows, halos, relayed rows); a tile without
  outputs only `[lo, hi)`. With stride 2 the window can miss own rows; the block still covers them. Positions are C4 = 4*ceil(Cin/4)
  bytes, block rows `nb_c * C4` bytes. `d = i0 - b0` and `wo = w0 - b0` locate the own rows and the window in the block.
- `R = W*Cin` bytes per image row; `c_in = ceil(R/256)` ring FIFO slots.
- **Chunks.** The input arrives as one byte stream per row group; tiles cut it into chunks of `m = 4/gcd(R,4)` image rows (a whole
  number of words), at most the group's rows: `L(r) = ceil(m(r)*R/4)` words, `Lf = m*R/4` for a full chunk, `chunks(r) =
  ceil(ni/m)`. The staging slot is `slot = max(4*Lf, 256)` bytes.
- K = `kwords = kh*kw*C4/4` words per output; groups `G = ceil(Cout/64)`; op groups of `min(64, N4)` outputs, blob groups of
  `cg = round_up(Cout, 16)` (Cout <= 64) or 64.

## 3. Programs

**Caching** is `codegen.caching_program` (as `gen_conv1x1`): one ringConsumer (addr `off + 2`, `(1, rows)`) and one parameter infeed
per piece, rows of `4*cg` bytes (`f438 = cg/16 - 1`).

**Execution** (seq numbers count tile instructions and syncs as in FC):

```
start, exe_prologue, input_head
identity row: 4 meshBus 0x17 fills at C (+ a 64-word zero row and a drain for Cin % 4 != 0), narrowToWide C -> wide const   (eltwise.ident_prologue)
sync_wn_fence
first copy ops            all chunks but the last of every tile with >= 2 chunks                        section 5
input DMA                 movi s11/s12, credit, avDataPop(S), host DMA (S = 8*ceil(H*W*Cin/8))
[sync WIDE_TO_NARROW]     tiles of the row groups whose ring FIFO is refilled >= 3 times (counter only, units 0)
input wideToNarrows       first the row groups without that fence, then the fenced ones                   section 4
(ringConsumer, infeed) per row group r: infeed [floor(lo/8)*8, ceil((lo+n)/8)*8), lo = i0*R, n = ni*R, tiles 0xf << 4r
sync_drain
last copy ops             the last chunk of every tile
sync_reset17
halo exchange             meshBus north, south, west, east                                               section 6
scsync_nop, sync_drain
ringConsumer1, bias wideToNarrow (mode 2), conv op per output block shape, ringProducer per parameter piece  section 7
output                    section 9
output_wait(n), scsync_wait_pb(rows), sync_rpb, scsync_fence, epilogue
```

**Instruction grouping.** Everything per tile is computed tile by tile; tiles whose instruction fields are identical share one
instruction (OR-ed mask), in the order of their lowest tile. Count-1 loop levels still carry their stride in the encoded increment,
so e.g. a 1-row copy op of a 4-wide and of a 3-wide block differ. The output narrowToWides are the exception: one per output block
shape (rows, cols), even when two shapes encode the same (2x1 and 1x2). Tiles without outputs take no part in the compute
(ringConsumer1 / bias / op masks and the parameter broadcast's `dest` are the tiles with outputs) nor in the horizontal halo phase.

## 4. Memory

**Narrow** (bytes, every tile): `Xs` = largest block, `Ys` = largest output block (`positions * N4`), `Cs = 16 (+256 if Cin % 4)`
(identity + zero row), staging `2*slot`. The placements below all need the same memory; the compiler's choice (fitted on 1500 shapes):

| condition (in order) | X (blocks) | Stg | Y | C |
|---|---|---|---|---|
| `Ys >= 2*slot + Cs and Ys > Xs` | Ys | 0 | 0 | 2*slot |
| `2*Xs >= slot` | 0 | Xs | Xs | Xs + 2*slot |
| `Ys > 2*slot or 4*Xs < Cs` | max(Ys, 2*slot + Cs) | 0 | 0 | 2*slot |
| otherwise | 2*slot | 0 | 0 | 2*slot + Xs |

Y always reuses the staging (they are never live together). Tile (r, c)'s block is at X with row stride `nb_c*C4`.

**Wide** (64-byte units, every tile):
- Input phase, stacked down from 8320, larger first, the identity row on a tie: the input ring FIFO (`8*c_in`) and the identity row
  `const` (`4*ceil(Cs/256)`).
- Compute phase, also from 8320: partial sums `4*ppt_max` (K-chunked only), bias (8, or 4 without a FIFO), weights (FIFO `8*a` or
  block `4*a`), largest first, ties psum, bias, weights (`codegen/conv.py`'s rule).
- `param_limit` = the lowest of these; the parameters must end at or below it. Output FIFO 8312.

## 5. Input path

**Ring.** Row group r's infeed carries its rows (8-byte aligned, so it may start 1..7 bytes early); `P = ceil(infeed bytes / 256)`
packets pass `refills = ceil(P/c_in)` times through the FIFO, a short last pass sets `grp = p_last - 1`, `gstride =
4*(c_in - p_last) + 1` (ringConsumer and wideToNarrow, `codegen/conv.py`). ringConsumer: `codegen/conv.py`'s three forms.

**wideToNarrow (per tile).** Starts `sk = off // 4` words into the stream, `off = (i0*R) % 8 + ic0*Cin` (ic0: first own column);
byte phase `ph = off % 4`. Chunks of `L(r)` words go to the staging (stride 0), `rem = 64P - sk`:
- `rem <= L`: one chunk of `rem` words;
- `ceil(rem/L) <= chunks + 1`: `n = ceil(rem/L)`, head `h = rem - (n-1)L`, no head fields when `h = L`;
- else `n = chunks + 1` (the last chunk drains the packets), `h = rem - (n-1)L`, minus L when h is a multiple of L.
Head fields: `head_words_m1 = h - 1`, tail bytes `4(L - h)` (18-bit, negative = skip) at bit 524; size field (bit 489, 2-byte
units) `max(4L, 256)/2`; skip fields when `sk > 0` (codegen/conv.py).

**Copy ops** move the tile's own positions of each chunk from the staging into its block, padding channels to words through the
identity row at `const`. First copy op: chunks `0..n-2` (from slot 0), last copy op: chunk `n-1` (from slot `(n-1) % 2`), written
at block row `d (+ m(n-1))`, column `d_c`. Progress fields: `words = L` (first) or `(n-1)L + ceil(((rows_last-1)R + ph + iwt*Cin)/4)`
(last); `sync0 = (Lf % 4) << 14 | words // 4`, cfg1 bits 8-9 = `words % 4`, `sync1 = 0x4000 + Lf // 4`, `2*slot` at bit 515.

| variant | when | main loops | in TTU (bytes) | out TTU (words) | class fields |
|---|---|---|---|---|---|
| word copy | Cin % 4 == 0 | (1, cw, iwt, chunks) | [4, C4, 0] x [cw, iwt, chunks] | [1, 1, cw, nb_c*cw] x [1, cw, iwt, chunks] | cfg0 7, cfg1 0x6B, dp_mode 4, in_tflags 0x80, out_tflags 3; par/psum header mode 1 iff cw = 1 |
| narrow copy | Cin <= 3 | (Cin, iwt, m rows, chunks) | [1, Cin, R, 0] x [Cin, iwt, m, chunks] | [1, 1, nb_c, (m*nb_c), nb_r*nb_c] x [1, iwt, m, (chunks), 1] | cfg0 7, dp_mode 3, cfg3 7, reduce 1, in_tflags 0xC1, out_tflags 7 (3 for one chunk) |
| byte-shift copy | Cin > 4, Cin % 4 != 0 | (4, cw, iwt, m rows, chunks) | as narrow | [1, 1, cw, rw, (m*rw), nb_r*rw] | `codegen/conv.py`'s shift copy (cfg0 9) with m rows per chunk |

## 6. Halo exchange (meshBus)

Vertical first (each tile's own columns), then horizontal (whole window rows, so the corners travel twice). Per direction one
instruction per distinct tile; each holds the tile's outbound half (rows its neighbour needs) and inbound half (rows from the
opposite neighbour). North (0x17): tile p sends `[i0, hi[p-1])` from block row d, receives `[i0+ni, hi[p])`; south (0x15): sends
`[lo[p+1], i0+ni)`, receives `[lo[p], i0)`; west / east the same on columns.
- **Padding.** At the image edges the inbound half is a fill (`fill_en`, `fill = in_zp * 0x01010101`) of the window's padding.
- **TTU levels** `[(1, words), (positions), (rows)]`, count-1 levels dropped; `sdims = 2^levels - 1`, `in_mode = 2*levels + 1`.
  A level moves at most 16 words: positions of more words go in 16-word pieces over an outermost level `(16, pieces)` (not counted
  in in_mode); a short last piece of r words sets `grp = r - 1` and the field at bit 249 (+205 inbound) to `4(16 - r) + its level`
  (FC relays use the same field).
- **Sync records** on outbound halves, value `4 + levels` (pieces level not counted): the same direction's IN counter when the tile
  relays (sends rows it receives in this phase), then, horizontally, MESH_NORTH_IN / MESH_SOUTH_IN for every vertical inbound
  (received or filled) of that tile. No fences between the moves.

## 7. Compute

**Mode.** Position-outer (all K per weight fill, positions outside, no partial sums) or K-chunked (fills of `a` rows, partial sums
per position across the `b = K/a` fills). Fill rows: K-chunked `a = kh*kw` (one channel word of every tap), for 1x1 kernels
`codegen/conv.py`'s largest divisor of cw <= min(15, cw/2); position-outer `a = K`. A fill streams through an a-slot FIFO iff
`a <= 15 or a >= 31` (16..30 rows sit in a plain block). Choice (1600/1600 compiles):
- Cin <= 4: position-outer;
- `K <= 30`: position-outer iff `K <= ppt_max + kh*kw` (`codegen/conv.py`'s `kw <= ppt + 1`);
- `K >= 31`: position-outer iff its footprint `8K + 8` < `4*ppt_max + (8a + 8 | 4a + 4)` of the K-chunked op.

**ringConsumer1** (tiles with outputs): K-chunked `(1, a)(0, b)(0, 1)(0, G)`, position-outer `(1, a)(0, 1)(0, G)`; FIFO: slots a,
`s_val -1`, aux `(bias + 2 | 1, bias + 4)`; block: aux `(bias + 2 | 1, bias)`; both sdims/cbuf 1, mode 3, s_id 33. A one-row fill
(a = 1) takes the single-slot form: mode 1, s_id 1, slots 1, aux `(bias + 1 | 0, bias + 4)`, `(0, K)(0,1)(0,G)` or `(1,1)(0,G)`.

**Bias load**: `codegen/conv.py`'s mode-2 wideToNarrow (b fills per group in the field at bit 639).

**Conv op** (cfg0 0xE5, dp_mode 4, sync0/1 0x4000, sync2 0xC6, sync3/4 0x80, cfg4 1), one per output block shape:

| | K-chunked | position-outer |
|---|---|---|
| loops | (aw, taps, ppt, b, G), aw = a/taps | (cw, taps, 1, ppt, G) |
| in (words, from X + (wo_r*nb_c + wo_c)*C4) | [(1, aw)] + taps + positions + [(aw, b), (0, G)] | [(1, cw)] + taps + [(cw, 1)] + positions + [(0, G)] |
| par | [1, aw, 0, 0, 0] x [aw, taps, ppt, b, G] | [1, cw, 0, 0] x [cw, taps, ppt, G] |
| psum | [0, 4 units] x [a, ppt, b, G] at the psum region (none if ppt = 1) | [0] x [K, ppt, G] (header mode 1 iff K = 1) |
| cfg1 / cfg2 / reduce | 0x16F / 7 (ppt 1), 2 (a 1), else 3 / 0b1011 | 0x18F / 6 if K = 1 and ppt > 1, else 7 / 0b111 |
| psum_tflags | 0x8C0 iff ppt = 1 and G > 1, else 0 | 0x840 (ppt > 1), 0x880 (ppt 1, G > 1), 0 |

taps = `[(cw, kw), (nb_c*cw, kh)]`, positions = `[(s*cw, owt), (s*nb_c*cw, oht)]`; within each, contiguous levels merge and trailing
count-1 levels drop (one kept). `in_tflags = 2^(levels-2) - 1` (levels without G). Output: `[1, N4/4, wg]` x `[wg, ppt, G]`
words at Y, groups of `min(64, N4)`, a short last group as in `codegen/conv.py`. `par_tflags 0xC0` / `par_fifo a` with a FIFO.

**Weight broadcast**: one ringProducer per parameter piece (`codegen/conv.py`'s `rprod_param`), `dest` = the tiles with outputs.

## 8. Parameter placement

As `gen_conv1x1` (codegen_conv.md section 9): tile `param_tile` at `param_offset`; the caching consumer's addr, the broadcast
producer's addr and nothing else move. A blob that does not fit is cut into pieces that fill `[offset, param_limit)` on each tile;
the compiler's alone-compiled tiles are `COMPILER_SPLIT_TILES` (2 pieces: tiles 1, 3). Verified on 20 co-compiled convs (section 1).

## 9. Output

- narrowToWide per output block shape: Y, levels `[N4/4 words] + [owt, N4/4] + [oht, owt*N4/4]` (count-1 levels dropped) to 8312.
- **Host path**: `signal_fence`, one host DMA of `sum(round_up(tile bytes, 8))`, then per tile with outputs (outfeed, ringProducer
  with ordinal k), `output_wait(tiles)`.
- **Scalar-memory path** iff a tile block other than the last one is not a multiple of 8 bytes (an odd last block is padded on the
  host path): `codegen/conv.py`'s packing into the 4096-word buffer with per-tile sizes, producers with the packet ordinals, and a
  final flush whose DMA is rounded up to 8 bytes; for an odd word count six more words copy the last one
  (`0x131f800, 0x800, 0xe000400000800, movi s6 0, 0x100018c0, 0xc000400000800`, reproduced as observed).

## 10. Host contract

- **Input**: x[H][W][Cin] uint8, row-major (NHWC, batch 1), one DMA of `8*ceil(H*W*Cin/8)` bytes (pad with anything).
- **Output**: 16 tile blocks in tile order (tiles without outputs are empty), each `[oht][owt][N4]` (N4 = Cout rounded up to 4; the
  extra channels are padding), each padded to 8 bytes on the host path, packed back to back on the scalar-memory path; total
  rounded up to 8. `output_layout` is the executable's: output (y, x) channel c sits at
  `tile_byte_offset[y_tile[y] + x_tile[x]] + y_local_y_offset[y]*x_local_row_size[x] + x_local_byte_offset[x] + c`.
- **Blob** (`conv2d_blob`): per group of cg outputs an int32 bias row (4cg bytes), then the weights as `[K/4][cg][4]` uint8;
  K-chunked convs order K as [channel word][tap], position-outer convs as [tap][channel word] (so the blob depends on the image:
  pass H, W, stride, padding when kh*kw > 1 and Cin > 4). Channels are padded to 4 with w_zp; outputs Cout..N4-1 are all w_zp, the
  rest of the last group 0. Rows are 4cg bytes; each takes one 256-byte wide row on the parameter tile.

## 11. Open questions

1. Semantics of the narrow allocator's tie-break (section 4) and of the FIFO choice (`a <= 15 or a >= 31`).
2. The position-outer threshold for 7x7 kernels with cw >= 2 did not fit the footprint rule in a probe (crossover near ppt 48
   instead of 99); 7x7 is outside the target set.
3. The paired-group op (cfg0 0x127, dp_mode 5) for 1x1 kernels with Cin <= 4, Cout > 64 and single-position tiles.
4. The scalar-memory path's 0x23 / vector-slot words.
5. Why the input wideToNarrow chunk count is capped at chunks + 1 and the head rule drops one L for multiples of L.
6. Streamed weights (STAND_ALONE executables, parameters that do not stay on chip): not generated; no shape of the tested
   range needed them.

## 12. Risks for hardware

- With the compiler's placements the programs are byte-identical to the compiler's, so they are exactly as trustworthy as those.
- Weights are broadcast from one tile every call (as `gen_conv1x1`): the runner must keep every resident blob below the lowest
  `param_limit` of the programs that run while it is resident.
- Blob groups of 16..48 outputs (Cout <= 48) use 64..192-byte rows: the FC rule "small output groups read wrong weights at >= 256 KiB"
  may apply; keep such blobs below 256 KiB.
- The output stream is tile-blocked (section 10), not NHWC: the host must reorder with `output_layout`, and for the scalar-memory
  path the blocks are packed without padding.
