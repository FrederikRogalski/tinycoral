# Batched matmul code generation: 1x1 CONV_2D (`coral/codegen/conv.py`)

`gen_conv1x1(Mp, N, K, param_tile=0, param_offset=0)` builds the PARAMETER_CACHING and EXECUTION_ONLY programs of
`y[Mp,N] = x[Mp,K] @ W[N,K].T` (uint8) run as a 1x1 CONV_2D over the H x W grid `GRIDS[Mp]` (`coral/codegen/conv.py`).
Like `gen_fc` it copies no compiler words: every instruction comes from the encoders in
`docs/isa/{op,wide_narrow,ring_mesh,scalar}.py`, the FC generator's helpers and the rules below, all fitted by diffing
against edgetpu_compiler at the field level. The exceptions are the undecoded scalar-side words of the scalar-memory
output path (two 0x23 templates and 7 vector-slot ALU constants, section 7), which are reproduced as observed.

```
python -m coral.codegen.conv                 # acceptance test (= python test/test_codegen_conv.py), offline, ~2 s when cached
python -m coral.codegen.conv --extra         # + ~120 edge-case compiles
python -m coral.codegen.conv --random 300 --seed 7
```

| API | |
|---|---|
| `gen_conv1x1(Mp, N, K, param_tile=t, param_offset=o)` | parameters on tile t at byte offset o (multiple of 256) |
| `gen_conv1x1(..., param_tile=(t0, t1, ..), param_limit_units=L)` | a split layer (section 9) |
| `gen_conv1x1(..., quant={...})` | main-op quantization (w_zp, in_zp, out_zp, mult, clamp_min, clamp_max) |
| `param_limit(ConvGeom(Mp, N, K))` | 64-byte unit where the program's own high wide-memory buffers start |
| `compiler_tiles(Mp, N, K)` | the tiles edgetpu_compiler uses for this layer compiled alone |
| `io_sizes(Mp, N, K)`, `output_layout(Mp, N)` | host DMA sizes and the executable's `output_layout` |

The parameter blob is the FC blob: per 64 outputs an int32 bias row (256 B), then the weights as [K4/4][64][4]
(`the old fc_params_padded(w, b, 64*ceil(N/64), K4)`, K padded with w_zp). The input is x[H][W][K]
(Mp*K bytes); the output is 16 tile blocks of `ppt*N4` bytes (`output_layout`).

## 1. Verification

| check | result |
|---|---|
| `tools.oracle.compile_conv(Mp, N, K)`, Mp in GRIDS x (N, K) in {(864,288), (288,288), (1536,288), (288,768)} | 20/20 byte-exact |
| random fresh compiles, N 64..2048, K 64..1024, Mp in {16, 64, 256}: seed 0, 100 per Mp (default run) | 167/167 single-tile (133 multi-tile excluded) |
| same, seed 7, 300 per Mp | 509/509 (391 multi-tile excluded) |
| same for Mp in {32, 128}, 200 per Mp; and 200 shapes with N 61..300 over all Mp | 233/233; 200/200 |
| `tools.oracle.coset(Mp)`: the 24 transformer slots per Mp, all 5 Mp | 120/120 (incl. the 10 split slots) |
| excluded multi-tile shapes, generated with `param_tile=compiler_tiles(...)` | 133/133, 391/391, 167/167 |
| `--extra` edge cases | 113/113 (+ 7/7 multi-tile) |
| development corpus: N 61..2048, K up to 1024 (dense K%4 sweeps, layout and boundary probes), W*K >= 256 | 5800/5800 single-tile, 447/447 multi-tile (with `compiler_tiles`) |
| non-reference quantization (zero points 3..250, other scales, fused RELU), `quant=` = the compiler's main-op fields | 3/3: only the main op depends on the quantization |

Every comparison is byte-exact on both programs and also checks the input/output DMA sizes (hints), the output
layout and the parameter blob size. On the device the programs ran bit-exact with the weights on tiles 0, 5, 9 and 14 at
arbitrary 256-byte-aligned offsets (12/12, `bench/RESULTS.md`), and the batched matmuls of `examples/` run on them.

**Excluded** (raise `NotImplementedError`): weights that do not fit one tile with an int `param_tile` (the compiler
splits them; reproducible with a tile tuple, section 9); STAND_ALONE programs (streamed weights: the 32000/16000/8000 x 288
classifiers of the coset, not modelled); N < 61 (parameter groups of 16..48 outputs); W*K < 256, i.e. K < 64 for
Mp in {16, 32} and K < 32 for Mp >= 64 (image row shorter than one ring packet; other staging rules).

## 2. Geometry and notation

- Grid H x W = GRIDS[Mp]; tile (r, c) holds image rows [r*hp, (r+1)*hp) and columns [c*wp, (c+1)*wp):
  hp = H/4, wp = W/4, ppt = hp*wp positions per tile. Ring row group r (tiles 4r..4r+3) receives image rows r*hp.. .
- K4 = 4*ceil(K/4), kw = K4/4 (input words per position), N4 = 4*ceil(N/4), G = ceil(N/64) output groups.
- R = W*K bytes per image row. S_row = hp*R bytes per row group; P = ceil(S_row/256) packets.
- Input FIFO: c_in = ceil(R/256) slots, refills = ceil(P/c_in) passes, p_last = P - c_in*(refills-1).
- blocks = G*(1 + kw) 256-byte parameter blocks; out_tile = ppt*N4 output bytes per tile.
- odd = K % 4 != 0. Cs = 16 (+256 if odd): constant row (+ zero row). slot = max(R, 256).
- Main-op mode: position-outer iff kw <= ppt + 1 (a = kw, b = 1); otherwise K-chunked with
  a = largest divisor of kw that is <= min(15, kw/2), b = kw/a (partial sums). FIFO iff a <= 15.
  Examples: kw 72 -> 12x6, 75 -> 15x5, 125 -> 5x25, 16 -> 8x2, 17 (prime) -> 1x17; Mp=256 with kw 16, 17 is
  position-outer without FIFO.
- Units: narrow byte addresses (TTU words of 4 bytes); wide addresses in 64-byte units, top at 8320.
- TTUs are written either as encoded `(inc, cnt)` pairs (cnt = count - 1, "rew" = the rewinding increment) or as
  logical strides `[s0, s1, ..]` over counts `(c0, c1, ..)`, innermost first (`op.ttu_increments` converts).

## 3. Programs

**Caching** (scalar.md 8.1, one piece per parameter tile; one piece unless split):

```
start, sync_init(caching), scsync_init, host_dma(tag 2, blob), param_pop(blob)
ringConsumer per piece: tile t_i, addr off+2, (1, n_i-1) rewinding, sdims = n_i > 1, nothing else
param_infeed per piece: offset = 256 * (blocks before), 256*n_i bytes, tiles 1<<t_i, 4 lanes (f438 = 3)
sync_final, scsync_fence, interrupt, halt, nop x4, end
```

**Execution** (seq numbers count tile instructions and syncs as in FC):

```
start, exe_prologue, input_head
meshBus 0x17 fill x4 at C+4j (fill 0x01<<8j; in_mode 3 on the 4th unless odd)
[odd: meshBus 0x17 zero fill of 64 words at C+16 (i_sdims 1), sync_drain]
narrowToWide constant row C -> wide `const` (Cs bytes)
sync_wn_fence
[hp > 1: first copy op(s): image rows 0..hp-2]
movi s11/s12, scsync_av_credit, scsync_set_avpop, av_pop(Mp*K), host_dma(tag 1, Mp*K)
[Mp >= 128: sync_w2n]
input wideToNarrow per (row-shift group, tile column)            section 5
(ringConsumer, av_infeed) per tile row r: infeed [floor(r*S_row/8), ceil((r+1)*S_row/8)) x 8 bytes, tiles 0xf<<4r
sync_drain
last copy op(s): image row hp-1
sync_reset17, scsync_nop, sync_drain
ringConsumer1 (weight FIFO), wideToNarrow bias (mode 2), main op, ringProducer per parameter piece
output: host path or scalar-memory path                          section 7
output_wait(16 * (packets per tile if scalar-memory path else 1)), scsync_wait_pb(blocks), sync_rpb, scsync_fence
epilogue, end
```

## 4. Memory layout (compiler allocation rules)

**Narrow** (bytes; Xs = ppt*K4, Ys = ppt*N4, Stg = 2*slot staging that Y aliases, C = Cs constant bytes):

| case | layout |
|---|---|
| Mp = 16, Ys > 2*slot | Y = Stg = 0, C = 2*slot, X = max(Ys, 2*slot + Cs) |
| Mp = 16, Ys <= 2*slot | Y = Stg = 0, X = 2*slot, C = 2*slot + Xs |
| Mp >= 32, Ys >= 2*slot + Cs | Y = Stg = 0, C = 2*slot, X = Ys |
| Mp >= 32, otherwise | X = 0, Stg = Y = Xs, C = Xs + 2*slot |

Boundaries confirmed with N steps of 1 (e.g. Mp=128 K=100: N 200 -> X first, 201 -> Y first; Mp=16 K=64: N 512 / 513).
The two Mp classes differ because only Mp >= 32 has a first copy op.

**Wide** (64-byte units below 8320; all tiles):
- input FIFO `in_fifo` = 8320 - 8*c_in; constant row `const` = in_fifo - 4*ceil(Cs/256).
- compute phase, stacked down from 8320, largest first, ties in the order psum, bias, FIFO:
  psum 4*ppt (K-chunked mode; also reserved, unused, for Mp = 16), bias 8, weight FIFO 8*a.
  Position-outer without FIFO (kw > 15): weights 4*kw at 8320 - 4*kw, bias 4 below.
- output FIFO 8312. `param_limit` = min(const, lowest compute region): parameters must end at or below it.
  Single tile iff off + 4*blocks <= param_limit (checked at equality and +4 for every Mp).

## 5. Input path

- **ringConsumer (row r)**: addr in_fifo, slots c_in, (1, c_in-1)(rewind, refills-1), sdims 1, cbuf 1, mode 3,
  s = (43, -64*c_in, c_in). Single slot: refills = 1 -> (1, 0) cbuf 1 mode 3; refills > 1 -> (0, refills-1), mode 1,
  s_id 11, no cbuf (the FC forward-consumer form). If the last pass is short (p_last < c_in):
  grp = p_last - 1, gstride = 4*(c_in - p_last) + 1 (the FC rule with p = packets of the last pass).
- **wideToNarrow (column c)**: reads its row group's FIFO as one stream that starts sk = shift + floor(c*wp*K/4) words
  in (shift = 1 word for row groups whose infeed starts 4 bytes early: Mp = 16 with odd K, rows 1 and 3; these get their
  own 4 instructions, so 8 in total). With L = R/4 words (one image row) and rem = 64*P - sk:
  rem <= L: one chunk of rem words; else n = ceil(rem/L) chunks of L, the last overhangs the stream:
  head = rem - (n-1)*L, tail = L - head (no head fields when head = L).
  Fields: tile_mask = column c of the row groups, wide_addr in_fifo, wide TTU (1, c_in-1)(rewind, refills-1),
  wide_circ 1, wide_rows c_in, grp/gstride as the consumer (bits 233/249, `rsv227`), narrow_addr Stg/4,
  narrow TTU (1, L-1)(0-stride, n-1), mode 1, size = R/2 in the 18-bit field at bit 489 (`rsv489` + `size64`<<5),
  head_words_m1 = head-1, tail bytes 4*tail in the 18-bit field at bit 524 (`rsv524`, `head_tail16`, `rsv534`),
  skip (sk > 0): skip_base = Stg/4, skip_m1 = sk-1, skip_end = Stg/4 + sk - 1;
  sync (14, wait 1, 0x8000, dec_lvl 1, dec -1, dec_mode 3), tail_f871.
- **Copy ops** (cfg0 7, dp_mode 4): the first one handles image rows 0..hp-2 and is issued before the input DMA,
  re-reading Stg each row; the last handles row hp-1 from Stg + ((hp-1)%2)*slot (the two staging slots alternate).
  loops (kw, wp, rows) at loop1..3; in strides [1, kw, 0] over (kw, wp, rows) from in_base; out [1, 1, kw, wp*kw] over
  (1, kw, wp, rows) from X, or X + (hp-1)*wp*K4 for the last op; par_base = const (14 bits: `par_base` + `par_sel`); 2R in the 21-bit field at
  bit 515 (`rsv515` + `in_twait`<<7); sync1 = 0x4000 + floor(wp*K/4); in_tflags 0x80, out_tflags 3, mult 1.0, clamps +-inf.
  **Progress count** (sync0, cfg1 bits 8-9): T64 = 16*wp*K (first op) or 16*(hp-1)*wp*K + 16*ceil((R + 4*ph)/16)
  (last op; ph = start byte of its positions in the staging word); sync0 = (wp*K % 4) << 14 | T64//64,
  cfg1 = base | ((T64 % 64)//16) << 8.
- **Odd K: byte-shifting copy ops** (cfg0 9, cfg1 base 0x8B, cfg3 7, reduce_mask 1, in_hmode 1, in_tflags 0xC1,
  out_tflags 7 + bit 893 when rows > 1). One op per distinct start byte ph = (c*wp*K) % 4, in column order, with the
  tile mask of those columns and in_base = Stg + ph (+ slot). loops (4 bytes, kw, wp, 1, rows). The in TTU walks bytes:
  strides [1, K, R, 0] counts [K, wp, 1, rows]; each increment is stored as inc >> 2, and the in_mode slots of dims 1..6
  all hold (1 - wp*K) % 4 = the low bits of the outer increments (dim 0: 1, dim 7: 3). out strides
  [1, 1, kw, wp*kw, (wp*kw if rows > 1), hp*wp*kw]; par (zero row) counts (4, kw, wp, rows), incs (0, -1, ...), modes 1,
  par_hmode 1; psum counts the same, psum_tflags 0x840. With r = (K-1)%4, t = K4-K: r at bits 174, 1229 (2 bits),
  1532 (2 bits); t at bit 1246 (2 bits); constant 1s at 188, 1243, 1546 (in `rsv174`, `par_fifo`, `rsv1230`, `rsv1523`).
  The 256-byte zero row (narrow C+16, copied to wide with the constant row: 272 bytes, 2 wide rows, no sync) supplies
  the K4-K padding bytes.

## 6. Compute

- **ringConsumer1** (0x12, all tiles) at the weight FIFO: K-chunked (1,a-1)(rew,b-1)(rew,0)(rew,G-1), slots a,
  aux (bias+2, bias+4); a = 1: (0,kw-1)(0,0)(0,G-1), slots 1, mode 1, s_id 1, no sdims/cbuf, aux (bias+1, bias+4);
  position-outer (1,kw-1)(rew,0)(rew,G-1): FIFO slots kw, aux (bias+1, bias+4); no FIFO: slots 0, aux (bias+1, bias),
  s_val 0. Otherwise sdims 1, cbuf 1, mode 3, s = (33, -1, en_a, en_b, y).
- **Bias wideToNarrow** (mode 2): wide bias, (1,0)(0,G-1), narrow 0x40 with strides [1, 64, 0] over (32, 1, G),
  wide_circ = wide_rows = FIFO,
  sync_id 15 (RING_READ_B), dec_lvl 2, dec -1, and b << 15 in the field at bit 624 (`sync_val` + `rsv643`):
  the number of FIFO fills per group.
- **Main op** (cfg0 0xE5, dp_mode 4, sync0/1 0x4000, sync2 0xC6, sync3/4 0x80, cfg4 1, out_ch 63):
  K-chunked: loops (a, 1, ppt, b, G); in strides [1, kw, kw, a, 0]; par [1, a, 0, 0, 0] at the FIFO, par_fifo a,
  par_tflags 0xC0; psum [0, 1, 0, 0] over (a, ppt, b, G) at the psum region, psum_mode7 2,
  psum_tflags 0x8C0 iff ppt = 1 and G > 1; cfg1 0x16F; cfg2 7 (ppt = 1), 3, or 2 when a = 1; psum_hmode = (a = 1);
  reduce_mask 0b1011. Position-outer: loops (kw, 1, 1, ppt, G); in [1, kw, kw, kw, 0]; par [1, kw, 0, 0], par_fifo kw
  or 0, par_tflags 0xC0 or 0; psum counts (kw, ppt, G) strides 0, psum_tflags 0x840; cfg1 0x18F, cfg2 7,
  reduce_mask 0b111. Both: out [1, N4/4, 16] over (16, ppt, G) at Y; a partial last group (N4 % 64) sets
  out_last_cnt = w-1, mode 2, skip 16-w (w = its words), out_ch_last = its outputs - 1.
- **Weight broadcast**: per parameter piece a ringProducer on its tile, addr off, (1, n_i-1), pcfg 4,
  r0 = (50 = RING_PRODUCER_B/op1, first block of the piece), to_c1, dest 0xffff. scsync_wait_pb counts all blocks.

## 7. Output

- **narrowToWide** (all tiles): Y, levels [N4/4 words] + [wp, stride N4/4] + [hp, stride wp*N4/4] (levels of count 1
  dropped), tail_lvl = levels, to the output FIFO 8312; one wide row -> (1,0) circ; more -> (0, rows-1) (FC rule).
- **Host path** (out_tile % 8 == 0, all Mp >= 32): sync_signal + scsync_fence, output DMA 16*out_tile, then per tile
  (outfeed(out_tile), ringProducer ordinal t) exactly as FC.
- **Scalar-memory path** (Mp = 16 with N4 % 8 == 4): sync_drain instead of the signal; per tile a ringProducer with
  pcfg 12, r0 = (49 = RING_PRODUCER_A/op1, t*rows), then scsync_smem_pre and outfeeds into a 4096-word scalar-memory
  buffer (base = fill level, d0_limit = n-1, d1/d2 stride 1-n, d3_stride 1, f222 0x3f). A tile that does not fit puts the
  largest multiple of 64 words that fits, then the buffer is flushed: scsync_smem_post, host DMA of floor(used/8)*8 words,
  movi s6 0, words 0x131f800 0x1323800, 0x200008c0 | (bytes/8 - 1) << 12, 0xe00040131f800 0x10000401323800
  0xe000400000800 0x10000400000800, host pointer += bytes (add64(4,5,bytes,4,5,6)), and if 1..7 words remain:
  movi s7 = flushed words, movi s8 0, 0x300008c0 | left << 12, 0x139b800, 0x800, 0xc028300000800 (they move to the
  buffer start). The last flush follows the 16th tile. PRODUCER_A threshold = 16*rows.

## 8. Rules missing from (or added to) the decode reports

1. `op.par_sel` / `op.psum_sel` are bit 13 of 14-bit wide addresses (`par_base`/`psum_base`): 8244 = sel 1, base 52.
2. wideToNarrow: bits 233/249 (`rsv227`) are grp/gstride as in the ring consumer; the size field is 18 bits at 489 in
   2-byte units (`size64` is its high part); the head tail is a byte count at bit 524 spanning `rsv524`, `head_tail16`
   and `rsv534`; `sync_val` continues into `rsv643` (the conv parameter stream count b << 15).
3. op: `rsv515` + `in_twait` form one field at 515 (2R for copy ops); in byte-mode TTUs (`in_hmode` 1) the increments
   are bytes >> 2 and the per-dim mode slots carry the low two bits (dim 0: 1; dims 1..6: those of the outer rewind);
   `rsv893` is a 4th out_tflags bit.
4. Partial input FIFO passes: grp/gstride follow the FC formula with p = packets of the last pass.
5. FC forward-consumer form (single slot, refills > 1) also applies to input and weight consumers.
6. Caching consumer with a single block: sdims 0.
7. The ringProducer output ordinal counts packets (t*rows) in the scalar-memory path, tiles otherwise.

## 9. Parameter placement (param_tile / param_offset) and split layers

- **One tile**: compiled alone, tile 0 offset 0. Co-compiled (coset): tile t, offset o (64-byte units, always a
  multiple of 4 = 256 B): caching consumer `tile_mask 1<<t, addr o+2`, infeed `tiles 1<<t`; execution broadcast
  ringProducer `tile_mask 1<<t, addr o`. Nothing else changes (120/120 coset slots).
- **Split rule** (verified on all 10 split coset slots and on every alone-compiled multi-tile program seen: 447 in the
  development corpus, 133 + 391 + 167 + 7 in the random and extra runs): the blocks are
  cut in order into pieces; piece i goes to tile t_i at the same offset o; every piece but the last fills
  [o, L) with L = the lowest `param_limit` of the programs that share the chip (a coset: min over the set; alone: the
  program's own). Caching: one consumer per piece (addr o+2, (1, n_i-1)), then one infeed per piece (offset 256 *
  blocks before). Execution: one broadcast ringProducer per piece, consecutive seq numbers, r0_val = first block of the
  piece. Tile choice when compiled alone (`compiler_tiles`): 2 pieces (1, 3), 3 (3, 7, 15), 4 (7, 15, 14, 13),
  5 (15, 14, 13, 12, 7); never more than 5 for N <= 2048, K <= 1024. In a coset the compiler packs freely, e.g.
  Mp=16 slot 18: tiles (6, 2) at 93440 B, 1688 + 64 blocks.
- **Runner contract**: every resident parameter region must end at or below the lowest `param_limit` of all programs
  that run while it is resident (each execution program uses the wide memory above its limit on every tile). With an
  int param_tile the generator only checks the program's own limit (or `param_limit_units`).

## 10. Open questions

1. Semantics of the copy-op progress count (T64), of `rsv174`/`rsv1230`/`rsv1523` in byte-shifting copies, and of the
   +4*ph term.
2. The scalar-memory path words (0x23 and the vector-slot ALU words) are not decoded.
3. Why the compiler switches to position-outer at kw <= ppt + 1 and limits FIFO fills to <= 15 rows and b >= 2.
4. STAND_ALONE (streamed) conv programs, N < 61 and W*K < 256 are not modelled.
5. The compiler's tile choice for 6+ pieces (outside N <= 2048, K <= 1024).
