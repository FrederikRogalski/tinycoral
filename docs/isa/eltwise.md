# Elementwise and pooling ops: LOGISTIC, RELU-type, MUL, ADD, MAX_POOL_2D, and the data flow between chained ops

This report extends `op.md` (the 17-word tile op, opcode 0x01) to the non-matmul operations, and documents the instructions that move data between chained ops.

- **Code:** `coral/isa/eltwise.py`.
  - Encoder and decoder for the new opcode 0x19 (NLU spline load).
  - Byte-granular TTU helpers (`ttu_get`, `ttu_strides`, `ttu_set`).
  - Placement rules (`tile_split`, `tile_block`, `chunk_1d`, and `tile_geometry(shape, tile)`, which gives P, w and R for any tensor).
  - Float32 quantization formulas (`*_quant`, `out_clamps`, `add_weights`).
  - Builders for every op variant, and for the data-flow instructions around them.
  - `classify`, `op_params` and `rebuild`.
  - `split()` with the corrected instruction lengths.
- **Test:** `test/test_eltwise.py` compiles 198 models (cached in `.compile/`) and rebuilds every instruction of these kinds byte for byte.
- **Scope:** everything is offline. Nothing here has been run on hardware.

## 0. Results and the main findings

All pass counts are from `python test/test_eltwise.py`: 198 programs, 0 failures. For the "model" check:
- geometry comes from the TFLite shapes;
- quantization comes from the TFLite scales and zero points;
- only allocation is read from the instruction: tile mask, `seq`, narrow bases, wide addresses, and the ADD operand distance.

| instruction | from model + allocation | full read-back round trip |
|---|---|---|
| 0x19 NLU spline load | 41/41 | 41/41 |
| LOGISTIC op | 246/246 | 246/246 |
| RELU / RELU6 / RELU_N1_TO_1 op | 191/191 | 191/191 |
| MUL step 1 (multiply) | 134/134 | 134/134 |
| MUL step 2 (pack) | 145/145 | 145/145 |
| ADD op | 109/109 | 109/109 |
| MAX_POOL_2D op (opcode 0x02) | 426/426 (390 of them in the pooled convs) | 426/426 |
| identity-row prologue (4 mesh fills + narrowToWide) | 155/155 | |
| MUL operand FIFO feed (narrowToWide) | 134/134 | |
| ADD weight matrix (10 mesh fills + narrowToWide) | 45/45 | |
| MAX_POOL relay (transposing narrowToWide, standalone pools) | 38/38 | |

### Corpus
- **LOGISTIC:**
  - 1-D n = 4..16384;
  - 4-D shapes, including C = 5 and the uneven 7x7 split;
  - input quantizations, and output zero point 10.
- **RELU, RELU6, RELU_N1_TO_1:** 1-D and 4-D, three quantization pairs each.
- **MUL:**
  - n = 1..4096, which covers FIFO depths 1-4 and every partial last round;
  - 4-D shapes;
  - four quantization triples × four fused activations.
- **ADD:**
  - sizes and 4-D shapes;
  - 18 scale ratios, with integer weights up to 25319;
  - quantization × activation.
- **MAX_POOL_2D:**
  - k = 2..8 and s = 1..5;
  - C = 3..256, so 1-4 channel groups and partial groups;
  - zero points 0, 100, 255.
- **Chains:**
  - FC→LOGISTIC, FC→FC, FC→RELU, FC→LOGISTIC→x·x;
  - two FCs feeding MUL or ADD.
- **The three target models:**
  - the FFN block in conv form (grids 4×4, 16×8, 16×16, plus 2×2 and 8×8, and a C=64/F=128 variant) and in FC form;
  - the pooled conv with M = 16, 64 and 256.

### Findings
1. **LOGISTIC has no LUT and no parameter blob.** A standalone LOGISTIC program has 0 parameter bytes and no extra wideToNarrow or ringConsumer.
   - The sigmoid is an 8-segment piecewise quartic polynomial. Its 40 f32 coefficients and 7 f32 breakpoints are immediates of a new 13-word tile instruction, opcode **0x19**, which loads the tiles' non-linear unit (NLU).
   - The table is the same constant in every program. All quantization lives in the op instruction: mult = input scale, input zp, out_zp, and a fixed input clamp of ±10.396732.
2. **MUL of two tensors is two op instructions.**
   - Operand b is first streamed by a narrowToWide into a 1-4 row FIFO in wide memory, one 4-byte word per 256-byte row.
   - Step 1 (dp_mode 4, cfg0 0x60) multiplies one byte of a per step by the FIFO row. It requantizes (in_zp = a_zp, w_zp = b_zp, mult = a_s·b_s/y_s, clamps, out_zp) and writes one 4-byte word per element.
   - Step 2 (dp_mode 2) packs 4 words back into one, lane-selecting against the 4x4 identity.
3. **ADD is one op in 16-bit mode (dp_mode 5).** Each input word is read as two 16-bit halves (two bytes zero-extended), and the four steps per output word use 16-bit integer weights.
   - The weights (w1, w2) are the best rational approximation of s2/s1 with terms ≤ 32767.
   - They are written into narrow memory by 10 mesh "fill" instructions and moved to wide memory by a narrowToWide.
   - The op carries offset = −(w1·z1 + w2·z2), mult = f32(s1/w1)·f32(1/s_y), the clamps and out_zp.
4. **MAX_POOL_2D is opcode 0x02**, a 17-word op with the same layout as 0x01.
   - It reads the window from wide memory through the par TTU, one byte lane per step. That gives 1 position × 64 channels per step, with up to 4 positions per 256-byte row.
   - It max-reduces kw×kh (reduce_mask 0b11), with offset = −zp and in_zp = out_zp = zp.
   - A transposing narrowToWide (tail_f799 = 1) builds that layout from the narrow tile block.
5. **TTU increments are byte granular, and op.md's per-dim `mode` fields are the increments' low bits.**
   - The 2 low bits of the increment of dim k are stored in the "mode" slot of dim k−1, or in `{p}_hmode` for dim 0. Wide TTUs count 64-byte units, so a 256-byte row is 4 "lanes".
   - With this reading, the unused outer dims of 12,568 TTUs in 3,142 op instructions decode to stride 0, i.e. pure rewinds. That covers every class, including FC, conv and copy ops.
   - The one exception is the `in` TTU of the pooled conv's 630 reformat ops, which use 7 dims.
   - The "lane modes" of MUL, ADD and MAX_POOL are then just strides of 1 or 2 bytes, or of one 64-byte lane.
   - See §1.
6. **All multipliers and clamps are float32 computations that use the reciprocal of the output scale:** `mult = f32(f32(x)·f32(1/s_out))`. The fp64 quotient does not reproduce them; §2.
7. **Data flow.**
   - A tensor stays in narrow memory at an allocator-chosen byte address, in one of three layouts (§8.1). The consuming op's TTU reads it at that address with that layout's strides.
   - Op instructions on a tile execute in order, so op→op needs no sync.
   - Cross-unit hand-offs use tile sync counters:
     - narrowToWide → MUL step 1 uses PARAMETERS;
     - mesh fills → the ADD weight transfer uses MESH_{N,S,W,E}_IN;
     - the NLU load is preceded by an op-unit fence.
   - §8.3 gives the exact instruction and sync sequences.

## 1. Conventions and corrections to op.md

- Bit numbering is as in `op.md`. op.py field names are used throughout, since every builder returns an `op.encode_op` field dict. Opcode 0x02 uses the same 17-word layout with `opcode=2`.
- **TTU increments** (correction to op.md).
  - Increment k is `4·inc_k + lo2_k`:
    - `lo2_0` = `{p}_hmode`;
    - `lo2_k` = `{p}_mode{k-1}` for k ≥ 1;
    - `{p}_mode7` is a separate 2-bit flag (3 for in/out of most ops).
  - Units are bytes for `in`/`out` (narrow memory) and 64-byte units for `par`/`psum` (wide memory). A 256-byte wide row is 4 such units; I call one unit a "lane", because unit k of a row is byte k of each of the 64 four-byte columns.
  - Increments are the usual rewinding increments of logical strides S: `incr_k = S_k − Σ_{e<k} cnt_e·S_e`.
  - `ttu_set(d, p, strides, counts)` writes them and `ttu_strides(d, p)` reads them back.
  - Example, MUL step 1 `in` of FFN 16×8: `(inc, cnt, mode)` = (0,767,1) (0,1,1) (0,3,1) (0,0,1) (−1536,0,1)…, with hmode 1.
    - These are the increments 1, 1, 1, 1, −6143 bytes;
    - so the strides are 1 B (element), 768 B (position), 1536 B (block row) and 6144 B (block).
- **Wide base addresses:** `par_base`/`psum_base` plus `par_sel`/`psum_sel` form one 14-bit address in 64-byte units, with sel = bit 13. For example the identity row is at `0x207C` = (124, sel 1).
- **Instruction lengths:** opcode **0x02 is 17 words** and **0x19 is 13 words**. `coral/isa.py` LEN says 1 and 15. `eltwise.split()` uses the right lengths (plus 6 for 0x15/0x18).
  - Both opcodes are tile instructions and take a `seq` slot.
  - With these lengths, `seq` is 0, 1, 2, … in every program of the corpus. The continuation bitstreams of the pooled convs continue the numbering.

## 2. Quantization formulas (float32, exactly as edgetpu_compiler)

`F(x)` = round to float32, `inv(s) = F(1/F(s))`, and every product is rounded to float32.

| op | mult | clamp_min / clamp_max (relative to out_zp) | zero points / offset | verified |
|---|---|---|---|---|
| RELU / RELU6 / RELU_N1_TO_1 / QUANTIZE | `F(s_in · inv(s_out))` | `out_clamps(s_out, z_out, act)` | in_zp = z_in, out_zp = z_out | 191 |
| LOGISTIC | `F(s_in)` | both clamps = ±10.396732 (f32 `0x41265904`, constant) | in_zp = z_in, out_zp = z_out (TFLite forces s_out = 1/256) | 246 |
| MUL step 1 | `F(F(s_a·s_b) · inv(s_y))` | `out_clamps(s_y, z_y, act)` | in_zp = z_a, **w_zp = z_b**, out_zp = z_y | 134 |
| ADD | `F(F(s_1 / w_1) · inv(s_y))` | `out_clamps(s_y, z_y, act)` | **offset = −(w_1·z_1 + w_2·z_2)**, in_zp = w_zp = 0, out_zp = z_y | 109 |
| MAX_POOL_2D | 1.0 | `out_clamps(s, z)` | in_zp = out_zp = z, **offset = −z** | 426 |

`out_clamps(s, z, act)`:
- lo = F((0−z)·s), hi = F((255−z)·s);
- the activation then intersects: RELU lo = max(lo, 0); RELU6 adds hi = min(hi, 6); RELU_N1_TO_1 uses [−1, 1];
- the result is (F(lo·inv(s)), F(hi·inv(s))).

So the clamps are generally not integers, e.g. (s, z) = (0.03, 7) gives −7.0000005 and 248.00002. The fp64 quotient reproduces only 52/66 of the low clamps and 39/66 of the high ones; this formula reproduces 66/66.

**ADD integer weights.**
- `add_weights(s1, s2)`: with r = Fraction(f32 s2)/Fraction(f32 s1), take `limit_denominator(L)` on r if r ≤ 1, or on 1/r otherwise. That gives the best rational approximation with both terms ≤ L.
- Observed pairs:
  - 0.3 → (10,3); 0.37 → (100,37); π/10 → (226,71); 0.0039 → (10000,39); 1.001 → (20999,21020); 0.6180339 → (25319,15648);
  - for 1.001, 1001/1000 is rejected because a better fraction exists below L.
- All 28 distinct scale pairs fit exactly for any L in [25319, 32768]. L = 32767 (int16 max) is assumed.
- Weights above 255 are real: the fill pattern carries them as 16-bit values.

**Which operand is which.**
- MUL: operand a (the `in` TTU, in_zp) was TFLite input 0, and operand b (the FIFO, w_zp) was input 1. That held in all 129 step-1 ops whose two inputs have different quantizations.
- ADD: operand 1 is whichever operand the allocator placed at `in_base`, and operand 2 is at `in_base + 4·delta`. The weight fills show which one that was:
  - TFLite input 0 in all 42 standalone ADD programs;
  - in the FC pairs, input 0 in 1 of 3 and input 1 in 2 of 3. There the op fields are order-symmetric.
  - The test tries both orders.

## 3. Opcode 0x19: NLU spline load (13 words, new)

| bits | w | name | value |
|---|---|---|---|
| 0-5 | 6 | predication | 0 |
| 6-11 | 6 | opcode | 0x19 |
| 12-27 | 16 | tile_mask | the OR of the masks of the LOGISTIC ops that follow (41/41) |
| 28-45 | 18 | rsv | 0 |
| 46-61 | 16 | seq | global sequence number, as in op |
| 62+32k, k=0..39 | 32 | coefficient c[s][j], s = k div 5, j = k mod 5 | f32; segment s evaluates `c0 + c1·x + c2·x² + c3·x³ + c4·x⁴` |
| 62+32k, k=40..46 | 32 | breakpoints b0..b6 | f32 ascending; segment s covers [b_{s−1}, b_s) |
| 1566-1597 | 32 | slot 47: mode (u32) | 3 for LOGISTIC, 0 for TANH |
| 1598-1661 | 64 | slots 48-49 | 0 |
| 1662-1663 | 2 | rsv | 0 |

**LOGISTIC table** (`eltwise.LOGISTIC_SLOTS`). It is constant: 41/41 test instances (every shape, input scales 0.05..0.3, input zero points 0..255, output zp 0 or 10), and also input scales 1/32..0.5 in exploratory compiles.
- Breakpoints: −5.63985, −2.96540, −1.34088, 0.08614, 1.80566, 4.80118, 8.47242.
- The middle segment [−1.34, 0.086] is `128.0016 + 63.972x − 0.330x² − 6.311x³ − 1.162x⁴`, i.e. 256·σ(x).
- The polynomial is evaluated on the real input x and returns the output in units of 1/256: the output scale is folded into the table.
- On [−10.4, 10.4], max |spline − 256·σ(x)| = 0.0076 output steps (1/256 each).
- The op adds out_zp afterwards.

**TANH**, for comparison, compiles to the same 0x19 + op pair.
- It uses another table: mode 0, all 7 breakpoints positive (0.36..4.84), and c0 = 0 in segment 0, i.e. an odd-symmetric one-sided spline.
- Its op has clamp ±6.8748 and out_zp 128, and bit 1946 clear.

The compiler binary names these tables (`two_tailed_logistic_spline.pb`, `tanh_spline.pb`, …, `CustomPiecewisePolynomial`) and the unit (`::TranscendentalNlu`, `::WideToNLU`).

## 4. Requantize op family: RELU, RELU6, RELU_N1_TO_1, QUANTIZE, LOGISTIC

These all use the `requant_tile_fields` instruction of op.md, generalized to P positions per tile: `requant_op_fields` / `logistic_op_fields`. The MXU multiplies every 4-byte input word by the 4×4 identity row in wide memory.

| field | formula |
|---|---|
| loop0, loop1 | P−1, Cw−1. P = positions of the tile block (rows·cols), Cw = ⌈C/4⌉ words per position. 1-D: P = 1, Cw = ⌈chunk/4⌉ |
| `in` / `out` TTU | strides [4, 4R, 4] bytes, counts [0, P−1, Cw−1]. R = position stride in words (R_in, R_out; §8.1). Positions are walked innermost |
| par / psum | base = identity row address (14-bit). Counts = the nonzero ones of (P−1, Cw−1), else 0. Strides 0 |
| single step (P·Cw = 1) | par_hmode = psum_hmode = 1, psum_tflags 0; otherwise psum_tflags 0x800 |
| cfg0 / cfg1 / cfg2 / dp_mode / sync0 / sync1 / out_ch | 0x5 / **0x4D, or 0x14D when `flat`** / 6 / 1 / 0x4000 / 0x4000 / 3 |
| `flat` = cfg1 bit 8 | 1 for single-position tensors (any 1-D tensor, [1,1,1,n]) and for C % 4 ≠ 0, else 0 (`requant_flat`). Meaning unknown |
| bit 1946 (op.py `rsv1946`) | 1 for LOGISTIC only. TANH, which also uses the NLU, leaves it 0. Guess: selects the logistic-specific NLU path |
| mult, clamps, zps | §2 |

**Example: FFN conv 16×8, LOGISTIC of h1** (seq 29, mask ffff).
- Fields: loops [7, 191]; in 8448 → out 0; R_in = R_out = 192; ident 0x207C; in_zp 128; mult 0.125; clamps ±10.396732; out_zp 0; cfg1 0x4D.
- The block: 8 positions (4 rows × 2 columns of the 16×8 grid), 192 words each.

## 5. MUL of two tensors

`mul_op1_fields`, `mul_op2_fields`, `mul_op_fields` (both steps), `mul_feed_n2w_fields`.

Notation: w = words per position, i.e. ⌈C/4⌉, or ⌈chunk/4⌉ for 1-D. m = 4w elements, including padding. (cols, rows) = tile block, P = cols·rows. R = (position, row, block) strides of a, in words. D = FIFO depth = min(4, ⌈w/2⌉). r = w mod D.

### 5.1 Operand-b FIFO feed (narrowToWide, `mul_feed_n2w_fields`)
- **Narrow side:** b's block, contiguous; levels [(1 word, w), (w, cols), (cols·w, rows)], dropping count-1 levels. `narrow_lvl_mask` = levels, or 0 for a single word.
- **Wide side:** a D-row ring at the FIFO address, **one narrow word per 256-byte row**; walk [(1 row, D), (0, ⌈w/D⌉ rounds), (0, P)]. For D = 1 the walk is [(0, w), (0, P)].
- `wide_rows` = D. wide_circ = wide_lvl_mask = sync_f1 = sync_wait_lvl = (row increment == 1).
- Sync record 0 = PARAMETERS, val 0xFFFF, sync_f45 = 1.
- **Partial last round:** `rsv502` = (r−1)<<6 | 1<<22 | (D−r)<<24. A single word also sets tail_lvl = 1 and wide_lvl_mask = 0.

### 5.2 Step 1, multiply (dp_mode 4, cfg0 0x60)
| field | formula |
|---|---|
| loops | [4D−1, ⌈w/D⌉−1, cols−1, rows−1]: one step per FIFO lane (4 lanes × D rows), rounds, block |
| `in` (a) | **one byte per step**: strides [1 B, 4R0, 4R1, 4R2] bytes, counts [m−1, cols−1, rows−1, 0]; modes 1, mode7 3, in_tflags 3 |
| `out` (intermediate) | **one 4-byte word per element**, dense: strides [4, 4, 4m, 4·cols·m, 4·P·m] B, counts [0, m−1, cols−1, rows−1, 0]; out_tflags 7 |
| par (FIFO) | strides [1 lane] (all outer strides 0 = re-read the ring), counts [4D−1, ⌈w/D⌉−1, cols−1, rows−1]; par_mode7 1; par_tflags 0x40 |
| par_fifo (1216-1229) | D, plus bit 13 (0x2000) when r ≠ 0 |
| partial last round (r ≠ 0) | rsv174 = 0x4000 \| (4r−1): loop0's last-round count, at bits 174-187, plus 2 mode bits at 188. rsv1230 = 1<<13 \| (2r−1) \| (D−r)<<18: par TTU last-round record (count, enable, skipped rows) |
| psum | counts [0, m−1, cols−1, rows−1], strides 0, psum_hmode 1, psum_mode7 1, psum_tflags 0x840 |
| cfg | cfg0 0x60, cfg1 0x12D, cfg2 6, cfg3 7, sync0/1 0x4000, out_ch = out_ch_last = 3 |
| quant | in_zp = z_a, w_zp = z_b, mult, clamps, out_zp (§2) |

**Interpretation (consistent with all fields).** Each step feeds one byte of a into the MXU. The weight row is the FIFO row holding b[4j..4j+3], replicated in all 64 columns, so lane k of the result for element 4j+k is a·b in every output channel. With out_ch = 3, one word (4 copies) is stored per element.

### 5.3 Step 2, pack (dp_mode 2)
- loops [3, w−1, cols−1, rows−1].
- `in`: the intermediate, strides [4, 4m, 4·cols·m, 4·P·m] B, counts [m−1, cols−1, rows−1, 0].
- `out`: strides [4, 4R0, 4R1, 4R2] B, counts [w−1, cols−1, rows−1, 0], with R = the output's layout.
- par: the identity row, strides [1 lane, 0…], counts [3, w−1, cols−1, rows−1], par_hmode 1, par_mode7 1, par_tflags 0x40.
- psum counts [3, w−1, cols−1, rows−1], psum_tflags 0x840.
- cfg0 = cfg1 = 0, cfg2 6, cfg3 7, mult 1.0, clamps ±inf, zero points 0.
- The 4 steps per output word select lane k of intermediate word 4j+k (identity row k), so the output is byte-packed.

### 5.4 Example: FFN conv 16×8, `a = h1 · g`
- **Shapes:** h1 has s 1/8, zp 128; g has s 1/256, zp 0; a has s 1/16, zp 20. w = 192, block 2×4, D = 4.
- **Step 1** (seq 34): loops [15, 47, 1, 3]; in 8448 (h1) → out 14592; the intermediate is 6144 words = 24 KiB per tile. FIFO at 0x205C.
  - Quant: in_zp 128, w_zp 0, mult = F(F(1/8·1/256)·16) = 0.0078125, clamps [−20, 235], out_zp 20.
- **Feed** (seq 35): narrow 0 (g) with levels (1,192)(1,2)(1,4) → wide 0x205C walk (1,4)(0,48)(0,8).
- **Step 2** (seq 37): in 14592 → out 0 (a overwrites g, which the feed has already consumed); ident 0x207C; loops [3, 191, 1, 3].
- The second MUL (`m = a · h3`) has the same shape: a at 0 → the in TTU, h3 at 30724 → the FIFO, m → 30724.

## 6. ADD of two tensors (dp_mode 5)

`add_op_fields`, `add_weights`, `add_weight_matrix`, `add_weight_fills`, `add_weight_n2w_fields`.

| field | formula |
|---|---|
| loops | [1, 1, cols−1, rows−1, 0, w−1]; loop0 = 16-bit half, loop1 = operand, loop5 = word. reduce_mask 0b11 = 4 accumulation steps per output word |
| `in` | **two bytes per step**: strides [2 B, 4·delta, 4R0, 4R1, 4R2, 4] B, counts [1, 1, cols−1, rows−1, 0, w−1]. delta = distance operand 2 − operand 1 in words; standalone = the tensor size. in_mode7 3, in_tflags 0xF |
| `out` | strides [4, 4R0, 4R1, 4R2, 4] B, counts [0, cols−1, rows−1, 0, w−1]; out_tflags 7 |
| par (weights) | strides [4 lanes = 1 row, 0], counts [3, P·w−1]: the 4 consecutive matrix rows, re-read per output word; par_mode7 1 |
| psum | counts [3, P·w−1], psum_tflags 0x840 |
| cfg | cfg0 0x9, cfg1 0x1CD, cfg2 0x16, sync0/1 0x4000, dp_mode 5, out_ch 3 |
| quant | offset (2045-2076, int32) = −(w1·z1 + w2·z2); mult; clamps; out_zp (§2) |

**Weight matrix** (`add_weight_matrix`, 64 bytes = 4 rows of 16).
- Each row is a 4-output × 4-byte matrix with 16-bit weights at byte lanes 0 and 2. The MXU input in this mode is two zero-extended bytes per step.
- Rows 0 and 1 hold w1, for the low and high halves of operand 1 (outputs 0,1 and outputs 2,3). Rows 2 and 3 hold w2 for operand 2.
- So y_i = w1·a_i + w2·b_i + offset.

**Building the matrix.**
- `add_weight_fills`: 10 inbound mesh fills, round robin over 0x17, 0x18, 0x16, 0x15, all with in_mode 3. They write narrow offsets
  0:w1, 4:w1<<16, 8..23:0, 24:w1, 28:w1<<16, 32:w2, 36:w2<<16, 40..55:0, 56:w2, 60:w2<<16.
- `add_weight_n2w_fields` then moves those 16 words to 4 wide rows. Its sync block waits on MESH_NORTH_IN, MESH_SOUTH_IN, MESH_WEST_IN and MESH_EAST_IN, each with val 0x8001 (§8.4).

**Example: `add_n64`.**
- a: s 1/16, zp 128; b: s 1/32, zp 100; y: s 1/8, zp 120.
- Weights (2, 1), so fills `0x2, 0x20000, 0, 0x2, 0x20000, 0x1, 0x10000, 0, 0x1, 0x10000`.
- offset −356, mult 0.25, clamps [−120, 135].
- loops [1, 1, 0, 0, 0, 15]; in 768 with delta 16; out 832; matrix at 0x2070.

## 7. MAX_POOL_2D (opcode 0x02)

`maxpool_op_fields`, `maxpool_layout`, `maxpool_relay_n2w_fields`.

**Wide layout** (`maxpool_layout`).
- Every 256-byte wide row holds 4 input positions: lane k (64-byte unit k) holds position k, one channel per column.
- An input row of Wb = (OW−1)·sw + kw positions takes Sy = 4·⌈Wb/4⌉ lanes.
- The Hb×Wb block of one 64-channel group takes blk = Hb·Sy lanes, with Hb = (OH−1)·sh + kh.
- Verified on 24 window/stride/block/group combinations in the test corpus, and 37 (overlapping) in exploratory compiles.

| field | formula |
|---|---|
| opcode | 2 (the `in` TTU is unused, all 0) |
| loops | [kw−1, kh−1, OW−1, OH−1, G−1]; G = ⌈C/64⌉ channel groups; reduce_mask 0b11 |
| par (window source) | strides [1, Sy, sw, sh·Sy, blk, G·blk] lanes, counts [kw−1, kh−1, OW−1, OH−1, G−1, 0]; par_mode7 3, par_tflags 0xF |
| `out` | strides [4, 4R0, 4R1, 4·Wg, 4·OH·R1] B, counts [Wg−1, OW−1, OH−1, G−1, 0]. Wg = ⌈min(C,64)/4⌉, R0 = ⌈C/4⌉ (position), R1 = OW·R0 (row) |
| out_ch / out_ch_last | 4⌈min(C,64)/4⌉−1 / 4⌈C_last/4⌉−1. A partial last group sets out_last_cnt = W_l−1, out_last_mode 3, out_last_skip = Wg−W_l |
| psum | counts as the loops, psum_tflags 0x880 |
| cfg | cfg0 0x127, cfg1 0, cfg2 6, cfg3 7, dp_mode 4 |
| quant | in_zp = out_zp = zp, offset = −zp, mult 1.0, clamps out_clamps(s, zp) |

**Relay** (`maxpool_relay_n2w_fields`, narrowToWide with tail_f799 = 1).
- **Narrow side:** levels [(Rp, 4 positions), (1 word, Wg), (4·Rp, ⌈Wb/4⌉ quads), (Rr, Hb), (16 words, G)], dropping count-1 levels.
- **Wide side:** [(1 row, blk/4), (blk/4, G)].
- Fields: wide_lvl_mask 1 (3 with groups), tail_lvl 2 (1 if Wg = 1), sync_en0 = sync_f1 = 1, sync_en1 = sync_en2 = 0.
- **Partial last group:** `rsv282` = (W_l−1)<<20 | 3<<36 | (16−W_l)<<39.
- Each 4 positions × 4 channel words becomes 4 columns × 4 lanes, which is the transpose.
- Standalone pools need a gather step before the relay:
  - The tile block is first written by copy/reformat ops (cfg0 7/9, not decoded here). The relay then reads it with that layout's Rp and Rr.
  - k5s3 needs 3 relays with different row strides.

**Pooled conv** (CONV_2D [1,160,200,288] × [M,1,1,288] → MAX_POOL 8×8/8 → [1,20,25,M]).
- **Instructions:** 3 instruction bitstreams of 16.4k + 16.4k + 0.8–1k words, 10 input DMAs (16 image rows each) and 10 output DMAs (2 output rows each).
- **Per chunk:**
  - 28 conv ops, on tile pairs 5555/aaaa;
  - mesh moves (0x16/0x17/0x18) that gather the 8×8 windows onto the pooling tiles;
  - one relay per window block;
  - 13 MAX_POOL ops: 12 × mask 0505 + 1 × 0101 = 50 output positions.
- **Each MAX_POOL op** computes 1 output position per tile: OH = OW = 1, Sy = 8, blk = 64.
  - M = 16 / 64: 1 channel group, out (1, M/4−1).
  - M = 256: loop4 = 3 (4 groups); out strides (1 word, 16 words per group).
- **Relays:** M = 16 relays have the standalone format. M = 64 and 256 relays additionally wait on 4 MESH_*_IN counters (vals 0x8000 / 0xFFFE), so the builder does not reproduce them.

## 8. Data flow between chained ops

### 8.1 Placement and narrow layouts
- **4-D tensors [1,H,W,C].**
  - Every tile holds a spatial block: rows `tile_split(H)[t//4]` × columns `tile_split(W)[t%4]`.
  - `tile_split(D)` = D//4 per tile row/column, with the remainder on rows {0}, {0,2} or {0,1,2}. For example D=2 → tiles 0 and 2 (mask 0505); D=7 → 2,2,2,1.
  - The rule matches the executables' `output_layout` for all 25 grid shapes tried (1×1 … 64×64, 20×25, 13×7).
  - The block is dense: R = (C/4, cols·C/4, rows·cols·C/4) words, and C is padded to 4.
  - Ops are emitted once per distinct block shape. The tile mask is the OR of the tiles with that shape, e.g. 7×7 gives 0777, 7888, 8000 (for MUL, 1×2 and 2×1 blocks are separate).
- **1-D tensors** (and [1,1,1,n]).
  - Tile t holds bytes [b·t, b·t+b) with b = 64·⌈n/1024⌉.
  - Two layouts occur:
    - **global**: base + b·t, R = ⌈n/4⌉ (`flat1d`). The op's own position-stride field then carries the full tensor size even though it is never stepped.
    - **tile-local**: the same address on every tile, R = ⌈chunk/4⌉. This is what an FC op broadcast to several tiles (mask 0fff) writes.
  - An elementwise op that reads tile-local data writes the global layout (out_base = Y + 64·t). For MUL, step 1 is then one broadcast op and step 2 is per tile (`fcmul_288_768`).

### 8.2 How the next op reads an intermediate
- **Same unit, no sync.** The producer's `out` TTU writes the tensor at address X with layout R, and the consumer's `in` TTU reads X with the same R. Both are op-unit instructions on the same tiles and execute in program order, so no sync is needed between them.
- **Example: FFN conv 16×8, per tile.**

  | narrow address | written by | read by |
  |---|---|---|
  | 8448 | conv w1 writes h1 | LOGISTIC reads it as `in`; MUL1 step 1 reads it as `in` |
  | 0 | LOGISTIC writes g | the FIFO feed reads it; MUL1 step 2 then writes a over it |
  | 30724 | conv w3 writes h3 | the MUL2 feed reads it; MUL2 step 2 then writes m over it, and conv w2 reads m |
  | 14592 / 6144 | the step-1 intermediates | step 2 |

  Buffers are reused as soon as their last reader has run.
- **Operands needed in wide memory** go through a narrowToWide, and the counters order the two units:
  - MUL operand b: PARAMETERS FIFO handshake. The step-1 op is issued *before* its feed.
  - ADD weights: mesh-in counters.
  - MAX_POOL window: drain fences around the relay.
  - The identity row for requantize and pack ops is written at most once per program (155 of 227 bitstreams have one prologue, the others none) and stays in wide memory.
- **Cross-tile data** for the next FC uses the known ring/mesh machinery (ring_mesh.md, codegen.md). In FFN-FC, `m` (64 bytes on each of tiles 0..11) is gathered west and north to tile 0. It is then broadcast by ring to tiles 1..4, the other tiles of the w2 FC (288 outputs on 5 tiles).
- **Conv weights in the FFN** are cached on tiles 1, 3 and 0. They are re-broadcast to all 16 tiles (ringProducer → ringConsumer1) before each conv.

### 8.3 Instruction and sync sequences (all corpus programs)

| op | before | after |
|---|---|---|
| LOGISTIC | `sync_op_fence` = sync(counters AVDATA\|PARAMETERS\|PARTIAL_SUMS, units = op only) → 0x19 (mask = OR of the ops) | op(s) → reset17 |
| MUL | reset17 → `sync(counters=NARROW_TO_WIDE)` → `sync(counters=PARAMETERS)` → step-1 op(s) → FIFO feed(s) → [identity prologue if not present] → `sync_wn_fence` (units bit 5) | step-2 op(s) → sync(NARROW_TO_WIDE) → sync(PARAMETERS) → reset17 |
| ADD | 10 mesh fills → narrowToWide (sync MESH_*_IN) | op(s) → reset17 |
| MAX_POOL | copy/reformat ops or mesh gathers → `sync_drain` → relay(s) → `sync_drain` | 0x02 op(s) → reset17 |
| identity row | 4 × meshBus 0x17 inbound fills (1<<8j at X+4j, the last with in_mode 3) → narrowToWide (sync_id 5 MESH_SOUTH_IN, tail_lvl 1) | (`ident_prologue`) |

Three of these syncs are not among scalar.md's 12 variants. All are `scalar.sync(seq, counters=…, units=…)`:
- sync(AVDATA|PARAMETERS|PARTIAL_SUMS, units=1);
- sync(NARROW_TO_WIDE);
- sync(PARAMETERS).

### 8.4 narrowToWide sync block (refinement of wide_narrow.md)
- The sync area from bit 543 (or 611 for wideToNarrow) is a list of **42-bit records**:
  - +0 en0, +1 f1, +2 f2;
  - +4..8 counter id;
  - +9..10 level;
  - +12 en1;
  - +13..31 val.
- The list ends with a record of f2 = 1, id 0.
- wide_narrow.py's `sync_en2`/`sync_f45` (+44/+45) are the f2 bit and the bit after it in the second record, at +42 + 2 and +42 + 3.
- Examples:
  - MUL feed: (PARAMETERS, lvl 1, 0xFFFF);
  - ADD weights: (MESH_NORTH_IN) (MESH_SOUTH_IN) (MESH_WEST_IN) (MESH_EAST_IN), each lvl 2, val 0x8001;
  - pooled-conv relays: 4 MESH_*_IN records with vals 0x8000 / 0xFFFE.

## 9. Constant per-class fields (quick reference)

| class | opcode | dp_mode | cfg0 | cfg1 | cfg2 | cfg3 | reduce | in/out tflags | par_tflags | psum_tflags | mode7 in/out/par/psum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| requantize / LOGISTIC | 1 | 1 | 0x5 | 0x4D \| flat<<8 | 6 | 0 | 0 | 1/1 | 0 | 0x800 (0 single step) | 3/3/0/0 |
| MUL step 1 | 1 | 4 | 0x60 | 0x12D | 6 | 7 | 0 | 3/7 | 0x40 | 0x840 | 3/3/1/1 |
| MUL step 2 | 1 | 2 | 0 | 0 | 6 | 7 | 0 | 3/3 | 0x40 | 0x840 | 3/3/1/0 |
| ADD | 1 | 5 | 0x9 | 0x1CD | 0x16 | 0 | 0b11 | 0xF/7 | 0 | 0x840 | 3/3/1/0 |
| MAX_POOL | 2 | 4 | 0x127 | 0 | 6 | 7 | 0b11 | 0/7 | 0xF | 0x880 | 0/3/3/0 |

## 10. Open questions

1. **Semantics of the class constants.**
   - cfg0-cfg3 and the tflags are known only per class.
   - cfg1 bit 8 (`flat`) is a reproducible rule with an unknown meaning.
   - The meaning of bit 1946 is unknown; it is set for LOGISTIC and not for TANH.
   - The hardware meaning of `{p}_mode7` is unknown.
2. **Element width.** What tells a TTU to read 1, 2 or 4 bytes per step: tflags, dp_mode or cfg? The increment encoding is clear; the width selector is not.
3. **NLU.**
   - Is the evaluation precision f32?
   - What do "mode" values 3 and 0 (LOGISTIC vs TANH) and slots 48-49 do?
   - Where does the 10.396732 clamp come from? It is just inside ln(32767).
   - Can a custom spline (any f(x)) be loaded with the same instruction? The format suggests yes; untested.
4. **ADD weight limit.** Somewhere in [25319, 32768]; 32767 is assumed.
5. **Exact meaning of the partial-round fields** (MUL rsv174 / rsv1230 / feed rsv502, relay rsv282). The formulas are exact; the field split is inferred.
6. **Not decoded.**
   - The copy/reformat ops (cfg0 7/9 and the dp_mode 3 variant for C < 4) that put 4-D input blocks into place.
   - The pooled conv's 630 dp_mode 4/cfg0 0 reformat ops, whose `in` TTU uses 7 dims.
   - Its mesh window gathers and the extra sync records of its relays.
   - MAX_POOL with SAME padding, and AVERAGE_POOL. op.md has avgpool constants.
7. **Hardware.** None of these instructions has been executed by our runtime. The next step is a hardware run of a generated LOGISTIC / MUL / ADD / MAX_POOL program against the CPU reference.
