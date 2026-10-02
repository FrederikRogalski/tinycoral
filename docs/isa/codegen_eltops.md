# SOFTMAX, L2_NORMALIZATION and SUM / MEAN (`coral/codegen/eltops.py`)

`coral/codegen/eltops.py` generates edgetpu_compiler's programs for three more TFLite ops, byte for byte. It also models their
arithmetic in numpy. `coral/isa/scalar_core.py` holds the scalar-core instructions that SOFTMAX needs, an interpreter for them,
and the corrected instruction split.

All three ops compile to STAND_ALONE executables without parameters, so every generator returns a single program:

| op | where it runs | how |
|---|---|---|
| SOFTMAX | the **scalar core**, in float32 | The tiles only move the input into the 16 KiB scalar memory. A loop over the rows runs three 0x23 hardware loops per row: max, sum of exp2, normalize. Two special-function ops, exp2 (sfu 3) and reciprocal (sfu 2), do the transcendentals. A push loop sends the result to the host. |
| L2_NORMALIZATION | tile 0 | One op sums the squares in the MXU. The NLU then evaluates a **reciprocal-sqrt spline** (opcode 0x19 table, mode bits 7) to a 16-bit integer. A second op multiplies every element by it. |
| SUM / MEAN (last axis) | tile 0 | One **opcode-3 op** (the 17-word op layout) accumulates x through the parameter path, then requantizes. |

```
python -m coral.codegen.eltops            # acceptance test (= python test/test_codegen_eltops.py), offline
python -m coral.codegen.eltops --quick    # a sample of every check
```

On the device (`test/test_hw_eltops.py`) SOFTMAX, L2_NORMALIZATION and SUM / MEAN are bit-exact against their numpy models; the
softmax runs also showed that the scalar core's float-to-int conversion rounds to nearest.

## 1. Verification

Results of `python test/test_codegen_eltops.py`. Programs are compared byte for byte; DMA hints and host sizes are checked too.

| check | result |
|---|---|
| SOFTMAX, 1 row, n = 1..256 | 256/256 |
| SOFTMAX, 1 row, n = 257..8192 (13 sizes; the compiler leaves n = 16384 on the CPU) | 13/13 |
| SOFTMAX, 6 rows, n = 4..256 | 253/253 |
| SOFTMAX, 3, 4, 5, 7, 8, 12, 16 and 32 rows at 28 sample n | 161/161; another 63 configurations raise NotImplementedError (§4.5) |
| SOFTMAX quantizations: inputs (1/16,128) (1/2,128) (0.3,7) (0.0123,250) (1/8,0) (0.17,255); outputs (1/256, 0/10/255), (1/128,0), (0.01,3); beta 0.5/1/2; shapes 1x32, 1x100, 6x64, 6x101, 6x256 | 210/210 |
| L2_NORMALIZATION [1,1,n], n = 4..2048 (n % 4 == 0), plus 5 input quantizations | 26/26 |
| L2_NORMALIZATION NLU table and both multipliers, 40 random input scales (fitted on 473) | 40/40 |
| SUM / MEAN [1,1,n] -> [1,1,1], n = 4..2048, plus 5 input quantizations each | 52/52 |
| the generated softmax scalar program, run by `scalar_core.run`, against `softmax_ref` | 6/6 shapes, bit-exact |
| `gen_scalar_probe` loops, run by the interpreter | 4/4 |
| `softmax_ref`, `l2norm_ref`, `reduce_ref` against float math | within rounding: 0.5 LSB for softmax and SUM/MEAN, 1 LSB for L2 |

## 2. API

| function | |
|---|---|
| `gen_softmax(rows, n, in_q, out_q=(1/256, 0), beta=1.0)` | the program for SOFTMAX over the last axis of [rows, n] (TFLite [1, n], [1, rows, n] or [rows, n]) |
| `softmax_io(rows, n)` | input: rows x n bytes row-major, padded to 8. Output: rows x m bytes with m = round_up(n, 4) when rows > 1 (row r at r*m, its first n bytes valid), n bytes for one row; padded to 8 |
| `gen_l2norm(n, in_q)`, `l2norm_io(n)` | L2_NORMALIZATION of [1, 1, n]. The output quantization is (1/128, 128), as TFLite requires. Input n bytes, output n bytes, both padded to 8 |
| `gen_reduce("sum" or "mean", n, in_q, out_q)`, `reduce_io(n)` | SUM / MEAN of [1, 1, n] over axis 2 with keep_dims. The output is 1 byte; the host reads 8 |
| `executable(prog, in_bytes, out_bytes)` | wraps a program as a `coral.executable.Executable` with the compiler's hints, for `coral.runtime.run_executable` |
| `softmax_ref`, `l2norm_ref`, `reduce_ref`, `rsqrt_nlu` | numpy models (§7) |
| `rsqrt_slots(s)`, `rsqrt_scale(s)`, `l2norm_quant(s)`, `reduce_quant(...)` | the compiler's constants |
| `gen_scalar_probe(kind, words, c)` | hardware probes of the scalar core's units (§8). Not a compiler program |
| `scalar_core.split`, `scalar_core.run`, `scalar_core.dma_seqs` | instruction split with the decoded lengths, interpreter, DMA descriptors |

## 3. The scalar core's compute instructions (new)

SOFTMAX is the only op found that computes on the scalar core. EXP, LOG, SQRT, RSQRT, SIN, HARD_SWISH, LOG_SOFTMAX, ELU, SQUARE
and NEG are not mapped at all; TANH, LOGISTIC and ABS run on the tiles. Every compute word is a 0x20 bundle. In geohot's field
view (`coral.isa.WORD`) its slots are:

| slot | fields | meaning |
|---|---|---|
| scalar ALU | `s_op` 54-59, `s_x` 60-64, `s_y` 65-69, `imm` 70-101 | `s_x = s_y op b`. `s_op` bit 5 selects the immediate form; otherwise b = `s[imm & 31]`. Ops 0x01-0x0f are integer ADD SUB AND OR XOR SHL SHR ASR EQ NE GT LT GE GES MOV. **0x10-0x16 are new and float32**: I2F, F2I, FADD, FSUB, FMUL, FMAX, FMIN. I2F and F2I take their source register in the immediate field; float immediates are f32 bits |
| load | `enable_vector` 12-13 = 2 / 3, `vs_reg_v1` 14-18, `imm_size` 19-30 | `s[vs_reg_v1] = smem32[s[imm_size & 31]]`, at a **word** address. With 3, `imm_size` bit 5 post-increments the address register |
| store / push | `v_op` 31-35, `v_offset` 36-43, `vs_reg` 49-53 | `v_op` 4: `smem32[s[v_offset]] = s[vs_reg]`. `v_op` 8: push `s[vs_reg]` into the host DMA issued before (`v_op` 0xa pushes its descriptor, 0xc issues it, as in scalar.md) |
| special-function unit | `v_op_2` 102-104, `vs_reg_w` 105-109, `unk_3` 110-114 | `s[unk_3] = f(s[vs_reg_w])`: 2 = reciprocal, 3 = exp2. The names come from where SOFTMAX uses them: on the sum, and on (x - max) * log2(e). The compiler issues one other bundle and a NOP before using the exp2 result |
| **0x23 loop** (1 word) | count at bits 12-27, body length at bit 28 | repeat the next `body` words `count` times. Bodies of 1 to 30 words are seen |
| **0x22 branch** (1 word) | signed 15-bit word offset at bit 17; bits 12-16 = 0x11 and bit 33 set | jump while p0. Only the backward form is seen |

`coral.isa.LEN` says 0x03 = 8 words and 0x22 = 29 words. Both values were statistical guesses and both are wrong: 0x03 is the
17-word op layout and 0x22 is one word. `scalar_core.split` uses the right lengths; `isa.split` mis-splits SOFTMAX and SUM/MEAN
programs. `scalar_core.OP3` is `op.OP` with opcode 3.

## 4. SOFTMAX

### 4.1 The scalar program (`softmax_scalar`)

The [rows][m] block sits at the top of scalar memory: base = 0x4000 - rows*m bytes, with m = round_up(n, 4). Registers: s4 / s5
are the row pointers (in and out are the same), s8 is the max, s9 the sum and then its reciprocal. s18 / s19 hold the OUTPUT
address, set by relocated MOVIs before the data moves.

```
s4 = s5 = base
row:  m = -inf
      loop n (13 words):  b = byte(s10++) ; x = f32(b - zp_in) * s_in ; m = fmax(m, x)        # no SUB when zp_in = 0
      sum = 0
      loop n (17 words):  x as above ; e = EXP2((x - m) * k) ; sum = fadd(sum, e)              # k = f32(beta * log2 e)
      r = RECIP(sum)
      loop n (27 words):  x as above ; e = EXP2((x - m) * k) ; v = e * r * f32(1/s_out)
                          [zp_out != 0: v = i2f(f2i(v) + zp_out)  (30 words)]
                          y = f2i(fmax(fmin(v, 255), 0)) ; replace the byte in its word ; store
      s4 += m ; s5 += m ; p0 = s5 < 0x4000 ; branch row
```

- Bytes are extracted from words by shift and mask; each store rewrites the whole word.
- exp2 runs twice per element.
- About 57 bundles per element: 87,600 for 6 x 256, 14,600 for 1 x 256.

The output part (`smem_to_host`):
- the host DMA descriptor (tag 3, out_bytes) from s18 / s19;
- a software-pipelined push loop: two loads, (load, push) pairs in a 2-word 0x23 loop, two pushes;
- a single word for an odd count, then zero pushes up to the DMA size.

This is codegen_conv's `_smem_flush`, with the start word as a parameter.

### 4.2 One row (`_sm_single_row`)

The input path is FULLY_CONNECTED's:
- the input DMA, then a wideToNarrow on each input tile (pieces of 64*ceil(n/1024) bytes);
- ringConsumer and infeed per row of 4 tiles;
- drain, reset17, drain;
- movi s18 / s19.

Then all the narrowToWides, and per tile a ringProducer (host form) with a scalar-memory outfeed at base/4 + 16t words. Then
`scsync(RING_OUTFEED := 0)` and `output_wait(T)`.

Allocation:
- narrow x = 768 if n <= 84, or if n % 4 == 0 and n <= 188; otherwise 0;
- wide: the output FIFO (8 units) and the input ring FIFO (8 per 256-byte packet), stacked down from 0x2080, larger first, ties
  with the output first.

### 4.3 Several rows (`_sm_multi_row`)

Tile t takes bytes [64t, 64t+64) of every row (T = ceil(n/64) tiles). The program:
- identity row (4 mesh fills; when n % 4 != 0 also the 64-word zero row and a drain), then narrowToWide;
- wn fence;
- the input DMA with a "direct" avDataPop and infeed: no staging-buffer walk, bit 126 / bit 162 set, no credit scsyncs. 16
  bytes use a two-unit walk instead;
- per tile a copy op, then per tile its wideToNarrow, then per tile its ringConsumer, then one infeed;
- [for n even with >= 3 packets: a WIDE_TO_NARROW fence per tile t >= 1 at rows*n/4 words, and a drain on the last tile];
- reset17, drain, movi s18 / s19;
- per tile: narrowToWide (rows x h words, row stride m) to the output FIFO, ringProducer (forward form), and an outfeed into
  scalar memory at base/4 + 16t (row stride m/4 words).

The copy ops pack each tile's slice into a dense block, rows m bytes apart, through the identity row. There are three stream
forms:

| n | stream | copy op |
|---|---|---|
| n % 4 == 0 | **rows**: a one-row staging ring of n/4 words, re-written per row. One tile and <= 384 B use the "bulk" form instead: a one-word handshake (stride 0 on both sides, in_tflags 0x40) | dp_mode 3, cfg0 3; for a 1-word slice dp_mode 2. Progress fields as codegen_conv's copy ops, with T64 = 4n: sync0 = (n%16/4) << 14 \| n/16, sync1 = 0x4000 + n/16, cfg1 bits 8-9 = n%16/4. They are 0x4000 / 0x4000 / 1 when the stream is <= 2 packets, and on tile 0 when n % 128 == 0 |
| n % 4 == 2 | **pairs of rows** (2n bytes = whole words), the staging ring re-read per pair | byte copy (below) with the input dims (n, 2) (0, rows/2), in_tflags 0xc1. twait = 4n in the codegen_conv field (rsv515 \| in_twait), +1 with the extra level, +0x2000 with a partial word. Progress with T64 = 8n. n = 126 is a special case (see below) |
| odd | **none**: tile t's wideToNarrow writes the whole input minus its first 16t words | byte copy from the whole input (row stride n). Progress = the words this tile receives: W - 16t |

The byte copy (`_sm_copy_bytes`) reads 4 bytes per output word through the identity row:
- loops [4, kw, (1,) rows]; 2 x 2 steps for a 3-byte slice, one step for a 1-byte slice;
- the bytes past n come from the zero row;
- a partial last word sets the same fields as codegen_conv's shift copies: rsv174, the in-TTU record rsv536, par_fifo, rsv1230,
  rsv1523, each from (bytes in the last round - 1);
- full 64-byte tiles: dp_mode 3, cfg0 5 (odd n) or 3 (n % 4 == 2); the last tile of several gets an extra count-1 level and
  cfg1 0x8b.

The other instructions:
- **wideToNarrow** (stream form):
  - L words per chunk, the chunk count as the rows, rewind to a one-chunk ring;
  - tile t >= 1 skips 16t words (skip_en0, rsv543, skip_*) and cuts its head chunk to L - 16t words (head_*);
  - the decrement record at bit 665 is 32 bits: -h (16 bits), then +h, with h = the tile's output words per chunk;
  - `rsv489` / `size64` = chunk bytes / 2, as codegen_conv;
  - sync_f1 when the stream is <= 2 packets, and on tile 0 when n % 128 == 0.
- **ringConsumer**:
  - <= 2 packets: all of them, no sync record;
  - else one slot refilled pk times, with a WIDE_TO_NARROW record of -(64 + 16t);
  - tile 0 at n % 128 == 0: the record (+h, -h) with h = 256 / n.
- **Output FIFOs**: up to 2 packets get their own rows; more go through one row refilled pk times, with sync records (narrowToWide
  sync_id 16, val 0xffff) and the ringProducer as a single-slot FIFO.

Allocation:
- wide: identity (4 units, 8 with the zero row), output FIFO (8) and input FIFO (8*pk), stacked down from 0x2080 by size; ties go
  identity, output, input;
- narrow: the packed block at 0, then the staging and the identity (+ zero row, 272 B), the bigger first (n % 4 == 0: staging at
  rows*m, identity at rows*m + round_up(rows*n, 8)).

### 4.4 Special cases kept as observed

- n = 126 with 6 rows: tile 0's wideToNarrow record has h = 2 instead of 32, and its copy op cfg0 is 5 instead of 3. Not
  understood. Only 6 rows were seen.
- Tile 0 at n % 128 == 0: the progress fields, sync_f1 and the ringConsumer record (above). Seen for 5, 6, 7 and 8 rows; not when
  only the stream ends on a packet boundary (4 x 192, 8 x 192).

### 4.5 Not covered (NotImplementedError)

- 2 rows: another copy form; the whole input lands in narrow memory.
- n < 4 with several rows, and n % 4 != 0 with rows*m < 48: other narrow layouts.
- n % 4 == 2 with an odd row count or 4 rows.
- Odd n with a multiple of 4 rows: streamed in groups of 4 rows.
- Several rows of more than 256 bytes: the pop walks 2-D.

## 5. L2_NORMALIZATION (`gen_l2norm`)

The input path and the mesh gather onto tile 0 are FULLY_CONNECTED's (`fc.gather`). The narrow layout is
`fc.narrow_layout(n, n)` = (X, Y, R); Z = R + 32 with a relay buffer, else Y + n. Then:

```
sync_op_fence ; 0x19 NLU load (tile 0, rsqrt_slots(s)) ; x -> the 4-row FIFO (eltwise's MUL feed; sync_en1 0, val 0x8000)
op 1  (opcode 1, dp_mode 1, cfg0 0x60, reduce_mask 1, cfg2 7, rsv1946 = 7):  in = x words, par = x from the FIFO,
      in_zp = w_zp = zp:  acc = sum (x - zp)^2 ;  v = clamp(f32(acc) * f32(s*s), f32(s*s), +inf) ;  NLU -> r, 4 lanes x 16 bit at Z
sync PARAMETERS >= rounds
x -> wide 0x2058, one word per row (narrowToWide tail_f799; its 20-bit sync value at bit 12 = 0x10000 + rounds)
op 2  (dp_mode 1, cfg0 0x67, cfg2 0xe, cfg3 6):  in = r (16-bit, stride 2 B, re-read), par = x, w_zp = zp:
      y = clamp(f32((x - zp) * r) * mult, -128, 127) + 128 ;  progress sync0 = 0x4000 | (rounds+1)/4, cfg1 = 0x2d | ((rounds+1)%4) << 8
sync PARAMETERS >= 2*rounds ; reset17 ; output (one outfeed from tile 0)
```

The constants were fitted on 473 compiled input scales. Every coefficient and multiplier is reproduced:
- K = f32(65535 / f32(1/s)). The NLU output is r = K / sqrt(v), an unsigned 16-bit integer. The clamp v >= s^2 keeps r <= 65535.
- The table is c_k = f32(C_k * K): one float32 multiply of fixed **float32** coefficients C_k (`RSQRT_C`) of 1/sqrt(m) on
  m in [1, 4). There are 8 segments with breakpoints 1.174, 1.453, 1.797, 2.250, 2.799, 3.437 and inf; segment 7 repeats
  segment 6. Mode slot 3.
- mult = f32(f32(s * f32(1/K)) * f32(1/(1/128))): the usual requantize formula with the NLU output scale 1/K. Every scale gives
  2^-9 within 1 ulp.
- op 1: mult = clamp_min = f32(s*s).

The domain [1, 4) implies range reduction by powers of 4 inside the NLU: v = m * 4^e, r = p(m) * 2^-e. Mode bits 7 at bit 1946
switch it on; LOGISTIC has 1 and TANH 0. This is an inference from the table; it has not been observed.

## 6. SUM / MEAN (`gen_reduce`)

The input path and gather are the same as L2_NORMALIZATION's. The layout is `fc.narrow_layout(1, n)`, and the output is at Y.
Then:
- `sync_reset_par`;
- x into the 4-row FIFO (eltwise's MUL feed);
- **one opcode-3 op**:
  - the in TTU is unused (in_zp 0);
  - par walks the FIFO with w_zp = zp, so acc = sum (x - zp);
  - dp_mode 1, cfg0 0x60, reduce_mask 1;
  - one output word;
  - partial last FIFO round: rsv174 = 0x4000 | (r-1), rsv1230 = 1<<13 | (r-1) | (D-r)<<18;
  - a single step sets psum_hmode;
- sync(NARROW_TO_WIDE | PARAMETERS), reset17, the 1-byte output.

The multipliers:
- SUM: f32(s_in * f32(1/s_out)) with `out_clamps`;
- MEAN: f32(that / n) and **no clamps** (+-inf).

## 7. The arithmetic: what the chip computes, and the numpy models

| step | how the chip does it | model | status |
|---|---|---|---|
| softmax: dequantize, max, subtract, scale, sum, normalize | scalar-core float32 ALU, sequential, in the program's order (§4.1) | `softmax_ref`: the same float32 ops in numpy, sum by `np.add.accumulate` | the interpreter running our emitted words equals the model bit for bit (6 shapes); the ALU is assumed IEEE round-to-nearest |
| **exp** | **sfu op 3 = exp2**, on (x - max) * f32(beta * log2 e) <= 0. There is no lookup table, no spline load and no NLU. It runs twice per element | exact float32 2^t (parameter `exp2=`) | **unknown precision**: a hardware unit, not measured |
| **reciprocal** | **sfu op 2**, once per row, on the float32 sum (between 1 and n) | f32(1/x) (parameter `recip=`) | **unknown precision** |
| softmax output | `f2i(fmax(fmin(e*r*256, 255), 0))`; with zp_out: `f2i` first, then + zp, then `i2f`, clamp, `f2i` | round half to even (`f2i='trunc'` / `'rna'` to compare) | **rounding mode unknown**: it decides between 0.5 and 1 LSB error |
| L2: sum of squares | MXU, int32, exact | `sum (x - zp)^2` | structure from the program |
| **rsqrt** | **NLU 8-segment quartic** on m in [1, 4), scaled to K/sqrt(v), 16-bit output (§5) | `rsqrt_nlu`: range reduction, float32 Horner, times 2^-e, rounded, clamped to 16 bits | coefficients exact; **range reduction, evaluation order (Horner / FMA) and output rounding are hypotheses**. The spline error is ~1e-7 relative, so only ties are at stake |
| L2 scaling | MXU (x - zp) * r (16-bit input mode), float32 requantization, round half to even | `l2norm_ref` | requantization as measured for the other ops (NOTES) |
| SUM / MEAN | MXU sum, float32 requantization | `reduce_ref` | MEAN results outside 0..255 (possible when s_out < s_in) are not clamped by the op; the model saturates them |

## 8. Hardware probes (`gen_scalar_probe`)

`gen_scalar_probe(kind, words, c)` builds our own program. It keeps the one-row softmax data path, so `words` float32 or int32
values go into scalar memory, and swaps in a scalar loop that replaces every word w with f(w):
- `kind`: 'exp2' (sfu 3), 'recip' (sfu 2), 'f2i', 'i2f', 'fmul' / 'fadd' by the immediate c;
- four NOP bundles on each side of the unit op;
- up to 2048 words per call.

The interpreter checks the loop (test 7). It is not a compiler program and has never run.

```python
import numpy as np; from coral.device import EdgeTPU; from coral.runtime import run_executable; from coral.codegen import eltops as E
tpu = EdgeTPU()
t = np.linspace(-30, 0, 2048, dtype=np.float32)                  # the softmax domain of exp2
y = run_executable(tpu, E.executable(E.gen_scalar_probe("exp2", 2048), 8192, 8192), t.tobytes())
np.frombuffer(y, np.float32)                                     # compare with np.exp2(t.astype(np.float64))
```

What to measure:
- exp2 on [-30, 0] and a fine grid near 0;
- recip on [1, 256] with random mantissas;
- f2i on k + {0, .25, .5, .75} including negatives;
- fmul / fadd on random pairs (round to nearest?);
- then one `gen_softmax(6, 256)` call against `softmax_ref(exp2=measured, recip=measured, f2i=measured)`.

For L2: `gen_l2norm(288)` with inputs that sweep sum((x - zp)^2) over several octaves. It shows r through the output; ties tell
the rounding.

## 9. Attention without BATCH_MATMUL (edgetpu_compiler 14.1, 6 heads, 48 dims, P = 64 and 256)

| graph | result |
|---|---|
| scores: MUL q [1,6,1,48] x K [1,6,P,48] (broadcast), then SUM over axis 3 | **maps** (also 3-D [6,1,48] x [6,P,48] + SUM axis 2). P = 256: one program of 9,874 words with 1,024 meshE moves (probably the broadcast of q), 128 dp_mode-4 ops and 2 opcode-3 reductions. The host sends K (72 KiB) every call and gets 6 KiB back (4 bytes per score) |
| MUL q x K with the same shape, + SUM | maps |
| out: MUL p [1,6,P,1] x V [1,6,P,48] (broadcast over channels), alone or + SUM over axis 2 | **internal compiler error** for both P |
| out with p expanded on the host to [1,6,P,48], MUL + SUM over axis 2 | maps: 776 words. The position sum is an opcode-3 op of the pooling class (dp_mode 4, cfg0 0x127, reduce_mask 3). Two 72 KiB inputs per call |
| SUM / MEAN over the channels of [1,6,P,48], [6,P,48], [1,6P,48] | map |

K and V are inputs, so the KV cache crosses USB on every call. None of these programs is decoded or generated here.

## 10. Open questions

- **The scalar units:** the precision of exp2 and reciprocal, the F2I rounding, the pipeline latencies (the compiler waits 2
  bundles after exp2), and the other `v_op_2` codes. Only 2 and 3 are seen.
- **The NLU in rsqrt mode:** how v is split into m and e, the polynomial evaluation, and the rounding of r.
- **Copy-op internals:** the progress counts, the decrement records and the two special cases of §4.4. They are reproduced from
  formulas without being understood.
- **Not modelled:**
  - the SOFTMAX configurations of §4.5;
  - L2_NORMALIZATION / SUM / MEAN with n % 4 != 0 (the compiler then emits a PARAMETER_CACHING program with a small blob);
  - several rows for L2 / SUM / MEAN, and SUM / MEAN over other axes.
- **The branch word:** what bits 12-16 and 33 mean.

## 11. Risks for running on hardware

- Programs equal to the compiler's are exactly as trustworthy as those.
- **The scalar core's compute path** (0x23 loops, the 0x22 branch, ld / st, the float ALU, the sfu ops) ran in the softmax programs
  of `test/test_hw_eltops.py` (1 and 6 rows) and in the attention programs.
- **The scalar-memory push output** is the same loop as codegen_conv's scalar-memory outputs, which ran bit-exact. The runtime
  must read the whole output (out_bytes) before anything else goes out.
- **Run time:** SOFTMAX occupies the scalar core for about 57 bundles per element. 6 x 256 is about 88k bundles: roughly 0.2 ms at
  500 MHz if the scalar core issues one bundle per cycle (not measured).
- **The probes are our own programs**, not the compiler's: a custom scalar loop, with 4 NOPs of slack around every unit op.
- **MEAN with s_out < s_in** can exceed 0..255 before the output conversion; the device behaviour there is unknown.
