# The attention core as one program (`coral/codegen/attention.py`)

`gen_attention(P, quant)` builds ONE STAND_ALONE Edge TPU program for the attention core of a TinyStories-15M layer at batch 1:
6 heads of 48 dims, P = 1..256 positions,

```
o = concat_h softmax(beta * q_h . K_h^T) . V_h        q [288], K, V [P][288] uint8 in, o [288] uint8 out, beta = 1/sqrt(48)
```

edgetpu_compiler cannot compile this graph: the whole graph (scores -> SOFTMAX -> p.V) and SOFTMAX -> FULLY_CONNECTED end in an
internal compiler error (section 7.3), and BATCH_MATMUL does not map (edgetpu_compiler leaves it on the CPU). The program composes three pieces
that the compiler does map, each of which our generator reproduces byte for byte, with glue taken from other compiler programs:

| piece | TFLite graph (the oracle) | generator |
|---|---|---|
| scores | `s[1,6,P] = SUM_axis3(MUL(q[1,6,1,48], K[1,6,P,48]))`, q broadcast over the positions | `gen_scores(P, quant)` |
| softmax | `SOFTMAX` over [1,6,P], beta | `eltops.gen_softmax(6, P, ...)` (existing) |
| p.V | `o[1,6,48] = SUM_axis2(MUL(p[1,6,P,48], V[1,6,P,48]))`, p expanded over the channels on the host | `gen_pv(P, quant)` |

In the composed program both broadcasts (q over the positions, p over the 48 channels) are stride-0 levels of the MUL's input TTU,
and the softmax runs on the scalar core between the two halves. Every instruction comes from the encoders in `coral/isa/` and the
builders of `codegen/{conv2d,conv,fc,chain,eltops}.py`; field values reproduced without a known meaning are listed in section 8.

```
python -m coral.codegen.attention            # acceptance test (= python test/test_codegen_attention.py), offline, all P = 1..256
python -m coral.codegen.attention --quick    # 24 sample P
.venv/bin/python test/test_hw_attention.py attn 64      # on the device (tools/hw.py): scores P | pv P | smmul | attn P [timed calls] | attn_smem P
```

| API | |
|---|---|
| `gen_attention(P, quant=None)` | `(program, io)`; `quant`: (scale, zero point) of `q, k, qk` (the q*K products), `s` (scores), `p` (probabilities, (1/256, 0)), `v, pv` (the p*V products), `o`; `beta` |
| `attention_io(P)`, `attention_executable(prog, P)`, `attention_inputs(q, K, V)` | host contract (section 5), a `coral.executable.Executable` for `coral.runtime.run_executable`, the input buffer from KV-cache rows |
| `attention_ref(q, K, V, quant, parts=False)` | the bit model (section 6); `parts` also returns the scores and p |
| `gen_scores(P, quant)`, `scores_ref`, `scores_io`, `scores_unpack` | the scores piece, its bit model and output layout |
| `gen_pv(P, quant)`, `pv_ref` | the p.V piece |
| `mul_ref`, `sum_ref` | MUL / SUM bit models |
| `gen_attention(P, quant, debug_smem=True)` | debug build: stops after the softmax and sends the scalar memory's score and p words (12P words) to the host |

## 1. Verification

Offline (`python test/test_codegen_attention.py`):

| check | result |
|---|---|
| scores vs edgetpu_compiler, P = 1..256 (program bytes, DMA descriptors = hints, host sizes) | 256/256 |
| scores, 2 more quantization sets (a calibrated TinyStories layer; odd scales and zero points), P = 1, 7, 64, 256 | 8/8 |
| p.V (p expanded) vs edgetpu_compiler, P = 1..256 | 256/256 |
| p.V, 2 more quantization sets, P = 4, 7, 64, 256 | 8/8 |
| SOFTMAX [1,6,P] with beta = 1/sqrt(48) (`eltops.gen_softmax`, after the fix in section 3.3) | 12/12 |
| the scalar memory -> tile 0 path (`smem_to_tiles`) vs the compiler's SOFTMAX -> RELU, 11 P from 2 to 256 (P = 1 is not mapped) | 11/11 |
| `gen_attention(P)`, P = 1..256: every instruction decodes and re-encodes; sequence numbers; DMA descriptors = host contract; every narrow address an instruction touches (from its TTUs) inside one buffer of the plan; producer = consumer size of every scalar-memory transfer and of p's broadcast | 256/256 |
| its scalar-core softmax run by `scalar_core.run` on random score words (random upper bytes), load latency 0..4 bundles | 6/6 P, bit-exact with `eltops.softmax_ref` |
| `scores_ref` / `pv_ref` within their rounding bound of float math | 8/8 |

The composed program's gather, position SUM and output are `gen_pv`'s builders (byte-identical to the compiler for every P), at
our addresses.

On the device (`test/test_hw_attention.py`, through `tools/hw.py`; 10 runs, no failed transfer; inputs: Gaussian uint8 with the
spread of TinyStories' real q, k, v under the quantization of a calibrated layer, so that the scores and p are not saturated):

| program | inputs | result |
|---|---|---|
| `gen_scores(64)` (byte-identical) | 3 x (q, K) | 3 x 384 scores bit-exact with `scores_ref` |
| `gen_pv(64)` (byte-identical) | 3 x (p expanded, V), p from Dirichlet rows | 3 x 288 bit-exact with `pv_ref` |
| edgetpu_compiler's own SOFTMAX -> MUL [1,6,64] (the scalar memory -> tile 0 path) | 3 x (x, v) | 3 x 384 bit-exact with `mul_ref(softmax_ref(x), v)` |
| `gen_attention(P)`, P = 1, 7, 64, 256 | 3 x (q, K, V) each | 4 x 3 x 288 bit-exact with `attention_ref` |
| `gen_attention(7, debug_smem=True)` | 3 x (q, K, V) | the 42 scores and 42 p words in scalar memory bit-exact |

Two earlier versions of the composed program failed on the device without hanging (section 7.1). Timing at P = 256 (default
clock, 250 MHz): **1.24 ms per call, wall time with USB** (median of 30; repeated calls give identical outputs). Each call sends
147 KB of K and V and the 27 KB program over USB.

## 2. Geometry

A [1, 6, P, 48] tensor lies on the 16 tiles as an image of 6 rows x P columns x 48 channels (`eltwise.tile_split`): tile row r
holds heads `ROWS = [2, 1, 2, 1]` (`HEAD0 = [0, 2, 3, 5]`), tile column c the positions `cols = tile_split(P)` from `p0(c)`
(P < 4 leaves columns empty, e.g. P = 2: [1, 0, 1, 0]). Every tile keeps its block `[heads][positions][48]` dense. Tiles with the
same instruction fields share one instruction; tiles of different block shapes stay separate even when the fields encode the same
(`_group` with a `shape` key), as the compiler does.

## 3. The pieces

### 3.1 Scores (`gen_scores`)

```
prologue
q input   input4d(W=1): conv2d's input path of a 6 x 1 x 48 image (identity row first) -> q on the column-0 tiles
4 x reset17
q broadcast: per own position j of column 0 a reformat copy of q into slot j of the replicated block (chain._copy), reset17s;
          per destination column c and position j one eastward mesh move of q into slot j: senders (column 0), fc.relay on the
          columns between, receivers; mesh fence (count 1 direct, 6 relayed; it leaves out the relaying tiles of tile rows 0
          and 2), drain, reset17s
K input   input4d(W=P) -> K blocks
MUL       reset17, N2W, PARAMETERS; step 1 (a = the replicated q, b = K through the FIFO); feed; wn fence; step 2 (pack over K)
SUM       reset17, PARAMETERS; feed of the products; opcode-3 op (dp_mode 1, cfg0 0x60, reduce_mask 1: eltops' SUM per position);
          one 4-byte word per score (the score in byte 0, bytes 1..3 zero on the device)
output    per tile its [heads][positions] words: host DMA and (outfeed, ringProducer) per tile, or conv's scalar-memory path when a
          block other than the last is not a multiple of 8 bytes (P = 4, 5, 6, 7, 12, 20, ...)
```

- P = 1: no broadcast. q's input lands in the one-position replicated block itself (staging right behind it), one reset17.
- `input4d` is `conv2d.input_stage` restricted to the tile columns that hold positions (conv2d assumes all four).
- Narrow allocation (`scores_alloc`, fitted on P = 1..256): replicated q at 0 (`rep = 96 smax` bytes), the K block at rep, K's
  staging and the MUL intermediate at 2 rep, q at `max(512, rep)`, the identity row behind q when q sits at 512, else at 512.
  The products go over K, the scores over the replicated q. **Relay slots:** 32 bytes per q move, direct moves included (unused),
  from `128 smax (+ 96 when smax >= 4)` upward, skipping q's 96 bytes.
- Wide: identity row 8316, q's input FIFO 8308, K's input FIFO stacked down from 8316, MUL FIFO 8284, SUM FIFO 8288.

### 3.2 p.V with p expanded (`gen_pv`)

```
prologue, p input (identity row first), reset17, v input, reset17
MUL       as above (a = p, b = V); step 2 writes the products of column c with row stride (P - p0(c)) * 48: room for the gather
gather    meshBus 0x16 chain: column c sends its rows of positions [p0(c), P) west, receives [p0(c+1), P) behind its own; relays
          wait on MESH_EAST_IN (conv2d's halo records) -> all positions of a head row on the column-0 tile
reset17 (two for P < 4), drain
relay     transposing narrowToWide (eltwise.maxpool_relay_n2w_fields): 4 positions per 256-byte row, lane = position
drain
SUM       opcode 3 with MAX_POOL's fields (cfg0 0x127, dp_mode 4, reduce_mask 3): a 1 x P "sum pool" per head, 48 channels,
          w_zp = the products' zero point, requantized with s_pv / s_o
output    the [6, 48] result from the column-0 tiles, one 288-byte DMA
```

- P = 1: no gather and no relay; two reset17s and a requantize op (`eltwise.requant_op_fields`: the sum of one element).
- The transposing relay reads whole quads of 4 positions, i.e. up to `round_up(P, 4)` positions per row: the compiler reserves
  `48 (P + round_up(P, 4))` bytes for the gathered rows.
- Narrow allocation (`pv_alloc`, P = 1..256): staging at 0 (two slots of 48P), identity row behind it; **the MUL intermediate gets
  384P + 4 bytes** (4 bytes per element of a whole row of P positions, though a tile holds at most smax of them);
  P % 4 == 0: `[M | Y (96P) | p | v]`, sums at 0; otherwise Y = 0, `M = 48 (P + round_up(P, 4))`, `[M | p | v]`, sums at M;
  P = 1 its own layout. Wide: the relay block (`8 ceil(P/4)` units) and the input FIFO from the top, the identity row below the
  lower of the two (P <= 4: at the top), the MUL FIFO below everything.

### 3.3 Softmax

`eltops.gen_softmax(6, P, ...)`, unchanged except one constant: the exp2 factor is `f32(f32(beta) * log2 e)` (TFLite stores
beta as float32; `eltops.exp2_scale`). eltops computed `f32(beta * log2 e)`, which differs in the last bit for beta = 1/sqrt(48)
(and 1.3); its tests only used beta = 0.5, 1, 2, where both agree. Fixed in `eltops.py` (`softmax_ref` too); its test still passes.

## 4. The composed program (`gen_attention`)

```
prologue
q input              input4d(W=1) -> q at Q on the column-0 tiles (identity row loaded here, used by all MUL packs)
q to columns 1..3    one eastward mesh move per column (the compiler's q moves for P = 4, where each column has one position)
K input, V input     input4d(W=P) -> K, V blocks (sharing staging and input FIFO)
scores MUL           a = q through the in TTU [(1, 48), (0, positions), (48, heads)]: q_h re-read for every position; b = K
channel SUM          -> one word per score
scores -> scalar mem per tile narrowToWide, ringProducer (ordinal k), 2-D scalar-memory outfeed into words [6][P]
                     (eltops' multi-row softmax input path, one block per tile); RING_OUTFEED reset, PRODUCER_A wait
softmax              scalar core (softmax_words): reads score words, writes p as words with 4 copies of the byte
p -> tile 0          wideToNarrow, ringConsumer, scalar-memory infeed (the compiler's SOFTMAX -> X return, bit 13)
p -> other tiles     fc.broadcast: tile 0's narrowToWide + ringProducer multicast, the receivers' ringConsumer + wideToNarrow
p.V MUL              a = p through the in TTU [(0, 48), (4, positions), (4P, heads)] from Pn + 4 (P h0 + p0): the word of p
                     re-read for all 48 channels; b = V; packs with the gather's row strides
gather, SUM, output  as gen_pv (P = 1: the requantize op)
```

- **The softmax on words** (`softmax_words`): eltops' program with word layouts. It keeps eltops' float32 operations in order
  (so `softmax_ref` is its bit model) and first uses a loaded register 5 bundles after the load, as the compiler's softmax does.
  The device's load latency is unknown. `scalar_core.run(load_latency=L)` now models one; the program is exact for L = 0..4.
- **p as 4-copy words:** the MUL step 1 reads operand a one byte per step. With a stride-0 innermost level the byte lane could be
  taken from the address or from the step count; with 4 equal bytes both readings give p.
- Memory (`attention_alloc`): nothing that is live at the same time shares memory (inputs share staging and input FIFO, the two
  MULs their FIFO). Narrow per tile, in order: staging, identity row, q, relays, K, V, scores intermediate / products / words,
  p (6 x P words on every tile), p.V intermediate, gathered products, sums: 124 KB at P = 256. Wide from 8320 down: identity,
  output FIFO, input FIFOs, MUL FIFO, SUM FIFO, scalar-memory FIFO, p's two FIFOs, relay block (down to 7344 units at P = 256).
  Scalar memory: scores [6][P] words, then p [6][P] words at the top (12 KB at P = 256).
- Size: 1009 words (P = 1), 1971 (P = 7), **1725 words = 27 KB for every P % 4 == 0** (the compiler's scores alone: 9874 words at
  P = 256, because it broadcasts q by copying it into every position).

## 5. Host contract (`attention_io`, `attention_executable`, `attention_inputs`)

- Inputs, three DMAs in this order: q [288] (288 bytes); K and V head-major [6][P][48] (288P bytes each). `attention_inputs`
  builds the buffer `q | K | V` from KV-cache rows [P][288]; `attention_executable`'s hints point into it.
- Output: o [288] (head h at 48h), one 288-byte DMA.
- Every call is self-contained: nothing stays on the chip between calls.

## 6. Bit model (`attention_ref`)

```
s = sum_ref(mul_ref(q[:, None, :], K_h, q_q, k_q, qk_q), axis 2, qk_q, s_q)
p = eltops.softmax_ref(s, s_q, p_q, beta)
o = sum_ref(mul_ref(p[:, :, None], V_h, p_q, v_q, pv_q), axis 1, pv_q, o_q)
```

- `mul_ref`: `(a - za)(b - zb)` exactly, `f32(acc) * mult` (`eltwise.mul_quant`), clamped in float32, rounded half to even, + zp.
- `sum_ref`: `sum(x - zx)` exactly, requantized with `sum_quant` (eltops' SUM formula: `f32(s_in * f32(1/s_out))`, `out_clamps`).
  The same model holds for the channel SUM (opcode 3, dp_mode 1) and the position SUM (opcode 3, pooling class), both on device.
- `softmax_ref`: eltops' model (exact float32 exp2 and reciprocal, round to nearest), verified on device before and here.

All four parts matched the device bit for bit (section 1).

## 7. Findings

### 7.1 What the device showed

1. **A scalar-memory infeed with a 16-tile bitmap delivers no data.** The first composed version sent p from scalar memory to
   all 16 tiles in one infeed (`tiles = 0xffff`, ringConsumer and wideToNarrow on every tile). It ran without hanging, but every
   tile's p buffer held stale bytes: the output was o's zero point for 4 of the 6 heads, the same for every input. The debug build
   (scores and p sent from scalar memory) was bit-exact, so the scalar side was right. The compiler only ever uses this infeed
   with `tiles = 1`. With tile 0 plus FULLY_CONNECTED's ring broadcast the program is exact. Why the multicast fails is unknown.
2. The scores' SUM op writes the score in byte 0 of a word and zeros in bytes 1..3.
3. Stride-0 TTU levels broadcast MUL operand a, at the innermost level (p, 4-copy words) and at the position level (q).
4. The opcode-3 op with MAX_POOL's fields sums (`sum_ref`), with the products' zero point as w_zp.

### 7.2 Precision: the decomposition requantizes every product

MUL writes uint8, so every product q_c k_c and p_j v_j is rounded to 8 bits before the SUMs. Measured on TinyStories-15M's own
activations (q, K, V of all 6 layers on a greedy 256-token story; all 8 quantization points calibrated per layer over that story,
full ranges): relative L2 error of o against float attention.

| layer | P = 7 | P = 64 | P = 256 | the same with only q, K, V and o in uint8 (P = 7 / 64 / 256) |
|---|---|---|---|---|
| 0 | 0.086 | 0.088 | 0.146 | 0.028 / 0.022 / 0.033 |
| 1 | 0.099 | 0.218 | 0.236 | 0.030 / 0.038 / 0.045 |
| 2 | 0.089 | 0.148 | 0.178 | 0.036 / 0.035 / 0.037 |
| 3 | 0.091 | 0.127 | 0.117 | 0.037 / 0.028 / 0.028 |
| 4 | 0.140 | 0.163 | 0.179 | 0.049 / 0.030 / 0.030 |
| 5 | 0.094 | 0.100 | 0.075 | 0.029 / 0.025 / 0.019 |

- About 2.5-6x the error of an attention that only quantizes its inputs and output. The bench simulation that found "attention in
  uint8 costs nothing" (`bench/attention_quant.py`, `bench/RESULTS.md`) quantized the scores once, not every product.
- Both halves contribute: with an exact p.V on the quantized p (the bit model's, = the device's), the error is still 0.08-0.13
  (layers 1 and 4).
- Narrower product ranges (99.9 / 99.5 % percentiles) are worse: the clipped large products are the ones attention needs.
- The effect on perplexity was not measured.

**A better decomposition:** dot products in the MXU, which accumulates in int32 and requantizes once. The scores are a 1x1
convolution over the K "image" (positions x 288) with q as the weights: 6 output channels, block-diagonal over the 288 inputs.
p.V is the same with p as the weights over V^T (48 positions x P channels per head). The weights must be in wide memory in the
`[K/4][cols][4]` layout. For q that is a per-call parameter blob (the argmax's trick, a caching program every call) or a
narrowToWide inside the program. For p, computed on the chip, only the latter works. That transfer exists (the MUL's FIFO feed
writes runtime data into wide memory), but the weight layout and the dp_mode-3/4 MXU semantics for runtime weights are not decoded.

### 7.3 What edgetpu_compiler 14.1 maps (P = 64)

| graph | result |
|---|---|
| SOFTMAX [1,6,64] -> RELU | maps (the scalar memory -> tile 0 return of `check_return_path`) |
| SOFTMAX -> MUL [1,6,64], no broadcast | maps (ran on the device, section 1) |
| SOFTMAX -> RESHAPE -> FULLY_CONNECTED (384 -> 64) | internal compiler error |
| scores -> SOFTMAX -> RESHAPE to p [1,6,1,P] -> MUL with V^T [1,6,48,P] -> SUM axis 3 (the whole attention) | internal compiler error |
| p [1,6,1,P] broadcast x V^T [1,6,48,P], SUM axis 3, alone | maps: 2386 words, 4-byte output words like the scores |
| p expanded [1,6,P,48] x V [1,6,P,48], SUM axis 2 | maps: 776 words (`gen_pv`'s oracle) |

So a p broadcast over the channels does have a compiler program when V is stored transposed (the scores piece's shape).
The compiler crashed on p broadcast over V's channels in V's own layout during the attention study; that form was not retested here.
`gen_attention` keeps V as [6][P][48] (a KV-cache row per token) and broadcasts p with a stride-0 level that has no oracle.

## 8. Copied rather than understood

- The allocation rules of both pieces (sections 3.1, 3.2) are fits. Examples: the 384P + 4-byte MUL intermediate, the relay slots
  from 128 smax (+ 96), q at max(512, rep), and the special cases P = 1 and 4.
- The reset17 counts between stages: 4 after q's input, two after the gather for P < 4, two before the P = 1 requantize.
- The q broadcast of the scores piece: per-position copies and mesh moves.
- The scalar-memory return path's two forms: one packet (mode 3, cbuf, s_id 43) or several (mode 1, s_id 11). The infeed fields
  are reproduced as observed: bit 13, k157, k434, f438. For rows of one word (P <= 4 in the compiler's SOFTMAX) the compiler writes
  the narrow walk as one level and sets tail_en870 = 1, tail_f871 = 0 (otherwise 0, 1). `smem_to_tiles` does the same;
  `gen_attention` keeps the two-level walk at P = 1 (`collapse=False`), the form that ran on the device.
- The position SUM is MAX_POOL's op with opcode 3; the meaning of cfg0 0x127 / reduce_mask 3 / psum flags is unknown.
- Inherited from conv2d / fc / eltops: copy-op progress fields, mesh record values (`fc.relay`: 5 + 4c, -3 + 4c), the
  scalar-memory output words of conv (gen_scores at small P), the outfeed's 2-D walk fields.
- Why a 16-tile scalar-memory infeed does not deliver (7.1).

## 9. What remains for a whole layer in one program (qkv FC -> RoPE -> attention -> wo; RMSNorm and residual on the host)

1. **The KV cache on the chip.** Today K and V cross USB every call: 147 KB at P = 256, about 0.4-0.5 ms of the 1.24 ms at
   NOTES.md's 300-360 MB/s host -> device (estimated, not measured separately; the 27 KB program adds about 0.08 ms).
   - The cache needs a layout that does not change with P. tile_split(P) moves positions between tile columns as P grows, so
     use for example a fixed 64 positions per column.
   - Each call must append the new k, v row from the qkv FC's output tiles into the cache blocks.
   - Unknown: whether narrow memory keeps its contents between programs. A one-program test
     would settle it.
2. **qkv FC -> attention.** The FC writes 64-output slices per tile. q must reach the column-0 tiles as [heads][48] rows, and k, v
   their cache slots: mesh / ring moves of the kind fc.gather and the q moves use.
3. **RoPE.** Fold the pair rotation into the qkv weights (the FC emits q, rot(q), k, rot(k), v). Then two MULs by cos / sin rows
   and an ADD (both decoded; cos / sin as per-call inputs or resident).
4. **attention -> wo FC.** o leaves the column-0 tiles as [heads][48]. It must be gathered to tile 0 and broadcast as the FC
   input (fc.gather / fc.broadcast, as here for p).
5. **Precision (7.2).** The MXU formulation of both halves is the main open piece: it removes the per-product requantization.
6. **Program size.** 1725 words (27 KB) of instructions per call for the attention alone; instructions cross USB every call (as
   for every program today). Whether a whole layer with its FCs fits one 16384-word bitstream (`codegen_fused.md`) is not checked.

## 10. Risks

- `gen_attention` is our own composition. It ran bit-exact for P = 1, 7, 64, 256 only; other P use the same builders, are checked
  offline (section 1) and have not run.
- The address checker sees start addresses and extents from the TTUs, not access widths. Over-reads such as the relay's quads are
  allowed only because the buffers were sized for them.
- The p.V MUL's stride-0 innermost level is safe only because p is replicated into words (section 4); other operands would need
  the same care.
