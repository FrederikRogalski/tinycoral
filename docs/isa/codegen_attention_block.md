# The attention block as one program (`coral/codegen/attention_block.py`)

`gen_attention_block(P, quant)` builds the PARAMETER_CACHING and EXECUTION_ONLY programs for the whole attention block of one
TinyStories-15M layer at batch 1 (d = 288, 6 heads x 48, the new token at position P-1, P = 1..256):

```
x [288]  (uint8, RMS-normalized by the host)
  -> q, swap(q), k, swap(k), v = five FULLY_CONNECTED 288 -> 288 from wqkv        (RoPE's pair rotation folded into the weights)
  -> q' = q*cos + swap(q)*sin',  k' = k*cos + swap(k)*sin'                          (MUL, MUL, ADD; cos / sin' per call)
  -> K = [cache rows 0..P-2 from the host, k'],  V = [cache rows, v]                (k', v written into row P-1 on the chip)
  -> att = attention over P positions                                               (coral/codegen/attention.py's core)
  -> o = wo(att)
outputs: o [288], k' | v [576] (the host appends them to its KV cache)
```

Every instruction comes from the encoders in `coral/isa/` and the builders of `codegen/{fc,attention,chain,conv2d}.py`, plus the
stage builders here: `scatter_input`, `mul_1d`, `add_1d` (they reproduce edgetpu_compiler's 1-D RoPE programs byte for byte,
section 1), `x_input` (fc.input_block with its own broadcast FIFO), `gather_north` (fc.gather's north phase with the head rows as
pieces), `slot_copies` and `output_tile0`.

```
python -m coral.codegen.attention_block             # acceptance test (= python test/test_codegen_attention_block.py), offline, P = 1..256
python -m coral.codegen.attention_block --quick     # 24 sample P
.venv/bin/python test/test_hw_attention_block.py rope 2         # on the device (tools/hw.py): rope Q | block P [timed calls]
.venv/bin/python test/test_hw_attention_block.py block 64 30
```

| API | |
|---|---|
| `gen_attention_block(P, quant=None, param_offset=0)` | `(caching, execution, io)`; quant: `block_quant`'s keys (section 7); param_offset: bytes into tiles 0..4's wide memory where the six blobs start |
| `block_params(wqkv, wo, quant)` | the parameter blob (560 640 bytes): `fused.conv_blob` of q, swap(q), k, swap(k), v, wo |
| `block_executables(caching, execution, P)` | `coral.executable.Executable`s for `coral.runtime.run_executable` / `tools.hw.run` (section 5) |
| `block_inputs(x, cos, sin, K, V)`, `block_outputs(buf)`, `rope_inputs(pos, quant)`, `block_io(P)` | host contract (section 5) |
| `attention_block_ref(x, cos, sin, K, V, wqkv, wo, quant, parts=False)` | the bit model (section 6) |
| `attention_block_float(x, pos, K, V, wqkv, wo, quant)` | the same block in float64 from the dequantized inputs |
| `fc_ref`, `add_ref`, `rope_ref` | bit models of FULLY_CONNECTED, ADD (dp_mode 5), RoPE |
| `block_quant`, `core_quant`, `matmul_quant`, `param_offsets`, `param_end`, `block_alloc` | quantization, placement, memory |

## 1. Verification

Offline (`python test/test_codegen_attention_block.py`):

| check | result |
|---|---|
| edgetpu_compiler's program for x -> FC, FC -> MUL(., cos), MUL(., sin) -> ADD, rebuilt from `scatter_input`, `mul_1d`, `add_1d` and fc.py's blocks, 3 quantizations (one with exact .5 ties in the ADD, one with ADD weights (5, 7)) | 3/3 byte-identical |
| the same MUL, MUL, ADD on four inputs | 3/3 byte-identical |
| parameter blob = conv_blob of the six matrices; the caching program's ringConsumers / infeeds at `param_offsets` (offsets 0 and 25 600) | 3/3 |
| `gen_attention_block(P)`, P = 1..256: every instruction decodes and re-encodes; sequence numbers; DMA descriptors = host contract; every narrow range inside one buffer of `block_alloc`; FC ops and bias loads inside their matmul's region, every other wide address above the parameters; scalar-memory transfer sizes; the four ring broadcasts (x, q'\|k'\|v, p, o: tiles and sizes); **data flow**: on every tile every byte an instruction reads was written earlier in program order (two allowed exceptions, below); the slot copies are the last writers of exactly row P-1 of K and V | 256/256 |
| the block's scalar-core softmax run by `scalar_core.run` on random score words, load latency 0, 2, 4 | 5/5 P, bit-exact with `eltops.softmax_ref` |
| bit models on a calibrated layer (layer 3, greedy 256-token story): uint8 RoPE within 2.5 LSB of the float rotation (positions 0, 1, 17, 255); the block near float math | 8/8 |

The data-flow check allows two kinds of reads of unwritten bytes: the transposing relay of the position SUM reads whole quads of 4
positions (the buffer is sized for it), and the K / V input path's copy ops are issued before the input DMA that fills their
staging (they wait on its counters). The test also checks the checker: a broadcast without one tile, slot copies at row P-2, no slot
copies and a shifted north gather are each caught (`check_mutations`).

Bit model vs float math (offline, the story's own inputs, relative L2 error of o): 0.055 (P = 1), 0.106 (7), 0.095 (64), 0.087 (256);
k' within 1 LSB of the float rotation, v exact. The error is the elementwise attention's (every product rounded to uint8,
`codegen_attention.md` 7.2) plus the uint8 FCs.

On the device (`test/test_hw_attention_block.py`, through `tools/hw.py`; 9 runs, no failed transfer; real layer-3 weights, calibrated
quantization; 3 inputs per run: the story's x and cos / sin' at position P-1 with its cache rows, x of another position and a
Gaussian x, both with cos / sin' of other positions):

| program | result |
|---|---|
| edgetpu_compiler's FC, FC -> MUL, MUL -> ADD (byte-identical to `gen_rope_piece`), set 2 (142 exact .5 ties in the ADD) | 3 x 288 bit-exact (`fc_ref`, `mul_ref`, `add_ref`) |
| the same, set 1 (ADD integer weights 5, 7) | 3 x 288 bit-exact |
| `gen_attention_block(P)`, P = 1, 2, 3, 7, 64, 255, 256 | 7 x 3 x (o, k', v) bit-exact with `attention_block_ref` |

These are the first device runs of ADD: dp_mode 5 with its 16-bit weight matrix, `add_ref`'s arithmetic, rounding half to even.

**Timing** (default clock, wall time per call with USB, median of 30 repeated calls, identical outputs): **0.657 ms at P = 64,
1.379 ms at P = 256** (the attention core alone: 1.24 ms at P = 256). Per call the host sends the 66 KB program and 864 + 576 P input
bytes (148 KB at P = 256) and reads 864 bytes; the split between USB and compute was not measured.

## 2. Program structure

```
prologue
x input       x_input: fc.input_block's stages (scatter to tiles 0..4, mesh gather to tile 0, ring broadcast to tiles 1..4), own FIFO
cos, sin'     scatter_input: the compiler's 1-D input (tile t takes bytes [64t, 64t + 64) to narrow X + 64t), no gather
K, V          attention.input4d (P >= 2; the identity row is loaded with K's input); P = 1: no cache rows, the identity row alone
5 x FC        fc.ops_block for q, swap(q), k, swap(k), v on tiles 0..4 (bias load, one op per tile); outputs in the 1-D layout
              (tile t: outputs [64t, 64t + 64) at Y + 64t), v straight into Q + 576
RoPE          mul_1d(q, cos) -> qc, mul_1d(swap(q), sin') -> qs (288 bytes after qc), add_1d(qc, qs) -> q' at Q; the same for k -> Q + 288
              (per tile on its 64-byte slices; each MUL: reset17, N2W / PARAMETERS fences, step 1 x 5, FIFO feed x 5, wn fence,
              step 2 x 5; each ADD: two reset17, 10 mesh fills of the weight matrix, narrowToWide to 4 wide rows, op x 5, reset17)
gathers       3 x fc.gather (the FULLY_CONNECTED input gather of a 288-byte vector): q', k', v onto tile 0 -> Q = q' | k' | v
broadcast     fc.broadcast of Q (864 bytes) from tile 0 to every other tile that holds positions
slot copies   chain._copy of k' and v (6 heads x 48 bytes) into row P-1 of the K and V blocks, on the tiles of the last tile column
              that holds positions
core          attention.gen_attention from the scores on, q_h read from Q + 48 h on every tile (stride 0 over the positions):
              MUL, channel SUM, scores -> scalar memory, softmax on the scalar core, p -> tile 0 -> ring broadcast, p.V MUL, gather,
              position SUM -> att on the column-0 tiles (tile row r: its heads at narrow O)
o gather      gather_north: tiles 4, 8, 12 send their head rows north to tile 0 at O + 48 HEAD0[r] (relays on tiles 4, 8): att [288]
wo            fc.broadcast of att to tiles 1..4, fc.ops_block (completion fence)
outputs       fc.output_block (o from tiles 0..4), then k' | v from tile 0 (one 576-byte DMA), epilogue
```

Size: 3248 words (P = 1), 4404 (P = 7, 255), **4124 words = 66 KB for every P % 4 == 0** (1725 of them the attention core's). The RoPE
stages take 1158 words: the compiler's 1-D form has one instruction per tile and step.

**Where k' and v land.** The program writes them into row P-1 of the K and V blocks itself (the "best" option), and also sends them
to the host at the end:
- One call per token. "Computed and sent back first" would need a second call (or a host round trip inside one call) before the
  attention could use them.
- k' and v are on tile 0 anyway after the gathers, and every tile that holds positions receives q' | k' | v by the broadcast that q'
  needs. Writing row P-1 is then one reformat copy per tile row of the last tile column (8 instructions in all).
- The host sends K and V with P rows: rows 0..P-2 from its cache and row P-1 as a placeholder (zeros), so that the blocks keep
  `tile_split(P)`'s layout (a P-1 row input would be distributed by `tile_split(P-1)`). It costs 576 extra bytes per call.
- At P = 1 there is no cache: no K / V input at all, the copies fill the one-position blocks.

## 3. RoPE

llama2.c's interleaved pairs: elements (2j, 2j+1) of a head rotate by `pos * 10000^(-2j'/48)` (j' = j mod 24).
- **Folding the rotation into the weights:** `q' = q*cos + swap(q)*sin'`, with `swap(q)[2j] = q[2j+1]`, `swap(q)[2j+1] = q[2j]`, and
  `sin'[2j] = -sin`, `sin'[2j+1] = +sin`. The sign lives in the per-call input sin', so swap(q) is a pure permutation of q:
  the FC computes it from wq's rows with every pair exchanged, with q's quantization, i.e. exactly q's values permuted.
- cos and sin' are per-call inputs (`rope_inputs(pos)`), quantized with (1/127, 128): -1, 0, 1 are exact.
- The products qc, qs (and kc, ks) default to one shared quantization, so the ADD's integer weights are (1, 1): an exact sum,
  rounded once.

## 4. Memory

- **Parameters** (`param_offsets`): six FULLY_CONNECTED 288 -> 288 blobs (`fused.conv_blob`: per 64 outputs an int32 bias row, then
  `[72][64][4]` weights; 93 440 bytes, 292 units per tile) on tiles 0..4, one after another from `param_offset`: q, swap(q), k,
  swap(k), v, wo, 1752 units (112 128 bytes) per tile. One caching program caches all six (per blob its ringConsumers, then its
  infeeds, as edgetpu_compiler caches several FCs).
- **Narrow** (`block_alloc`, bytes, every tile, 64-byte aligned; nothing that is live at the same time shares memory): input staging,
  identity row, x, the relay buffer of every gather, cos, sin', the four FC outputs, the RoPE MUL intermediate (256), QC = qc | qs,
  KC = kc | ks, the ADD weight fills, Q = q' | k' | v, then the attention core's buffers (`attention.narrow_sizes`), att, wo's output.
  128 768 bytes at P = 256.
- **Wide** (64-byte units, from the top): identity row (8316), output FIFO (8312), 2 rows unused, K / V input FIFO, MUL FIFO, SUM FIFO,
  scalar-memory FIFO, p's two FIFOs, the p.V relay block, the ADD weight rows (4 rows), the q' | k' | v, o and x broadcast FIFOs.
  The three 1-D inputs come first and use the compiler's ring FIFO at [8304, 8320) before the identity row is loaded there.
  `block_alloc(Geom(P))["wide_bottom"]` = the lowest unit (8196 at P = 1, 7316 at P = 256): resident parameters must end below it.
- **Scalar memory:** the score words [6][P], then p's words [6][P] (the attention core's).

## 5. Host contract (`block_io`, `block_executables`, `block_inputs`, `block_outputs`)

- Inputs, in DMA order: x [288], cos [288], sin' [288]; for P >= 2 also K and V, each [6][P][48] uint8 head-major (row P-1 a
  placeholder). `block_inputs(x, cos, sin, K_cache, V_cache)` builds the one host buffer from the P-1 cache rows [P-1][288].
- Outputs, in DMA order: o [288], then k' | v [576]. `block_outputs` splits them.
- Parameters: `block_params(wqkv, wo, quant)` once, through the caching program (`tools.hw.run(caching, execution, inputs,
  params=blob)`). wqkv is llama2.c's wq | wk | wv stacked ([864][288], one (scale, zero point)); wo [288][288].
- Every call is self-contained apart from the cached weights: nothing else stays on the chip between calls.

## 6. Bit model (`attention_block_ref`)

```
q = fc_ref(x, Wq), k = fc_ref(x, Wk), v = fc_ref(x, Wv)                         (swap(q) = q[SWAP] exactly)
q' = add_ref(mul_ref(q, cos), mul_ref(q[SWAP], sin')), k' likewise
att = attention.attention_ref(q', [K; k'], [V; v])
o = fc_ref(att, Wo)
```

- `fc_ref`: `sum (x - zx)(w - zw)` exactly, `f32(acc) * f32(mult)`, float32 clamps (`eltwise.out_clamps`), rounded half to even, + zp.
- `add_ref`: `w1 a + w2 b + offset` exactly (`eltwise.add_quant`: integer weights of the scale ratio, offset = -(w1 z1 + w2 z2)),
  then the same requantization. Operand 1 is the one at the op's in_base (qc in the block).
- `mul_ref`, `attention_ref`: attention.py's, verified on the device before.

All parts matched the device bit for bit (section 1).

## 7. Quantization (`block_quant`; (scale, zero point) per tensor)

| key | tensor |
|---|---|
| `x` | the block input |
| `wqkv`, `wo` | the weights |
| `qf`, `kf`, `v` | the FC outputs q / swap(q), k / swap(k), v (v is the attention's v) |
| `cos`, `sin` | the per-call RoPE inputs, default (1/127, 128) |
| `qc`, `qs`, `kc`, `ks` | the RoPE products; qs / ks default to qc / kc |
| `q`, `k` | after RoPE (the attention's q, k) |
| `qk`, `s`, `p`, `pv`, `att`, `beta` | the attention core's products, scores, probabilities (1/256, 0), products, output, 1/sqrt(48) |
| `o` | the block output |

`test_codegen_attention_block.calibrated(layer)` calibrates all of them on a greedy 256-token story of the float model (full
ranges, `coral.fused.quantize_params`).

## 8. Findings

1. **ADD runs as decoded** (eltwise.md 6): the 16-bit mode with the integer weight matrix filled by 10 mesh fills, and
   `add_ref`'s arithmetic with rounding half to even (exact .5 ties). First device run of ADD.
2. **edgetpu_compiler's 1-D elementwise chain** (FC outputs in the 1-D layout feeding MUL and ADD) is one instruction per tile and
   step; the compiler loads the identity row inside the program's first MUL (between the feeds and the wn fence), and then the
   NARROW_TO_WIDE fence after that MUL carries count 1 (`mul_1d(ident_narrow=...)`). Inputs after the identity row was loaded move
   their ring FIFO down to end at 8316 (`scatter_input(top=...)`). The ADD's weight rows go to 4 wide rows (16 units); the compiler
   puts them over the identity row once that is dead.
3. **Ring broadcasts from tile 0 work with FIFO addresses of our choice** (x, q' | k' | v, p and o each have their own), and several
   outputs in one program are plain repeated output blocks with ringProducer ordinals restarting at 0 (as in the compiler's
   multi-output programs).
4. **Flat attention at large P loses precision:** with an x that does not match the cache (inputs #1, #2 of the device test) p
   spreads over all 256 positions and rounds to 0..7 of 1/256; o is then 0.4-0.7 away from float math (relative L2), bit-exact with
   the model. The story's own inputs stay at 0.09.

## 9. Copied rather than understood

- The RoPE stages are the compiler's 1-D form, reproduced byte for byte: the two reset17 before an ADD's fills, the extra reset17
  between the MULs, the NARROW_TO_WIDE count 1, the ADD weight transfer's four MESH_*_IN sync records (val 0x8001, `rsv589` /
  `rsv631`), the round robin of the fills over the four mesh buses.
- The mode-2 bias load (`wide_narrow.w2n_bias`) writes a "bias store", not narrow memory; its semantics are not decoded.
- `gather_north` uses fc.gather's north-phase rules (relay records 5 + 4c / -3 + 4c, fence = packets forwarded so far, a drain with
  the last fence's mask, the mesh counter reset) with the head rows as pieces of 24 / 12 / 24 / 12 words; the rules are fits.
- Inherited from the attention core (`codegen_attention.md` 8): its allocation rules, reset17 counts, the scalar-memory paths' fields,
  the position SUM's MAX_POOL fields; why a 16-tile scalar-memory infeed delivers nothing.
- The wide FIFO addresses and the narrow plan are ours; the compiler's own choices for this block are unknown (it cannot compile it).

## 10. What remains

**A whole layer in one program** (x -> attention block -> residual -> RMSNorm -> FFN -> residual):
1. **Residual** h = x + o: one more `add_1d` on tiles 0..4 (the residual x as an input [288] or kept on the chip, variant R).
   o is already in the 1-D layout on tiles 0..4.
2. **RMSNorm** = L2_NORMALIZATION (`eltops.gen_l2norm`, verified on the device) times sqrt(288) w. Fold w into the next FC's
   weight columns and sqrt(288) into its input scale: only the L2 norm stays. It runs on one tile in the compiler's form, so h
   needs a gather (fc.gather) and the result a broadcast.
3. **The FFN** from `fused.gen_ffn(1)`'s FULLY_CONNECTED form (w1, w3 on tiles 0..11, LOGISTIC, two MULs, m gathered on tile 0,
   w2 on tiles 0..4), with its stages called on our buffers. Its parameters (695 KB spread over 12 tiles) go next to this block's
   (112 KB on each of tiles 0..4).
4. **Wide memory:** 1.26 MB of parameters per layer (incl. the 187 KB of swap(q), swap(k) rows). Six layers (7.5 MB) only fit when
   the per-layer matmuls move to other tiles (fc.py's tile shift with `move_gather=False`, which runs on the device) and the swap
   rows go (e.g. read q in pair-swapped order through the MUL's in TTU instead: negative TTU strides, untested).

**All 6 layers in one call:**
- **Program size:** 4124 words for this block plus 2298 for `gen_ffn(1)` plus the norm and residual glue, about 40 000 words
  (~650 KB) for six layers: several bitstreams (`fused.split_bitstreams`, as the argmax), all sent over USB every call, about
  1.8-2.2 ms per token at 300-360 MB/s (estimated). A tile-local layout for the 1-D vectors (one instruction for tiles 0..3 instead
  of one per tile) would roughly halve the RoPE / FC part.
- **KV cache on the chip:** K / V cross USB every call today (148 KB per layer at P = 256). This block already writes row P-1 on
  the chip; keeping the blocks needs (a) narrow memory to survive between programs (untested) and (b) a
  layout that does not move with P (`tile_split(P)` redistributes positions; e.g. 64 fixed positions per tile column).
- **Precision:** the elementwise attention rounds every product (`codegen_attention.md` 7.2); norms and the residual in uint8 cost
  more (`bench/RESULTS.md`: perplexity 2.39 / 2.69 vs 2.16).

## 11. Risks

- Ran on the device at P = 1, 2, 3, 7, 64, 255, 256 only; other P use the same builders and pass the offline checks.
- The data-flow check follows program order; it does not model the hardware's sync counters, so it cannot see a race between
  units that the fences would not prevent.
- The parameter offsets are ours (tiles 0..4 from `param_offset`); another program that writes wide memory below `wide_bottom`
  on tiles 0..4 destroys them (the runtime's `floor` rule applies).
