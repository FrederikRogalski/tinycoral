# A whole transformer layer, and all six, as one Edge TPU program (`coral/codegen/layer.py`)

`gen_model(P, quants, plan, local=True)` builds the PARAMETER_CACHING program and the EXECUTION_ONLY bitstreams of TinyStories-15M's
transformer at batch 1: every layer, from the residual stream in to the residual stream out, for the new token at position P-1
(P = 1..256), in **one program call per token**. edgetpu_compiler can't compile any of it (it has no attention). The host keeps the
embedding lookup, the final RMSNorm and the 32000 x 288 classifier (numpy), the sampling and the uint8 KV cache
(`examples/stories_chip.py`).

```
x (uint8 residual)  -> L2_NORMALIZATION (RMSNorm: its weight and sqrt(288) folded into the next FC's weight columns)
                    -> the attention block: q, k, v FCs (q, k rows in the order SIGMA) -> RoPE (MUL, MUL with swap(q) read in
                       pair-swapped order, ADD) -> attention over the host's KV rows + this token's k', v (attention.py's core) -> wo
                    -> ADD (residual) -> L2_NORMALIZATION -> FFN (w1, LOGISTIC, MUL, w3, MUL, w2: fused.py's FC form) -> ADD
outputs: x_out [288] after the last layer, and k' | v [576] of every layer (the host's KV cache rows)
```

```
python -m coral.codegen.layer --quick                     # acceptance test (= python test/test_codegen_layer.py), offline
.venv/bin/python test/test_hw_layer.py model 1,2,64,256 30 local    # on the device: 6 layers, 3 inputs per P, 30 timed calls
.venv/bin/python examples/stories_chip.py --steps 200      # generation; --mock (bit model), --float, --eval N, --check DIR
```

## 1. Results

**On the device** (`test/test_hw_layer.py` through `tools/hw.py`; real weights, calibrated; each job 3 inputs: the story's state at
P-1 with its KV rows, another position's residual, a Gaussian residual; every output compared with `model_ref`):

| program | result | wall time per call (USB included, median of 30) |
|---|---|---|
| one layer (layer 3), P = 1, 2, 7, 64, 256, both layouts | bit-exact (x_out, k' \| v) | local: P=2 0.51 ms, 7 0.53, 64 0.72, 256 1.50 (global: 0.59, 0.64, 0.82, 1.57) |
| one layer with q, w1, w2 streamed from other tiles, P = 1, 64, both layouts | bit-exact | as spread |
| **all 6 layers** (19 matmuls streamed), P = 1, 2, 64, 256, both layouts | **bit-exact** (x_out, 6 x k' \| v) | local: **1.63 / 1.88 / 3.22 / 7.59 ms** (global: 2.12 / 2.28 / 3.62 / 8.16) |
| every prefix program (`STOPS`, 16 stages of a layer) at P = 1, both layouts | bit-exact | |
| edgetpu_compiler's LOGISTIC, 18 input quantizations x all 256 inputs | `logistic_ref` exact | |
| edgetpu_compiler's L2_NORMALIZATION -> FC 768 -> LOGISTIC (control) | bit-exact | |

**Generation** (`examples/stories_chip.py`, 201 tokens, greedy): **176.1 tok/s** at batch 1 (local layout; TPU call 4.29 ms median
with USB, host 1.27 ms per token), 160.4 tok/s with the global layout. Today's batch-1 path (`examples/stories.py`, DEV=CORAL, 18
programs per token with attention on the host) does ~70 tok/s. With `--check` (every call compared with the bit model) 199 of 205
calls were bit-exact; the 6 others differ from a softmax rounding tie (section 7.4), the same calls in both layouts.

**Offline** (`python test/test_codegen_layer.py`): every instruction decodes and re-encodes, sequence numbers (and < 2^14, the DMA
instructions' field), DMA descriptors = host contract (one hint per DMA, sizes 288 / 384 / 576 / 768), every narrow access inside one
buffer, FC weight reads and parameter streams inside their matmul's region, every other wide address above the parameters, the
identity row loaded before its first use, program-order data flow on every tile, the swap-read walks, the slot copies writing exactly
row P-1; for 1 layer at P = 1..256, 6 layers at sample P, the prefix programs, streamed plans, both layouts. Bit models: the L2 norm
with any output scale, the parameter blob and caching program, the whole model near float (section 8).

Program size per call, 6 layers: local 306 KB (P=1) / 388 KB (P=64, 2 bitstreams); global 508 / 591 KB (3 bitstreams). Per layer at
P=64: local ~4100 words, global ~6300 (MULs 1846, FCs 1056, ADDs 620, attention core ~1050, gathers 505, LOGISTIC 251).

## 2. API

| function | |
|---|---|
| `gen_model(P, quants, pl=None, stop=None, reload_ident=True, local=False)` | `(caching, [bitstreams], io)`; quants: per layer (section 6); pl: `plan(L)`; stop: `(layer, stage)` prefix program (bisection); local: the tile-local layout |
| `plan(L, limit=None, stream=None)`, `check_plan`, `regions`, `model_limit(P)` | parameter placement (section 5) |
| `model_params(Ws, quants)`, `model_caching(pl)`, `model_executables(caching, bitstreams, io)` | the blob, the caching program, Executables with hints |
| `model_inputs(x, cos, sin, Ks, Vs)`, `model_outputs(buf, L)`, `model_io(P, L)`, `rope_inputs(pos, quant)` | host contract (section 4) |
| `model_ref`, `layer_ref`, `attention_ref`, `ffn_ref`, `l2norm_ref`, `logistic_ref`, `stop_ref` | bit models (section 7) |
| `FloatModel`, `ChipModel`, `calibrate_split`, `quants_from`, `quant_weights`, `float_stories`, `evaluate`, `folded` | reference, calibration, quality (sections 6, 8) |

## 3. Program structure

```
prologue; inputs x, cos, sin' (attention_block.scatter_input: 1-D layout on tiles 0..4); the identity row (EL.ident_prologue)
per layer l:
  gather x onto tile 0 (fc.gather; layers >= 1 in the local layout: gather_local)        [stop x]
  L2_NORMALIZATION on tile 0 (eltops.l2norm_stage: gen_l2norm's ops, its FIFOs)          [stop xn]
  identity row again; broadcast xn to tiles 1..4 (fc.broadcast)
  K, V cache rows of layer l (attention.input4d, P >= 2)
  q, k, v FCs on tiles 0..4 (spread: fc.ops_block / streamed: fused's streamed stage)   [stops q, k]
  RoPE: MUL(q, cos), MUL(q read pair-swapped, sin'), ADD -> q' ; the same for k'         [stops qc, qs]
  q', k', v onto tile 0 into Q_l (gathers)                                                [stop qkv]
  broadcast Q_l to every tile with positions; k', v into row P-1 of the K / V blocks (attention_block.slot_copies)
  the attention core (attention.py: scores MUL + channel SUM, softmax on the scalar core, p broadcast, p.V MUL, position SUM)
  gather_north the attention onto tile 0                                                  [stop att]
  broadcast; wo FC; ADD h = o + x; gather h onto tile 0                                   [stop h]
  L2_NORMALIZATION (tile 0)                                                               [stop hn]
  identity row again; broadcast hn to tiles 1..11
  w1 FC on tiles 0..11; LOGISTIC (with the identity row loaded right before, edgetpu_compiler's sequence); MUL a = h1 g;
  w3 FC; MUL m = a h3                                                                     [stops h1, g, a, h3, m1]
  gather m onto tile 0                                                                    [stop m]
  broadcast m to tiles 1..4; w2 FC; ADD x = y + h (the next layer's residual)
gather x_out onto tile 0; outputs: x_out (288 bytes), then k' | v of every layer (576 bytes each, from Q_l); epilogue
```

Every stage is an instruction sequence that ran on the device before (fc.py, attention_block.py, attention.py, fused.py's FC form,
eltops.py), with our own allocation. Stages are separated by reset17 / drains, so buffers of different stages share wide memory.

**Two layouts of the 1-D vectors.** Global (`local=False`, the compiler's form): tile t's 64 bytes at base + 64t, one op per tile.
Local (`local=True`): tile t's 64 bytes at base on every tile, so tiles with equal fields share one instruction (FC ops for tiles
0..3, or 0..11; MUL / LOGISTIC / ADD per tile group): about a third fewer words. edgetpu_compiler uses it itself for an FC feeding a
per-tile op (one FC op for tiles 0..11 writing narrow 0, in its L2 -> FC -> LOGISTIC program). The gathers then use `gather_local`
(fc.gather's mesh phases with local sources; row r's first tile collects its row contiguously). The inputs x of layer 0, cos and sin'
stay in the 1-D layout of their input DMA: the ops that read them (layer 0's first ADD, the RoPE MUL feeds) take one address per tile.

**Prefix programs** (`stop=(layer, stage)`, `STOPS`): the same program cut after a stage, the stage's vector put on tile 0 (or, for
the FFN's 768-byte vectors, output from tiles 0..11) and sent out. They localized every fault in section 7.

## 4. Host contract (`model_io`)

- Inputs, one buffer, in DMA order: x [288] = the token's embedding quantized with `quants[0]['res']`; cos, sin' [288] =
  `rope_inputs(pos)` (permuted by SIGMA); then per layer K, V [6][P][48] uint8, head-major (`attention.heads` of the cache rows), row
  P-1 a placeholder the program overwrites (P >= 2).
- Outputs, in DMA order: x_out [288] (`quants[-1]['out']`), then per layer k' | v [576] (k' in the order SIGMA). The host appends
  them to its cache; it never interprets k' (the permutation is inside every head, section 4.1).
- Parameters: `model_params(Ws, quants)` through `model_caching(plan)`, once (6.42 MB). One placement for every P, so the 256
  execution programs share the cached weights.
- P = 1 (the BOS token) runs with its own quantization (section 6), P >= 2 with the main one; both use the same weights and K / V
  quantization.

### 4.1 RoPE without a second FC

llama2.c rotates the pairs (2j, 2j+1): q' = q cos + swap(q) sin' (sin' carries the sign, attention_block section 3). The q and k
FCs compute their rows in the order `SIGMA`: per group of 16 outputs, the first elements of 8 consecutive pairs, then their second
elements. 16 divides 48 and 64, so every group lies inside one head and one tile's chunk. swap(q) is then the half swap of every
16-byte group; MUL step 1 reads it with the walk (1, 8), (-8, 2), (16, groups) from base + 8, whose only negative increment (-15) is
of the kind the compiler's rewinds use. The attention is unchanged: q and k carry the same permutation inside every head. This drops
the attention block's two duplicate FCs (187 KB per layer), without which six layers would not fit.

## 5. Memory

- **Narrow** (`narrow_sizes`, every tile, 64-byte aligned, nothing live at the same time shares memory): the attention core's
  buffers (attention.narrow_sizes), Q_l of every layer, the 1-D vectors at their whole size (tile 0 holds the gathered vector), the
  FC inputs XN / HN / O at 384 bytes (a streamed FC's op walks its input in fills of 32 words). An ADD's operand 2 sits above
  operand 1.
- **Wide** (`model_alloc`, from the top): the identity row (8316), the output FIFO, the attention block's FIFOs and relay block
  (attention_block.block_alloc's order), the L2 norm's FIFOs (8288 / 8280, eltops'), the FFN broadcast FIFOs (8044 / 8048, the FFN FC
  form's), the streamed FCs' weight FIFO [8064, 8192) and bias rows [8056, 8064). Every multi-row FIFO starts at a multiple of 8
  units (asserted). The parameters end below `model_limit()` = 7316 units (P = 256's lowest buffer): 468 KB per tile, 7.49 MB.
- **Parameters** (`plan`): 1.07 MB per layer (q, k, v, wo, w2: 5 groups of 64 outputs; w1, w3: 12), 6.42 MB for six layers. The FCs
  always compute on tiles 0..4 (288 outputs) or 0..11 (768). A matmul is spread (its groups resident on their compute tiles) while it
  fits, else streamed: every group from the tile with the most room, any tile, any offset, streamed to its compute tile every call
  (fused.py's streamed stage, one ringProducer per group). `plan(6)` streams 19 of 42 matmuls (2.5 MB per call over the ring); on the
  device that cost no measurable time.

## 6. Quantization (`quants`: per layer, (scale, zero point) per tensor)

| key | tensor |
|---|---|
| `res`, `mid`, `out` | the layer's input residual, after the attention residual, the layer's output (= the next layer's `res`) |
| `x`, `hn` | the L2 norms' outputs: TFLite's (1/128, 128) |
| `wqkv`, `wo`, `w1`, `w3`, `w2` | the weights, folded (`folded`: g sqrt(288) in the columns of wqkv, w1, w3), per tensor |
| attention_block's keys | `qf kf v qc kc q k qk s p pv att o beta cos sin` (attention_block section 7) |
| `h1 h3 g a m y` | the FFN (g = LOGISTIC's (1/256, 0)) |

`calibrate_split(w, stories)`: full ranges over the float model's stories. **The BOS token gets its own quantization.** It drives
residual channels 173, 245 and 91 to +-10..15 from layer 2 on (attention-sink massive activations), while every other token stays
within a few units. One quantization for both leaves the residual stream 2-3 steps for a typical element (perplexity 2.88 vs 2.13
below). P = 1 is always the BOS token, so the P = 1 program uses quantization calibrated on position 0, every other P one calibrated
on the other positions; the weights and the K / V quantization (which must cover the BOS row in the cache) are shared.

## 7. Bit model and findings

`model_ref` = per layer `layer_ref`: `l2norm_ref` -> `attention_ref` (fc_ref with SIGMA rows, mul_ref, add_ref, attention.attention_ref)
-> add_ref -> `l2norm_ref` -> `ffn_ref` (fc_ref, `logistic_ref`, mul_ref) -> add_ref. New here:
- `logistic_ref`: x = clamp(f32(q - zp) f32(s), +-10.396732); segment by the 7 breakpoints (segment s covers [b_{s-1}, b_s));
  p = c0 + x(c1 + x(c2 + x(c3 + x c4))) in float32; y = clamp(rne(p) + zp_out, 0, 255). Exact on the device for all 256 inputs of 18
  input quantizations.
- `l2norm_ref` with any output quantization (eltops.l2norm_ref for (1/128, 128)).

**7.1 An L2 norm breaks the identity row of its tile.** After the L2_NORMALIZATION stage on tile 0, the next MUL on tile 0 wrote an
input-independent [255, 0, 0, 0] pattern (tiles 1..4 correct), in both layouts; loading the identity row again after the L2 norm
fixed it. edgetpu_compiler loads the identity after an L2 norm in its own programs. The mechanism is not known.

**7.2 So does a streamed FC** on its compute tiles: with q streamed, the following RoPE MULs were wrong on all five tiles; with w1 / w2
streamed the layer was exact because the LOGISTIC stage reloads the identity. `layer_stages` tracks this (`st['dirty']`): after an L2
norm or a streamed FC, the identity row is loaded again before the next stage that reads it.

**7.3 A LOGISTIC after an L2 norm stalls the chip** unless the identity row is loaded right before its op fence (edgetpu_compiler's
sequence for L2_NORMALIZATION -> FC -> LOGISTIC, and the FFN FC form's). With it, every FFN stage is bit-exact. Two stalls on the
device; the USB firmware kept answering and the chip came back without a replug.

**7.4 The softmax's exp2 / reciprocal are not exactly float32.** In generation 6 of 205 calls differed from the bit model. Replaying
the first one (P = 5) through prefix programs: layers 0..2 and layer 3's q', k', v exact, layer 3's attention output 1 of 288 bytes
off by one, in the head whose softmax value is 36.49991 (9e-5 below a 1/256 rounding tie; the model rounds it to 36, the device to
37). The scalar core's EXP2 / RECIP differ from correctly rounded float32 by a few ulp; `softmax_ref` assumes exact. Every other
operation matched bit for bit. (`eltops.gen_scalar_probe` at 2048 words returned zeros on the device: the 16-tile path; not pursued.)

**7.5** Grouped instructions (one op for several tiles) and FIFOs of the earlier allocation were not the cause of any fault; my first
composed program also put the L2 norm's 4-row FIFO at a unit not divisible by 8 (fixed, asserted; whether it mattered is unknown).

## 8. Quality (offline, teacher forced, `evaluate`)

| | perplexity | argmax = float |
|---|---|---|
| float (8 sampled stories, 1559 tokens; calibration on 8 others) | 2.155 | |
| the chip's arithmetic (`ChipModel`, bit-exact with the device but for 7.4) | **2.496** | **83.5%** |
| another sample (8 stories, 1382 tokens): float / chip | 1.877 / 2.133 | 86.5% |
| the same sample with one quantization for all positions | 2.875 | 74.8% |

A fake-quantization ablation (every point but one quantized) found the residual stream and the FFN product m the largest costs before
the BOS split; after it no single point dominates. Greedy story on the device (prompt "Once upon a time"): "...there was a little
girl named Lily. She loved to play outside and pick flowers. One day, she saw a big, red strawberry in the garden..."

## 9. Copied rather than understood

- The streamed FC stage and the LOGISTIC stage are fused.py's (private there, copied with notes); `gather_local` is fc.gather's
  rules with other addresses; the L2 norm's ops are gen_l2norm's (`eltops.l2norm_stage`); the attention core, slot copies and north
  gather are attention_block's (their own "copied" lists apply).
- Why an L2 norm and a streamed FC leave the identity row unusable, and why the LOGISTIC needs it loaded right before: fixed by
  following the compiler's sequences, not understood.
- The ADD's 16-bit integer weights now run with values up to 32 000 (verified only through the bit-exact layers).

## 10. What remains

- **The softmax model** (7.4): measure EXP2 / RECIP (a working probe: the 4-tile softmax data path) and tabulate them per
  quantization, or accept rare 1-LSB differences at rounding ties.
- **Instruction stream:** 306-388 KB per call is now the largest per-token cost below P = 128 (~1.2 ms at 320 MB/s); the K / V
  inputs above (886 KB at P = 256). Keeping the KV cache on the chip (narrow memory surviving between calls, untested) would remove
  the latter; fewer instructions per stage (e.g. the RoPE MULs' per-tile feeds of cos / sin') the former.
- **Host:** 1.27 ms per token (the classifier in float32 numpy).
- P = 256 is the last position; the programs are generated per P (~0.25 s each, 6 s for 205 in 8 processes).
