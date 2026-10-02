# How it went

The journey in order: discoveries, dead ends, numbers. Everything here was measured or observed; `bench/RESULTS.md` has the
final numbers and how to reproduce them.

## The start
- **Starting point:** a $60 Google Coral USB Accelerator plugged into a MacBook.
- **The task:** port tinygrad to the Coral, with our own compiler instead of Google's, and do something impressive. The advice that came with it: tight, fast feedback loops.
- **Prior art:** geohot's `edgetpuxray` and his streams on the Edge TPU.

## Driver: talking to an undocumented chip over USB
- **The boot path:** the stick boots as a DFU bootloader (1a6e:089a). We upload a 10.7 KB 8051 firmware, and it re-enumerates as 18d1:9302.
  - The firmware lives only in RAM, so every replug starts from scratch (safe to experiment).
- **Control registers:** read and written with vendor control transfers. The register map comes from libedgetpu's headers (names for every unit: 16 tiles, ring bus, mesh bus, DMA engines, 19 sync counters per tile).
- **The data path:** a single bulk endpoint carries everything, as 8-byte headers plus payload: instructions, inputs, parameters.
- **First win:** Google's own MobileNet runs bit-exact through a pure-Python driver. It is also faster than Google's C++ runtime: 3.89 vs 3.96 ms at std clock, 2.35 vs 2.45 ms at max.
  - The trick: queue every USB transfer asynchronously up front.

## Reverse engineering
- **A hardware debugger:** breakpoint on pc 0, then single-step the scalar core and read its registers over USB.
- **Pain:** tracing a program with big DMAs while halted wedges the USB firmware. The human had to replug the stick again and again ("warum muss ich das immer wieder machen?", "why do I have to do this again and again?").
- **Funny bug:** a 12-bit (instead of 13-bit) relocation field produced an LLM that only said "laush laush". Parameters above 256 KB wrapped around.
- **Four agents decoded the four instruction families in parallel:** tile compute op, wide/narrow DMA, ring/mesh bus, and scalar/sync.
  - Every instruction of the 335 corpus programs (FC, RELU and 1x1 conv) round-trips bit-exactly through the decoded layouts: op 1084/1084, wide/narrow 2531, the scalar side 9332, …
  - The chip: 128-bit VLIW words, variable-length instructions (1–17 words), loop nests ("tensor traversal units"), and sync counters.
  - Requantization is a float32 multiply inside the op instruction.
- **Google's own paper agrees:** an IISWC 2022 Google paper describes the template, a PE array with PE memory for activations, core memory for weights and SIMD lanes. That matches our 4x4 tiles, 192 KiB narrow and 512 KiB wide per tile.

## Our own code generator
- **`gen_fc(N, K)`:** emits FULLY_CONNECTED programs byte-identical to edgetpu_compiler, built from the decoded fields plus layout rules fitted to the compiler's choices.
  - 355/355 table shapes and 1918/1920 random shapes match.
- **First hardware run of generated programs:** 22/22 bit-exact. Then the hardware taught us two rules the compiler never violates:
  - parameter regions must be 256-byte aligned (offset 128 hangs the chip);
  - programs with 16-output groups read wrong weights once their parameters sit at ≥ 256 KiB (16x4096: fine at 192 KiB, wrong at 256 KiB).
- **After the fixes:** 50/50 random shapes, weights and quantizations bit-exact on hardware.
- **FC latency, ours vs Google's stack:** faster on all 7 shapes (e.g. 288x288: 0.176 vs 0.215 ms; 2048x1024: 0.229 vs 0.263 ms).

## What we learned about Google's compiler
- It crashes on a 1x1 conv with 128 positions, 768 → 288 ("Internal compiler error").
- It can't batch FULLY_CONNECTED, and it maps a row of positions onto 4 of the 16 tiles.
- **Separately compiled layers all claim tile 0, offset 0.** Compiled together, the compiler does fit all 24 TinyStories matmuls on chip.
- **It re-streams a big layer's weights once per 64 positions** when the outputs don't fit: the 9.3 MB classifier becomes 75 MB per call at batch 256.
  - Splitting the classifier so that each piece streams once fixed it.

## The USB link shapes everything
- **The link is asymmetric:** device→host runs at only ~55 MB/s, also with Google's driver and independent of the TPU clock. Host→device does 300–360 MB/s.
- **The TPU mostly waits:** at batch 256 it computed for ~3% of a decoding step. Effective ~1% of its 4 TOPS.
- **Design rule that followed:** keep data on chip, and send big things in the fast direction.

## The deadlock (the cause of all those replugs)
- The batch-128 run hung the stick again. libedgetpu's source had the answer in a comment: in single-endpoint mode "bulk-out could delay completion of bulk-in till deadlock occurs".
- Our runtime sent all inputs at once while outputs were still pending. Programs that interleave input and output DMAs deadlock the USB firmware.
- **Fix:** send in segments, never an OUT while an earlier IN is pending. No hang since.

## Fused blocks and the upside-down classifier
- **The FFN block as one program:** w1, w3, logistic, mul, mul, w2. The 1536-wide activations never leave the chip.
- **The classifier turned upside down:** the 32000-word vocabulary is the input image (160x200 "pixels" of 288 channels) and the 256 token activations are the conv weights. An 8x8 max-pool sends back only the best of every 64 logits.
  - 128 KB come back instead of 8 MB, and the big data goes in the fast direction.
  - The host recomputes the 4 best blocks per row exactly.
- **Result, TinyStories-15M in tinygrad, batch 256:** 1852 tok/s on the Coral vs 640 tok/s on the Mac CPU (tinygrad's CPU backend), 2.9x.
  - 256 stories generated in parallel, coherent: "Once upon a time, there was a kind bear named Kate…".
  - The M1 Pro's GPU (tinygrad METAL) does 28 000 tok/s. The Coral is for hosts without a GPU.
  - At this point the fused programs were still built by Google's compiler from our weights (and run by our runtime).

## Small things
- **The LED:** the stick has a white LED. libedgetpu never touches it and the TPU has no LED register. It is driven by the 8051 USB firmware and blinks with activity (we blinked a pattern by switching load on and off).

## Evening, Oct 1: chasing the last 2x
- **Batch 1 stays a wash.**
  - Fusing the layer blocks for a single token as FULLY_CONNECTED programs takes 18 TPU calls instead of 24, ~6 ms per token in total.
  - Still 61 tok/s vs 69 on the CPU: at batch 1 both sides are dominated by per-kernel host overhead (~70 kernel launches per token), not by the accelerator.
- **Three agents work in parallel:**
  - our own codegen for the batched 1x1-conv programs;
  - decoding LOGISTIC/MUL/ADD/MAX_POOL inside the op instruction;
  - making plain tinygrad code compile to TPU programs: a quantized linear written with normal Tensor ops, recognized in the kernel AST.
- **The upside-down classifier is exact in practice:** on 10 240 real tokens, the 4 best blocks from the TPU always contained the float argmax over all 32 000 logits. The host refine costs 2.6 ms per step.
- **The sigmoid is a polynomial baked into an instruction.** LOGISTIC needs no lookup table and no parameter bytes. A 13-word instruction (opcode 0x19) loads each tile's "non-linear unit" with an 8-segment quartic spline: 40 coefficients and 7 breakpoints, identical in every program.
  - MUL is two ops (multiply into 32-bit words, then pack back to bytes). ADD runs in a 16-bit mode whose integer weights are the best rational approximation of the scale ratio.
  - MAX_POOL is its own op (opcode 0x02).
  - An agent rebuilt all 1,600+ such instructions from 198 compiled programs byte for byte.
- **Our own batched programs run on the chip.** The 1x1-conv generator is byte-identical to edgetpu_compiler for every TinyStories shape at every batch size, and for all 120 co-compiled placements, including layers split over several tiles.
  - On hardware it is bit-exact with the weights on tile 0, 5, 9 or 14 at arbitrary 256-byte-aligned offsets.
- **The chip rounds half to even.** We had assumed TFLite's half away from zero, and thousands of random tests never noticed: exact .5 ties are rare.
  - Plain tinygrad code compared bit for bit against the TPU caught it: one value in 110 592 was off by one, at acc·mult = 64.5.
  - A test with multiplier 0.5 and odd accumulators (thousands of ties) settled it: 100% half-to-even, in FC and batched programs alike.
- **Plain tinygrad compiles to the TPU.** A quantized linear layer written with ordinary Tensor ops runs as clang kernels on DEV=CPU.
  - On DEV=CORAL, tinygrad's scheduler fuses it into one kernel; our selector recognizes the matmul in the kernel AST and emits an Edge TPU program instead.
  - Same code, bit-identical results on both devices; no custom kernel in the model code.
- **Placing layers where Google's compiler never would.** Our generator can move a layer's compute onto other tiles while the input gather stays on tile 0. edgetpu_compiler never emits that program structure.
  - It ran bit-exact on the first try (15/15).
  - With it, all 24 TinyStories matmuls at batch 1 are packed into the 16 tiles by our own planner: 62.3 tok/s with no Google-made program at all, the same speed as with the compiler's co-compiled programs (62.6).
- **Plain tinygrad all the way down:** model code with ordinary Tensor ops, tinygrad's scheduler, our kernel selector, our generated FC and batched programs, and our runtime. Batch 256: 759 tok/s vs 640 on the CPU, still bottlenecked by the classifier on the host.
- **No Google compiler anywhere.** The last agent composed the fused FFN program (conv, conv, logistic, mul, mul, conv), the upside-down argmax and a placement plan for all 19 programs from the decoded pieces.
  - All 113 programs of the compiler's own co-compiled set are reproduced byte for byte.
  - Our plan puts them elsewhere, and they run.
  - TinyStories-15M at batch 256: **1815 tok/s with only our own programs** vs 640 on the Mac CPU (2.8x). The 256 stories are word for word identical to the run with Google's programs.
- **Which CPU?** The 2.8x is against one core: tinygrad's CPU backend runs each clang-compiled kernel on one core.
  - The same model in numpy with Apple's Accelerate (AMX, all cores) does 5 435 tok/s at batch 256, 3x the Coral.
  - So a $60 USB stick with our compiler stack beats the single-core CPU path, and Google's stack can't run the model at all. The Coral's job is a weak host, not an M1 Pro.
- **A web app for the demo** (`examples/server.py`, http://localhost:8642): write a story together with the model at ~70 tok/s, or watch 256 stories grow at once at ~1500 tok/s. It shows live tokens/s, USB traffic per step, and a 4x4 map of which layer's weights sit in which tile.
  - Seen on it: at batch 1, 135 KB per token go to the TPU but only 6 KB come back. Most of it is the 18 programs' instructions, re-sent on every call.
- **Speculative decoding at batch 1: measured first, then not built.** Simulated on the CPU from real greedy stories:
  - **Prompt lookup** (copy what followed the last matching n-gram): 1.13 tokens per verify step.
  - **Early exit** (draft with the first L of 6 layers): it agrees with the full model 7% / 21% / 38% / 62% of the time after 2 / 3 / 4 / 5 layers.
  - **Why it doesn't pay:** a verify step over several positions costs more on the Coral than a plain step (bigger programs, the classifier on the host for several rows), so none of these would be faster.
  - **What would help:** a small, trained draft model with the same 32 000-token vocabulary. karpathy's stories260K uses its own 512-token vocabulary, so it can't draft for stories15M.

## Any tinygrad model: quantize, select, fuse
- **Stage 1:** `coral.quantize(model, calibration_images)` records the input/output ranges of every `nn.Linear` / `nn.Conv2d` in float, then swaps each one for a uint8 layer written in plain tinygrad. Convolutions become im2col (cut out by tinygrad on the host) plus one quantized matmul, which the selector sends to the TPU.
  - tinygrad's own MNIST CNN (beautiful_mnist), trained on METAL, runs unchanged: 98.40% quantized = 98.40% float on 2 000 test images.
  - Slow (130 images/s vs 272 on one CPU core): 423 TPU calls per 100 images, and im2col sends every pixel 25 times.
- **The bar: Google's stack on the same CNN.** With the batch norms folded (all their scales are positive, so they commute with ReLU and max pool) and TFLite-style quantization, edgetpu_compiler puts all 7 ops into one program.
  - 2 611 images/s at 98.93%, i.e. 20x our stage 1.
  - Its own cycle estimate is 31 µs per image, but an invoke takes 383 µs: 57 KB of instructions go over USB for every single image.
  - It can't batch: with batch > 1 nothing maps to the TPU. That is the opening: one program for many images, instructions sent once.
- **coral.quantize became a converter.** Like TFLite's converter, it now rewrites the float model before quantizing, and checks every rewrite on the calibration data:
  - It folds batch norms: beautiful_mnist's sit behind a ReLU, which is fine because all their scales are positive.
  - When a layer reaches the next one only through ReLU / max pool / flatten, both share one quantization. That makes the host work between two TPU layers an exact uint8 max pool, which is what lets a later stage fuse them.
  - 98.93% on all 10 000 test images, the same as the float model.
- **Google's stack runs exactly our quantized model.** `tools/export.py` writes coral.quantize's result as a TFLite file; edgetpu_compiler maps all 7 ops.
  - On the device, the 10 uint8 logits of every one of the 10 000 test images are bit-identical to what tinygrad computes on the host with plain kernels.
  - A whole CNN, and the TPU's arithmetic and our tinygrad semantics agree exactly. That also makes Google's program for this model the byte-exact oracle for our own fused version.
- **10x Google's deployment on Google's own stack, with a tall image.** The compiler refuses batches, but VALID convolutions never mix rows: stack B digits into one 28B x 28 image, and turn the linear layer into a 3x3 convolution over each digit's last map.
  - To edgetpu_compiler this is just a big single image. Same compiler, same runtime, same chip: 2 578 → 12 072 (8 per invoke) → 26 127 images/s (128 per invoke).
  - All 10 000 logit vectors are still bit-identical to tinygrad's.
  - 38 µs per image, close to the compiler's own estimate of the arithmetic: now the chip is busy, not the USB. This is the target for our own fused programs.
- **Stage 2: convolutions as convolutions.** A quantized conv in plain tinygrad (coral.qops.qconv2d) schedules as one kernel. The selector reads stride, padding and image size off its index arithmetic, verifies every index and mask exactly, and turns it into a CONV_2D.
  - An agent decoded the compiler's k×k conv programs: how the image is cut over the 4x4 tiles, the halo rows relayed over the mesh, the two compute modes, the output blocks. Its generator is byte-identical to edgetpu_compiler for all 4 MNIST convs and ~3300 random shapes.
  - On the device, our first generated 5x5 conv was bit-exact on the first try, and so were the other MNIST convs and 7 random shapes.
  - Then a 1x1 conv whose output goes through scalar memory (instructions copied from the compiler, not understood) hung the chip, and the stick needed a replug. That path stayed off until it was understood. Lesson: run copied-not-understood paths last, in a test of their own.
- **Where what happens:** nothing is decided at run time. It all happens while tinygrad compiles, at two hooks:
  - Kernel selection, in tinygrad's codegen: each kernel is either an exact quantized matmul/conv (→ our generated TPU program) or clang on the CPU.
  - Chain fusion, in TinyJit's lowering: runs of TPU kernels joined by max-pool/copy kernels become one TPU call.
  - The runtime only executes: it places weights, moves bytes over USB and collects results. Google's compiler is never involved, except offline as the reference our generated bytes are checked against.
- **The hang was ours, not the chip's.** That conv's output goes through scalar memory and leaves the chip as several DMAs, each ending with a short packet. Our asynchronous USB path read only the first, then sent the next program while the chip still wanted to send.
  - That is exactly the deadlock rule above (libedgetpu: "bulk-out could delay completion of bulk-in till deadlock occurs"), hit from a new side.
  - Now the reader keeps reading until the hint's size. The same case runs bit-exact, and so do 48 random shapes over all 16 tiles, many of them through scalar memory.
- **A second ordinary tinygrad model: tinygrad's own CIFAR-10 speedrun net** (SpeedyConvNet from `examples/beautiful_cifar.py`, with residuals, GELU and batch norms).
  - Trained by tinygrad's unchanged script on the M1's GPU overnight: 40 minutes, 89.4%.
  - Its batch norms normalize with whatever batch they see, which a deployed model can't do; `examples/cifar.py` freezes them to training-set statistics (90.20% on 2 000 test images).
  - coral.quantize folds all six batch norms (two of them sit behind a max pool, which a positive scale commutes with). The uint8 model scores 90.20% too.
  - The M1 got sluggish when training, a CPU test and the agents' compiler runs coincided: training waited for the night.
- **Stage 3: one program for the whole network, straight from tinygrad.** An agent taught our generator to write convs, max pools and the final linear layer into one program, with the activations on chip and our own memory placement. For MNIST at 1..32 images per call it is byte-identical to what edgetpu_compiler makes of the same model.
  - It got gated access to the device (`tools/hw.py`: byte-identical programs only, one process at a time, stop on the first failed transfer). It ran its programs itself, and every logit matched tinygrad's.
  - Wired into the backend, the TinyJit-fused chain runs as that program, 32 digits per call stacked into a tall image.
  - `DEV=CORAL python examples/mnist.py`: tinygrad's beautiful_mnist at **22 994 images/s, 98.95%**. That is 8.8x Google's own deployment of the same model on the same chip, and 33x tinygrad's CPU path on one core. Only the input quantization and the logits stay on the host.
- **CIFAR on the device found two real bugs, both silent.** The model scored 11% (chance) on the stick while the bit-exact mock scored 90.2%. An instrumented run compared every TPU call against its reference on the same inputs, and found the one wrong call, then the reasons:
  - Our 1x1-conv generator had quietly left its validated range. At 256 positions × 4096 input bytes edgetpu_compiler switches to a mode we never decoded. Our program differed, and computed garbage. The generator now refuses that range.
  - Worse: programs write their own buffers into wide memory on all 16 tiles every call, and for big inputs those start lower (458 KB) than the 485 KB we had assumed. Other layers' weights got overwritten. The runner now tracks the lowest buffer of every program that ran.
  - Then the big convs went native, with their weights split over 2, 3 and 5 tiles as edgetpu_compiler does: CIFAR at 90.20% and 130 images/s, 3x the CPU, every weight uploaded once.

## Attention and the whole transformer on the chip
- **Before building attention on the chip, a simulation asked whether uint8 kills the model** (`bench/attention_quant.py`). TinyStories-15M in numpy, quantizing more and more of it, teacher forced on stories the float model wrote:
  - Perplexity: float 1.81; matmuls in uint8 (what the TPU ran then) 2.18; plus attention in uint8 2.16; plus the norms 2.39; plus the residual stream 2.69.
  - Attention in uint8 is free. The goal became the whole model on the stick at batch 1.
  - The compiler helps where it can: SOFTMAX, L2_NORMALIZATION, SUM and MEAN map to the TPU (so they can be decoded), RSQRT and BATCH_MATMUL don't. RMSNorm becomes an L2 norm times a constant.
- **The attention building blocks run bit-exact on the device:** SOFTMAX (on the scalar core, in float32, with exp2 and reciprocal instructions), L2_NORMALIZATION and SUM/MEAN, all from our generator. The softmax test also showed that the scalar core converts float to int rounding to nearest.
- **Attention runs on the chip.** An agent composed the attention core of a transformer layer into one program: q·K on the tiles, softmax on the scalar core, p·V on the tiles. Bit-exact against its numpy model on the device at 1, 7, 64 and 256 positions; 1.24 ms per call at 256 positions.
  - Google's compiler crashes on that graph, but it maps each half alone, and our pieces are byte-identical to those halves.
  - Two of its first composed versions gave wrong answers without hanging. One debug run found why: a broadcast from scalar memory to all 16 tiles delivers nothing. It now goes the compiler's way, to tile 0 and on over the ring.
  - The price of doing it elementwise: every product is rounded to 8 bits before the sum. Simulated perplexity 2.33 instead of 2.20; the stories stay readable.
- **A whole attention block in one program:** the qkv projection, RoPE (as the compiler would map it: two FCs, two MULs and an ADD, byte for byte), attention over the cached rows plus this token's own k and v (written into the cache rows on chip), and the wo projection.
  - Bit-exact on the device with TinyStories' real layer-3 weights at 1..256 positions; 0.66 ms per call at 64 positions.
  - The first device runs of ADD also showed it rounds half to even, as everything else on this chip.
- **The whole model in one program, offline first.** An agent assembled one transformer layer (RMSNorm via L2 normalization, the attention block, residual ADDs, the FFN) and then all six layers into one program per token.
  - The bit model of that program, run teacher-forced on float-written stories: perplexity 2.13 vs 1.88 in float. That beats the matmul-only TPU path (2.18).
  - Finding: the BOS token drives three residual channels to ±15 while every other token stays within a few units. The first position gets its own quantization; with one quantization for all positions perplexity was 2.88.
  - Its first composed program hung the stick (night two, replug number who-knows). Likely cause: a wide-memory FIFO not 8-unit aligned, which no program that ever ran had. Fixed and asserted; the next device session bisected stage by stage.
- **The stick recovers by itself now.** The layer agent's bisection hung the chip again (one of its new grouped elementwise forms). This time the USB firmware still answered control transfers: only the program had stalled.
  - Draining the pipes and running the chip's open sequence again brought it back, and a canary program came back bit-exact. No replug.
  - The driver does this by itself now after every failed transfer, and asks for a human only if the canary fails.
- **The whole language model on Google's chip.** One program per token runs all six layers of TinyStories-15M: norms, RoPE, attention with softmax, residuals, FFNs. Google's compiler can't compile any of it. **176.7 tokens/s at batch 1**, 2.5x our matmul-only path.
  - The story it wrote: "Once upon a time, there was a little girl named Lily. She loved to play outside and pick flowers. One day, she saw a big, red strawberry in the garden…"
  - The bisection that got there found three hardware rules no document mentions: an L2 norm or a streamed matmul leaves a tile's identity row unusable for the next MUL, and a LOGISTIC right after an L2 norm stalls the chip unless the identity is reloaded first, as edgetpu_compiler always does.
  - 21 device runs and 4 stalls, all recovered by the driver without a replug.
  - In a 205-token generation 199 calls matched the bit model exactly. The rest trace to one softmax value 9e-5 below a rounding tie: the scalar core's exp2 is not quite correctly rounded.
