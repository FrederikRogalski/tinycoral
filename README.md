# tinycoral

tinygrad on the Google Coral USB Accelerator (Edge TPU), with a reverse-engineered instruction set, our own code generator and our own USB driver. A tinygrad model runs on the stick with `DEV=CORAL`: no libedgetpu, no TFLite and no edgetpu_compiler at run time. Google's compiler is used only offline, as the reference our generated programs are checked against byte for byte.

| | Google's stack (edgetpu_compiler + libedgetpu) | tinycoral |
|---|---|---|
| MobileNet v1 (Google's program), std clock | 3.96 ms | **3.89 ms**, bit-exact |
| one FC layer, 288x288 / 1024x4096 | 0.215 / 0.257 ms | **0.176 / 0.214 ms**, our program |
| MNIST CNN (tinygrad's beautiful_mnist), uint8 | 2 615 img/s, one image per call | **22 994 img/s**, the whole network as one program, 32 images per call |
| TinyStories-15M (llama2.c), batch 256 | can't compile it | **1 815 tok/s**, every program ours |
| TinyStories-15M, batch 1, the whole transformer on the chip | can't compile it | **176.7 tok/s**, one program per token |

TinyStories-15M in tinygrad, tokens/s:

| batch | tinygrad CPU (one core) | tinygrad CORAL | tinygrad METAL (M1 Pro GPU) |
|---|---|---|---|
| 1 | 68.8 | 61 | 316 |
| 256 | 640 | **1 815** | 28 000 |

- A MacBook (M1 Pro-class) with the Coral on USB 3; details and how to reproduce every number in [`bench/RESULTS.md`](bench/RESULTS.md).
- The Coral is for weak hosts. On this laptop numpy with Accelerate on all cores does 5 435 tok/s at batch 256, and Google's stack runs the MNIST CNN at 26 127 img/s when it gets the same trick (images stacked into one tall image, `tools/export.py`).

## Quickstart
```
DEV=CORAL python examples/mnist.py                        # tinygrad's beautiful_mnist, quantized, as one Edge TPU program (train it first: examples/mnist.py train)
DEV=CORAL python examples/stories_batch.py --batch 256    # 256 TinyStories at once, every matmul on the Edge TPU
DEV=CORAL python examples/stories.py                      # one story, batch 1
python examples/stories_chip.py                           # one story, batch 1, all 6 layers in one Edge TPU program per token
DEV=CORAL python examples/server.py                       # a web app: write with the model, or watch 256 stories grow at once
MOCKCORAL=1 DEV=CORAL python examples/stories.py          # without the stick: the TPU's arithmetic in numpy
```

## How a tinygrad model gets onto the Edge TPU
The model is plain tinygrad. Everything is decided while tinygrad compiles; the runtime only executes.

1. **`coral.quantize(model, calibration_inputs)`** works like TFLite's converter: it folds batch norms, shares quantizations across ReLU and max pool, and swaps every `nn.Conv2d` / `nn.Linear` for a uint8 layer written in plain tinygrad ops (`coral/qops.py`). Every rewrite is checked on the calibration data.
2. **tinygrad's scheduler** cuts the forward pass into kernels, as for any device.
3. **Kernel selection** (`coral/select.py`, hooked into tinygrad's codegen) proves from a kernel's AST that it is exactly a quantized matmul or conv and turns it into an Edge TPU program. Everything else compiles with clang and runs on the host. On any other device the same code runs as ordinary kernels, with bit-identical results.
4. **Chain fusion** (`coral/chain.py`, hooked into TinyJit) replaces runs of TPU kernels joined by exact uint8 max pools or copies with one TPU call, so the activations stay on the chip.
5. **The code generator** (`coral/codegen/`) writes the instructions. Where edgetpu_compiler has an equivalent, the program is byte-identical to its.
6. **The runtime** (`coral/tpu.py`, `coral/runtime.py`, the USB driver) keeps the weights resident in the tiles' memory, moves bytes over USB and collects the results.

## What's here
| path | what |
|---|---|
| `coral/usbdev.py`, `dfu.py`, `device.py`, `runtime.py`, `recover.py` | the driver, pure Python on tinygrad's libusb bindings: firmware upload over DFU, control registers, the single-endpoint DMA protocol and its deadlock rule, recovery of a stalled chip |
| `coral/executable.py`, `fb.py` | edgetpu_compiler's output format (DarwiNN executables inside a TFLite file) |
| `coral/regs.py` | register names and addresses, generated from libedgetpu's headers (`tools/gen_regs.py`) |
| `coral/isa/` | the decoded instruction set: field layouts, encoders and decoders, builders for every instruction we generate |
| `coral/codegen/fc.py`, `conv.py`, `conv2d.py` | FULLY_CONNECTED, batched matmul (a 1x1 conv over a grid of positions) and k×k CONV_2D programs, byte-identical to edgetpu_compiler |
| `coral/codegen/chain.py` | a whole CNN (convs, max pools, a final linear layer) as one program with the activations on chip |
| `coral/codegen/eltops.py`, `fused.py` | SOFTMAX (on the scalar core), L2_NORMALIZATION, SUM/MEAN; the FFN block, the on-chip block-argmax classifier and a placement plan for a whole model |
| `coral/codegen/attention.py`, `attention_block.py`, `layer.py` | attention, a transformer layer's attention block, and all 6 layers of TinyStories-15M as one program per token |
| `coral/programs.py`, `coral/tpu.py` | the host side of the programs; the runner that keeps every weight matrix resident |
| `coral/ops_coral.py`, `install.py`, `qops.py`, `select.py`, `chain.py` | the tinygrad backend `DEV=CORAL` |
| `coral/quantize.py`, `fused.py`, `nn.py` | post-training quantization of any tinygrad model; fused blocks as tinygrad layers |
| `examples/` | MNIST, CIFAR-10, TinyStories (matmuls, fused blocks, the whole transformer on the chip) and the web app |
| `test/` | `test_codegen*.py`, `test_isa.py`, `test_eltwise.py`, `test_select*.py` run offline; `test_hw_*.py` and `test_mobilenet.py` on the device |
| `bench/` | `RESULTS.md` and the scripts behind it |
| `tools/` | not needed at run time: edgetpu_compiler in Docker as the oracle (`compiler.py`, `oracle.py`, `tflite_gen.py`, `export.py`, `reloc.py`), the instruction corpus and the FC table (`corpus.py`, `fcgen.py`, `data/`), a single-step tracer (`trace.py`), a runner for hardware tests (`hw.py`) |
| `docs/ARCHITECTURE.md`, `docs/isa/*.md` | how the chip works, and every decoded instruction |
| `NOTES.md`, `STORY.md` | the hardware facts the code relies on; how it went |

## Setup
```
git clone https://github.com/tinygrad/tinygrad tinygrad-src && git -C tinygrad-src checkout d2892164e20e9c53350abaa7e7a8cc7b7376f8d4
python3.12 -m venv .venv && .venv/bin/pip install -e tinygrad-src -e .   # tinygrad at the commit tinycoral was developed against
.venv/bin/python -m coral.install                                        # links the CORAL backend into tinygrad
brew install libusb                                                      # or your distro's libusb
git clone https://github.com/google-coral/libedgetpu ref/libedgetpu      # the USB firmware the driver uploads, and the register map
```
- **Models:** `models/stories15M.bin` and `models/tokenizer.bin` from karpathy/llama2.c. MNIST and CIFAR-10 train with tinygrad: `python examples/mnist.py train`, `python examples/cifar.py train`.
- **Offline tests:** `python -m coral.codegen.fc` (`.conv`, `.conv2d`, `.chain`, `.eltops`, `.fused`, `.attention`, `.attention_block`, `.layer`) compare our programs byte for byte with edgetpu_compiler's, `MOCKCORAL=1 python test/test_select.py` (and `test_select_conv.py`, `test_select_glue.py`) run the backend against the numpy model, `python test/test_isa.py` and `test/test_eltwise.py` check the decoded instructions. They need `pip install -e .[tools]`, `git clone https://github.com/google-coral/edgetpu ref/edgetpu` (the compiler) and Docker, which runs the x86-64 compiler; results are cached in `.compile/`. Docker is needed for nothing else.
- **On the device:** the `test/test_hw_*.py` files, or `python bench/hw_session.py` for the tests and benchmarks one by one. `MOCKCORAL=1` swaps the device for numpy models (bit-exact for matmuls and convs, float math for the fused blocks).

## Status and limits
- Through tinygrad (`DEV=CORAL`): quantized matmuls and convolutions, fused FFN and classifier blocks, and whole CNNs as one program. Activations, attention, norms and RoPE run on the host. The whole-transformer program (`examples/stories_chip.py`) runs outside tinygrad's scheduler for now.
- Whole-network programs cover chains of VALID stride-1 convolutions (odd kernels up to 5x5), 2x2 max pools and a final linear layer. Other chains run layer by layer.
- At batch 1 the matmul path is bound by USB round trips and host overhead: about tinygrad's single-core CPU speed. Device→host USB is ~55 MB/s, so designs that keep data on chip win.
- Developed and measured on macOS with one Coral USB Accelerator. The device is used by one process at a time; after a stalled program the driver re-initializes the chip and checks it with a canary program.

## Credits
- geohot's [edgetpuxray](https://github.com/geohot/edgetpuxray) and streams: the first map of this territory.
- [libedgetpu](https://github.com/google-coral/libedgetpu) (Apache 2.0): the register names (`coral/regs.py` is generated from its headers), the USB firmware the driver uploads (cloned with libedgetpu, not part of this repository), and one crucial comment about USB deadlocks.
- karpathy's [llama2.c](https://github.com/karpathy/llama2.c): the TinyStories models and tokenizer.
- [tinygrad](https://github.com/tinygrad/tinygrad).

Written by AI agents (Claude, in Claude Code): the reverse engineering, the driver, the code generator and the tinygrad backend. A human plugged the stick in, replugged it whenever an agent wedged it, and steered. [`STORY.md`](STORY.md) tells how it went.

Not affiliated with or endorsed by Google or tiny corp. MIT license (`LICENSE`).
