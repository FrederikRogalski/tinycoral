# Notes: facts about the Coral USB Accelerator we rely on

The reference behind the code. `docs/ARCHITECTURE.md` explains the chip and `docs/isa/*.md` every instruction; `STORY.md` tells how we got here.

## USB (single-endpoint firmware)
- **Boot:** the stick boots as a DFU bootloader (1a6e:089a). We upload `apex_latest_single_ep.bin` (10.7 KB 8051 firmware) in 256-byte blocks plus an empty one, and it re-enumerates as 18d1:9302. The firmware lives in RAM: every replug starts from scratch.
- **CSR access:** vendor control transfers.
  - bRequest 0 is 64-bit, bRequest 1 is 32-bit.
  - wValue = addr & 0xffff, wIndex = addr >> 16.
  - Read is 0xC0, write is 0x40.
- **Bulk OUT ep 1:** an `<I length><B tag><3x>` header, then the payload.
  - Header and payload may share one transfer; two DMAs in one transfer are not accepted.
  - Tags: 0 instructions, 1 input, 2 parameters, 3 output, 4–7 scalar-core interrupt 0–3.
- **Bulk IN:**
  - 0x81: outputs. Every output DMA ends with a short packet: one transfer per DMA.
  - 0x82: 16-byte events `<Q addr><I len><B tag>`.
  - 0x83: interrupts.
- **Deadlock rule:** no bulk OUT may start while an earlier bulk IN (an output DMA) is pending.
  - libedgetpu: "bulk-out could delay completion of bulk-in till deadlock occurs".
  - `coral/runtime.py` runs a program's hints in segments: a segment's INs are queued before its OUTs, and the next segment starts when they have completed.
  - Breaking the rule hangs programs whose input and output DMAs interleave, and wedges the firmware until a replug.
- **One output, several DMAs:** a program can send one output hint's bytes as several device DMAs, each ending with a short packet (conv outputs that go through scalar memory do). Read until the hint's size before anything goes out; reading only the first piece and moving on is the deadlock above. It hung the device once.
- **Bandwidth:** device→host is ~55 MB/s (libedgetpu too, independent of the TPU clock); host→device is 300–360 MB/s. Keep outputs small.
- **Wedging** (`coral/device.py` guards against it: one process at a time via `.coral.lock`; no automatic port reset):
  - **A stalled program can be recovered without a replug.** Its output never comes, but the usb firmware still answers control transfers. Draining both IN pipes and running the open sequence again brings the chip back.
  - After a failed transfer, `.coral.hung` makes the next `EdgeTPU()` do exactly that. Then a canary program (MNIST's first conv, `coral/recover.py`) must come back bit-exact before the marker is cleared.
  - If the firmware doesn't answer (the deadlock below), only a replug helps.
  - Tracing a program with big DMAs while halted wedges the firmware.
  - A USB port reset on a wedged stick makes it vanish until a replug.

## The chip ("beagle")
- **Tiles:** 4x4 = 16.
  - Narrow memory (activations) 192 KiB per tile.
  - Wide memory (parameters) 512 KiB per tile. Our parameter regions stay below 496 640 B; program buffers sit above.
- **Scalar core:**
  - 32 registers (CSR 0x44400 + 8i) and 8 predicates (0x44500 + 8i).
  - DMA units avDataPop, parameterPop, infeed and outfeed, each with its own PC, run control and loop state.
- **Per tile:** op, narrowToWide, wideToNarrow, ringBusConsumer0/1, ringBusProducer, meshBus0..3, 19 sync counters, a non-linear unit (spline).
- **Run control:** 0 idle, 1 run, 2 halt, 3 single-step.
  - Breakpoint is `(pc<<1)|1`. Break on pc 0, then single-step.
  - Registers are readable only while halted.
- **Clock:** "max" is 500 MHz; "high" is 250 MHz, libedgetpu's std. Peak is 4 TOPS at max.

## Instruction set (details in docs/isa/)
- **Words and opcodes:** 128-bit words. The opcode in bits 6–11 of the first word fixes the length (`coral.isa.LEN`).
  - 1 word: start, end, halt, sync, scalar ALU, 0x23.
  - 2 words: 0x24, avDataPop, outfeed.
  - 4 words: ring, infeed.
  - 6 words: mesh 0x15–0x18.
  - 7 words: wide/narrow.
  - 13 words: 0x19, the NLU spline load.
  - 17 words: the op 0x01 and max-pool 0x02.
- **Tile instructions** carry a 16-bit tile mask at bits 12–27, plus a running sequence number shared with the syncs.
- **Data movement:** every instruction that moves data carries loop nests ("TTUs"): (increment, count) per level, byte-granular.
- **Requantization** is inside the op: zero points, a float32 multiplier, float32 clamps, then rounding **half to even** (measured with exact .5 ties).
- **Parameter blob** (FC and 1x1 conv alike): per group of 64 outputs, an int32 bias row, then the weights as [K/4][64][4] uint8. K is padded with the weight zero point.
- **LOGISTIC** is an 8-segment quartic spline whose 40 coefficients are immediates of the 0x19 instruction (no lookup table).
- **MUL** is two ops: multiply into 32-bit words, then pack.
- **ADD** runs in a 16-bit mode with integer weights.
- **MAX_POOL** is opcode 0x02.

## Rules the hardware taught us
- **Alignment:** parameter regions must start 256-byte aligned. At offset 128 the caching program hangs.
- **Small output groups:** programs with 16/32/48-output groups (N < 64) read wrong weights at ≥ 256 KiB. We run the 64-output program with padded rows instead.
- **Rounding is half to even.** TFLite's reference rounds half away from zero.
- **A program's own wide buffers clobber every tile, every call.** The 1x1-conv programs put theirs lower for big inputs: from 458 496 B at 256 x 2304, and from 409 344 B at 256 x 3840, below the 496 640 we had assumed. Weights placed above got overwritten. CIFAR, with 4 MB of weights, showed it (11% accuracy). `TPURunner` now remembers the lowest buffer of every program that ran (`floor`) and places nothing above it.
- **Our 1x1-conv generator covers inputs up to 256 positions × 3840 bytes.** At 256 x 4096 edgetpu_compiler switches to a mode `coral/codegen/conv.py` doesn't model. Our program then differs, and on the device it computes garbage (found by CIFAR at 11% accuracy). `TPURunner.fc` keeps every call within 256·3840 bytes.
- **Our tile-shifted FC programs work:** the compute tiles move to [s, s+T) while the input gather stays on tile 0, a structure edgetpu_compiler never emits. That is what lets our placer keep a whole model resident.

## What edgetpu_compiler does (we use it only as an offline oracle)
- **Placement:**
  - Compiled alone, every program puts its parameters at offset 0 from tile 0. FC programs spread over tiles 0..T−1; 1x1 conv programs sit on tile 0 and broadcast over the ring every call.
  - Co-compiled models get other tiles and offsets: all 24 TinyStories matmuls (6.3 MB) fit at once.
- **Streamed layers:** when the outputs don't fit on chip, the weights are re-streamed once per 64 positions (75 MB per call for a 256 x 16000 x 288 layer).
- **Failures:**
  - It crashes on a 1x1 conv with 128 positions, 768 → 288.
  - It can't batch FULLY_CONNECTED, and maps a row of positions onto 4 of the 16 tiles.

## Next
- **The KV cache on the chip:** K and V still cross USB every call (886 KB at 256 positions). Keeping them needs narrow memory to survive between programs (untested) and a layout that doesn't move with the number of positions.
- **Fusion from tinygrad's graph:** multi-kernel fusion (the FFN block, a whole transformer layer) chosen by tinygrad, instead of explicit FusedBlock layers and `examples/stories_chip.py`'s own decoding loop.
