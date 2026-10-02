# How the Coral USB Accelerator works (as far as we know)

Google never published the Edge TPU's RTL or ISA. Everything below comes from three sources:
- the register map in libedgetpu's headers (`beagle_csr_offsets.h`), which names every unit;
- the instruction format we decoded and verified (`docs/isa/`): a code generator built from it produces programs byte-identical to edgetpu_compiler, and they run bit-exact;
- Google's architecture paper (Seshadri et al., IISWC 2022, arXiv:2102.10423). It describes the accelerator template: a PE array, PE memory for activations, core memory for weights, and SIMD lanes.

"Probably" marks inferences.

## The stick
```
 USB 3 ──► USB bridge (8051 firmware, loaded by us over DFU at every plug-in)
              │   control transfers: read/write any chip register (CSR)
              │   one bulk OUT endpoint: [len|tag] headers + payload (tag 0 instructions, 1 inputs, 2 parameters)
              │   bulk IN: outputs (0x81), completion events (0x82), interrupts (0x83)
              ▼
         ┌─────────────── Edge TPU ("beagle") ───────────────┐
         │  scalar core: runs the program, issues DMAs       │
         │  DMA units: infeed, outfeed, avDataPop,           │
         │             parameterPop                          │
         │                                                   │
         │  4x4 tiles (PEs), each:                           │
         │    narrow memory 192 KiB  (activations)           │
         │    wide memory   512 KiB  (weights)               │
         │    op unit: SIMD MAC lanes + requantization       │
         │    wide→narrow / narrow→wide DMA                  │
         │    ring bus consumer ×2, producer                 │
         │    mesh bus N/E/S/W to the neighbours             │
         │    19 sync counters                               │
         └───────────────────────────────────────────────────┘
```

- **Peak:** 4 TOPS int8 at 500 MHz, i.e. ~256 MACs per tile per cycle (probably 4 cores × 64 lanes, like the paper's "V1" class).
- **The compiler's 1x1-conv mapping** reaches 1024 MACs per cycle on all 16 tiles.
- **On-chip memory:** 16 × 512 KiB = 8 MiB for weights. Everything that should stay resident has to fit there; 6.3 MB of TinyStories-15M does.
- **USB is asymmetric:** host→device runs at 300-360 MB/s, device→host at only ~55 MB/s. The latter is the same with Google's driver and doesn't change with the TPU clock, so it is probably the bridge.

## Programs
A program is a stream of 128-bit words: variable-length instructions of 1-17 words. The opcode sits in bits 6-11 of the first word.

| opcode | unit | what it does |
|---|---|---|
| 0x3e / 0x3f / 0x21 | scalar | start, end, halt |
| 0x20 | scalar ALU | 32 registers, 8 predicates; ADD…MOV with an immediate form; also builds host DMA descriptors |
| 0x24 / 0x1a | sync | wait on and reset sync counters (scalar / tiles) |
| 0x25 / 0x26 / 0x27 | DMA | avDataPop / parameterPop, infeed, outfeed |
| 0x13 / 0x14 | tile DMA | narrowToWide, wideToNarrow, with loop nests |
| 0x10 / 0x11 / 0x12 | ring bus | producer, consumer 0, consumer 1 |
| 0x15 - 0x18 | mesh bus | send/receive to the S/W/N/E neighbour |
| 0x01 | tile op | the compute: four loop nests (input, output, parameters, partial sums), a datapath mode, and requantization (zero points, a float32 multiplier, clamps) |

- **Tensor traversal units (TTUs):** every data-moving instruction carries loop nests as (increment, count) pairs per level. They describe the addresses it walks.
- **Sync counters:** units signal each other through per-tile counters (AVDATA, PARAMETERS, MESH_*_IN/OUT, RING_*). A sync instruction waits until a counter reaches a value.
- **Sequence numbers:** every tile-dispatched instruction carries one, and a code generator must number syncs and tile ops together.

## What a matmul looks like: y[N] = W[N,K] x[K] (FULLY_CONNECTED)
1. **Caching program, run once:** parameters arrive over USB → infeed → ring bus → the wide memory of tiles 0..T-1. Each tile gets its own 64-output groups, laid out as int32 bias plus [K/4][64][4] uint8 weights.
2. **Execution program, every call:**
   1. The input DMA brings x in. It is cut into pieces and spread over input tiles via the ring bus.
   2. The mesh bus gathers the pieces back to tile 0, westward along rows, then northward.
   3. Tile 0 forwards x over the ring to all compute tiles.
   4. Every tile runs one op: 64 outputs per pass. It multiplies uint8 x uint8 into int32, adds the bias, scales by a float32 multiplier, clamps, rounds and adds the zero point.
   5. narrowToWide, then the ring producers send each tile's slice to the scalar core, then outfeed back over USB.
3. **Batched (1x1 conv over a grid of positions):** the weights sit on one tile and are broadcast over the ring to all 16 tiles every call. Each tile computes a block of positions for all outputs.

## Rules the hardware taught us (none of them documented)
- **Alignment:** parameter regions must start 256-byte aligned. At offset 128 the caching program hangs.
- **Small output groups:** programs with 16/32/48-output groups (N < 64) read wrong weights once their parameters lie at ≥ 256 KiB.
- **USB ordering:** single-endpoint USB must never start a bulk OUT while an earlier bulk IN is pending. Otherwise the device deadlocks and has to be replugged.
- **Clamping and rounding:** outputs are clamped in float32 before rounding, and rounding is half to even. TFLite's reference rounds half away from zero; we assumed that until exact .5 ties on the device proved otherwise.
