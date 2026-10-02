# Scalar-core-side instructions and the program skeleton

This file covers:

| opcode | words | name used here | role |
|---|---|---|---|
| 0x3e | 1 | start | program header |
| 0x1a | 1 | sync | **tile fence** (the patent's *TileFenceOp*): it is dispatched to the tiles and takes a slot in the tile instruction sequence |
| 0x24 | 2 | scsync | **scalar fence / scalar sync** (the patent's *ScalarFenceOp*). It is not a tile instruction. |
| 0x25 | 2 | pop | avDataPop (stream=1) or parameterPop (stream=0): host stream → staging buffer |
| 0x26 | 4 | infeed | staging buffer → ring bus → tiles |
| 0x27 | 2 | outfeed | ring → host (DMA tag 3), or ring → scalar memory |
| 0x21 | 1 | halt | |
| 0x3f | 1 | end | |
| 0x20 | 1 | alu | the sequences that build host DMA descriptors (`v_op` 0xa push, 0xc issue) |

Code: `coral/isa/scalar.py` has the layouts, `encode_*`/`decode_*`, compiler-exact builders, skeleton blocks and `verify()`.

**Data.** The data is the corpus (335 programs) plus 162 extra programs compiled for this study (listed at the end). All work was offline.

**Verification.** `python coral/isa/scalar.py` checks the corpus. `scalar.verify(progs)` checks any list of programs.

| check | corpus (335 programs) | extra (162 programs) |
|---|---|---|
| instances checked | start/end/halt 335 each, sync 3378, scsync 2217, pop 335, infeed 1295, outfeed 1102 | sync 1880, scsync 1540, pop 178, infeed 409, outfeed 928 |
| `encode(**decode(w)) == w` | 100% | 100% |
| bits left over after decode (`rest != 0`) | 0 instances | 0 instances |
| builders from program metadata (hint sizes) | bit-exact for every pop, infeed and host outfeed, and for start/end/halt | bit-exact for every pop, infeed and host outfeed |
| every sync / scsync equals one of 12 / 10 named builders | all instances | all instances |
| DMA descriptors (replayed ALU) match the hints; ALU sequences rebuilt bit-exact | all programs | all programs |
| sequence numbers | all programs | all programs |
| complete caching program rebuilt byte-exact (`caching_program`) | 162/162 | 11/11 |
| exe skeleton blocks present | all programs | all programs |

Conventions:
- Bit numbers are absolute within the instruction; word j covers bits [128j, 128j+128).
- Bits 0-5 (predication) are 0 in every instance of these opcodes.
- "act unit" = 8 bytes; "param unit" = 64 bytes.
- `neg(x, w)` means `(-x) mod 2^w`.

Confidence levels:
- **high**: the formula holds on all instances, or the meaning is forced.
- **med**: consistent with all data, but other readings are possible.
- **low/guess**: a hypothesis.

## 0. Things that apply to all target opcodes

1. **Global tile-instruction sequence number (high).** Every instruction that is dispatched to the tiles carries a running index n. This includes the sync 0x1a and the tile ops 0x01, 0x10-0x18. n is the number of earlier tile-dispatched instructions in the program.
   - sync stores n in bits [28,43). Tile ops store it at bits [46,58).
   - 0x24, 0x25, 0x26, 0x27, 0x20, start, end and halt are *not* counted.
   - This holds for all 497 programs. A code generator must number syncs and tile ops together.
2. **Instruction length fixes.** `coral/isa.py` LEN is wrong for three opcodes:
   - **0x15 and 0x18 are 6 words.** Together with 0x16 and 0x17 they are the four meshBus units. ADD programs use all four.
   - **0x23 is 1 word.** It appears with scalar-memory outfeeds and is probably a branch.
   - With these fixes, every program in both sets parses into exactly one start, halt and end, and the sequence numbers are consistent.
3. **Tile sync-counter bit order (med-high).** It follows the tile CSR order of `SyncCounter_*`:
   - 0 AVDATA, 1 PARAMETERS, 2 PARTIAL_SUMS.
   - 3-6 MESH_{N,E,S,W}_IN, 7-10 MESH_{N,E,S,W}_OUT.
   - 11 WIDE_TO_NARROW, 12 WIDE_TO_SCALING, 13 NARROW_TO_WIDE.
   - 14 RING_READ_A, 15 RING_READ_B, 16 RING_WRITE, 17 RING_PRODUCER_A, 18 RING_PRODUCER_B.
   - Evidence: MESH_EAST_IN|MESH_WEST_OUT pairs appear around westward meshBus 0x16 groups, and MESH_SOUTH_IN|MESH_NORTH_OUT around northward meshBus 0x17 groups. RING_PRODUCER_A is reset after the output ringProducers.
4. **Scalar sync-counter bit order (med).** It follows the scalar CSR order:
   - 0 AVDATA_POP, 1 PARAMETER_POP, 2 AVDATA_INFEED, 3 PARAMETER_INFEED, 4 SCALAR_INFEED.
   - 5 PRODUCER_A, 6 PRODUCER_B, 7 RING_OUTFEED, 8 SCALAR_PIPELINE.
5. **Stream select (high).** Bit 12 of pop and of infeed is 1 for activations (DMA tag 1) and 0 for parameters (DMA tag 2).

## 1. start 0x3e, end 0x3f, halt 0x21

| insn | bits | width | name | encoding | conf |
|---|---|---|---|---|---|
| start | 6-11 | 6 | opcode | 0x3e | high |
| start | 19-50 | 32 | length_bits | `128*(nwords-2)`: the number of bits between start and end (exclusive), which includes halt and the 4 NOPs | high |
| end | 19-50 | 32 | length_bits | always 128 | high value, unknown meaning |
| halt | 14-18 | 5 | code | always 4. geohot's `vs_reg_v1`; the old compiler (inception) used 1. | high value, unknown meaning |

Notes:
- **start length width.** Bits ≥ 36 are never set in our corpus. Geohot's inception chunk (16322 words) sets bits up to 39, so the field is at least 21 bits wide. Width 32 is a guess.
- **Program tail.** The tail is always `halt; nop×4; end`, where a nop is a 0x20 word = `0x800`. Constant words: halt = `0x10840`, end = `0x4000fc0`.

## 2. sync 0x1a: tile fence (1 word)

### Fields

| bits | width | name | encoding / formula | conf |
|---|---|---|---|---|
| 12-27 | 16 | tiles | tiles that receive the fence. Always 0xffff except mesh syncs and the drain that follows them (§2 table) | high |
| 28-42 | 15 | seq | global sequence number (§0.1) | high |
| 43 | 1 | b43 | 1 only in the `init` and `final` sync of PARAMETER_CACHING programs | high (meaning unknown) |
| 44-62 | 19 | counters | tile sync-counter mask (§0.3) | med-high |
| 63 | 1 | b63 | 1 exactly when counters = MESH_EAST_IN\|MESH_WEST_OUT | high (meaning unknown) |
| 64-78 | 15 | count | 0, except mesh syncs (formula below) | high (formula), low (meaning) |
| 79-88 | 10 | units | tile unit mask; 0x3ff = all 10 units. Bit 7 = meshBus 0x16 and bit 8 = meshBus 0x17 follow from usage, so bits 6-9 are meshBus 0x15-0x18. Bit 5 is used before every input: it is wideToNarrow if bit k = opcode 0x0f+k, or ringBusProducer if the tile CSR order (op, w2n, n2w, rc0, rc1, rp, mb0-3) applies. This is **ambiguous**. Bit 0 = op under both readings. | med |
| 107 | 1 | signal | send a completion token to the scalar core. Every sync with signal=1 is immediately followed by an `scsync` with `tokwait` | high (pairing), med (meaning) |
| 109-110 | 2 | f109 | 3 only in the last sync of a program (`final`), before the completion interrupt | high |

Constant pattern: `mask=ffff97fffe007f81000007f009889fff value=…09889680`. The 1-bits of that value are opcode 0x1a plus the tile-mask bits that are set in every instance.

### The 12 variants (every instance is one of them; `scalar.sync_variant()`)

| variant (builder) | tiles | counters | count | units | signal | other | where | n (497 programs) |
|---|---|---|---|---|---|---|---|---|
| `sync_init` | ffff | ALL19 | 0 | 3ff | 1 | b43 = caching | 2nd word of every program, seq 0 | 497 |
| `sync_signal` | ffff | 0 | 0 | 3ff | 1 | | full drain + token; always followed by `scsync_fence` | 823 |
| `sync_final` | ffff | 0 | 0 | 3ff | 1 | f109=3, b43 = caching | last sync | 497 |
| `sync_drain` | any | 0 | 0 | 3ff | 0 | | phase boundaries | 653 |
| `sync_wn_fence` | ffff | 0 | 0 | 0x020 | 0 | | starts the DMA part of every input | 337 |
| `sync_reset17` | ffff | ALL17 (0..16) | 0 | 3ff | 0 | | phase boundaries | 1584 |
| `sync_reset_mesh` | ffff | the 8 MESH counters | 0 | 3ff | 0 | | after a mesh phase | 123 |
| `sync_rpa` | ffff | RING_PRODUCER_A | 0 | 3ff | 1 | | after the PRODUCER_A wait of each output | 330 |
| `sync_rpb` | ffff | RING_PRODUCER_B | 0 | 3ff | 1 | | conv: after the PRODUCER_B wait | 14 |
| `sync_w2n` | ffff | WIDE_TO_NARROW | 0 | 0 | 0 | | 4-D relu ≥ 4 KiB, after the input DMA | 13 |
| `sync_mesh_west` | partial | MESH_EAST_IN\|MESH_WEST_OUT | see below | 0x080 | 0 | b63=1 | between meshBus 0x16 groups | 314 |
| `sync_mesh_north` | partial | MESH_SOUTH_IN\|MESH_NORTH_OUT | see below | 0x100 | 0 | | between meshBus 0x17 groups | 73 |

#### Mesh syncs in FC programs

Let m = Kp/1024 rounded up, i.e. the number of 64-byte input chunks per tile.

| K | west syncs (tiles, count) | north syncs (tiles, count) |
|---|---|---|
| 96, 128 | (ffff, 0) | |
| 192 | (ffff, 0), (fffd, 2) | |
| 256-320 | (ffff, 0), (fffd, 2), (fff9, 4) | |
| 512 | (ffff, 0), (ffdd, 2), (ff99, 4) | |
| 768 | (ffff, 0), (fddd, 2), (f999, 4) | (ffef, 8) |
| 1024 | (ffff, 0), (dddd, 2·m), (9999, 4·m) | (ffef, 8·m), (feef, 16·m) |
| 2048, 4096 | same as 1024, with m = 2 and m = 4 | same as 1024 |

In words:
- **Count formula (high on data).** The count of the g-th west sync is 2·m·g. The count of the h-th north sync is 8·m·h.
- **Reading (guess).** count/2 is the number of 64-byte chunks per tile moved by the *earlier* groups. A north hop carries a whole row, 4 chunks.
- **Tiles.** The mask drops the columns (west) or rows (north) whose transfers are finished.

### Semantics (hypothesis)

The patent's TileFenceOp blocks further instructions to the tile units in `units` until their queues have retired. The tiles then:
- set the counters in `counters` to `count` (`count`=0 resets them), or alternatively wait on them;
- send a token to the scalar core when `signal` is set.

Arguments for the "set/reset" reading of counters+count:
- The compiler emits many `sync_reset17` with count 0, which would be no-ops as waits.
- `sync_w2n` has units=0, so no fence, only the counter operation.

The mesh syncs fit a "wait ≥ count" (pipelining barrier) reading better. This question is open.

## 3. scsync 0x24: scalar fence / scalar sync (2 words)

| bits | width | name | encoding | conf |
|---|---|---|---|---|
| 12-27 | 16 | tiles | 0xffff when tokwait or wait is set, else 0 | high |
| 32-40 | 9 | smask | scalar sync-counter mask (§0.4) | med |
| 43-55 | 13 | value | value for the selected counter(s); only 0x1fff with smask=AVDATA_INFEED seen | high (value), low (meaning) |
| 57 | 1 | tokwait | wait for the tokens of the tiles in tmask (from `sync.signal`) | med-high |
| 73-88 | 16 | tmask | 0xffff with tokwait | high |
| 93+16k … 108+16k | 16 each | thr0..thr8 | per-counter wait threshold. Seen: **thr5** (bits 173-188) = number of outfeeds of the output. **thr6** (bits 189-204) = parameter bytes / 256 (conv parameter broadcast over the ring). Slots for other k are extrapolated (guess). | high (thr5/thr6) |
| 237-245 | 9 | wait | bit k: wait until counter k ≥ thr_k. Seen: bit 5 (abs 242) and bit 6 (abs 243) | med |
| 246-250 | 5 | sunits | 0b01100 normal; 0b11111 in the first scsync; 0b10000 before each scalar-memory outfeed; 0b01000 after the last one. Guess: a scalar-unit idle mask in order (avDataPop, parameterPop, infeed, outfeed, scalar pipeline). Under that reading, normal = infeed+outfeed idle and "after smem outfeeds" = outfeed idle. | high (values), low (meaning) |

Constant pattern:
- w0 `mask=fffffffffe0001fffd000600f0000fff value=…0900`
- w1 `mask=f833ffffffffff505ffc1fffffffffff value=0`

### The 10 variants (every instance; `scalar.sync_variant()`)

| builder | fields | where | n |
|---|---|---|---|
| `scsync_init` | tiles=ffff, smask=0x1ff, tokwait, tmask=ffff, sunits=0x1f | 3rd word of every program | 497 |
| `scsync_fence` | tiles=ffff, tokwait, tmask=ffff | after every `sync` with signal=1 | 1664 |
| `scsync_set_av` | smask=AVDATA_POP\|AVDATA_INFEED | first instruction of every input | 337 |
| `scsync_av_credit` | smask=AVDATA_INFEED, value=0x1fff | just before avDataPop | 337 |
| `scsync_set_avpop` | smask=AVDATA_POP | just before avDataPop | 337 |
| `scsync_nop` | nothing but sunits | FC/conv, after the input phase | 176 |
| `scsync_wait_pa(n)` | tiles=ffff, smask=wait=PRODUCER_A, thr5=n | after the outfeeds of an output; n = #outfeeds of that output | 330 |
| `scsync_wait_pb(n)` | tiles=ffff, smask=wait=PRODUCER_B, thr6=n | conv exe; n = conv parameter bytes / 256 | 14 |
| `scsync_smem_pre` / `_post` | sunits only | around scalar-memory outfeeds | 59 / 6 |

Semantics (med/low):
1. With tokwait, the scalar core stalls until it has the tokens of the `sync.signal` fence from all tiles in tmask. This is the patent's ScalarFenceOp waiting for a TileFenceOp count.
2. For each counter in smask: if its `wait` bit is set, wait for `counter ≥ thr`; otherwise set the counter to `value`. The AVDATA_INFEED value 0x1fff is probably the flow-control credit (8191 act units ≈ 64 KiB) between avDataPop and infeed.
3. Wait until the scalar units in `sunits` are idle.

## 4. pop 0x25: avDataPop / parameterPop (2 words)

Let U = transfer units: U = bytes/8 for activations, bytes/64 for parameters. Bytes = the INPUT / PARAMETER hint size, already padded to a multiple of 8 by the compiler.

| bits | width | name | act (stream=1) | param (stream=0) | conf |
|---|---|---|---|---|---|
| 12 | 1 | stream | 1 | 0 | high |
| 28-43 | 16 | d0_stride | 1 | 1 | high (value) |
| 44-58 | 15 | d0_limit | 16383 | 1023 | high (value) |
| 59-74 | 16 | d1_stride | neg(16383,16) = 0xc001 | neg(1023,16) = 0xfc01 | high (value) |
| 75-89 | 15 | d1_limit | passes−1, passes = ceil(U/16384) | floor(U/1024) | high |
| 90-105 | 16 | d2_stride | 0xc001 | 0xfc01 | high (value) |
| 106-120 | 15 | d2_limit | 0 | 0 | |
| 121, 124 | 1 | k121, k124 | 1 | 1 | |
| 136 / 140 | 1 | par136 / av140 | 0 / 1 | 1 / 0 | high |
| 142-155 | 14 | count_m1 | last−1, last = U − 16384·(passes−1); 0 if last = 16384 | (U mod 1024) − 1 | high |
| 157 | 1 | k157 | 1 (0 if last = 16384) | 1 | high |
| 159-172 | 14 | count_neg | neg(last, 14), 0 if last = 16384 | neg(U mod 1024, 10) | high |
| 174, 178, 199, 215 | 1 | k* | 1 | 1 | |
| 177 | 1 | par177 | 0 | 1 | high |
| 183 | 1 | b183 | 0 | caching: U even; streaming: 0 | high |
| 184-198 | 15 | rows_neg | 0 | caching: neg(min(U//2, 1024), 15); streaming: 0 | high |

Special case U = 1 (an input ≤ 8 bytes): every walk and count field is 0. Only stream, d0_stride=1 and the k*/av140 bits stay set.

Builders: `av_pop(nbytes)` and `param_pop(nbytes, streaming=False)`. These are bit-exact for all exe pops (including 128-256 KiB inputs), all caching pops, and the three parameter-streaming STAND_ALONE pops. The streaming pops use the same d1_limit/count fields as caching pops, but b183 = rows_neg = 0.

Interpretation (med):
- **TTU walk.** d0..d2 look like a TTU loop nest walking a circular staging buffer. Each slot is 31 bits wide: a 16-bit signed stride and a 15-bit limit. Strides are relative: −d0_limit rewinds to the start of the buffer.
- **Buffer passes.** The pop makes d1_limit+1 passes over the buffer. Every pass is full except the last, which has count_m1+1 units (if k157 = 1).
- **Activation buffer** = 16384 × 8 B = **128 KiB**.
- **Parameter passes** are 1024 units = 64 KiB. However rows_neg counts 128-byte rows capped at 1024 rows (128 KiB), and infeed offsets wrap at 2048 × 64 B = 128 KiB. So the parameter buffer is probably 128 KiB with a 128-byte walk granule.

## 5. infeed 0x26 (4 words)

U is in the same units as for the pop. off = byte offset in the input (or parameter blob) / unit.

| bits | width | name | act | param | conf |
|---|---|---|---|---|---|
| 12 | 1 | stream | 1 | 0 | high |
| 14-33 | 20 | buf_off | off (not wrapped; seen up to 24576) | off mod 2048 | high |
| 34-54 | 21 | d0_stride | 1 | 1 | high (value) |
| 55-74 | 20 | d0_limit | 16383 (0 if U=1) | 1023 | high (value) |
| 75-95 | 21 | d1_stride | neg(16383,21) (0 if U=1) | neg(1023,21). Streaming programs use other values (bit 79) | high (caching) |
| 96-115 | 20 | d1_limit | 0 | floor(U/1024) | high |
| 116-136 | 21 | d2_stride | neg(16383,21) (0 if U=1) | neg(1023,21) | high (value) |
| 137-156 | 20 | d2_limit | 0 | 0 | |
| 157, 160 | 1 | k157, k160 | 1 | 1 | |
| 172 / 176 | 1 | par172 / av176 | 0 / 1 | 1 / 0 | high |
| 183-196 | 14 | count_m1 | U−1 | (U mod 1024)−1 | high |
| 203 | 1 | k203 | 1 (0 if U=1) | 1 | |
| 205-218 | 14 | count_neg | neg(U,14) (0 if U=1) | neg(U mod 1024, 10) | high |
| 311 | 1 | k311 | 1 | 1 | |
| 314 | 1 | par314 | 0 | 1 | high |
| 320-335 | 16 | pop_wait | off + 1 (unwrapped) | off + 1 (unwrapped) | high (formula), med (meaning: starts once the pop has produced unit `off`) |
| 336, 352, 392 | 1 | k* | 1 | 1 | |
| 409-424 | 16 | tiles | ring destination bitmap (the ring header); equals the tile mask of the paired ringConsumer | | high |
| 434-437 | 4 | k434 | 0xf | 0xf | |
| 438-439 | 2 | f438 | 3 | (U−1) mod 4: the 64-byte lanes used in the last 256-byte row, minus 1 | high |

Builders `av_infeed(offset, nbytes, tiles)` and `param_infeed(offset, nbytes, tiles)` are bit-exact for every act infeed and every caching parameter infeed.

### How the compiler chunks a 1-D activation input

S = input bytes, padded to 8.
- **Tiles.** Each tile gets b = 64·⌈S/1024⌉ bytes, and T = ⌈S/b⌉ ≤ 16 tiles are used.
- **Infeeds.** There is one infeed per group of 4 tiles: chunk j covers tiles 4j..4j+3 (only the tiles actually used), at offset 4jb, with min(4b, S−4jb) bytes. Each infeed directly follows its ringConsumer.
  - Example: n=272 → (0, 256, 000f), (256, 16, 0010).
  - Example: n=2304 → 3 × 768 B.
- **Same rule for FC.** FC inputs (S = K padded to 8) are chunked the same way.
- **conv1x1** sends the whole input in one infeed to tiles 0005 (M=2) or 000f (M≥8).
- **Overlap.** For 4-D tensors whose per-tile boundaries are not 8-byte aligned, consecutive chunks overlap by one unit.

## 6. outfeed 0x27 (2 words)

| bits | width | name | host variant (all FC/relu/conv) | scalar-memory variant | conf |
|---|---|---|---|---|---|
| 12-26 | 15 | base | 0 | destination offset in scalar memory (elements; tile t gets Σ of the previous tiles' counts) | high (formula) |
| 27-42 | 16 | d0_stride | 0 (1 if U=1) | 1 | high (value) |
| 43-57 | 15 | d0_limit | U−1, U = bytes/8 | elements − 1 | high |
| 58-73 | 16 | d1_stride | 0 | neg(d0_limit,16) | high (value) |
| 74-88 | 15 | d1_limit | 0 | 0 | |
| 89-104 | 16 | d2_stride | 0 | neg(d0_limit,16) | high (value) |
| 105-119 | 15 | d2_limit | 0 | 0 | |
| 120-135 | 16 | d3_stride | 0 | 1 | high (value) |
| 173-174 | 2 | k173 | 3 | 0 | |
| 222-228 | 7 | f222 | 0x7f | 0x3f | |

Notes:
- There is one outfeed per output tile, each directly followed by the ringProducer of that tile. The bytes per tile follow the input rule: for 1-D relu, b = 64·⌈S/1024⌉ with the remainder on the last tile. For FC, 64 bytes per tile with the remainder (padded to 8) on the last tile, and N/16 when N > 1024.
- `outfeed(nbytes)` is bit-exact for all 1971 host outfeeds, and the outfeed sizes add up to the OUTPUT hint.
- The **scalar-memory variant** (bit 120 set) appears in 4-D relu/add programs with tiny channels. Each outfeed is bracketed by `scsync_smem_pre`. After them come `scsync_smem_post`, a `0x23` instruction and vector-slot ALU words; this path was not decoded further.

## 7. Host DMA descriptors (scalar ALU 0x20 sequences)

- **Descriptor.** A descriptor is 4 words, pushed with `v_op=0xa, v_offset=k, vs_reg=r` (desc[k] = s[r]) and issued with `v_op=0xc`:
  - desc[0] = address lo, desc[1] = address hi, desc[2] = length in bytes, desc[3] = tag.
- **Tags (high).** 1 = input stream, 2 = parameter stream, 3 = output stream, 4..7 = scalar-core interrupt 0..3.
  - The USB bridge turns descriptors into transfers on the bulk endpoints.
  - Tags 4-7 become 16-byte events on EP 0x82; `descr_ep=0xF0` filters them.
  - The completion interrupt `sc_int_0` is the descriptor (0, 0, 0, 4).
- **Length.** The length equals the hint size: input/output padded to 8 bytes, the parameter blob size, and 0 for the interrupt. Verified on all 497 programs by replaying the ALU.
- **Pipelining.** Pushes are VLIW bundles. A MOVI in the same bundle as a push is visible one bundle later. For example `s8 = len` rides with the push of desc[1], and s8 is pushed in the next bundle.

### `add64`: 64-bit address + 31-bit offset

The ALU has only 32-bit registers, so each address add uses 8 bundles. Example: hi=s11, lo=s12, out_hi=s4, out_lo=s5, tmp=s6.

```
p0 = lo < 0              s_op 0x2c (LT imm, signed)   -> bit31 of lo
p1 = lo != lo            s_op 0x0a (NEQ reg)          -> false
tmp = lo & 0x7fffffff    s_op 0x23
[p0] tmp += off          s_op 0x21, predicated on p0
[p0] p1 = tmp < 0        carry out of bit 31 (valid for off < 2^31)
out_lo = lo + off
out_hi = hi + 0
[p1] out_hi += 1
```

### Sequences emitted by the compiler (all reproduced bit-exactly by `host_dma` / `interrupt`)

| use | base regs (hi:lo) and relocation | sequence | registers |
|---|---|---|---|
| input, exe | s11:s12 = INPUT x, via relocated MOVIs (`field_offsets` point at bit 70 of the MOVI words) | MOVI s11, MOVI s12, add64(+0 → s4:s5, tmp s6), add64(+0 → s7:s6, tmp s10), push s6,s7,s8,s9, issue | s8 = length, s9 = 1 |
| output, exe | s11:s12 = OUTPUT y (relocated MOVIs) | same as input. The 10 words up to the first add64 come before the narrowToWide tile instructions, and the remaining 13 come after them | s9 = 3 |
| parameters, caching | s0:s1 = PARAMETER (relocated) | MOVI s1, MOVI s0, add64(→ s2:s3, tmp s4), add64(→ s5:s4, tmp s8), push s4,s5,s6,s7, issue | s6 = length, s7 = 2 |
| interrupt, exe | s4 = s5 = 0 | MOVI s4, MOVI s5, add64(→ s7:s6, tmp s10), push/issue | len 0, tag 4 |
| interrupt, caching | s2 = s3 = 0 | MOVI s2, MOVI s3, add64(→ s5:s4, tmp s8), push/issue | len 0, tag 4 |

More base registers:
- Exe programs load s1/s0 = PARAMETER lo/hi and s3/s2 = SCRATCH lo/hi right after `scsync_init` (relocated MOVIs).
- Over USB, all addresses are irrelevant and every base is 0.
- The second add64 is where a per-chunk offset would go. Geohot's inception chunk uses the first add64 with offset 0x409a80 (a tensor inside the parameter buffer).

## 8. Program skeletons

Notation:
- `seq` = the running sequence number at that point.
- Tile instructions are opaque here.
- `exe_prologue`, `input_head`, `input_dma`, `output_dma_head/tail`, `output_wait` and `epilogue` are block builders from `scalar.py`. Each appears verbatim, with correct seq, in every exe/standalone program of both sets, once tile instructions are removed from the stream. The compiler sometimes interleaves independent tile instructions inside a block, for example in 4-D relu.

### 8.1 PARAMETER_CACHING (exact; `caching_program()` rebuilds all 173 byte for byte)

```
start(53 + 8T)
sync_init(caching=True) ; scsync_init
host_dma(tag 2, P)                      # MOVI s1,s0 (PARAMETER, relocated) + 21 ALU words
param_pop(P)
ringConsumer × T                        # tile side, seq 1..T, writes wide memory of tile t
param_infeed(t·P/T, P/T, tiles=1<<t) × T  # FC: equal split over T = ceil(N/64) tiles; conv: one chunk to tile 0
sync_final(T+1, caching=True) ; scsync_fence
interrupt(caching=True) ; halt ; nop×4 ; end
```

### 8.2 relu, 1-D, n elements

S = n padded to 8, b = 64·⌈S/1024⌉, T = ⌈S/b⌉, C = ⌈T/4⌉.

```
start ; exe_prologue                                  # init fence, s1/s0/s3/s2 MOVIs
input_head(1) ; input_dma(2, S)
wideToNarrow × T ; (ringConsumer(g_j) ; av_infeed(4jb, ..., g_j)) × C
sync_drain ; sync_reset17 ; sync_reset17
meshBus0x17 × 4 (tiles ffff) ; narrowToWide (ffff) ; op × T
sync_reset17 ; sync_signal ; scsync_fence
output_dma_head ; narrowToWide × T ; output_dma_tail(S)
(outfeed(b_t) ; ringProducer(tile t)) × T
output_wait(seq, T) ; epilogue(seq) ; end
```

### 8.3 FC execution program

T_in = input tiles (§5 rule with S = K padded to 8), T = ⌈N/64⌉ output tiles, u = Kp/64.

```
start ; exe_prologue ; input_head(1) ; input_dma(2, S)
wideToNarrow × T_in ; (ringConsumer ; av_infeed) × C
sync_drain ; sync_reset17
[u ≥ 2: sync_reset17 ; mesh phase = meshBus0x16 groups separated by sync_mesh_west, then
        meshBus0x17 groups separated by sync_mesh_north (K ≥ 768) ; sync_drain(partial) ; sync_drain ; sync_reset_mesh]
scsync_nop ; sync_reset17
[T ≥ 2: wideToNarrow ; ringConsumer ; ringProducer ; narrowToWide ; sync_reset17]   # copy the input from tile 0 to the other tiles
[T = 1: sync_reset17]
sync_signal ; scsync_fence
wideToNarrow ; op × T
sync_reset17 ; sync_signal ; scsync_fence
output_dma_head ; narrowToWide × T ; output_dma_tail(N padded)
(outfeed(64 or N) ; ringProducer(tile t)) × T
output_wait(seq, T) ; epilogue(seq) ; end
```

### 8.4 Other forms seen

- **Several inputs (ADD) and several inputs/outputs (2-3 independent relus).**
  - Every input repeats `input_head ; input_dma ; (ringConsumer ; infeed)*`.
  - Every output repeats `output_dma_head … output_wait(seq, its outfeed count)`.
  - The PRODUCER_A threshold counts only that output's outfeeds.
- **conv1x1.**
  - Before the `sync_wn_fence` there are `meshBus0x17 × 4 ; narrowToWide`.
  - The parameter broadcast `ringProducer → 0x12` adds `scsync_wait_pb(P/256) ; sync_rpb ; scsync_fence` after the PRODUCER_A wait.
- **Inputs ≥ 128 KiB.**
  - Pops make several passes (§4). Infeed offsets stay unwrapped.
  - One extra `sync_w2n` follows the input DMA in 4-D relus of ≥ 4 KiB.
- **Parameter streaming** (FC 2048-4096 × 2048-4096, STAND_ALONE).
  - The program contains a tag-2 DMA for the whole blob, a `parameterPop`, and 16 parameter infeeds with small per-round counts (8-16 units, round-robin over tiles).
  - Their d1_stride (bit 79) and pop_wait values follow a different, undecoded pattern.

## 9. Constant bit patterns per word (all instances of each class; mask=constant bits, value=their value)

| class (n) | word | mask | value |
|---|---|---|---|
| start (497) | w0 | ffffffffffffffffffffffe003ffffff | 00000000000000000000000000000f80 |
| end (497) | w0 | all ones | 00000000000000000000000004000fc0 |
| halt (497) | w0 | all ones | 00000000000000000000000000010840 |
| sync (5258) | w0 | ffff97fffe007f81000007f009889fff | 00000000000000000000000009889680 |
| scsync (3757) | w0 | fffffffffe0001fffd000600f0000fff | 00000000000000000000000000000900 |
| | w1 | f833ffffffffff505ffc1fffffffffff | 0 |
| pop/act (337) | w0 | fffffcfffbfff1fff4000fffffffffff | 12000000000000000000000010001940 |
| | w1 | ffffffffffffffffffffe00050003fff | 00000000008000800004400000001000 |
| pop/param (176) | w0 | fffffffffff007ffffffffffffffffff | 120003f0040007e0083ff00010000940 |
| | w1 | ffffffffffffff80007ffe007f003fff | 00000000008000800006400020000100 |
| infeed/act (651) | w0 | ffefffff01fff7e0007fffffe0003fff | 00000000000000000000000400001980 |
| | w1 | fffffffff80017e0007ffffffffffe03 | 00000000000000000001000120000000 |
| | w2 | ffffffffffff8000ffffffffffffffff | 00000001000100000080000000000000 |
| | w3 | fffffffffffffffffffffe0001ffffff | 000000000000000000fc000000000100 |
| infeed/param (1053) | w0 | ffffffe0fffffffffffffffffe00ffff | c0100000ffe00801ff80000400000980 |
| | w1 | ffffffffff801ffe007fffffffffffff | 000000000000080000001001200001ff |
| | w2 | ffffffffffff0003ffffffffffffffff | 00000001000100010480000000000000 |
| | w3 | ffffffffffffffffff3ffe0001ffffff | 0000000000000000003c000000000100 |
| outfeed/host (1971) | w0 | ffffffffffffffffff8007fff7ffffff | 000000000000000000000000000009c0 |
| | w1 | all ones | 0000001fc00000000000600000000000 |
| outfeed/smem (59) | w0 | fffffe0001fffc0003ff07fffff80fff | 010000000000000000000000080009c0 |
| | w1 | all ones | 0000000fc00000000000000000000000 |

The act pop and infeed classes include the U=1 degenerate form, which has fewer constant bits. The field tables above give the per-class defaults.

## 10. Open questions

1. **Meaning of `count` and `b63` in mesh syncs.** "Set counter := count" or "wait ≥ count"? Does the count unit start at bit 64 or at bit 65 (all observed values are even)?
2. **sync `units` bits 1-5.** Do they follow opcode order or tile CSR order? This decides whether bit 5 (`sync_wn_fence`) means wideToNarrow or ringBusProducer.
3. **Bits with unknown meaning.**
   - sync: b43 (caching), f109 = 3 (final).
   - scsync: `sunits`, and the AVDATA_INFEED value 0x1fff.
   - halt.code = 4, end.length = 128.
4. **Positions of scsync thr0-4, thr7, thr8.** Only thr5 and thr6 were observed. Is the 16-bit slot layout right?
5. **Staging-buffer sizes.** The parameter walk is 1024 units with 1024-unit passes, but offsets wrap at 2048 units and rows at 1024 × 128 B. The activation credit 0x1fff (8191 units) is half of the 16384-unit walk. How do the parameter buffer (64 or 128 KiB) and the activation buffer (128 KiB walk vs 64 KiB credit) fit together?
6. **Parameter streaming encodings.** These were not decoded: infeed d1_stride ≠ −1023 and per-round counts.
7. **Width of the start length field and of `seq`.** Does `seq` wrap? Programs with more than 4096 tile instructions were not observed.
8. **Scalar-memory outfeed path.** 0x23, the vector-slot ALU words, and when the compiler chooses it.

## Experiments compiled for this study (in addition to `tools/data/corpus.pkl.xz`)

- **relu, 1-D:**
  - n = 1..64 (every value).
  - n = 80-3072 (19 sizes).
  - n = 6144, 12288, 16384, 24576, 32768, 49152.
  - 1-D relu fails to compile for n > 65536.
- **relu, multi-dimensional shapes:**
  - 4-D: (1,2,2,4), (1,4,4,4/8/16), (1,8,8,8/16/32), (1,16,16,16), (1,16,16,3), (1,7,7,5), (1,3,5,7), (1,32,32,8), (1,2,3,64), (1,64,64,4), (1,1,1,100), (1,1,100,1), (1,10,10,1).
  - 3-D: (1,4,64), (1,16,64), (1,3,100).
  - Large: 128-256 KiB, (1,64,64,64), (1,128,128,8), (1,32,32,128), (1,80,64,32), (1,96,64,32), (1,129,128,8), (1,127,128,8), (1,64,64,40), (1,48,64,64), (1,100,100,8).
- **ADD (2 inputs):**
  - n = 16, 64, 256, 1024, 4096.
  - shapes (1,8,8,16), (1,4,4,4).
- **Independent relus** with 2 or 3 inputs/outputs.
- **FC:**
  - N = 32, 48, 80.
  - K = 4096 with N = 16, 64, 256, 1024.
  - Parameter streaming: 2048×4096, 4096×4096, 4096×2048.
- **conv1x1:**
  - M = 256, K = 320, N = 64 (80 KiB input).
  - M = 200.

`python coral/isa/scalar.py --extra` recompiles this set (docker `etpc`; results cached in `.compile/`) with `extra_programs()` and verifies it.
