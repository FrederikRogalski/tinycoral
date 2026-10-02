# Tile-local DMA: 0x14 wideToNarrow and 0x13 narrowToWide

Both instructions are 7 words (896 bits). Each one moves data inside a tile, between narrow memory (activations, 192 KiB) and wide memory (parameters plus the ring-bus FIFOs). Each side has its own address and its own TTU loop nest.

- **wideToNarrow** moves wide memory into narrow memory.
  - In `mode=2` it moves parameters into the bias/"scaling" store instead.
- **narrowToWide** moves narrow memory into wide memory.

Code: `coral/isa/wide_narrow.py` provides:
- `encode_/decode_wide_to_narrow`, `encode_/decode_narrow_to_wide`, and the dispatchers `encode_/decode_wide_narrow`;
- the helpers `ttu()`, `ttu_strides()` and `level_mask()`;
- the compiler-template recipes `w2n_input`, `w2n_ring_recv`, `w2n_bias` and `n2w_output` (§4).

Every bit belongs to exactly one field; unknown bits are in `rsv*` fields. The round trip is exact: run `python coral/isa/wide_narrow.py`. See "Round trip" below.

Evidence base:
- The corpus: 1313 wideToNarrow and 1218 narrowToWide instances.
- About 2600 more instances from extra compiles:
  - relu with n = 64·j (j ≤ 32) and n = 1024·k (k ≤ 64);
  - FC sweeps over K at N=64, over N at K = 64/320/1024, and odd shapes;
  - conv3x3, maxpool, a strided 1x1 conv, and a two-input ADD.
- No hardware runs. Nothing here has been executed on the device.

## 1. Layout overview

| region | wideToNarrow 0x14 | narrowToWide 0x13 |
|---|---|---|
| header: predication, opcode, `tile_mask`, `seq` | [0, 60) | [0, 62) |
| source address | `wide_addr` [60, 74) | `narrow_addr` [62, 78) |
| source loop nest (TTU) | wide, 4 levels [78, 210) | narrow, 6 levels [78, 276) |
| source trailer | [210, 267) | [276, 335) |
| destination address | `narrow_addr` [267, 283) | `wide_addr` [335, 349) |
| destination loop nest | narrow, 6 levels [283, 481) | wide, 4 levels [353, 485) |
| destination trailer | [481, 611) | [485, 543) |
| sync block (same relative layout) | [611, 699) | [543, 631) |
| tail flags | [867, 872) | [799, 803) |

Narrow-side and wide-side structures are identical in both opcodes; only the order (source first) differs.

- **Narrow side.** A 16-bit address, then 6 × (inc s17, lim u16), then a 6-bit level mask.
- **Wide side.** A 14-bit address and 4 reserved bits, then 4 × (inc s17, lim u16), then a trailer:
  - +0..3: level mask;
  - +4: `wide_circ`;
  - +6..16: `wide_rows`.

### TTU (loop nest) encoding (high confidence)

Level d (innermost = 0) has a signed 17-bit increment `inc_d` and a 16-bit `lim_d`, which is the iteration count − 1.
- On every step, the address advances by `inc_d` of the outermost level that steps; all inner levels wrap at the same time.
- The compiler pre-computes the increments from logical strides:

```
inc_d = stride_d - sum_{j<d} stride_j * lim_j
```

- Unused outer levels get count 1 and stride 0. Their increment is therefore `-sum stride_j*lim_j`, the "return to base" value.
  - Example: relu tile 64 B gives `1/15 -15/0 -15/0 ...`. These are the "6-bit fields equal to k−1 and −k" and the "17-bit −(⌈k/4⌉−1)" fields from earlier diffs.
- An unused level 0 keeps its natural stride 1, e.g. `1/0`.
- Verified: `ttu(strides(inc, lim)) == fields` on every instance.

Non-trivial examples that pin down the semantics:

| example | narrow inc/lim per level | logical strides | counts | meaning |
|---|---|---|---|---|
| conv1x1 M=8 output (0x13) | `1/15 1/1 -31/0` | 1, 16 | 16, 2 | 2 positions × 64 B |
| maxpool relay (0x13) | `16/3 -47/15 1/3 -255/0` | 16, 1, 64 | 4, 16, 4 | a 4×16 transpose |
| bias load, N>1024 (0x14) | `1/31 33/0 -31/G-1` | 1, 64, 0 | 32, 1, G | level 1 has count 1, level 2 repeats G times |

Units (high confidence):
- **Narrow side.** One step and one address unit are 4 bytes (narrow memory has 4 byte-banks).
  - relu n=16 → `lim0`=3; n=65536 → 4096 B/tile → `lim0`=1023.
- **Wide side.** One TTU step is one 256-byte wide row.
  - relu 4096 B/tile → `w_lim0`=15.
  - conv3x3: 4 rows per 1 KiB refill.
  - **But `wide_addr` is in 64-byte units.** That is the relocation unit from NOTES, and FIFO strides of `+4` per row. Every observed wide address has its low 2 bits 0.

## 2. Field tables

Abbreviations:
- H/M/L: high/medium/low confidence.
- "all": every corpus instance. "ext": the extra compiles as well.
- S = total input bytes (relu n, FC K).
- `P = 64·⌈S/1024⌉`: bytes per tile/chunk.
- `j = c mod 4`: position inside a 4-tile ring-consumer group.
- `o = (j·P) mod 256`: byte offset inside the wide row.
- `B = min(P, S − c·P)`: bytes for chunk c.
- X / Y: narrow base addresses of the input / output tensor (in words; allocator choices, see §4).

### 2a. wideToNarrow (0x14)

| bits | w | name | encoding / formula | evidence | conf |
|---|---|---|---|---|---|
| 0 | 1 | gate | predication (shared header); 0 in corpus | all | M |
| 1–3 | 3 | pred_reg | 0 in corpus | all | M |
| 4 | 1 | pred_pol | 0 in corpus | all | M |
| 5 | 1 | rsv5 | 0 | all | – |
| 6–11 | 6 | opcode | 0x14 | all | H |
| 12–27 | 16 | tile_mask | tiles that execute it; per-chunk `1<<c`; shared templates use OR-ed masks | all | H |
| 28–45 | 18 | rsv28 | 0 | all | – |
| 46–59 | 14 | seq | program-global dispatch sequence number, 0-based (see below); width ≥7 seen, 14 assumed | 335/335 programs | H (meaning), L (width) |
| 60–73 | 14 | wide_addr | source, in 64-B units (low 2 bits always 0). Values: input FIFO `0x2080 − 8·⌈S/256⌉ + 4·⌊j·P/256⌋`; ring-receive FIFO `0x1f70`; bias load = parameter base (the known 13-bit relocation field; 0 in compiler output) | relu+FC: all verified | H |
| 74–77 | 4 | rsv74 | 0 (maybe address MSBs) | all | – |
| 78+33d (d=0..3) | 17 | w_inc*d* | wide TTU increment (signed), see §1 | all | H |
| 95+33d | 16 | w_lim*d* | wide TTU count−1, in 256-B rows. Input: `w_lim0 = ⌈(o+B)/256⌉−1`; ring receive: `⌈K/256⌉−1` with `w_inc0=0` (FIFO, inc 0) when >1 row; bias load N>1024: `w_lim1 = G−1, w_inc1 = 1` | verified | H |
| 210–213 | 4 | wide_lvl_mask | bit1 = (wide stride₁≠0) [all+ext]; bit0 = (mode≠0) [all+ext]; bits 2,3 never set | all+ext | M |
| 214 | 1 | wide_circ | = (wide_rows>0 and w_inc0==1), all+ext. Guess: circular-FIFO addressing enable | all+ext | M (rule) / L (meaning) |
| 215 | 1 | rsv215 | 0 | | – |
| 216–226 | 11 | wide_rows | 0 for linear access. 1 for FC ring receive. FIFO depth in rows for conv streaming (e.g. M·K/256 for conv1x1; 4 for conv3x3 16×16×64). Guess: circular-buffer size in 256-B rows; width assumed | all+ext | M/L |
| 227–266 | 40 | rsv227 | 0 | | – |
| 267–282 | 16 | narrow_addr | destination, in 4-byte words. Input chunk: `X + c·P/4`; relu `X=0` (n>64; 0xc0 for n≤64), i.e. tile t gets `t·P/4`. Ring receive: `X`. Bias load: always `0x40` (see open questions) | verified | H |
| 283+33d (d=0..5) | 17 | n_inc*d* | narrow TTU increment (signed). Always `n_inc0=1` in corpus; 16 in a pool relay | all | H |
| 300+33d | 16 | n_lim*d* | narrow count−1, in words. Input: `n_lim0 = ⌈B/4⌉−1`; ring receive: `⌈K/4⌉−1`; bias: `31` (32 steps), `n_lim2=G−1`; conv streaming: `n_lim1` = refills−1 with stride 0 | verified | H |
| 481–486 | 6 | narrow_lvl_mask | bit d = (narrow logical stride_d ≠ 0); bit0 always 1; bias load has bit1 (phantom stride 64) | all+ext (2705/2705) | H (rule) |
| 487–488 | 2 | mode | 0 = FIFO ring receive (FC x broadcast to tiles 1..T−1); 1 = input from ring-consumer FIFO with sub-row selection; 2 = parameter → bias/"scaling" store | all | M |
| 489–493 | 5 | rsv489 | 0 | | – |
| 494–506 | 13 | size64 | Bytes per FIFO entry/fill ÷ 64, minimum 4 (= one wide row). Linear input: `max(4, ⌈B/64⌉)` (relu n=5120 → 5, n=65536 → 64; FC → 4). Circular input: bytes of one fill (conv1x1 M·K/64; conv3x3 4·rows). mode 2: 2; mode 0: 0. Width assumed | all+ext | H (formula) / M (meaning) |
| 507–522 | 16 | head_words_m1 | Only circular inputs with ≥2 narrow refills and a nonzero in-row offset. `head_words_m1 + 1 + 16·head_tail16 = words per refill`, with `head_tail16·16 = skip words`, i.e. the words before the FIFO wrap point. ≥8 bits observed (value 191); one exception (conv1x1 M=2 K=288 tile 0) | ext (conv3x3, pool, conv) | L |
| 523 | 1 | head_en | 1 iff the head_* fields are used | ext | L |
| 524–529 | 6 | rsv524 | 0 | | – |
| 530–533 | 4 | head_tail16 | words after the wrap ÷ 16; equals skip/16 when skip>0 | ext | L |
| 534–541 | 8 | rsv534 | 0 | | – |
| 542 | 1 | skip_en0 | 1 iff `o > 0` (the chunk does not start at a 256-B row boundary) | all | H |
| 544 | 1 | skip_en1 | always equal to skip_en0 | all+ext | H |
| 543, 545 | 1+1 | rsv543, rsv545 | 0 | | – |
| 546–561 | 16 | skip_base | narrow base of the whole destination tensor (`X`, words) when skip_en, else 0. FC N=1024 K=96 → 0x100; relu → 0; second ADD input → 0x100 | verified | H (formula) / L (purpose) |
| 562–577 | 16 | skip_m1 | words skipped at the start of the wide data − 1: `o/4 − 1` (relu/FC; e.g. 15, 31, 47 for j=1,2,3 with 64-B chunks) when skip_en, else 0. conv1x1 M=32 K=64 (512 B per tile slice): `128·j − 1` | verified | H |
| 578–593 | 16 | skip_end | = skip_base + skip_m1 (all+ext); redundant as far as we can tell | all+ext | H (rule) |
| 594–610 | 17 | rsv594 | 0 | | – |
| 611 | 1 | sync_en0 | always 1 | all | H (const) |
| 612 | 1 | sync_f1 | 0 for 0x14 (used by 0x13) | all | – |
| 613 | 1 | sync_f2 | = (mode==2) | all+ext | H (rule) |
| 614 | 1 | rsv614 | 0 | | – |
| 615–619 | 5 | sync_id | sync counter the transfer waits on, index in the tile `SyncCounter_*` CSR order: 14 = RING_READ_A (ring consumer 0 FIFO), 15 = RING_READ_B (ring consumer 1, opcode 0x12), 0 = none (cached params) | all+ext | M |
| 620–621 | 2 | sync_wait_lvl | 2 = linear input/FC bias; 1 = circular input, 1-row ring receive, param stream; 0 = multi-row ring receive (per row). Guess: loop level at which the wait is applied | all+ext | L |
| 622 | 1 | rsv622 | 0 | | – |
| 623 | 1 | sync_en1 | always 1 | all | H (const) |
| 624–642 | 19 | sync_val | 0x8000 for every FIFO read; 0xffff for cached-param bias loads; conv param streams 0x10000/0x30000/0x40000 (N=64/288/1024), 0x8000 (M=128). Meaning unknown (threshold/ratio?) | all+ext | L |
| 643–654 | 12 | rsv643 | 0 | | – |
| 655 | 1 | sync_en2 | always 1 | all | H (const) |
| 656 | 1 | sync_f45 | = (sync_id==0) | all+ext | M (rule) |
| 657–661 | 5 | rsv657 | 0 | | – |
| 662–663 | 2 | sync_dec_lvl | 2 linear input / param stream; 1 circular input; 0 = no decrement. Guess: level of the FIFO-credit decrement | all+ext | L |
| 664 | 1 | rsv664 | 0 | | – |
| 665–681 | 17 | sync_dec | signed; −1 for FIFO inputs (modes 1 and 2-stream), 0 otherwise. Guess: amount added to `sync_id` counter when consuming | all+ext | M |
| 682–696 | 15 | rsv682 | 0 | | – |
| 697–698 | 2 | sync_dec_mode | 3 whenever sync_dec≠0, else 0 | all+ext | L |
| 699–866 | 168 | rsv699 | 0 | | – |
| 867 | 1 | tail_f867 | = (mode==2) | all+ext | M (rule) |
| 868–869 | 2 | rsv868 | 0 | | – |
| 870 | 1 | tail_en870 | always 1 | all | H (const) |
| 871 | 1 | tail_f871 | = (sync_dec_lvl==1), i.e. circular/conv streaming inputs | all+ext | M (rule) |
| 872–895 | 24 | rsv872 | 0 | | – |

The sync block is the same in both opcodes. With r = bit − 611 for 0x14 and r = bit − 543 for 0x13:
- r0, r12, r44 are constant 1;
- r4..8 is `sync_id`;
- r9..10 is the wait level;
- r13..31 is `sync_val`;
- r45 is a flag;
- r51.. holds the decrement fields, which only 0x14 uses.

### 2b. narrowToWide (0x13)

Same conventions. Bits 0–59 are the same header as 0x14 (`opcode=0x13`).

| bits | w | name | encoding / formula | evidence | conf |
|---|---|---|---|---|---|
| 46–59 | 14 | seq | dispatch sequence number (as 0x14) | 335/335 | H/L(width) |
| 60–61 | 2 | rsv60 | 0 (maybe byte-address LSBs of narrow_addr) | all | – |
| 62–77 | 16 | narrow_addr | source, 4-byte words. FC output tile t: `Y + t·16·G`; FC ring send: `X`; relu output: `t·P/4` (0xc0 for n≤64) | verified | H |
| 78+33d (d=0..5) | 17 | n_inc*d* | narrow TTU increment (signed) | all+ext | H |
| 95+33d | 16 | n_lim*d* | count−1 in words. Output: `⌈N_t/4⌉−1` with `N_t = min(64·G, N − 64·G·t)` (N=10 → 2, N=100 tile 1 → 8); relu: `⌈B/4⌉−1`; ring send: `⌈K/4⌉−1`; broadcast row: 3 (16 B) | verified | H |
| 276–281 | 6 | narrow_lvl_mask | bit d = (narrow stride_d ≠ 0); bit2 seen for 3-level conv3x3/pool outputs | all+ext | H (rule) |
| 282–334 | 53 | rsv282 | 0 | | – |
| 335–348 | 14 | wide_addr | destination, 64-B units. `0x2078` output FIFO (feeds ringProducer/outfeed); `0x1f70` FC ring-send FIFO; broadcast row: allocator (0x207c relu, 0x2074/0x205c/... conv) | verified | H |
| 349–352 | 4 | rsv349 | 0 | | – |
| 353+33d (d=0..3) | 17 | w_inc*d* | wide increment. Single row → `1/0`; multi-row FIFO → `w_inc0=0, w_lim0 = rows−1` (relu n≥8192 output, FC ring send K>256); pool relay `1/3` | verified | H |
| 370+33d | 16 | w_lim*d* | count−1 in 256-B rows: `⌈bytes/256⌉−1` | verified | H |
| 485–488 | 4 | wide_lvl_mask | bit1 = (wide stride₁≠0) (never set); bit0 set only for the maxpool relay (linear 4-row write) | all+ext | M |
| 489 | 1 | wide_circ | = (wide_rows>0 and w_inc0==1) | all+ext | M |
| 490 | 1 | rsv490 | 0 | | – |
| 491–501 | 11 | wide_rows | 1 for FIFO destinations (output, ring send); 0 for broadcast row and relay | all+ext | M |
| 502–542 | 41 | rsv502 | 0 | | – |
| 543 | 1 | sync_en0 | always 1 | all | H (const) |
| 544 | 1 | sync_f1 | 1 for single-row outputs/ring sends, broadcast row and relay; 0 for multi-row FIFO writes | all+ext | L |
| 545 | 1 | sync_f2 | 0 in 0x13 (used by 0x14 for mode 2) | all+ext | – |
| 546 | 1 | rsv546 | 0 | | – |
| 547–551 | 5 | sync_id | 16 = RING_WRITE (outputs and ring sends); 5 = MESH_SOUTH_IN (broadcast row after the 4 meshBus instructions); 0 = none (relay) | all+ext | M |
| 552–553 | 2 | sync_wait_lvl | 1 single row/broadcast; 0 multi-row FIFO (per row) and relay | all+ext | L |
| 554 | 1 | rsv554 | 0 | | – |
| 555 | 1 | sync_en1 | always 1 | all | H (const) |
| 556–574 | 19 | sync_val | 0xffff for outputs and ring sends, 0 for broadcast/relay | all+ext | L |
| 575–586 | 12 | rsv575 | 0 | | – |
| 587 | 1 | sync_en2 | always 1 | all | H (const) |
| 588 | 1 | sync_f45 | 1 for sync_id 16, else 0 | all+ext | L |
| 589–630 | 42 | rsv589 | 0. Where 0x14 has its decrement fields; never used by 0x13 | all+ext | – |
| 631–798 | 168 | rsv631 | 0 | | – |
| 799 | 1 | tail_f799 | 1 only for the maxpool relay | ext | L |
| 801–802 | 2 | tail_lvl | = number of active narrow loop levels (1 FC/relu, 2 conv1x1 M≥8, 3 conv3x3/pool outputs); relay: 2 of 3. Guess: completion is signalled every time levels 0..tail_lvl−1 finish | all+ext (1259/1260) | M |
| 803–895 | 93 | rsv803 | 0 | | – |

`seq`: tile-unit instructions and sync instructions (0x1a, at bit 28) share one counter.
- The units are 0x01, 0x10, 0x11, 0x12, 0x13, 0x14, 0x16 and 0x17.
- Counted in program order, these instructions number 0, 1, 2, … in all 335 programs.
- The hardware use is unknown (tracing or sync references?). For codegen, keep the numbering consistent.

## 3. Constant bit patterns (corpus)

These are the bits that are 1 in every instance, as a mask per word (value = mask).

| word | 0x14 always-1 mask | 0x13 always-1 mask |
|---|---|---|
| w0 | `00000000000000000000000000000500` | `000000000000400000000000000004c0` |
| w1 | `0` | `0000000780000003c0000001e0000000` |
| w2 | `00003800000000000000200008000000` | `0000000000000000000000000010000f` |
| w3 | `000000020001c0000000e00000007000` | `0` |
| w4 | `00008008000000000000000000000000` | `00000000000008000000080080000000` |
| w5 | `00000000000000000000000000008000` | `0` |
| w6 | `00000040000000000000000000000000` | `0` |

- **Structural constants** (the encoder sets them by default):
  - opcode;
  - `n_inc0` bit 0 (0x14 bit 283, 0x13 bit 78);
  - `narrow_lvl_mask` bit 0 (481 / 276);
  - `sync_en0/1/2` (611, 623, 655 / 543, 555, 587);
  - 0x14 `tail_en870`.
- **Incidental:**
  - sign bits of the always-negative increments `n_inc2..5` (0x14 bits 363–365, 396–398, 429–431, 462–464; 0x13 bits 157–160, 190–193, 223–226, 256–259);
  - 0x14 bit 301 (`n_lim0` bit 1).
- **Never set** in the corpus: 0x14 has 557 bits never set; 0x13 has 704. These are the `rsv*` fields (always 0), unused TTU levels, and the high bits of fields whose observed values are small, such as `narrow_addr`, `size64` and `wide_rows`.

## 4. Roles in programs and formulas

All formulas below are checked field by field:
- every corpus FC execution-program instance: 2248;
- every relu instance, corpus plus n-sweep: 657 wideToNarrow + 657 narrowToWide;
- the FC sweeps: 1311 instances.

The only mismatches are tile-geometry guesses for N=2560. They are not encoding errors: for N>1024 the compiler uses `G=⌈N/1024⌉` 64-output groups per tile on `T=⌈⌈N/64⌉/G⌉` tiles, and splits the bias load by group count.

The input FIFO is shared by the 4 tiles of a ring-consumer group. The ring consumer (0x11, tiles 0x000f/0x00f0/…) writes the group's data into its tiles' wide memory at the same address. Each wideToNarrow then picks its tile's slice out of that.

**wideToNarrow, input distribution** (relu, FC chunks, ADD inputs; mode 1). One instance per tile/chunk c, issued before the ring consumers:
- `tile_mask=1<<c`, `wide_addr = 0x2080 − 8·⌈S/256⌉ + 4·⌊j·P/256⌋`.
- Wide TTU: `inc0=1, lim0=⌈(o+B)/256⌉−1`, `inc1..3=−lim0`.
- `narrow_addr = X + c·P/4`; narrow TTU `1/(⌈B/4⌉−1)`, rest `−lim0/0`.
- `narrow_lvl_mask=1`, `wide_lvl_mask=1`, `wide_circ=0`, `wide_rows=0`, `size64=max(4,⌈B/64⌉)`.
- If `o>0`: `skip_en0=skip_en1=1`, `skip_m1=o/4−1`, `skip_base=X`, `skip_end=X+o/4−1`.
- Sync: `sync_id=14, wait_lvl=2, sync_val=0x8000, dec_lvl=2, sync_dec=−1, dec_mode=3`.
- Notes:
  - The FIFO region is `2·⌈S/256⌉` rows directly below `0x2080` (64-B units). That is 0x2080·64 B = 520 KiB, above the nominal 512 KiB.
  - Relu n=1024·k: `wide_addr = 0x2080−32k (+4·row)`. In 256-B rows that is `0x820−8k`. The 9-bit slice at bit 65 (`wide_addr>>5`) is the earlier "0x104−k".

**wideToNarrow, FC ring receive** (mode 0). FC with T>1: tile 0 sends the gathered x through the ring to tiles 1..T−1.
- `tile_mask = ((1<<T)−1)&~1`, `wide_addr=0x1f70`, `wide_rows=1`, `wide_lvl_mask=0`, `size64=0`, `narrow_addr=X`, narrow `1/(⌈K/4⌉−1)`.
- K≤256: wide `1/0`, `wide_circ=1`, `wait_lvl=1`.
- K>256: wide `0/(⌈K/256⌉−1)` (FIFO of one row, inc 0), `wide_circ=0`, `wait_lvl=0`.
- `sync_id=14, sync_val=0x8000`, no decrement.

**wideToNarrow, bias load** (mode 2, FC execution programs). One template for all tiles:
- `tile_mask = (1<<T)−1`, `wide_addr` = parameter base (relocated; 0 = start of the tile's parameter block).
- Wide `1/0`, or `1/0 1/(G−1) −(G−1)/0 −(G−1)/0` for G>1 (`wide_lvl_mask` 1/3).
- `narrow_addr=0x40`, narrow `1/31 33/0 −31/(G−1) −31/0…` (strides 1, 64, 0), `narrow_lvl_mask=3`, `size64=2`.
- `sync_f2=tail_f867=1`, `sync_id=0, wait_lvl=2, sync_val=0xffff, sync_f45=1`, no decrement.

**narrowToWide, output** (one per tile, after the ops; feeds outfeed via ringProducer):
- `tile_mask=1<<t`, `narrow_addr = Y + t·16·G` (relu: `t·P/4`), narrow `1/(⌈N_t/4⌉−1)`.
- `wide_addr=0x2078`, `wide_rows=1`.
- Single row (≤256 B): wide `1/0`, `wide_circ=1`, `sync_f1=1`, `wait_lvl=1`.
- More rows: wide `0/(rows−1)`, `wide_circ=0`, `sync_f1=0`, `wait_lvl=0`.
- `sync_id=16, sync_val=0xffff, sync_f45=1, tail_lvl=1`.

**narrowToWide, FC ring send** (tile 0, mask 1). As output, but `narrow_addr=X`, count `⌈K/4⌉`, `wide_addr=0x1f70`.

**narrowToWide, broadcast row** (relu/conv prologue, after 4 meshBus instructions):
- `tile_mask=0xffff`, 16 B (`n_lim0=3`) to `wide_addr` near the FIFO top.
- `sync_id=5` (MESH_SOUTH_IN), `sync_val=0`, `wide_rows=0`.
- Purpose unknown.

**Conv/pool streaming** (only for reference). wideToNarrow with `wide_circ=1`, `wide_rows`=FIFO depth.
- Narrow level 1 = refills with stride 0.
- `wait_lvl=dec_lvl=1`, `tail_f871=1`.
- head_* fields for tiles whose slice wraps.
- Conv parameters stream through ring consumer 1 (0x12) into a mode-2 load with `sync_id=15`.

**Narrow allocation** (compiler choice, not ISA). The FC input x and output y bases are in bytes. Across 198 FC shapes there are two patterns.
- **"y first":** y at 0, x at max(768, Np).
  - Every K≤132.
  - Large N with moderate K: 768/192–256, 1024/≤512, 1280/512, 1536/288, 2048/1024, 2560/128, 3072/256, 4096/1024.
- **"x first":** x at 0, y at Kp. Every other shape with K≥192.
- **Relu:** each tile stores its slice at its offset in the full tensor (`t·P`); n≤64 puts it at 768 B.

Our own codegen can choose any non-overlapping placement; these are only the compiler's choices.

**Fields from the earlier relu diffs**, explained for n=1024·k:
- `narrow n_lim0 = 16k−1` and `n_inc1..5 = −(16k−1)`. Bits 4..9 of these give the "6-bit k−1 / −k" fields.
- `w_inc1..3 = −(⌈k/4⌉−1)` for tile 0.
- `size64 = max(4,k)`.
- `wide_addr = 0x2080−32k`, and `wide_addr>>5` (bits 65..73) is `0x104−k`.

## 5. Round trip

`python coral/isa/wide_narrow.py` checks two things.
- **Corpus:** 0x14 round trip 1313/1313, 0x13 1218/1218. All `rsv*` fields are 0. `ttu()`/`ttu_strides()` and the `narrow_lvl_mask` rule hold on every instance.
- **Extra compiles:** 1392 + 1210 more instances, also exact with zero `rsv` bits.
- **Recipes:** the §4 recipes (`w2n_input`, `w2n_ring_recv`, `w2n_bias`, `n2w_output`) rebuild instructions from the formulas, given only `seq` and the allocator's narrow bases X/Y.
  - Corpus: 2352/2352 FC (N≤1024) and relu instructions match bit-exactly.
  - Corpus plus sweeps: 4651/4651.
  - `w2n_bias(groups=G)` also matches the 3 corpus N>1024 bias loads (N = 1536/2048/4096).
  - Not covered: the relu broadcast row.

## 6. Open questions

1. **Sync semantics.** What `sync_val` encodes: 0x8000 for FIFO reads, 0xffff for cached parameters, 0x10000·{1,3,4} for parameter streams. Also open:
   - whether `*_lvl` really are loop levels;
   - what `sync_dec_mode=3`, `sync_f1` and `sync_f45` mean;
   - which counter is signalled on completion. Nothing visible encodes it; presumably it is fixed per unit, WIDE_TO_NARROW / WIDE_TO_SCALING / NARROW_TO_WIDE.

   Needs hardware experiments (single-step plus `SyncCounter_*` reads).
2. **Mode 2 destination and size.** It always writes 32 steps to address 0x40 with `size64=2` (128 B). But the bias is 256 B per 64 outputs, and bit-exact hardware runs show all 256 B are used.
   - So the step is probably 8 B, or the units differ, in mode 2.
   - Address 0x40 overlaps the input vector for K≥256, so the destination is probably the separate bias/"scaling" store.
3. **Wide address space.** Addresses go up to 0x2080 (64-B units, 520 KiB). Is it really 520 KiB, or does it wrap? What are rsv74/rsv349, which might be address MSBs?
4. **mode 0 vs 1** beyond "skip fields allowed". The exact meaning of `wide_lvl_mask` bit 0. What `wide_circ` and `wide_rows` do in hardware: circular buffer? FIFO credits?
5. **Head fields.** `head_*` (conv streaming) only fit a "words before/after FIFO wrap" rule, with one exception (conv1x1 M=2 K=288). Their widths are guesses.
6. **Widths** beyond the observed ranges: `seq` (≥7 bits used), `size64` (≥10), `head_words_m1` (≥8), `head_tail16` (≥4), `wide_rows` (≥8).
7. **skip_base / skip_end.** Why both exist: `skip_end = skip_base + skip_m1` is redundant in every instance.
