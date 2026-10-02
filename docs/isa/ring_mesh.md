# Ring and mesh DMAs: 0x10 ringProducer, 0x11 ringConsumer, 0x16 / 0x17 mesh (plus 0x12, 0x15, 0x18)

Codec and helpers: `coral/isa/ring_mesh.py` (`encode_*/decode_*`, `describe`, `split`, `translate_tiles`).
Run `python coral/isa/ring_mesh.py [model_edgetpu.tflite ...]` for the round trip and the FC formula checks.

## 0. Summary

- **Common format.** All of these are tile DMAs built from the same address generator ("TTU descriptor"):
  an 18-bit address, 4 loop dimensions of (17-bit signed increment, 16-bit count−1), and a 33-bit sync/buffer block.
  - Ring instructions (4 words) have one descriptor plus a ring tail.
  - Mesh instructions (6 words) have two descriptors (outbound at +0, inbound at +205 bits) plus a mesh tail.
- **Ring routing is a destination bitmap, not a hop count.**
  - `ringProducer.dest` (bits 467..483) is a 17-bit multicast bitmap: bit t = tile t, bit 16 = the scalar core (outfeed).
  - `ringProducer.to_c1` (bit 466) sends the packets to ringBusConsumer1 (opcode 0x12) instead of ringBusConsumer0 (0x11).
  - The infeed (0x26, scalar side) has the same kind of 16-bit destination bitmap at bits 409..424.
  - Consumers carry no source field. No hop count or ring-position field exists in any of these opcodes.
- **Mesh direction is the opcode.** 0x15 = south, 0x16 = west, 0x17 = north, 0x18 = east (the direction the data moves).
  - The mesh is row-major: tile t sits at (row t//4, column t%4). Moves only go to the 4-neighbour.
  - Each instruction can carry an outbound half (send to the neighbour in that direction) and an inbound half
    (receive from the opposite neighbour). Tiles that relay data run both halves through a 4-slot buffer.
- **0x15 and 0x18 are 6 words long, not 36.** `coral/isa.py` `LEN` has 36 for both, learned statistically. That is wrong.
  - With 6, MobileNet's execution program splits into 590 tile instructions with perfectly sequential `seq` numbers.
  - With 36, 406 of 445 instructions are misaligned.
  - The corpus is unaffected because it has no 3x3 convolutions. `ring_mesh.split()` uses the corrected lengths.
- **Bits 46..59 hold `seq`**, a 14-bit dispatch sequence number: 0, 1, 2, … over all tile instructions and syncs of a
  program (syncs keep it at bits 28..41). Verified on 1122 programs, up to 589 in MobileNet. It is not a routing field.
- **In FC programs, tile topology appears in exactly four places:**
  1. tile masks (all tile instructions, including 0x1a syncs);
  2. `ringProducer.dest`;
  3. the infeed destination bitmap;
  4. the mesh geometry (which tiles are neighbours).

  Everything else that differs between tiles is logical and stays valid when the whole program is moved:
  narrow addresses (64·chunk), the output ordinal in `r0_val`, and the wideToNarrow lane selection.
- **Why the rotated FC hung.** An early rotation test skipped all opcodes ≥ 0x20, so the
  infeed destination bitmaps still pointed at tiles 0..T−1. The rotated ring consumers then waited forever.
  This happens already in the caching program. For T > 1 the forward producer's `dest` was also not rotated.

## 1. Common header and descriptor (all seven opcodes)

| bits | w | name | encoding / formula | evidence, confidence |
|---|---|---|---|---|
| 0..5 | 6 | gate, pred_reg, pred_pol, rsv5 | predication (known ISA convention) | always 0 in all 1122 programs |
| 6..11 | 6 | opcode | 0x10 0x11 0x12 0x15 0x16 0x17 0x18 | — |
| 12..27 | 16 | tile_mask | bit t = tile t executes the instruction | high |
| 28..45 | 18 | rsv28 | 0 | always 0 |
| 46..59 | 14 | seq | dispatch sequence number, `seq[k] = k` over tile instructions + syncs | high: 1122/1122 programs, ≥10 bits used |
| 60..77 | 18 | addr (`o_addr`) | start address. Mesh: narrow-memory byte address. Ring caching consumer: wide-memory 64-byte row (hardware-verified relocation). Ring execution staging (8312, 8048, 8320−8c): unit unknown | format high; units see §7 |
| 78+33j .. 94+33j | 17 | inc*j* (j = 0..3) | signed increment applied when dimension j advances, in TTU elements. Outer dims compensate for the inner ones: inc_j = stride_j − Σ_{i<j} cnt_i·stride_i; unused outer dims rewind (inc = −total, cnt = 0) | high (caching FC n=2: inc1 = 256−255 = 1, inc2 = −511) |
| 95+33j .. 110+33j | 16 | cnt*j* | iterations − 1, innermost first | high |
| 210..213 | 4 | sdims | 1 for plain moves; 3/7 for 2-/3-dim mesh halo moves; 0/1 on consumers (see roles); 1 or 3 on caching consumers (n=1 / n>1) | format high, meaning guess (loop-level mask for sync) |
| 214..215 | 2 | cbuf | 1 = circular staging buffer (ring staging, mesh relays) | medium |
| 216..226 | 11 | slots | depth of that buffer: ring = packets in the stream (c), mesh relay = 4, producer = 1 | medium |
| 227..232 | 6 | rsv_s | 0 | always 0 |
| 233..242 | 10 | grp | ring group consumer: packets per tile group − 1 | high for the formula, meaning medium |

The mesh opcodes repeat the descriptor for the inbound half at +205 bits, with prefix `i_`:
- `i_addr` 265..282, dims at 283+33j, sync block 415..447.

The outbound half uses prefix `o_` (bits as above).

**TTU element size** (verified):
- Mesh: 4 bytes. `cnt0+1 = bytes/4`, including partial chunks (K=96: 32 bytes gives (1,7)).
- Caching ring consumer: one 256-byte [64][4] weight block. K/4 blocks per 64 outputs.
- Execution ring staging: one packet slot.

## 2. 0x10 ringProducer (4 words)

Tail, bits 243..511:

| bits | w | name | encoding / formula | evidence, confidence |
|---|---|---|---|---|
| 243..334 | 92 | rsv243 | 0 | always 0 |
| 335..336 | 2 | mode | 3 output to scalar core, 1 activation forward tile→tiles, 0 parameter broadcast | high (empirical) |
| 337..342 | 6 | pcfg | 20 output, 0 forward, 4 / 12 parameter broadcast (MobileNet) | constant per role, meaning unknown |
| 343..349 | 7 | r0_id | sync record: id = op<<5 \| flag. Output: 81 = (op2, RING_PRODUCER_A); forward: 45 = (op1, NARROW_TO_WIDE); broadcast: 50 = (op1, RING_PRODUCER_B) | medium (flag names fit every role) |
| 350..365 | 16s | r0_val | output producers: **output ordinal k** (0,1,2,… in program order) | high: 4372/4372. In conv M=2 tiles 0 and 2 get k=0,1, so it is logical, not the tile index |
| 366 | 1 | r0_en_a | 1 when the record is used | high |
| 367..381 | 15 | r0_x | 0 | always 0 |
| 382 | 1 | r0_en_b | 1 when used | high |
| 383 | 1 | r0_y | 0 | always 0 |
| 384..424 | 41 | r1_* | second record, same layout. Output producers: (45 = NARROW_TO_WIDE/op1, val 1) | medium |
| 425..465 | 41 | r2_* | third record slot, never used | — |
| 466 | 1 | to_c1 | 1: receiver is ringBusConsumer1 (0x12) | high: 52/52 sends match a 0x12 mask (plus 8 3x3-conv 0xffff sends) |
| 467..483 | 17 | dest | multicast bitmap, bit t = tile t, bit 16 = scalar core | high: 474/482 tile sends equal a consumer mask in the same program, the rest are 0xffff 3x3-conv broadcasts |
| 484..511 | 28 | rsv484 | 0 | always 0 |

**Roles** (FC execution programs; T = tiles used, c = ⌈input bytes/256⌉):
- **Output producer.** One per tile, on tile k, in tile order. All fields verified on 475 FC programs:
  - addr = 8312, (inc0,cnt0) = (1,0), sync block 1/1/1, mode = 3, pcfg = 20;
  - r0 = (81, k), r1 = (45, 1), dest = 1<<16;
  - conv uses (inc1,cnt1) = (0, R−1) for R packets per tile.
  - Hypothesis: r0 makes producer k wait until RING_PRODUCER_A ≥ k, which serialises the outputs at the outfeed.
- **Activation forward.** Only on tile 0, only when T > 1. Fields (verified):
  - dest = tiles 1..T−1, addr = 8048, (1,0),(0,c−1);
  - mode = 1, r0 = (45 NARROW_TO_WIDE, 1).
  - The stride-0 outer loop resends one single-slot FIFO c times.
- **Parameter broadcast** (conv1x1, MobileNet): tile 0 reads the cached parameters (addr 0, cnt0 = 256-byte blocks − 1) and sends with
  `to_c1 = 1` to tiles 0..3, including itself.

## 3. 0x11 ringConsumer (= ringBusConsumer0), same layout for 0x12 (ringBusConsumer1)

Tail, bits 243..511:

| bits | w | name | encoding / formula | evidence, confidence |
|---|---|---|---|---|
| 243..248 | 6 | rsv243 | 0 | always 0 |
| 249..266 | 18 | gstride | multi-packet groups only: 4·(G−1)·p + 1 (G groups, p packets per group); else 0 | formula high on all cases (FC K=2048/3072/4096, relu 1536..65536), meaning unknown |
| 267 | 1 | aux_en0 | 1 in caching and parameter consumers | high |
| 268..285 | 18 | aux_addr0 | caching FC: 0 (+ relocation, the documented "w2 bit 12" field) | high (relocation); meaning guess: first bias row |
| 286..299 | 14 | rsv286 | 0 | always 0 |
| 300..317 | 18 | aux_addr1 | caching FC: 4·(n−1) (+ relocation, "w2 bit 44"); n = 64-output groups per tile | high; guess: last bias row (256 B = 4 rows per bias) |
| 318 | 1 | aux_en1 | 1 with aux_en0 | high |
| 319..334 | 16 | rsv319 | 0 | always 0 |
| 335..336 | 2 | mode | 3 infeed-fed / parameter; 1 forward consumer with c > 1 | empirical |
| 337..338 | 2 | rsv337 | 0 | always 0 |
| 339..345 | 7 | s_id | sync record id: 43 = (op1, WIDE_TO_NARROW), 11 = (op0, WIDE_TO_NARROW), 33 = (op1, PARAMETERS) on 0x12 | medium |
| 346..361 | 16s | s_val | execution input consumers: −64·c (forward consumer: −64) | high |
| 362 | 1 | s_en_a | 1 on parameter consumers (0x12, MobileNet 0x11) | — |
| 363..367 | 5 | rsv363 | 0 | always 0 |
| 368..377 | 10 | s_cnt | c (packets in the stream; forward consumer: 1) | high |
| 378 | 1 | s_en_b | 1 in all execution consumers | high |
| 379 | 1 | s_y | 1 in all execution consumers | high |
| 380..511 | 132 | rsv380 | 0 | always 0 |

**Caching program (FC).** All formulas verified on 3652 consumers in 475 programs.
- One consumer per tile t < T, on tile t, in tile order. With n = max(1, ⌈N/1024⌉), T = ⌈N/(64n)⌉ and Kq = ⌈K/4⌉:
  - addr = 2 + 4n, the documented relocation field "w0 bit 60";
  - dims: (1, Kq−1), then (1, n−1) if n > 1 else (−(Kq−1), 0), then (−(n·Kq−1), 0) twice;
  - sdims = 1 (n = 1) or 3; aux = (0, 4(n−1)); everything else 0.
- So 256-byte blocks go to rows 2+4n, …, and n bias blocks are placed through the aux pair.
- conv1x1 caches all parameters on tile 0 only: addr 2, cnt0 = blocks−1, no aux.
- The data arrives from one infeed per tile, with destination bitmap 1<<t.

**Execution program (FC), input consumers.** Verified on 1155 consumers:
- **Placement.** One consumer per row group: tiles 4g..4g+3, clipped to the u = ⌈K/64⌉ chunks used (masks 0x1, 0x3, 0x7, 0xf).
  - Each is fed by exactly one infeed whose destination bitmap equals the consumer mask (all 5075 infeeds in 1120 programs).
  - The input bytes are cut into 256-byte packets. Group g receives p consecutive packets (multicast);
    the infeed source offsets advance by p packets per group.
  - Inferred: the wideToNarrow on tile 4g+i then keeps 64-byte lane i. It writes narrow address xb+64·t, and its
    lane-select fields depend only on i = t%4 (429/475 programs; the exceptions are partial chunks and second roles).
- **Fields:**
  - addr = 8320 − 8c, cbuf = 1, slots = c, s = (43, −64c, cnt c);
  - (inc0,cnt0) = (1,0) if p = 1 else (1, c−1);
  - grp = p−1; gstride as in the table; sdims = 0 if c = 1 else 1.
- **Forward consumer.** On tiles 1..T−1, only when T > 1, receives x from tile 0's forward producer:
  - c = 1: addr 8048, (1,0), mode 3, cbuf 1, s_id 43;
  - c > 1: addr 8048, (0, c−1), mode 1, cbuf 0, s_id 11;
  - always s = −64 / cnt 1.

## 4. Mesh DMAs 0x15 / 0x16 / 0x17 / 0x18 (6 words)

| bits | w | name | encoding / formula | evidence, confidence |
|---|---|---|---|---|
| 0..242 | | header + outbound descriptor `o_*` | §1; outbound = read local narrow memory, send to the neighbour in the opcode's direction | high |
| 243..264 | 22 | rsv243 | 0 | always 0 |
| 265..447 | | inbound descriptor `i_*` | §1 at +205: write data from the opposite neighbour into local narrow memory | high |
| 448..472 | 25 | rsv448 | 0 | always 0 |
| 473..474 | 2 | out_mode | 3 on FC 0x16 outbound halves and on all FC relays (0x16 and 0x17); 0 on 0x17 plain senders and on halo moves | high (empirical), meaning unknown |
| 475..476 | 2 | rsv475 | 0 | |
| 477..479 | 3 | in_mode | 3 FC 0x16 inbound halves and relays; 5 / 7 halo moves (3x3 conv); 0 FC 0x17 receivers | empirical |
| 480+41k .. 520+41k, k=0..4 | 41 each | s*k*_id (6), s*k*_val (18s), s*k*_en_a, s*k*_x (15), s*k*_en_b | sync records, id = flag<<1 \| op | medium (see below) |
| 685..726 | 42 | rsv685 | 0 | always 0 |
| 727 | 1 | fill_en | the inbound half generates a constant instead of receiving | high |
| 728..759 | 32 | fill | fill pattern. 0x80808080 (= input zero point 128) for the edge rows/columns of SAME 3x3 convs; 0x01 << 8j for the 4 "init" moves at program start (relu, conv) | high |
| 760..767 | 8 | rsv760 | 0 | always 0 |

**Direction evidence.** SAME 3x3 conv, 32x32, tiles as a 4x4 grid. Each opcode is issued on rows (000f / 0ff0 / f000)
or on columns (1111 / 6666 / 8888):

| opcode | edge that only receives | edge that only sends, or receives with fill | ⇒ data moves |
|---|---|---|---|
| 0x17 | row 0 (in only) | row 3 (out + fill) | north (t → t−4) |
| 0x15 | row 3 (in only) | row 0 (out + fill) | south (t → t+4) |
| 0x16 | column 0 (in only) | column 3 (out + fill) | west (t → t−1) |
| 0x18 | column 3 (in only) | column 0 (out + fill) | east (t → t+1) |

The sync records agree with the hardware's own counter names:
- FC west relays use (MESH_EAST_IN, MESH_WEST_OUT); 361 north relays use (MESH_SOUTH_IN, MESH_NORTH_OUT).
- Horizontal halo moves wait on MESH_NORTH_IN / MESH_SOUTH_IN, because corners need the vertical exchange first.
- So the geometric reading (row-major, north = towards row 0) matches the CSR naming.

**FC input gather** (K > 64). Verified on 418 programs (2825 0x16 senders, 680 0x17 senders):
- **Chunk layout.** Tile t holds chunks t·cpt … t·cpt+cpt−1 at narrow address xb + 64·cpt·t.
  - cpt = chunks per tile = ⌈K/1024⌉.
  - xb = the narrow address of x: 0 in 286 of 418 programs, 768 / 1024 / … in the others.
- **Westward within rows (0x16).** For each position i = 1, 2, 3, tile 4g+i sends its chunks towards 4g:
  - Sender: o_addr = xb + 64·cpt·t, (inc0, cnt0) = (1, bytes/4 − 1), then rewinds. Partial last chunks shrink cnt0.
  - Receiver: tile 4g, i_addr = the same address.
  - Relays: the tiles in between run both halves through a 16-byte, 4-slot staging buffer at o_addr = i_addr:
    (1,3),(−3, words/4−1), cbuf 1, slots 4, two sync records.
- **Northward along column 0 (0x17).** Tile 4r sends the chunks its row has gathered to tile 0, relayed by tiles 4..4(r−1).
  - Sender: o_addr = xb + 64·cpt·4r, cnt0+1 = (bytes of that row)/4.
  - Tile 0 receives at the same address.
- **Syncs.** The 0x1a syncs between the steps are masked to "all tiles except the relays" (0xdddd, 0x9999, 0xffef, 0xfeef).
- **After the gather.** Tile 0 holds x. The forward ringProducer multicasts it to tiles 1..T−1.

**Other uses:**
- relu / conv1x1 programs start with four 0x17 moves on all tiles with only the inbound half in fill mode.
  They write constants 0x01 << 8j at 4 consecutive narrow bytes.
- 3x3 convs and MobileNet use all four directions for halo exchange, with fill for zero padding.

## 5. Constant bit patterns

Always-1 bits: 0x10 {10, 78, 210, 366, 382}; 0x11 {6, 10}; 0x16 {7, 8, 10}; 0x17 {6, 7, 8, 10}. That is the opcode plus,
for 0x10, inc0 = 1, sdims = 1 and the two record enables. Everything named `rsv*` above is always 0.

Per word, the bits that are constant over all 1122 programs (corpus + 785 extra compiles + MobileNet), as mask / value:

| op | word | constant mask | value |
|---|---|---|---|
| 0x10 | w0 | 00007c007ffffc003f003ffff0000fff | 00000000000040000000000000000400 |
| 0x10 | w1 | fffffffffeb7fffc0001fffe0000ff00 | 00000000000400000000000000000000 |
| 0x10 | w2 | ffffc00000467fffffffffffffffffff | 40004000000000000000000000000000 |
| 0x10 | w3 | fffffff00003ffffffffff7fff600040 | 0 |
| 0x11 | w0 | 00007c007fffbc001f003ffff0000fff | 00000000000000000000000000000440 |
| 0x11 | w1 | 05ff81fe00b3fffc0001fe000000ffc0 | 0 |
| 0x11 | w2 | f200f80002a67fffbdfc3ffffdfce7f8 | 0 |
| 0x11 | w3 | ffffffffffffffffffffffffffffffff | 0 |
| 0x16 | w0 | 00007fe07ff1b8003f003ffff0000fff | 00000000000000000000000000000580 |
| 0x16 | w1 | fffffffffb83ffc00001ffc00000ffe0 | 0 |
| 0x16 | w2 | 3ff800001ffc00000ffc0ffe370007ff | 0 |
| 0x16 | w3 | fef8003119ffffffffffff707ff80000 | 0 |
| 0x16 | w4 | fff7c003f3fffbf000d1fffdf50040ff | 0 |
| 0x16 | w5 | ff7f7f7f7f7fffffffffffffffffffe7 | 0 |
| 0x17 | w0 | 00007f807fffb8003f003ffff0000fff | 000000000000000000000000000005c0 |
| 0x17 | w1 | fffffffffb83ffc00001ffe00000ffc0 | 0 |
| 0x17 | w2 | 3ffc00001ff800000ff00ffff00007ff | 0 |
| 0x17 | w3 | fef8003519ffffffffffff707ff80000 | 0 |
| 0x17 | w4 | fffffffff3fffbeb60f9fffc000060ff | 0 |
| 0x17 | w5 | ff7e7e7e7e7fffffffffffffffffffff | 0 |

These masks are what the observed data happens to keep constant. Fields such as unused dims or the records are constant
only because the compiler never varies them. Rely on the `rsv*` fields of the tables for real reserved bits.

## 6. Placing an FC on tiles [s, s+T) instead of [0, T)

**What the compiler emits for FC (N, K).**
1. **Caching.** One consumer per tile 0..T−1, each fed by an infeed with bitmap 1<<t.
2. **Scatter.** The input is multicast in 256-byte packets to row groups of 4 tiles covering chunks 0..u−1, u = ⌈K/64⌉.
   For K ≥ 769 that is all 16 tiles, independent of N.
3. **Gather.** Mesh moves west within rows and then north along column 0 bring x to tile 0.
4. **Broadcast.** Tile 0's ringProducer sends x to tiles 1..T−1.
5. **Compute.**
6. **Output.** Ring producers k = 0..T−1 send to the scalar core, with r0_val = k.

**What must change to move it to tiles t+s:**
- tile masks of every tile instruction, including 0x1a syncs (rotate partial masks, keep 0xffff);
- `ringProducer.dest` bits 0..15 (keep bit 16 = scalar core);
- the infeed (0x26) destination bitmap at bits 409..424.

Nothing else is physical:
- narrow addresses are 64·(logical chunk/output index);
- the output ordinal is logical;
- wideToNarrow lane selection depends on t%4;
- parameter addresses are the same on every tile;
- the scalar-side bits 12..27 of 0x24/0x25/0x26/0x3e are offsets or flags, not tile masks.

**Verdict.**
- **Encodable as a pure translation**, provided no used tile (compute tiles plus scatter/gather staging tiles) passes tile 15, and either:
  - s is a multiple of 4 (whole mesh rows); or
  - the program has no west/east chain that would cross a row end (K ≤ 64, or the used tiles of each 4-group stay within one row).
- **K ≤ 64.** Any s with s+T ≤ 16 works. The program has no mesh moves; N=64,K=64 is just the tile masks plus the infeed bitmaps.
- **K ≥ 769.** The scatter/gather occupies all 16 tiles, so no translation is possible.
- **Other placements, and wrap-around.** They need a re-planned gather. Mesh moves exist in all 4 directions, and masks and
  destination bitmaps are free-form, so this is encodable. But it means generating new 0x15..0x18 instructions and syncs,
  not rotating bits.
- **Ring order.** Avoid wrap-around even for ring-only programs: if the ordinal wait counts packets passing on the ring,
  the producers' upstream order must be preserved (unverified).
- **Simpler alternative for co-resident models.** Skip the scatter/gather and multicast the whole input directly to tile s
  with the infeed bitmap. That needs a re-generated consumer and wideToNarrow pair, like the existing forward consumer
  (single-slot FIFO, (0, c−1) loop).

`ring_mesh.translate_tiles(bitstream, s)` implements the translation. It refuses moves that wrap or break a mesh hop.
- Offline it accepts, for example, N=128,K=128 with s ∈ {1, 2, 4, 8, 12} and N=256,K=256 with s ∈ {4, 8, 12}.
- It rejects N=128,K=128 with s=3 and N=64,K=1024 with any s.
- On the device, translated FC programs ran bit-exact (9/9, `bench/RESULTS.md`).

## 7. Open questions

1. **Units of the execution staging addresses** (8312 = 0x2078, 8048, 8320−8c with 8 address units per packet slot).
   - The caching consumer's address is in 64-byte wide rows (hardware relocation test), but 8320 rows exceed 512 KiB.
   - Either wide memory is ≥ 520 KiB (with a staging area above the 496 KiB parameter limit), or execution consumers write
     narrow memory in 8-byte units (8 × 8 B = one 64-byte lane).
   - The wideToNarrow reads these addresses with its source descriptor, which also reads wide parameters. Unresolved.
2. **Sync records.**
   - The id layouts differ: ring `op<<5|flag`, mesh `flag<<1|op`.
   - The flag names fit every role, but the meaning of op, val, en_a/en_b, and the mesh slot pairing are hypotheses.
   - Mesh relays carry e.g. (MESH_EAST_IN, 9) and (MESH_WEST_OUT, 1). The values grow by 16 per extra chunk.
3. **Ordering of output packets.** The output-ordinal mechanism (RING_PRODUCER_A, op 2, val k) and whether it depends on
   ring position. The ring visiting order of tiles is unknown (the patent mentions a serpentine ring).
4. **sdims, gstride, out_mode/in_mode.** Their exact semantics. The formulas are known, the meanings are not.
5. **Bit 466.** Whether it is really "consumer 1" or the low bit of a larger VC field. Bits 463..465 never vary;
   the chip has 8 ring VCs and no other VC field was found.
6. **The aux address pair** in caching consumers (relocated, (0, 4(n−1)), presumably the bias destination range) and in
   0x12 / MobileNet parameter consumers (values near addr, e.g. addr−6 and addr−4).

## 8. Verification

- **Round trip.** `encode_ring_mesh(**decode_ring_mesh(w)) == w` for every instance:
  - corpus: 0x10 1237, 0x11 1418, 0x12 12, 0x16 1393, 0x17 413;
  - 785 extra compiles (FC grid N=64..1024 × K=64..4096, conv1x1, relu, 3x3 convs): 0x10 3625, 0x11 4079, 0x12 56,
    0x15 102, 0x16 5136, 0x17 1828, 0x18 104;
  - MobileNet v1: 0x10 47, 0x11 64, 0x12 14, 0x15 55, 0x16 90, 0x17 108, 0x18 55.
  - Every bit belongs to exactly one field (asserted at import).
- **Formula checks.** The self test re-checks the 150 + 150 corpus FC programs. The counts below are over the corpus plus the
  extra FC grid (475 caching and 475 execution programs):
  - caching consumer fields: 3652/3652;
  - output producers: 3652/3652;
  - forward producer and consumer: 420/420;
  - group consumers: 1155/1155;
  - infeed bitmap = consumer mask: 1120/1120 programs;
  - mesh gather roles and addresses: 418/418 programs.
