# Opcode 0x01: tile `op` (TensorOp), 17 words / 2176 bits

The per-tile compute instruction. It is used for FULLY_CONNECTED, CONV_2D (1x1, 3x3, depthwise), elementwise requantize ops (RELU, RELU6, RELU_N1_TO_1, QUANTIZE, ABS), pooling and binary ADD/SUB/MAX/MIN/MUL. It also does the data "copy/reformat" passes that the compiler inserts in front of convolutions.

Code: `coral/isa/op.py` provides `decode_op(words) -> dict` and `encode_op(**fields) -> list[int]`. It also has TTU helpers and two builders, `fc_tile_fields()` and `requant_tile_fields()`, that generate complete instructions.

Bit numbering follows NOTES.md: bit 0 is the LSB of byte 0 of word 0, and word j covers bits [128j, 128j+128). All offsets below are absolute within the instruction.

## Evidence base
- **Corpus:** all 1084 op instructions in `tools.corpus` (963 fc, 97 relu, 24 conv1x1).
- **Extra compiles:** 652 more instances, all offline.
  - FC sweeps:
    - K = 4..60 (step 4), 64..1024 (step 64), 1536..8192;
    - N = 4..120, and N = 1088..4096, 1100, 2000;
    - fused activation 0..3;
    - zero-point and scale variants;
    - chained FC/RELU models.
  - Elementwise ops:
    - relu over 1-D and 4-D shapes, including n not a multiple of 4;
    - relu6, relu_n1_to_1, quantize, abs, add, sub, max, min, mul.
  - Pooling and convolutions: max/avg pool, depthwise 3x3, conv 3x3 and 1x1, and conv1x1 N sweeps with N = 68..288.
  - Long programs, to get `seq` values ≥ 128.
- **Compiler strings:** the edgetpu_compiler binary names the TensorOp's address generators.
  - "TensorOp main TTU"
  - "TensorOp narrow_memory_read"
  - "TensorOp narrow_memory_write_from_non_linear"
  - "TensorOp wide_memory_read_for_parameters"
  - "TensorOp wide_memory_read_for_sums"

  These names match the five blocks found below, in the same order.

## Structure

```
[0,62)       header: predication, opcode, tile_mask, seq
[62,174)     main loop nest  loop0..loop7          (14-bit count-1 each, innermost first)
[174,557)    TTU in_   narrow-memory read   (input activations)      base + 8 dims x 37 bits + tail
[557,957)    TTU out_  narrow-memory write  (after non-linear unit)  base + 8 dims x 37 bits + tail
[957,1260)   TTU par_  wide-memory read for parameters (weights)     base + 8 dims x 29 bits + tail
[1260,1563)  TTU psum_ wide-memory read for partial sums             base + 8 dims x 29 bits + tail
[1563,2176)  datapath / output stage: flags, sync slots, zero points, reduction mask, channel counts, f32 scale and clamps
```

### How a TTU works
Each TTU is an 8-deep loop nest. Dimension k is a record of three fields, innermost first:
- `inc`: signed;
- `cnt`: the iteration count minus 1;
- `mode`: 2 bits.

The address starts at `base`. After each step, the TTU finds the innermost dimension d whose counter is below `cnt_d`. It increments that counter, zeroes the counters of all inner dimensions, and adds `inc_d` to the address.

Given real strides S, the increments are therefore `inc_d = S_d - sum_{e<d} cnt_e * S_e`, so the outer dimensions carry "rewinds". Unused outer dimensions (cnt = 0) still hold the full rewind value. `op.ttu_increments` and `op.ttu_addresses` implement this.

Checks against the data:
- For all 1736 instances, the `in`, `par` and `psum` TTUs generate exactly `prod(loop_k + 1)` addresses. They advance once per main-loop step.
- The `out` TTU advances once per output word.
- For FC, the simulated sequences are exactly as expected:
  - `in` walks the K/4 input words once per pass;
  - `par` walks consecutive 256-byte weight rows;
  - `psum` stays constant;
  - `out` walks the tile's output words.

### Units
- **Narrow memory (`in`, `out`):**
  - `base` is a byte address. Bits 191-192 and 574-575 are always 0, so addresses are 4-byte aligned. It could also be a word address at bit 193/576; nothing so far tells the two apart.
  - `inc` counts 32-bit words. One main-loop step consumes one 4-byte input word.
- **Wide memory (`par`, `psum`):**
  - `base` is in 64-byte units. `par_base` is the known 13-bit parameter relocation field at w7 bit 61.
  - `inc` counts 256-byte rows. One step consumes one row: 64 outputs x 4 inputs of weights, i.e. 256 MACs per tile.
- **Requantization:** after the reduction, each group of `out_ch+1` channels is requantized in float32 (multiplier, clamps, out_zp) and written as `(out_ch+1)/4` words through `out`.

## Field table
Confidence: **H** means the formula was verified on all relevant instances, or the field is reproduced bit-exactly by a builder. **M** means consistent with all data, but the semantics are inferred. **L** is a guess. "rsv" fields were 0 in every instance seen.

| bits | width | name | encoding / formula | evidence | conf |
|---|---|---|---|---|---|
| 0 | 1 | gate | predication gate (as in other instructions) | always 0 | L |
| 1-3 | 3 | pred_reg | predicate register | always 0 | L |
| 4 | 1 | pred_pol | predicate polarity | always 0 | L |
| 5 | 1 | rsv5 | | always 0 | |
| 6-11 | 6 | opcode | 0x01 | | H |
| 12-27 | 16 | tile_mask | bit t = tile t | FC/relu: `1<<t`; conv uses multi-tile masks, e.g. 0x5, 0xf, 0x5050 | H |
| 28-45 | 18 | rsv28 | | always 0 | |
| 46-61 | 16 | seq | count of earlier tile-dispatched instructions in the bitstream (opcodes 0x01, 0x10-0x14, 0x16, 0x17, 0x1a), counted from 0. Width beyond 9 bits is a guess. | exact on all 1084 corpus ops and on the extra FC/relu/conv1x1 compiles; values up to 437 (bit 54 set); does **not** wrap at 128 (bit 53 is used). In programs with compute opcodes of unverified length (0x00, 0x15, 0x18, 0x19) the count is off by 12-27, presumably because `isa.split` mis-splits those programs | H |
| 62+14k .. 75+14k (k=0..7) | 14 each | loop0..loop7 | main loop nest, iterations-1, innermost first. FC: `loop0 = ceil(K/4)-1` (4-wide input groups), `loop3 = passes-1`. RELU: `loop1 = words-1` | step counts of in/par/psum TTUs equal `prod(loop+1)` on all 1736 instances | H |
| 174-190 | 17 | rsv174 | | always 0 | |
| 191-208 | 18 | in_base | narrow-memory byte address of the first input word | FC: x at 768 (K ≤ 128) or 0; relu: `in0 + per_tile*t` | H |
| 209-210 | 2 | in_hmode | header mode | 0 except mul (1) and binary ops (2) | L |
| 211+37k .. 227+37k | 17 s | in_inc{k} | increment, in 32-bit words | FC: `[1,1,1,-(kw-1),...]` | H |
| 228+37k .. 245+37k | 18 | in_cnt{k} | iterations-1 | FC: `cnt0 = kw-1`, `cnt3 = passes-1` | H |
| 246+37k .. 247+37k | 2 | in_mode{k} | per-dimension mode | FC/relu/conv/copy: dims 0-6 = 0 and dim 7 = 3. mul/binary ops: every dim 1 or 2 (dim 7 = 3). avgpool compute op: dim 7 = 0 | L |
| 507-514 | 8 | in_tflags | | FC/relu 0x01; copy 0x80; conv 3 or 7; binary 0xf | L |
| 515-521 | 7 | rsv515 | | | |
| 522-535 | 14 | in_twait | non-zero only for copy/reformat ops, e.g. 4, 8, 9, 10, 32, 36, ..., 640. Grows with the bytes delivered per tile; looks like a wait count | | L |
| 536-573 | 38 | rsv536 | | | |
| 574-591 | 18 | out_base | narrow-memory byte address of the tile's first output word. FC: `y0 + 64*passes*t`, so bits 580-581 = `t & 3` | e.g. N=100: tile 1 → 300+64 = 364 | H |
| 592-593 | 2 | out_hmode | | always 0 | L |
| 594+37k .. 610+37k | 17 s | out_inc{k} | words. FC: `[1, ceil(N/4)-(w0-1), 1, rew x5]`, where `rew = -((w0-1)+(passes-1)*w0)`. The dim1 value is the output-row stride minus the dim0 rewind | | H |
| 611+37k .. 628+37k | 18 | out_cnt{k} | FC: `cnt0 = w0-1`, where `w0 = ceil(min(64,N_tile)/4)` = output words per pass. `cnt2 = passes-1` | | H |
| 629+37k .. 630+37k | 2 | out_mode{k} | dim 7 = 3, others 0 | | L |
| 890-892 | 3 | out_tflags | FC/relu/conv 1; copy 3; binary/avgpool 7 | | L |
| 893-900 | 8 | rsv893 | | | |
| 901-917 | 17 s | out_last_inc | always 0 | | M |
| 918-935 | 18 | out_last_cnt | partial last output group: `W-1`, where W = valid words in the last group | conv1x1 N=68..124 (W=1..15); FC N=1100/2000 | H |
| 936-937 | 2 | out_last_mode | 2 when the last group is partial, else 0 (3 in the K=4 multi-pass schedule) | | M |
| 938-940 | 3 | rsv938 | | | |
| 941-956 | 16 | out_last_skip | `w0 - W` = words skipped in the partial group (as a field at 938 this would be `8*(w0-W)`) | W = 1..15 sweep | H |
| 957-969 | 13 | par_base | wide-memory address of the weights in 64-byte units = the parameter relocation field. Single-layer FC: `4*passes`, because one 256-byte bias row per 64-output group comes first. Chained models: other values | reloc offset from NOTES; FC formula verified | H |
| 970 | 1 | par_sel | 1 when par does not read resident weights: relu (`par_base` 0x7C), conv/dwconv weight FIFO, avgpool, binary/mul | | L |
| 971-972 | 2 | par_hmode | 1 for single-step requant ops, avgpool, mul and some copy ops; else 0 | | L |
| 973+29k .. 985+29k | 13 s | par_inc{k} | 256-byte rows. FC: `[1, -(kw-1), 1, 1, rew x4]`, where `rew = -((kw-1)+(passes-1)*kw)` | | H |
| 986+29k .. 999+29k | 14 | par_cnt{k} | FC: `cnt0 = kw-1`, `cnt3 = passes-1`. RELU: `cnt0 = words-1` with inc 0, so par is idle | | H |
| 1000+29k .. 1001+29k | 2 | par_mode{k} | dim 7: FC 1, conv 3, relu/copy 0 | | L |
| 1205-1212 | 8 | par_tflags | resident weights (FC) 3; weight FIFO (conv/dwconv) 0xC0; streamed FC weights 0x80 | | L |
| 1213-1215 | 3 | rsv1213 | | | |
| 1216-1229 | 14 | par_fifo | rows in the weight ring buffer, e.g. conv K=64 → 8, K=288 → 12, 3x3 → 9, streamed FC → 32 (= par dim0 rows). 0 for resident weights | | M |
| 1230-1259 | 30 | rsv1230 | | | |
| 1260-1272 | 13 | psum_base | partial-sum address. FC 0. conv1x1 with M ≥ 8 (K chunked over loop3): small offsets 0..56 with psum_sel = 1. K≤4 multi-pass FC: 8048 (top of wide memory, sel = 0) | | M |
| 1273 | 1 | psum_sel | 1 when partial sums are kept across K chunks (conv); then cfg2 bit 1845 = 0. Guess: selects a different memory | | L |
| 1274-1275 | 2 | psum_hmode | 1 for FC with K ≤ 4 and for single-step requant ops; also on some copy and mul ops | | M |
| 1276+29k .. 1288+29k | 13 s | psum_inc{k} | FC: all 0 except inc1 = 1 (that dimension has cnt 0) | | H |
| 1289+29k .. 1302+29k | 14 | psum_cnt{k} | FC: `cnt0 = kw-1`, `cnt3 = passes-1` | | H |
| 1303+29k .. 1304+29k | 2 | psum_mode{k} | dim 7: FC/conv 2, relu 0 | | L |
| 1508-1522 | 15 | psum_tflags | FC: 0x8C0 iff passes > 1 for this tile, else 0. relu 0x800 (0 if single step) | | H for FC |
| 1523-1562 | 40 | rsv1523 | | | |
| 1563-1571 | 9 | cfg0 | op class (see below). FC 0xA5, relu 0x05, copy 0x07, conv/dwconv 0xE5 | | M |
| 1572-1577 | 6 | rsv1572 | | | |
| 1578-1587 | 10 | cfg1 | FC 0x16C (0x14E when streamed), relu 0x14D, copy 0x6B, conv 0x16F | | M |
| 1588+16j (j=0..4) | 16 each | sync0..sync4 | slots shaped like TTU counts (14-bit value plus 2 mode bits). FC/relu/conv: sync0 = sync1 = 0x4000. Copy ops: sync0 = count, sync1 = 0x4000 + 4*count. Weight-FIFO ops: sync2 = 0xC6 or 0xB6, sync3 = sync4 = 0x80. Guess: sync-watcher parameters (the compiler prints "fifo entry size, fifo entries to wait, sync flags per fifo entry, sync flags to wait, init value") | | L |
| 1668-1842 | 175 | rsv1668 | | | |
| 1843-1847 | 5 | cfg2 | values: FC 7; FC K≤4 6; relu/copy 6; conv/dwconv 7, or 3 when partial sums are read (psum_sel = 1); avgpool 0xE; binary 0x16. Bit 1843 = MAC op with a multi-step reduction (FC K>4, conv, dwconv). Bit 1845 = 0 exactly when psum_sel = 1. Bit 1844 is always 1 | conv1x1/conv sweeps | M |
| 1848-1849 | 2 | rsv1848 | | | |
| 1850-1853 | 4 | cfg3 | 7 for mul, 6 for avgpool, else 0 | | L |
| 1854-1861 | 8 | w_zp | weight zero point (u8) | known; zero-point sweep | H |
| 1862-1869 | 8 | rsv1862 | | | |
| 1870-1877 | 8 | in_zp | input zero point (u8) | known; sweep | H |
| 1878-1922 | 45 | rsv1878 | | | |
| 1923-1925 | 3 | dp_mode | datapath select: 3 = MAC with wide-memory weights (FC, depthwise), 4 = conv/copy/avgpool, 1 = elementwise requant, 5 = binary add/sub/max/min, 2 = second mul op | | M |
| 1926-1933 | 8 | reduce_mask | bit k set = main-loop dim k is a reduction (accumulation) dimension. One output group is emitted each time all masked dims complete. FC 0b101 (loop0, plus loop2 when K is split as for K=8192), conv 0b1011 (loop0 = K chunk, loop1 = kernel taps, loop3 = K chunks), some unsplit 1x1 convs 0b111, dwconv/avgpool/binary 0b11, relu/copy 0 | `out` steps = `prod(loop+1)/prod_{k in mask}(loop_k+1) * (out_ch+1)/4` holds on 1731/1736 instances (exception: second mul op) | H |
| 1934-1939 | 6 | out_ch_last | channels in the last output group - 1 | FC N=1100 → 11; conv N=100 → 35 | H |
| 1940-1945 | 6 | out_ch | output channels per group - 1 (multiple of 4, minus 1). FC: `min(64, roundup4(N_tile))-1`. relu/copy: 3 | | H |
| 1946-1948 | 3 | rsv1946 | | | |
| 1949-1980 | 32 | mult_bits | f32 requant multiplier (`in_scale*w_scale/out_scale`) | known | H |
| 1981-2012 | 32 | clamp_max_bits | f32 = `255-out_zp`, or the activation bound | RELU6 and act sweep | H |
| 2013-2044 | 32 | clamp_min_bits | f32 = `-out_zp`; fused RELU makes it 0.0, which is the only change act=1 makes | act sweep | H |
| 2045-2076 | 32 s | offset | int32. Non-zero only for binary ops (-128, -256, -383) | | L |
| 2077-2084 | 8 | out_zp | output zero point (u8) | known | H |
| 2085-2092 | 8 | rsv2085 | | | |
| 2093 | 1 | cfg4 | 1 for FC/conv/dwconv (ops with bias/parameters), else 0 | | M |
| 2094-2175 | 82 | rsv2094 | | | |

Absolute bit ranges of the TTU dimension records (inc is signed):

| k | in_inc / in_cnt / in_mode | out_inc / out_cnt / out_mode | par_inc / par_cnt / par_mode | psum_inc / psum_cnt / psum_mode |
|---|---|---|---|---|
| 0 | 211-227 / 228-245 / 246-247 | 594-610 / 611-628 / 629-630 | 973-985 / 986-999 / 1000-1001 | 1276-1288 / 1289-1302 / 1303-1304 |
| 1 | 248-264 / 265-282 / 283-284 | 631-647 / 648-665 / 666-667 | 1002-1014 / 1015-1028 / 1029-1030 | 1305-1317 / 1318-1331 / 1332-1333 |
| 2 | 285-301 / 302-319 / 320-321 | 668-684 / 685-702 / 703-704 | 1031-1043 / 1044-1057 / 1058-1059 | 1334-1346 / 1347-1360 / 1361-1362 |
| 3 | 322-338 / 339-356 / 357-358 | 705-721 / 722-739 / 740-741 | 1060-1072 / 1073-1086 / 1087-1088 | 1363-1375 / 1376-1389 / 1390-1391 |
| 4 | 359-375 / 376-393 / 394-395 | 742-758 / 759-776 / 777-778 | 1089-1101 / 1102-1115 / 1116-1117 | 1392-1404 / 1405-1418 / 1419-1420 |
| 5 | 396-412 / 413-430 / 431-432 | 779-795 / 796-813 / 814-815 | 1118-1130 / 1131-1144 / 1145-1146 | 1421-1433 / 1434-1447 / 1448-1449 |
| 6 | 433-449 / 450-467 / 468-469 | 816-832 / 833-850 / 851-852 | 1147-1159 / 1160-1173 / 1174-1175 | 1450-1462 / 1463-1476 / 1477-1478 |
| 7 | 470-486 / 487-504 / 505-506 | 853-869 / 870-887 / 888-889 | 1176-1188 / 1189-1202 / 1203-1204 | 1479-1491 / 1492-1505 / 1506-1507 |

### FULLY_CONNECTED summary (tile t of `y[N] = W[N,K] x[K]`)
Definitions:
- `kw = ceil(K/4)`
- `G = ceil(N/64)`
- `P = ceil(G/16)` (passes)
- `T = ceil(G/P)` (tiles)
- `gt = min(P, G-P*t)`
- `N_tile = min(64P, N-64P*t)`

Fields:
- **K loop:** `loop0`, `in_cnt0`, `par_cnt0` and `psum_cnt0` are all `kw-1`. The rewinds are `-(kw-1)`.
- **Outputs per tile:**
  - `out_cnt0 = w0-1`, with `w0 = ceil(min(64, N_tile)/4)`;
  - `out_ch = out_ch_last = min(64, roundup4(N_tile))-1`;
  - `loop3 = in_cnt3 = out_cnt2 = par_cnt3 = psum_cnt3 = gt-1` (passes over 64-output groups, N > 1024).
- **Addresses:** `in_base` = address of x; `out_base` = address of y[64P*t]; `par_base` = relocated weights.
- **Constant fields:** everything else, e.g. cfg0 = 0xA5 and `reduce_mask` = 0b101.

`op.fc_tile_fields(N, K, t, seq, in_base, out_base, par_base, w_zp, in_zp, mult, out_zp, ...)` reproduces bit-exactly:
- all 963 corpus FC ops;
- 197 extra FC ops: K up to 6144, partial last passes, activations, zero points.

Exception: K ≤ 4 with N > 1024. The compiler then uses another schedule (loop1 = 1, psum TTU at base 8048, cfg0 = 0xA7), and the builder asserts.

K = 8192 is also different. The weights are streamed through a 32-row FIFO at par_base = 8064 (the top of wide memory), with `loop0 = 31`, `loop2 = 63`, `par_fifo = 32`, `par_tflags = 0x80`, extra sync2..4 values, and cfg1 = 0x14E.

`op.requant_tile_fields` reproduces all 171 RELU, RELU6, RELU_N1_TO_1 and QUANTIZE ops on [1, n] tensors, including the single-word tiles. These ops all use the same instruction and differ only in the f32 clamp, multiplier and zero-point fields.

### Operation type (how FC, RELU and CONV differ)
There is no single opcode field. The op class is the combination of `dp_mode`, `reduce_mask`, cfg0-4, `par_sel` and the tail flags:

| class | dp_mode | reduce_mask | cfg0 | cfg1 | cfg2 | cfg4 | par | out_ch |
|---|---|---|---|---|---|---|---|---|
| FC | 3 | 0b101 | 0xA5 | 0x16C | 7 (6 if K=4) | 1 | resident, par_tflags=3 | N_tile-1 |
| RELU/requant | 1 | 0 | 0x05 | 0x14D | 6 | 0 | idle (base 0x7C, sel=1) | 3 |
| copy/reformat (first conv/pool op) | 4 | 0 | 0x07 | 0x6B | 6 | 0 | idle | 3 |
| conv (3x3, or 1x1 with K split) | 4 | 0b1011 | 0xE5 | 0x16F | 3/7 | 1 | FIFO, par_fifo=rows, sync2=0xC6 | ≤63 |
| depthwise conv | 3 | 0b11 | 0xE5 | 0x16E | 7 | 1 | FIFO, sync2=0xB6 | 3 |
| avg pool | 4 | 0b11 | 0x127 | 0 | 0xE | 0 | | 15 |
| add/sub/max/min | 5 | 0b11 | 0x09 | 0x1CD | 0x16 | 0 | | 3 |

## Constant bits (corpus, 1084 instances)
`mask` marks the bits that are identical in every instance; `value` gives those bits' values. Across the corpus, 1513 bits are constant (67 of them are 1) and 663 vary.

| word | mask | value |
|---|---|---|
| w0  | `fc3fe0ff83c00f803fe03ffff0000fff` | `00000000000000000000000000000040` |
| w1  | `00ffe00fffff000fffffffffffffffff` | `00000000000800000000000000000000` |
| w2  | `f0fa007fff000003ff0000003fffc1b0` | `00fa0000000000000000000020000000` |
| w3  | `e7fffffd003fffffe801ffffff400fff` | `0600007d00000003e80000001f400000` |
| w4  | `007fff87ffff8004fffffffffff403fb` | `00000000000400000000000000000000` |
| w5  | `fff8003ffffc0001ff8000007fffa0d8` | `00780000000000000000000010000000` |
| w6  | `f7fffffc001fffffe000ffffff0007ff` | `0700003c00000001e00000000f000000` |
| w7  | `ff8003f003ffd8007ffefdfffe3fffff` | `00000000000000000000000000000000` |
| w8  | `07fff8003ffc0001fff0000ffe00007f` | `00000000000000000000000000000000` |
| w9  | `fdfc7ffffffffff1e787ffe000ffff00` | `00000000000000000000000000000000` |
| w10 | `e000fff00007ff00003ff83ffdf801ff` | `00000000000000000000000000000000` |
| w11 | `ffff73f7fff0007fff8003fffc001fff` | `00000000000000000000000000000000` |
| w12 | `f7fff7fff39f403bd01863f8efffffff` | `00000000000400000000200028000000` |
| w13 | `ffffffffffffffffffffffffffffffff` | `00000000000000000000000000000000` |
| w14 | `ffffffffffdfffdfffd7ffffffffffff` | `00000000000000000010000000000000` |
| w15 | `e00ffffff8503fffff6ffffffc30fc07` | `0000000008500000076000000030c000` |
| w16 | `ffffffffffffffffffffdfefffffffff` | `00000000000000000000000000000000` |

The FC-only template (963 corpus FC ops) is much tighter: 1946 bits constant, 179 of them set. Exactly 34 fields vary:
- tile_mask, seq, loop0 and loop3;
- in_base, out_base and par_base;
- in/out/par/psum counts and increments;
- psum_tflags, out_ch and out_ch_last.

The quantization fields are constant here only because the corpus uses fixed scales.

## Role in programs
- **FC (EXECUTION_ONLY):** one op per used tile, `tile_mask = 1<<t`, issued back to back. They follow the input distribution (ringConsumer and 0x16 moves) and the bias wideToNarrow, and are bracketed by `sync` (0x1a) instructions. narrowToWide / ringProducer then move the outputs.
- **RELU:** one op per tile, possibly in place (`in_base == out_base`).
- **conv1x1:** two ops broadcast to 2-4 tiles (mask 0x5 or 0xf).
  - A copy/reformat op runs right after the ringConsumer. It has `dp_mode` 4, `reduce_mask` 0, and carries the in_twait/sync0/sync1 counts.
  - The compute op reads weights through a wide-memory FIFO (`par_sel` = 1, `par_fifo` = rows). The FIFO is filled by opcode 0x12, presumably ringConsumer1. K is split into chunks with partial sums (`reduce_mask` 0b1011: loop0 and loop3 are reduced, loop2 iterates positions, loop4 output-channel groups).
- **Other compute opcodes:** logistic, tanh, and n=64 add/sub/max/min compile to other compute opcodes (e.g. 0x19), not 0x01.

### Sync flags
In FC and RELU programs the op carries no producer/consumer sync-flag ids. Chaining FC→FC, FC→RELU or RELU→FC changes only `seq`, `in_base`, `out_base` and `par_base`. Every other bit, including all tails and the sync slots, stays the same whether the op's input comes from the ring or from another op, and wherever its output goes.

Synchronization must therefore come from the surrounding `sync` (0x1a) instructions, plus the global `seq` ordering. That counter matches the compiler's "global token" and is also embedded in the sync instructions.

Wait-like data only appears in ops that consume data while it streams in:
- copy ops: `in_twait`, sync0, sync1;
- weight-FIFO ops: `par_tflags`, `par_fifo`, sync2-4.

Their exact sync semantics are not decoded.

## Open questions
1. **Mode bits:** what do the 2-bit per-dimension `mode` fields and the header `hmode` fields mean? Dim 7 is 3 for in/out in every class except the avgpool compute op. Binary ops and mul set 1 or 2 on every in/par dimension.
2. **Flag fields:** the bit meanings of cfg0-cfg4, the tflags, `par_sel`/`psum_sel` and `dp_mode` are unknown. Only the per-class values are known.
3. **Sync slots:** the semantics of sync0-sync4 and `in_twait`. The hypothesis is sync-watcher config (init count, increment, loop level), but hardware tests are needed.
4. **Field widths:**
   - `seq` is verified to at least 9 bits; 16 bits is assumed.
   - The A/B bases may really be 16-bit word addresses at bits 193/576.
   - The loop/count widths (14/18/14 bits) are inferred from field spacing. Maximum observed values: loop0 = 1535, in_cnt = 1535, A stride 16384.
5. **Exceptions to the model:**
   - The second mul op (`dp_mode` 2) breaks the `reduce_mask` output-count rule.
   - K ≤ 4 FC with N > 1024 uses a psum schedule that the builder does not model.
6. **Ordering and offsets:**
   - Does the hardware require `seq` to match the dispatch order, and what happens if it does not?
   - What is the int32 `offset` (2045-2076) used by binary ops?

## Round trip
Run `python coral/isa/op.py` from the project root. Result:
- round trip: **1084/1084** corpus op instructions are identical;
- `fc_tile_fields`: **963/963** corpus FC ops reproduced;
- `requant_tile_fields`: **97/97** corpus RELU ops reproduced.

Over all 1736 instances (corpus plus extra compiles) the round trip is also 1736/1736.
