# FULLY_CONNECTED code generation (`coral/codegen/fc.py`)

`gen_fc(N, K)` builds the PARAMETER_CACHING and EXECUTION_ONLY programs for `y[N] = W[N,K] x[K]` (uint8) from the decoded ISA.
It uses no compiler templates and copies no instruction words. Every instruction comes from the encoders and builders in
`docs/isa/{op,wide_narrow,ring_mesh,scalar}.py` and the formulas in their reports. Section 4 lists the rules that were
missing from those reports.

```
python -m coral.codegen.fc            # acceptance test (same as python test/test_codegen.py), offline, ~20 s
python -m coral.codegen.fc --extra    # + 43 edge-case shapes compiled with edgetpu_compiler (cached in .compile/)
python -m coral.codegen.fc --random 160
```

| API | status |
|---|---|
| `gen_fc(N, K, param_offset=0)` | byte-identical to edgetpu_compiler (section 1) |
| `gen_fc(..., tile_shift=s)` | identical to `ring_mesh.translate_tiles` wherever that is legal (section 5) |
| `gen_fc(..., tile_shift=s, move_gather=False)` | a program structure the compiler never emits (section 5) |
| `gen_fc(..., bias=False)` | the compiler's form for N=1 with an all-zero bias |
| `gen_fc(..., quant={...})` | overrides the reference quantization: w_zp, in_zp, out_zp, mult, clamp_min, clamp_max |

## 1. Verification

| check | result |
|---|---|
| FC table, N ≤ 1024 and K ≤ 1024 | 256/256 byte-exact (caching + execution) |
| FC table, K ∈ {2048, 3072, 4096} | 48/48 |
| FC table, N ∈ {2048, 3072, 4096} | 51/51 |
| `_compile(N, K)`, N ∈ {10, 100, 288, 864} × K ∈ {30, 288, 300} | 12/12 |
| extra edge cases (`--extra`) | 43/43 |
| random shapes: 1920 compiles (N 1..4096, K 1..4096, a third with N ≤ 64) | 1918/1920; the other 2 are the unsupported K ≤ 4 with N > 1024 |
| narrow-memory layout rule (section 3) vs sampled compiles | 1448/1448 |
| `param_offset` vs `tools.fcgen.relocate(table program, measured relocation fields)` | 1065/1065 (355 entries × 3 offsets) |
| `tile_shift` vs `translate_tiles` (programs, legality verdict) | 424/424 programs, 5325/5325 verdicts |
| `tile_shift` with `move_gather=False`: placement and seq self-consistency | 2280/2280 |

Zero-bias compiles of N=1 are compared with `bias=False`; see section 4.8.

## 2. Program structure

Notation: `[stride×count, ...]` is a logical loop nest, innermost first. `(inc, cnt)` pairs are encoded TTU fields, with cnt = count − 1 and inc the compiler's rewinding increment.

Geometry:
- G = ⌈N/64⌉ output groups, P = ⌈G/16⌉ groups per tile, T = ⌈G/P⌉ compute tiles. Tile t has gt = min(P, G − Pt) groups.
- Input: S = 8⌈K/8⌉ DMA bytes, b = 64⌈S/1024⌉ bytes per input tile, T_in = ⌈S/b⌉ input tiles. Tile t holds x[bt : bt + piece(t)].
- Narrow addresses: X (x), Y (y), R (32-byte relay buffer); see section 3.
- `seq` counts every tile-dispatched instruction (0x01, 0x10–0x18) and every 0x1a sync, and nothing else.

**Caching** (scalar.md 8.1):

```
start, sync_init(caching), scsync_init, host_dma(tag 2), param_pop
ringConsumer per compute tile t
param_infeed per compute tile t
sync_final, scsync_fence, interrupt, halt, nop×4, end
```

The ringConsumer for tile t has:
- addr = 2 + 4·gt + off;
- loops [1×Kq], or [1×Kq, 1×gt] when gt > 1;
- aux = (off, 4(gt−1) + off), the bias rows.

**Execution**:

```
start, exe_prologue
input_head(1), input_dma(2, S)
wideToNarrow input × T_in                     wide_narrow.w2n_input (lane t%4 of its group's ring FIFO)
(ringConsumer, av_infeed) per row group j     4 tiles per group, 256-byte packets, multicast
sync_drain, sync_reset17
[T_in ≥ 2] mesh gather                        see below
scsync_nop, sync_reset17
[T ≥ 2] w2n ring recv, ringConsumer, ringProducer(tile 0 → tiles 1..T−1), narrowToWide(tile 0 → 0x1f70)
sync_reset17, sync_signal, scsync_fence
wideToNarrow bias (mode 2)                    one per distinct gt
op × T                                        op.fc_tile_fields, par_base = 4·gt + off
sync_reset17, sync_signal, scsync_fence
output_dma_head, narrowToWide output × T, output_dma_tail(8⌈N/8⌉)
(outfeed(8⌈n_t/8⌉), ringProducer(t, ordinal t)) × T
output_wait(T), epilogue, end
```

**Mesh gather.** Rows are groups of 4 input tiles.

West phase, for k = 1..min(4, T_in)−1. Group k moves position k of every row that has one to position 0:
1. One plain 0x16 sender per row: o_addr = X + b·(4r+k), loop [1×words].
2. For k ≥ 2, relay instructions on positions 1..k−1. Rows whose pieces have the same word count share one instruction.
3. One plain 0x16 receiver per row: i_addr = the sender's address.
4. `_sync_mesh(west)`.

After the last group: `sync_drain(last mask)`, then `sync_drain(ffff)`.

North phase, for k = 1..rows−1. Group k moves row k (gathered on tile 4k) to tile 0:
1. 0x17 sender on tile 4k (out_mode 0).
2. For k ≥ 2, a relay on tiles 4..4(k−1).
3. Receiver on tile 0 (in_mode 0).
4. A north fence, but only for k ≥ 2.

After the north phase: `sync_drain(last mask)` if there were ≥ 3 rows, then `sync_reset_mesh`.

## 3. Narrow-memory allocation (compiler rule)

The rule was fitted on 1448 compiles: dense N sweeps from 640 to 1100 at K = 64 and 128; binary searches of the y-first/x-first
boundary for 70 values of N; and grids. All are reproduced exactly.

- **Rounding.** Sizes are rounded up to 4 bytes: K4 = 4⌈K/4⌉, N4 = 4⌈N/4⌉.
- **Relay buffer.** R (32 bytes) exists whenever T_in > 1, even when no relay instruction uses it.
- **M.** Let M = 256·⌈K/256⌉ + 512.

```
x first  [x | y | R]           X = 0, Y = K4, R = K4 + N4          unless K ≤ 188 or N4 + 32 ≥ M
y first  [y | R | ... | x]     Y = 0, R = N4, X = max(M, N4 + |R|)   when N4 < M
         [y | x | R]           Y = 0, X = N4, R = N4 + K4            when N4 ≥ M
N ≤ 4    the one-word y is placed after R instead of before it: (Y, R) = (base + 32, base)
```

Examples:
- N ≤ 704, K ≤ 188: x sits at 768.
- N = 752, K = 128: x at 784.
- 1000 × 320: x at 1032.
- 2045 × 1330: x at 2048 and R at 3380.

The semantics are not understood. M behaves like a reserved region for x whenever y comes first.

## 4. Rules missing from (or corrected in) the decode reports

All of these were found by matching compiler output. Each is reproduced on every instance in section 1.

1. **Mesh-sync count (scalar.md §2).** `b63` + `count` form one 16-bit value at bit 63.
   - West fences: value = 16-byte packets forwarded so far by the phase's first relay instruction, + 1.
   - North fences: the same packet count, without the + 1.
   - With full 64-byte chunks this gives the documented (b63 = 1, 2m·g) and (b63 = 0, 8m·h). Partial pieces make the value odd or uneven, e.g. K = 200 → (b63 = 0, 3), K = 1000 north → (b63 = 1, 15).
2. **Mesh-fence tile mask.** The mask is `~(relays forwarding as many packets as the first relay instruction)`.
   - Example K = 1000: relays 0x666 (64-byte pieces) and 0x6000 (40-byte piece) → fence f999.
   - With 52 bytes (also 4 packets) both relay groups are excluded → dddd.
3. **Relays (ring_mesh.md §4).** A relay walks [1×4 words, 0×packets] through R.
   - Records: (EAST_IN, 9 + 4p), (WEST_OUT, 1 + 4p) going west; (SOUTH_IN, 5 + 4p), (NORTH_OUT, −3 + 4p) going north. p = packets that relay chain already forwarded in this phase. The third record is (id 1, no enables).
   - A partial last packet of r words (1..3) sets `grp = r − 1` and a field at bit 249 (inside `rsv243`; `rsv448` for the inbound half) to 4(4 − r) + 1. This is the ring consumer's gstride slot.
   - A 1-word transfer uses [1×1, 0×1] and no partial fields.
   - Mesh `sdims` is 0 for a 1-word plain move.
4. **Input ringConsumer (ring_mesh.md §3).** Row group j receives p_j = ⌈bytes_j/256⌉ of the input's c = ⌈S/256⌉ packets.
   - (inc0, cnt0) = (1, 0) if p_j = 1, else (1, c − 1) (encoded fields).
   - grp = p_j − 1.
   - gstride = 4(c − p_j) + 1 if p_j > 1. The documented 4(G−1)p + 1 is the case where every group has p packets.
5. **Parameter DMAs (scalar.md §4–5).**
   - Infeed `f438` = 64-byte lanes per weight block − 1, i.e. g/16 − 1 for g outputs per group. (U−1) mod 4 only coincides with it on the corpus; N = 10, K = 300 differs.
   - `pop_wait` wraps mod 2^16 for blobs > 4 MiB.
   - Pop `b183` = 1 when U is even or the 128-byte row count is capped (U ≥ 2048). scalar.md only has "U even"; N = 41, K = 3896 (U = 2925) differs.
   - For U ≡ 0 mod 1024, pop and infeed use d1_limit = U/1024 − 1, count_m1 = count_neg = 0, and k157 (pop) / k203 (infeed) = 0.
6. **1-word narrow walks (wide_narrow.md).** The output narrowToWide and the ring-receive wideToNarrow have `narrow_lvl_mask` = 0 when they move a single word. The input-scatter wideToNarrow keeps 1.
7. **Uneven N > 1024.** The last compute tile has gt < P groups.
   - Its caching consumer (addr 2 + 4gt, cnt1 = gt − 1, aux 4(gt − 1)), its infeed size, its op (`par_base` = 4·gt, loop3 = gt − 1) and its bias load all use gt.
   - Its bias load is a separate wideToNarrow, after the one for the full tiles.
8. **N = 1 with an all-zero bias.** The compiler drops the bias: there is no bias load, the blob is weights only, and the caching consumer has addr 2 and no aux.
   - The op has `par_base` 0 and cfg1 = sync0 = sync1 = cfg4 = 0. So these four op fields belong to the bias ("scaling") path.
   - With a nonzero bias, N = 1 is an ordinary FC, which `gen_fc(1, K)` reproduces.
9. **Not modelled.**
   - K ≤ 4 with N > 1024: another op schedule (op.md); raises `NotImplementedError`.
   - Weights that don't fit on chip: the compiler streams parameters in a STAND_ALONE program.

## 5. `param_offset` and `tile_shift`

**param_offset** (bytes, multiple of 64) is added in 64-byte units to:
- the caching consumers' `addr`, `aux_addr0` and `aux_addr1`;
- the bias wideToNarrow `wide_addr`;
- every op's `par_base`.

These are exactly the measured relocation fields.

**tile_shift = s, move_gather = True** is ring_mesh.md §6: a pure translation of every physical tile reference. That covers tile masks (including partial sync masks), ringProducer `dest`, and infeed bitmaps.
- **Legal** iff no used tile (compute or input) passes tile 15, i.e. s ≤ 16 − max(T, T_in).
- Also, every westward chain must stay in one mesh row: each logical row with w ≥ 2 input tiles needs s mod 4 + w ≤ 4. North hops stay neighbours for any s.
- `legal_shifts(N, K)` lists the legal values. The programs equal `translate_tiles(gen_fc(N, K), s)`.
- Only 153 of the 355 table shapes have a legal s > 0. For K ≥ 769 the gather uses all 16 tiles, so no shift is possible.
- Examples: 128 × 128 → {1, 2, 4, 5, 6, 8, ...}; 256 × 256 → {4, 8, 12}; 288 × 768 → {4}.

**tile_shift = s, move_gather = False** moves only the compute tiles (parameters, bias load, ops, outputs) to [s, s+T).
- The input scatter and mesh gather stay on tiles [0, T_in): they only use narrow memory and the wide FIFOs at 0x1f70 and above, which lie above any parameter region (≤ 0x1e50).
- The ring broadcast then sends x from tile 0 to all compute tiles that do not hold it. That is all of them when s > 0, including T = 1, where the compiler has no broadcast.
- **Legal** iff s ≤ 16 − T. 285 of the 355 table shapes have a legal s > 0.
- For s = 0 it is identical to the compiler.
- For s > 0 the program has a structure the compiler never emits (a broadcast whose source is not a compute tile). Offline it is checked for placement, seq numbering and round trips.

On the device both forms ran bit-exact: 9/9 translated programs and 15/15 with `move_gather=False` (`bench/RESULTS.md`).

## 6. Batched matmuls

y[M,N] = x[M,K] @ W.T for M > 1 runs as a 1x1 convolution over a grid of M positions with W stationary: `coral/codegen/conv.py`
(`docs/isa/codegen_conv.md`).

## 7. Open questions

1. What the narrow allocator's M = 256⌈K/256⌉ + 512 rule means, and why a one-word y swaps with R.
2. The semantics of the mesh fence value (+1 for west only), of the relay record bases (9/1, 5/−3), and of the fence masks that keep the relays of a smaller relay class.
3. The `psum_tflags` bits (0x800 multi-group, 0x40, 0x80) and how `out_last_*` interacts with loop1.
4. Whether ring FIFOs can be reused by looping consumers (RING_READ_A credits).
