# Fused LLM programs and the whole-set placement (`coral/codegen/fused.py`)

`coral/codegen/fused.py` generates the two fused TinyStories-15M programs and places the whole model, with no
edgetpu_compiler involved:
- `gen_ffn`: the FFN block of `coral.fused.ffn_block`. It runs w1, w3, LOGISTIC, MUL, MUL and w2 in one program.
- `gen_argmax`: the classifier of `coral.fused.argmax_block`. It is a pooled conv: the vocabulary is the image, the
  tokens are the weights, and an 8x8 MAX_POOL follows.
- `plan` and `gen_set`: our own placement of every parameter blob of the set, and the programs generated for it.

Every instruction comes from the encoders in `docs/isa/{op,wide_narrow,ring_mesh,scalar,eltwise}.py` and from the
builders of `coral/codegen/fc.py` and `coral/codegen/conv.py`, plus the rules below. They were fitted by diffing against
edgetpu_compiler at the field level. No compiler words are copied.

The model has fixed shapes (D=288, Hd=768, V=32000), so two kinds of tables appear: the compiler's narrow allocation
of the FFN (sec. 2.2) and two numbers per Mp for the argmax allocation (sec. 4.5). Both are read from the compiler's
programs.

```
python -m coral.codegen.fused            # acceptance test (= python test/test_codegen_fused.py), offline; ~10 s with the compile cache
python -m coral.codegen.fused --quick    # one quantization per block, co-compiled sets for Mp 1 and 16 only
python -m coral.codegen.fused --extra    # + 12 FFN and 6 argmax quantization sets
```

| API | |
|---|---|
| `gen_ffn(Mp, D=288, Hd=768, quant, param_tile, param_offset, param_limit_units, pieces)` | `(caching, execution)`. Mp 1 (FULLY_CONNECTED; spread, or streamed when `param_tile`/`pieces` is given) or Mp 16..256 (1x1 conv) |
| `ffn_params(W1q, W3q, W2q, b1, b3, b2, zps)` | the FFN parameter blob (sec. 2.5) |
| `gen_argmax(Mp, V, K, grid, pool, quant, placement)` | `(caching, [3 execution bitstreams])`, Mp in 16..256 |
| `argmax_params(tokens, Mp, zp)`, `argmax_hints`, `argmax_io`, `argmax_output_layout` | token blob and runtime contract (sec. 4) |
| `gen_fc_streamed(N, K, tile, offset, quant)` | single-tile FULLY_CONNECTED, the compiler's form for batch-1 layers that do not fit spread (sec. 3) |
| `gen_conv_pieces(Mp, N, K, pieces, quant)` | `gen_conv1x1` with a free (tile, offset) per parameter piece |
| `llm_specs()`, `plan(blocks, Mp)`, `check_plan`, `gen_set(blocks, Mp, plan, quants)` | our placement and the whole set as `coral.executable.Executable`s (sec. 5, 6) |
| `program_hints`, `caching_hints`, `block_io`, `set_limit`, `program_limit` | hints, host sizes, the param_limit rule |

Quantization keys:
- `gen_ffn`: `x, h1, h3, g, a, m, y, w1, w3, w2` (scale, zero point); g is (1/256, 0).
- `gen_argmax`: `vocab, tokens, logits`.
- conv blocks of `gen_set`: `x, w, y`.

## 1. Verification

`python -m coral.codegen.fused`, all offline. Programs are compared byte for byte, every bitstream.

| check | result |
|---|---|
| 1. FFN compiled alone (`coral.fused.ffn_block`), Mp in {1, 16, 32, 64, 128, 256} x 4 quantization sets (random weights; asymmetric ranges, set 0 = x (-4, 5), h1 (-6, 4), ...): caching + execution, blob = `ffn_params`, DMA hints, host sizes, output layout | 24/24 |
| 1b. FFN with random int32 biases (own TFLite model, same structure), Mp in {1, 16, 256}: programs + blob with bias rows | 3/3 |
| 2. argmax compiled alone, Mp in {16, 32, 64, 128, 256} x 3 quantization sets: caching, all 3 execution bitstreams, blob = `argmax_params`, hints (DMA interleaving, instruction chunks), output layout, sizes | 15/15 |
| 3a. the co-compiled LLM set (`coral.fused.compiled(Mp)`: 6 x (qkv 864x288, wo 288x288, ffn) + cls for Mp > 1), every program regenerated from the compiler's (tile, offset) only (caching, execution, hints) | Mp 1: 18/18; 16, 32, 64, 128, 256: 19/19 each (113 programs) |
| 3b. our `plan()` per Mp: no overlaps, every region 256-byte aligned below the set limit; every generated program scanned for wide addresses below the limit outside its own parameters | 6 plans, 113 programs, 0 violations |
| 3c. our layer placements are identical for Mp in {16, 32, 64, 128, 256} | 72/72 |
| `--extra`: 12 more FFN quantization sets (Mp 1, 16, 64, 256) and 6 more argmax sets (Mp 16, 256) | 48/48, 12/12 |

Section 3a covers seven program forms. Six come from this module:
- FFN in conv form with split pieces;
- FFN in FC spread form;
- FFN in FC streamed form;
- FC streamed;
- conv with free pieces;
- argmax.

The seventh, FC spread for qkv/wo, comes from `coral.codegen.gen_fc`.

On the device our plan and programs run TinyStories-15M at batch 1..256: 1815 tok/s at batch 256, the same stories as with the
compiler's programs (`bench/RESULTS.md`).

## 2. FFN

### 2.1 Conv form (Mp in 16..256)

The program chains codegen_conv's stages and eltwise.md's ops on all 16 tiles. Every stage puts its own fences before
and after it.

```
start, exe_prologue
input_head; identity row: 4 mesh fills at C + narrowToWide -> wide IDENT; sync_wn_fence
[hp > 1: first copy op]; input DMA (S = Mp*D); [Mp >= 128: sync_w2n]; input wideToNarrow x4; (ringConsumer, infeed) x4
sync_drain; last copy op
conv w1:  reset17, scsync_nop, drain, ringConsumer1, bias wideToNarrow, conv op, ringProducer per piece   (X -> h1)
LOGISTIC: reset17, sync_op_fence, 0x19 NLU load, LOGISTIC op, reset17                                   (h1 -> g)
MUL 1:    reset17, sync(NARROW_TO_WIDE), sync(PARAMETERS), step 1, FIFO feed (g), sync_wn_fence, step 2,
          sync(NARROW_TO_WIDE), sync(PARAMETERS), reset17                                               (h1*g -> a over g)
conv w3:  as w1                                                                                         (X -> h3)
MUL 2:    as MUL 1                                                                                      (a*h3 -> m over h3)
conv w2:  as w1, K = Hd                                                                                 (m -> y)
output: codegen_conv's host path; output_wait(16); scsync_wait_pb(all blocks); sync_rpb; fence; epilogue
```

- The constant row of codegen_conv's copy ops is the 4x4 identity. It sits at the top, `IDENT` = 8316, and serves
  the copy ops, LOGISTIC and both MUL step 2s.
- Each ringProducer carries `r0_val` = the first block of its piece in the whole FFN blob: w1 at 0, w3 at 876, w2 at 1752.
- `scsync_wait_pb` counts all 2717 blocks.

### 2.2 Memory

Wide memory, in 64-byte units:
- identity at 8316;
- input ring FIFO at 8316 - 8*c_in;
- w1 and w3: compute regions stacked down from 8316 by `conv_regions` (codegen_conv's rule: largest first, ties
  psum, bias, FIFO);
- w2: compute regions stacked down from 8320, because the identity is dead by then;
- MUL operand FIFO at 8316 - 8*D (D = 4 rows: 8284);
- output FIFO at 8312.

`ffn_param_limit` is the lowest of these. For example Mp=16 gives 8208 (w1's reserved psum).

Narrow memory is the compiler's allocation for D=288, Hd=768, in bytes. g/a and h3/m share a buffer, because MUL writes
its output over its FIFO operand. int1 and int2 are 4*ppt*Hd + 4 bytes.

| Mp | Stg | C (identity) | X | h1 | g/a | int1 | h3/m | int2 | y |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0 | 2304 | 4612 | 768 | 0 | 1536 | 768 | 1536 | 0 |
| 32 | 0 | 2304 | 3072 | 1536 | 0 | 3648 | 1536 | 3072 | 0 |
| 64 | 0 | 4608 | 6144 | 3072 | 0 | 7296 | 3072 | 6144 | 0 |
| 128 | 0 | 4608 | 6144 | 8448 | 0 | 14592 | 30724 | 6144 | 0 |
| 256 | 0 | 9216 | 12288 | 16896 | 0 | 29184 | 61444 | 12288 | 0 |

Every buffer lies right above the buffers it is live with. I could not find the allocator's ordering: no fixed
permutation, first-fit or best-fit order reproduces all five rows. X goes below h1 for Mp >= 128, h3 above int2.

### 2.3 FULLY_CONNECTED form (Mp = 1)

Spread form, i.e. three `gen_fc` layers chained, from the alone compile and layers 0-3 of the batch-1 set:
- w1 and w3 run on tiles 0..11 and w2 on tiles 0..4, each at its own `param_offset`.
- The alone default is w3 at 0, w1 at 18688, w2 at 37376.
- h1, g, a, h3 and m are 64-byte slices in eltwise's global 1-D layout, so LOGISTIC and MUL run one op per tile.

```
codegen._input_block (x at X=1536, relay R=0)
w1: sync_signal, fence, bias load, FC op per tile, reset17
LOGISTIC: reset17, identity prologue (narrow 1824 -> wide 8060), op fence, NLU (tiles 0..11), op per tile, reset17
MUL 1 per tile (intermediate at 1824, FIFO 8288)
reset17, scsync_nop, reset17, x broadcast again (tile 0 -> 1..11, FIFO 8044), reset17
w3, MUL 2 (intermediate 1536)
reset17, codegen._gather of m (12 input tiles, X=768, R=288), scsync_nop, reset17, broadcast m to tiles 1..4 (FIFO 8048), reset17
w2 + completion fence, codegen._output_block (y at 0)
```

Streamed form, from layers 4-5 of the batch-1 set:
- The same chain, but each FC stage is sec. 3's streamed stage.
- w3 takes its bias rows at 8052, because the identity (8060) is live.
- The parameter limit of the FC forms is 8044, the second x broadcast FIFO.

### 2.4 Quantization (float32, as edgetpu_compiler)

The conv and FC main ops:
- `mult = f32(f32(s_in * s_w) * f32(1/s_out))`, with the inputs rounded to f32 first. This matched 18/18; the fp64
  quotient matches only 14/18.
- The clamps are `eltwise.out_clamps(s_out, zp_out)`, i.e. `f32(f32((0-zp)*s) * f32(1/s))`. They are not integers:
  for example 127.00001 instead of 255-128. The integer clamps that `coral/programs.py` generates for FCSpec matmuls differ by one ulp.

LOGISTIC and MUL use eltwise's formulas. MUL 1 is `mul_quant(h1, g, a)` and MUL 2 is `mul_quant(a, h3, m)`; operand a
of each MUL is TFLite input 0.

### 2.5 Parameter blob (`ffn_params`)

`conv_blob(w1) + conv_blob(w3) + conv_blob(w2)`, in this order, the same for every form. Each `conv_blob(W[N,K])`
consists of:
- for every group of 64 outputs: an int32 bias row (256 bytes), then the weights as `[K4/4][64][4]` uint8;
- K padded to a multiple of 4 with w_zp;
- the padded outputs of the last group (w2: outputs 288..319) all 0, weights and bias.
  `the old fc_params_padded` uses w_zp there instead: functionally the same, but not byte-identical.

Sizes are w1 = w3 = 12 x 73 blocks (224256 bytes) and w2 = 5 x 193 blocks (247040 bytes), 695552 bytes in total.
There is no NLU table: the sigmoid spline is an immediate of the 0x19 instruction.

## 3. Streamed FULLY_CONNECTED (`gen_fc_streamed`, also the FFN's streamed form)

All 1 + kw blocks of every 64-output group sit on one parameter tile. Each call they stream to compute tile t.

The program is codegen's input block, then the streamed stage, then sync_signal + fence and codegen's output block.

The streamed stage:
- sync_signal, fence, drain;
- per compute tile a ringConsumer1:
  - weight FIFO at 8064, 32 slots, walking (1, 31)(rew, b-1);
  - aux = bias + 2 and bias + 4;
  - partial last fill: grp = r-1, gstride = 4(32-r)+1;
- the bias load: codegen_conv's mode 2 with `b << 15`;
- one FC op per tile;
- per tile, a ringProducer on the parameter tile:
  - `addr = off + 4(1+kw)t`, `r0_val = (1+kw)t`;
  - `dest = 1 << t`, `to_c1`;
- reset17, `scsync_wait_pb`, sync_rpb, fence.

The op is `op.fc_tile_fields` with these changes:
- K runs in b = ceil(kw/32) fills of 32 rows; r is the number of rows in the last fill.
- Loops (32, 1, b). The `in` strides are (1, kw, 32); `par` is (1, 0, 0) at the FIFO; `psum` is (0, 1, 0).
- `par_tflags` 0x80, `par_fifo` 32, `cfg1` 0x14F, `sync2` 0xB6, `sync3` = `sync4` = 0x80.
- A partial last fill needs these fields:
  - `rsv174 = 0x8000 | (r-1)`;
  - `in_twait = 0x2000`;
  - `rsv536 = (32-r)<<22 | 1<<18 | 3`;
  - `par_fifo |= 0x2000`;
  - `rsv1230 = (32-r)<<18 | 1<<14 | 3`;
  - `rsv1523 = (r-1)<<9 | 1<<24`.

Only r = 8 (K = 288) and r = 32 (K = 768) have been seen. Other r raise `NotImplementedError`.

## 4. argmax: the pooled conv

### 4.1 Structure

The program is fully unrolled, about 7400 instructions:

```
prologue; identity row (narrow C = 172800 -> wide 8316)
chunk c = 0..9 (image rows 16c..16c+15):
  input_head, wn_fence, first copy op, input DMA 921600 B at 921600c, sync_w2n, input w2n x4, (rCons, infeed) x4, drain,
  last copy op                                        codegen_conv's input path for a 16 x 200 x 288 image, Stg = 57600
  reset17 x2, stash: narrowToWide of the tile's chunk block (4 rows x 50 positions, 57600 B) -> wide 7416 (225 rows), reset17
  13 steps (step s >= 1 first: reset17, drain, restore wideToNarrow 7416 -> X_s, drain, reset17 x2)
  output phase (13 output steps), output DMA (50*Mp bytes at 50*Mp*c), [reset17]
epilogue
```

All 10 chunks are identical apart from three things:
- the host DMA offsets, which sit in the first add64 of each descriptor;
- the output FIFO: 8308 while the identity is still needed, 8312 in the last chunk;
- the end of the last chunk.

`av_infeed` here needs multi-pass corrections, because the per-row infeeds are 230400 bytes:
- d1_limit = passes - 1, plus the count of the last pass;
- `buf_off` mod 2^15 units;
- `pop_wait` mod 2^16.

### 4.2 A step: 4 windows of 8x8 (the last step: 1 window of 8 columns)

Step s covers image columns 16s..16s+15. Tile column c' receives the 4 columns 16s+4c'.. (2 columns in step 12) as the
conv input X2: 4 rows x 4 positions. Image column j lives on tile column j // 50, local column j % 50.

1. **Input moves** (`am_transfers`): one transfer per (destination column, source column) pair.
   - Order: west moves first, sorted by (src, dst). Then east moves, sorted by (-src, -dst). Then the local copies, in
     column order.
   - Each move is a sender (`o_addr = X + 288 lc`, dims [1, 72, 3600] x [72, n, 4], sdims 7, out_mode 3, rsv475), relays
     on the columns in between, and a receiver (`i_addr = X2 + 288 doff`, dims [1, 72, 72 wpt], in_mode 7).
   - Every move is followed by a fence:
     - mask: all tiles except the relays;
     - counters: west MESH_EAST_IN|MESH_WEST_OUT with units bit 7; east MESH_WEST_IN|MESH_EAST_OUT with units bit 9;
     - value: a running count c per direction. A relayed move adds its packets (72 n), a direct move adds 1.
   - Relay records are (IN, 5+4c) and (OUT, -3+4c). This also explains codegen.md's west relays (9 + 4p): their first
     group is direct.
2. **Local copy** (dp_mode 4, cfg0 0, through the identity):
   - The `in` TTU walks words, positions and rows in dims 0, 2 and 4.
   - Each of these dims is followed by a count-1 rewind dim; dims 6/7 repeat 4/5.
   - These are the "7-dim" reformat ops of eltwise.md 10.6.
3. **Conv**:
   - Preceded by reset17, drain, scsync_nop, drain.
   - codegen_conv's ringConsumer1 and K-chunked op (a=12, b=6), with channel groups of cg = min(Mp, 64). Rows of a
     group are cg*4 bytes; the bias store holds cg/2 units per group.
   - The op writes the window layout. The even tile columns (0x5555) write 8-wide window rows (cfg0 0xE7, out_tflags 3).
     The odd ones (0xaaaa) write contiguously (0xE5, 1; contiguous position dims merged).
   - The channel-group dim (stride cg/4 words) stays even when G = 1.
   - Step 12 has one op per column with rows 2(4-c) positions wide.
   - Then one ringProducer per token-blob piece.
4. **Gather** onto the pool tiles 0, 2, 8, 10:
   - meshN: the south tiles send north; even columns go into window rows 4..7, odd ones behind their own block.
   - meshW: the odd columns send their 8 rows x 4 positions into the right halves.
   - Step 12 uses a westward chain 3 -> 2 -> 1 -> 0 onto pool tiles 0 and 8.
   - Mesh records: senders (0, 7); forwarders (0, 7)(10, 7); receive-and-forward (8, 7)(0, 7)(10, 7).
5. **Relay and pool**:
   - The relay is eltwise's `maxpool_relay_n2w_fields` (Rp = Mp/4, Rr = 2 Mp).
   - Mp <= 32: drain, relay, drain.
   - Mp >= 64: no drain before the relay. The relay itself carries six 42-bit sync records: (id 0), MESH_EAST_IN,
     MESH_WEST_IN, MESH_SOUTH_IN, MESH_NORTH_IN, end. All are at level 4, value 0x8000. The WEST and NORTH values are
     0x8000 / 0xFFFF / 0xFFFE for G = 1 / 2 / 4.
   - Then a MAX_POOL op with OH = OW = 1, Sy = 8, blk = 64; then reset17 x2, `scsync_wait_pb`, sync_rpb, fence.

### 4.3 Output phase and output DMA

Output step s handles the pooled results of tile column 0 (pooled column 2s) and column 2 (2s+1). Both tile rows 0 and 2
are handled by the same instructions.

- Pooled column p belongs to output column `tile_split(25)` = 7, 6, 6, 6, at position p - start.
- Moves: west first, then east (src descending); each move moves one position of Mp bytes. Then local copies; a drain
  with the last fence's mask when the step ends with a move.
- Each output step starts with reset17 x2.

Output DMA:
- reset17, sync_signal, fence;
- narrowToWide of the 7-position (0x0101) and 6-position (0x0e0e) blocks;
- one host DMA of 50*Mp bytes;
- (outfeed, ringProducer) for the tiles 0, 1, 2, 3, 8, 9, 10, 11, as ordinals 0..7;
- `output_wait(8)`.

### 4.4 Wide memory

- identity 8316;
- input FIFO 8316 - 8*225 = 6516 (`am_param_limit`);
- stash 7416..8316.

Per step:
- window `win = top - 64G`, with top = the stash (7416), or the identity in step 12 when the stash is dead;
- below it codegen_conv's regions (FIFO 96, psum 4*ppt, bias 8).

### 4.5 Narrow allocation (`am_alloc`)

The allocation is fixed by two numbers per Mp, `AM_ALLOC` = {16: (5, 0), 32: (7, 0), 64: (9, 0), 128: (10, 2),
256: (12, 6)}, read from the compiler. With p = Mp:

| buffer | address |
|---|---|
| X2_s (conv input) | 57600 + p min(s, k) |
| X_s (restored chunk block) | p min(s, b) |
| Y_s (conv output / window) | p min(s+1, b) (0 if b = 0) |
| R_s (input relay buffer) | 62208 + p s; step 12: X2_12 + 2304 (+ p if k <= 11) |
| P_s (pooled result) | first match of: p s if s < b; X2_s if s < k and s <= 11; R_s if s <= 10; X2_k + 2304 if s = 11; Y_12 + 64 p if s = 12 |
| O (output blocks) | p b |
| output relay of output step s | O + 7p + 32 s |

These reproduce every narrow address of all 5 Mp; k and b themselves are the compiler's choices.

### 4.6 Bitstreams and hints

`split_bitstreams`:
- The instruction stream is cut into groups that each start at a tile sync (0x1a).
- Groups are packed greedily into bitstreams of at most 16384 words, start and end included.
- A cut bitstream ends with halt (code 1), 4 nops and end. The next starts with start; seq numbers continue.
- The result is 16373 / 16362 / 951 words (Mp 16, 32) and 16378 / 16377 / 801 words (Mp 64..256).

`argmax_hints` / `program_hints`:
- Instruction chunk k is placed where bitstream k begins; then every host DMA in issue order; the interrupt is last.
- The result: `instr 0, in 0, out 0, ..., in 4, instr 1, out 4, in 5, ..., in 9, instr 2, out 9, interrupt`.
- Input chunk c is 921600 bytes at 921600c; output chunk c is 50*Mp bytes at 50*Mp*c.

### 4.7 Token blob (`argmax_params`)

`conv_blob(tokens [Mp, 288], zp)` with groups of cg = min(Mp, 64) tokens:
- an int32 bias row (4cg bytes, zeros), then [72][cg][4];
- rows M..Mp-1 padded with the token zero point;
- 73G rows of 4cg bytes: 4672 / 9344 / 18688 / 37376 / 74752 bytes.

Each row takes one wide row (256 bytes) at the parameter tile.

The caching program is the conv one, with infeeds of cg/16 lanes (`f438`). It runs before every execution, because the
tokens change every step.

## 5. Placement

**The rule** (codegen_conv.md 9):
- Every execution program uses the wide memory above its lowest buffer on every tile.
- Every resident parameter region must end at or below `set_limit` = the minimum of `program_limit` over the set.

| set | qkv / wo | ffn | argmax | set limit |
|---|---|---|---|---|
| Mp = 1 | 8048 (forward FIFO) | 8044 | - | 8044 units = 514816 B |
| Mp = 16..256 | 8212 .. 8152 | 8208 .. 8148 | 6516 | 6516 units = 417024 B |

**What the compiler does** (3a, `coral.fused.compiled`):
- Mp >= 16: the blobs are packed over all 16 tiles, the same for every Mp. The first ten (layers 0-2 and L3.qkv) sit at
  offset 0 on distinct tiles, the rest at 1460..4964 units.
- Big matmuls are split. The qkv pieces share an offset (15 and 13 at 3504); the FFN pieces each have their own, e.g.
  L4.ffn w1 = (7, 3504, 753 blocks) + (3, 3860, 123).
- The classifier sits at (10, 5108 units = 326912 B).
- Mp = 1: layers 0-3 spread (gen_fc / FFN spread form at per-matmul offsets), layers 4-5 streamed from single tiles.

**`plan(blocks, Mp)`**:
- Mp >= 16:
  - The token blob comes first, at tile 0 offset 0. This keeps 16/32-output rows below 256 KiB (sec. 9).
  - Room for the Mp=256 blob (292 rows) is reserved, so every layer has the same place for every Mp >= 16. A runtime
    can switch batch sizes without re-caching weights (3c).
  - Then the layers in order, every matmul blob laid out consecutively over the tiles' [0, 6516) and cut where a tile
    is full.
  - Pieces are `(tile, byte offset, blocks)`; an FFN has one list per matmul.
  - This fills tiles 0..14 to 6516 units, plus 1924 units of tile 15.
- Mp = 1:
  - As many blocks spread as fit, placed widest first so the 5-tile matmuls stack on tiles 0..4. That is all four of
    layers 0-3.
  - The rest go streamed, best fit decreasing, one tile per matmul.
  - High water per tile: 7760 on tiles 0..4, 7008..7592 on 5..12, 1168 on 13; tiles 14 and 15 stay free.

**`check_plan`** checks overlaps, the limit and 256-byte alignment. The test also scans every generated program and
finds no wide address below the limit outside its own parameters.

## 6. Runtime contract

`gen_set` returns, for every block, `coral.executable.Executable`s that `coral.runtime.run_executable` can run directly.
- Run the PARAMETER_CACHING program with `parameters=blob` once per resident block; the argmax caching program runs
  before every call.
- The hints follow the compiler's convention.
- The runtime splits them into segments: a segment's OUTs (instructions, inputs) start only after every earlier IN
  (output) completed.

| block | input (host -> TPU) | output (TPU -> host) | blob |
|---|---|---|---|
| conv / ffn, Mp >= 16 | x[H][W][K] uint8 (`GRIDS[Mp]`; position m = y*W + x), 8*ceil(Mp*K/8) B, one DMA | 16 tile blocks of ppt*N4 B, one DMA; `output_layout` = codegen_conv's (`block_io`) | `conv_blob` / `ffn_params` |
| conv / ffn, Mp = 1 | x[K], 8*ceil(K/8) B (pad with the input zero point) | y[N] contiguous, 8*ceil(N/8) B | same blobs |
| argmax | the quantized vocabulary [32000][288] = image [160][200][288], 10 DMAs of 921600 B | 10 DMAs of 50*Mp B (`argmax_output_layout`: pooled (y, x) at `tile_byte_offset[8y + x_tile[x]] + Mp*(x - start)`) | `argmax_params(tokens)`, every call |

Hints:
- execution of conv / ffn: `[instr 0, in x, out y, interrupt]`;
- argmax: sec. 4.6;
- caching: `[instr 0, in PARAMETER (blob bytes), interrupt]`.

## 7. Rules found here (beyond the decode reports)

1. The conv / FC clamps are `out_clamps` (float), not -zp / 255-zp. Also the MAX_POOL multiplier is the requantize one
   with equal scales, `f32(s * f32(1/s))`: 1.0 or 0.99999994. eltwise.md's constant 1.0 holds only for its test scales.
2. The conv multiplier rounds its inputs to f32 first.
3. Padded output rows of a blob are 0, not w_zp.
4. A matmul's pieces may each have their own offset: one rCons + infeed and one rProd per piece.
5. Mesh fence values and relay records are a running per-direction count: + packets per relayed group, + 1 per direct
   group. This also explains codegen.md's west relay bases 9 / 1: its first west group is direct. Its north phase
   does not count the direct group.
6. The 42-bit sync records of narrowToWide carry a level of 3 bits (`rsv554` = level bit 2).
7. Large infeeds (> 128 KiB): multi-pass fields, wrapped `buf_off` and `pop_wait`.
8. The streamed FC op's last-fill records (sec. 3).
9. Bitstream splitting at tile-sync groups, and the hint placement (sec. 4.6).

## 8. Not covered, open questions

- **Shapes:** D=288, Hd=768, V=32000, K=288, grid 160x200, pool 8 only. Other shapes raise `NotImplementedError`.
  The two allocation tables (2.2, 4.5) are the compiler's choices, not derived. The streamed FC only covers last fills
  of 8 or 32 rows.
- **The narrow allocator's ordering** (FFN) and the origin of (k, b) (argmax) are not understood.
- **Not modelled:** mixed FFN forms at Mp = 1 (some matmuls spread, some streamed), and splitting an argmax blob.
- **Semantics:** unknown for the mesh record values 7, for the reformat op's rewind dims, and for the streamed FC's
  record constants.

## 9. Risks for running on hardware

- **Placements**
  - With the compiler's placements our programs are byte-identical to the compiler's (3a), so they are exactly as
    trustworthy as those.
  - Our plan uses the same instruction forms at new addresses:
    - conv blocks with per-piece offsets (seen only in the compiler's FFNs);
    - FFN and streamed-FC pieces at our tiles and offsets;
    - the token blob at tile 0, offset 0.
- **Small channel groups.** The argmax at Mp = 16 / 32 uses 16 / 32-token groups (64 / 128-byte rows). The hardware
  rule "16-output groups read wrong weights at >= 256 KiB" was found for FC.
  - The compiler's own set puts the blob at 326912 B. That has never run: the runtime uses the on-chip argmax only for
    batch >= 64.
  - Our plan keeps it at 0.
- **Plans fill the tiles right up to the limit.**
  - The argmax's input FIFO (6516 units) caps every Mp >= 16 plan.
  - A buffer missing from `program_limit` would overwrite weights. 3b scans every generated program for wide addresses
    below the limit, but it sees start addresses, not extents.
- **Every call of the argmax re-runs its caching program** (the token blob), and the tokens must be quantized with
  the block's token quantization.
- **The argmax execution program** has 3 instruction chunks (16373..16378 words, just under 16384) and interleaves 10
  input and 10 output DMAs.
  - The runtime must send each chunk at its hint position.
  - It must keep the single-endpoint rule (no bulk OUT while an earlier bulk IN is pending), as `coral/runtime.py` does.
- **All programs of one plan share the wide memory.** Different plans must not be mixed. The batch-1 plan (FC forms,
  limit 8044) and the Mp >= 16 plan place things differently, so switching between them means re-caching.
