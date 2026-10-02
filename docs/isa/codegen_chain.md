# Whole-network chains as one program (`coral/codegen/chain.py`)

`gen_chain(layers)` builds the PARAMETER_CACHING and EXECUTION_ONLY programs of a whole CNN: a chain of VALID convolutions (fused
ReLU clamps), 2x2 / stride 2 max pools and a final linear layer. The activations stay in narrow memory between the layers; only the
input image goes in and the last layer's output comes out. The programs are byte-identical to what edgetpu_compiler makes of the
same TFLite model (`tools/export.tflite_chain`, or any chain of CONV_2D VALID / MAX_POOL_2D / FULLY_CONNECTED), including the memory
placement, which `default_alloc` reproduces without the compiler. Every instruction comes from the encoders in `coral/isa/` and the
builders of `coral/codegen/{conv2d,conv,fc}.py`; no compiler words are copied. Some rules were fitted, not understood (section 9).

```
python -m coral.codegen.chain                 # acceptance test (= python test/test_codegen_chain.py), offline
python -m coral.codegen.chain --n 80 --seed 1 # more random chains
python -m coral.codegen.chain --skip-mnist    # without tinygrad / the MNIST download
```

| API | |
|---|---|
| `gen_chain(layers, alloc=None)` | `(caching, execution)`; `alloc` defaults to `default_alloc(layers)` |
| `chain_blob(layers, params)` | the parameter blob: per Conv `conv2d_blob`, then the Linear's rows. `params = [(w_u8, bias_i32), ...]`: conv `w[Cout,kh,kw,Cin]`, linear `w[N,K]` with K in the map's (y, x, c) order |
| `chain_io(layers, images=1)` | the host contract (section 7) |
| `chain_layers(H, W, Cin, specs, images=1)` | the layer description from per-layer specs (section 2) |
| `chain_params(layers, weights)` | `chain_blob`'s params from NCHW-style weights (`w[Cout,Cin,kh,kw]`, linear `w[N,K]` with K in (c, y, x) order) |
| `check_chain(layers)` | raises `NotImplementedError` outside the covered set (section 8) |
| `default_alloc`, `narrow_alloc`, `wide_alloc`, `param_tiles`, `window_units` | the memory rules (section 6) |
| `gen_execution(layers, alloc)`, `gen_caching(layers, alloc)`, `Alloc` | the programs for a given placement |

## 1. Verification

Offline, everything against edgetpu_compiler 14.1 compiling the same TFLite model: both programs byte-exact, the parameter blob, the
DMA sizes of the hints, the executable's `output_layout` and the blob size. Nothing is taken from the compiler's output.

| check | result |
|---|---|
| MNIST (`examples/mnist.py`, `coral.quantize`, `tools/export.tflite_chain`), B = 1, 2, 4, 8, 16, 32 images per program, layers parsed from the TFLite model | 6/6 |
| the same, layers built with `chain_layers` / `chain_params` from the quantized model | 6/6 |
| random chains B = 1 (seed 0 `--n 40` + seed 1 `--n 60`): 1-4 convs k in 1/3/5, Cin 1..64, Cout 1..128, ReLU or not, pools, linear or not | 46/72, 28 excluded |
| random chains B = 2 / B = 4 (tall images, the linear a conv over each image's map), both seeds | 15/21 (4 excluded), 14/22 (3 excluded) |
| 60 hand-written chains (c3..c7, cpc, ccpc, cpcpc, c128pc, rt, mn1, f1..f7, g1..g9, cpcf, ...) | 56/57, 3 excluded (2x2 kernels, pool after pool); f6 fails (narrow placement) |
| `coral/codegen/conv2d.py` after the refactor (`--n 275`, 3300 shapes) | 3299/3300; the failure (39x51x1 -> 128, 3x3 s2 SAME, input stage) fails the same with the unmodified file |

Most random failures are narrow-memory placements (section 6.1); the others: the reused identity row's wide slot (6.2), the pool
relays' sync records and the parameter FIFO depth in some tall images, the input stage's placement. Exclusions: section 8 (of the
28 excluded B = 1 draws, 14 are layers the compiler streams, 9 maps with channels not a multiple of 4 before the linear layer).

**Hardware** (via `tools/hw.py`, programs and blob byte-identical to the compiler's): MNIST B = 1 (32 images), B = 8 (64 images),
B = 32 (128 images): output sizes as `chain_io` says, `chain_io`'s output layout read back, logits identical to tinygrad's
(`models/mnist_test.npz` "ours") for 224/224 images. Three device sessions, no failed transfer.

## 2. The layer description

`Conv(H, W, Cin, Cout, kh, kw, stride=1, q)`, `Pool(k=2, stride=2, q=(scale, zp))`, `Linear(N, K, q)`. `q` is `conv2d_quant`'s dict
(`w_zp, in_zp, out_zp, mult, clamp_min, clamp_max`): `mult = f32(f32(s_x * s_w) * f32(1/s_y))` and the clamps come from the scales
exactly as the compiler computes them, so the specs carry the TFLite float32 scales, not just the integer clamps. A pool keeps its
conv's output quantization. `images = B > 1`: B images stacked into one tall image (`tools/export.py`): VALID convs and 2x2 pools
never mix two images' rows as long as the image height is a multiple of 2^pools; the linear layer becomes a conv whose kernel is one
image's whole map, so output row `b * pitch` holds image b's result (`chain_io(...)["image_rows"]`).

```python
specs = [{"conv": dict(Cout=32, k=5, x_q=(sx, zx), w_q=(sw, zw), y_q=(sy, zy), act=1)}, {"conv": ...}, {"pool": dict(k=2, stride=2)},
         ..., {"linear": dict(N=10, x_q=..., w_q=..., y_q=..., act=0)}]
layers = chain_layers(28, 28, 1, specs, images=B)
params = chain_params(layers, [(w_nchw, bias), ...])
caching, execution = gen_chain(layers); blob = chain_blob(layers, params)
```

## 3. Program structure

1. **Input stage**: `conv2d.input_stage` for the first conv (identity row, staging slots, ring FIFO, copy ops into its input block),
   then a reset17.
2. **Per conv**: the halo exchange of its input block (section 4), reset17 fences (one after a pool, one more when the halo gathers
   the input onto a single tile row / column: `gathered_line`), then `conv_stage`: scsync, drain, ringConsumer1 (parameter FIFO),
   bias load (wideToNarrow with two sync records), one op per distinct tile, the parameter broadcast (ringProducer from the
   parameter tile). The op writes its outputs straight into the consumer's layout (`into_block`): the next conv's input block with
   the halo rows / columns left free (cfg0 0xE7 and a 4-level out TTU when the rows land inside wider block rows), or the window
   block of a max pool.
3. **Per max pool** (`pool_stage`): the halo exchange of the pool's spans (`pool_geom`: the conv2d spans of a k x k / s conv on the
   map), a transposing narrowToWide per tile into the wide window (`maxpool_relay_n2w_fields`), the MAX_POOL_2D op per tile writing
   into the next consumer's block. The relays wait on the gather's counters with their own sync records when Cout is a multiple of
   64, else after a drain.
4. **Last layer**: a conv: `conv2d.output_stage` (host DMA per tile, or the scalar-memory path when a tile block is not a multiple
   of 8 bytes). A linear layer at B = 1: `fc_tail` (section 5) then `fc.py`'s gather, op and output block.

## 4. Sync counters and records

`State` tracks per tile the counters since the last reset17: AV (op completions) and the four mesh IN counters. Rules (all
observed):
- a conv op counts once per 64-output group, a pool op once per channel group; records that wait for an op use the count once its
  first group is done (`av_done`);
- a mesh receive counts once per 16-word piece of a position (`cdiv(cw, 16)`), a record waits for the first piece;
- an outbound half waits on: the relayed IN counter (relays), AV when an op wrote the block since the reset (own data), and, for
  horizontal moves, the vertical IN counters; values are `(count << 2) | levels` with the 16-word pieces level not counted; more
  than five records spill the end marker into a sixth slot;
- inbound halves of north / west moves also wait for the op on tiles whose conv computed fewer outputs than the busiest tile
  (`inrec_after`), not after pools;
- the bias load waits for parameter fill `1 + R` (R = fills of the convs since the reset) and op count `AV - 1`; the op's
  progress count is `1 + R`, `sync2 = 0xC6 + 0x80 * AV`.

## 5. The final linear layer (B = 1)

`fc_tail` turns the last map (P x Q x C, dense per-tile blocks) into the FC's input layout (input tile t holds `x[bt, bt+b)`):
1. each map tile row's column-0 tile copies its own rows into the row buffer A (copy op through the identity row), the other
   columns' rows arrive by westward moves (`_phase`: senders, relays grouped by word count, receivers, mesh fence per step);
2. FC input row R's column-0 tile assembles whole map rows covering the row (`_fc_rows`): the tail of the tile row above when the
   FC row starts there (southward, first), its own rows, the head of the following tile rows (northward, per FC row from the last:
   moves over 2+ tile rows first, relayed by the column-0 tiles between; records and fences: section 9);
3. tile (R, 0) copies its chunk 4R into place, chunks 4R+1.. go east; then the gather onto tile 0, two reset17s, the FC op, output.
The identity row is the input stage's when that is one 256-byte row (`reuses_ident`), else rebuilt at WIDE_TOP - 4 before steps
1 and 3 (`prologue`). Fences before the tail: one, one after a pool, one more when the map leaves a tile row or column empty.

## 6. Memory (`default_alloc`)

Time is counted in steps: the input stage, one per conv, one per pool, the FC reshape and its steps. A buffer is live over
`[lo, hi]`.

### 6.1 Narrow (bytes, the same on every tile)

Buffers: the first conv's staging slots Stg, identity row C and input block X0; each conv's input block; each pool's window block
P (its size: the relays' reach, `_relay_extent`; the objective counts the block's own size); the last map Y; with a linear layer
the row buffer A, relay buffer R1 and the FC's x / y / relay buffers (`fc_narrow`: x first, `[x | y | R]`).
- X0 / Stg / C: `conv2d.narrow_layout`'s four cases with Ys = the first conv's output buffer; when that buffer does not end up at
  address 0, the `[X0 | Stg | C]` case.
- Conv-only chains: the lowest peak, then the lowest sum of size x address, over first-fit placements in every order (branch and
  bound, the first optimum in buffer order). This reproduces every MNIST batch and ~85 % of 94 chains; the rest follow a plain
  largest-first greedy instead and no single rule found reproduces both (the compiler's "MemoryAllocator Integer Programming
  Solver"; its objective is not known).
- With a linear layer: largest first (equal sizes: the one that fits lowest first), then A and Y from the FC's y address (A first
  when an FC row takes rows from the tile rows below and none from above, else Y first; Y only where it is free, else after A,
  else at its first fit), R1 right after them, the rebuilt identity row 32 bytes after R1.

### 6.2 Wide (64-byte units)

Greedy by falling key, each buffer at the highest address below WIDE_TOP clear of the buffers it is live with. Keys: the conv
regions' sizes (psum `4 * ppt` when K-chunked, allocated even when one position per tile leaves it unused, bias 8 / 4, parameter
FIFO `8a` / block `4a`;
ties psum, bias, FIFO), the input ring FIFO and identity row (ties: identity row first), and a pool window's 4 x its size. A window
(`window_units`: 4 x the largest narrow block's rows x ceil(its columns / 4), per 64-channel group; more than the relays write) is
live during its conv and its pool. A reused identity row lives the whole program: with 3+ layers before the linear one it goes
first, to the very top, unless a window's key exceeds 128; otherwise its key is 48.

### 6.3 Parameter tiles

One layer per tile, offset 0, tiles handed out in the order 0, 1, 3, 7, 15, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14: the linear layer
first (tile 0, where its op runs), then convs with at most 4 outputs, then by falling parameter rows (`blocks` = groups x (1 +
reduction words)), ties in layer order. 17+ layers share tiles (not modelled).

## 7. Host contract (`chain_io`)

- input: `input_bytes` = the image `x[H][W][Cin]` uint8 NHWC (B images stacked: `[B*H][W][Cin]`; for Cin = 1 that is the NCHW
  batch byte for byte), zero-padded to a multiple of 8;
- output: a final linear layer: `y[N]` in the first N of `output_bytes = 8 * ceil(N / 8)` bytes (`output_layout`: one position on
  tile 0); a final conv: the 16 tile blocks of `[OH][OW][Cout]` as `conv2d_io` describes (`output_layout`, read by
  `coral.fused.relayout` or by hand), `image_rows[b]` the row holding image b's result;
- `param_bytes`: `chain_blob`'s size; the caching program loads it, each layer's part on its parameter tile.

## 8. Not covered (`NotImplementedError`)

- strided or even-kernel convolutions in a chain (2x2 kernels: conv2d's mode rule for them is not modelled), kernels over 5x5 (a
  tall image's linear layer over a bigger map: the compiler orders that blob differently); padding;
- convs with at most 4 outputs (the compiler packs 4-output parameter groups; also not in conv2d.py);
- pools other than 2x2 / stride 2, a pool right after a pool, a chain ending with a pool;
- a final linear layer with more than 64 outputs, or one of whose input rows starts two or more tile rows above its own (relayed
  southward moves, not observed), or after a map whose channels are not a multiple of 4;
- more than 16 layers with parameters; programs over one bitstream; parameters the compiler streams instead of caching (large
  linear layers);
- batch > 1 any other way than tall images.

## 9. Copied as observed (fitted, not understood)

- the reset17 counts (after pools, `gathered_line`, before the FC tail), the AV / IN counting per group / per 16-word piece and the
  first-piece record values, `inrec_after`, the pool relays' record list and when they use records instead of a drain;
- the FC reshape's moves: south before north, north per FC row from the last with the moves over 2+ tile rows first; their
  relays' IN record and fence count the identity row's fill rounds, after fills the fence covers all tiles (three chains); the
  second prologue waits for `1 + max(1, packets)`; tile columns whose own columns no window needs move nothing;
- the narrow allocator's objective (6.1), the wide window size and the 4x key, the identity row's top rule (6.2), the parameter
  tile order and the 4-output rule (6.3);
- the scalar-memory output path of `conv2d.output_stage` (0x23 and vector-slot ALU words, as in conv.py): MNIST at B >= 4 uses it.

## 10. Risks for hardware

Byte-identical programs run like the compiler's own (MNIST B = 1, 8, 32 verified). A program with a wrong placement or record would
not be byte-identical, and `tools/hw.py` refuses it. Through the backend the chain program runs `examples/mnist.py` (32 images per
call, logits identical to tinygrad's, `bench/RESULTS.md`).
