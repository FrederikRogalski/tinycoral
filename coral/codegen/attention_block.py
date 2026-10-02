# The attention block of a TinyStories-15M layer at batch 1 as ONE Edge TPU program (docs/isa/codegen_attention_block.md):
#   caching, execution, io = gen_attention_block(P, quant)
#     x[288] (RMS-normalized by the host), cos / sin[288] of position P-1, K / V cache rows 0..P-2
#     -> q, swap(q), k, swap(k), v = five FULLY_CONNECTED 288 -> 288 from wqkv (the pair rotation of RoPE folded into the weights)
#     -> q' = q*cos + swap(q)*sin', k' likewise (MUL, MUL, ADD in edgetpu_compiler's 1-D form on tiles 0..4)
#     -> k', v written into row P-1 of the K / V blocks on the chip, attention over P positions (coral/codegen/attention.py's core)
#     -> o = wo(attention)
#     outputs: o (288 bytes) and k' | v (576 bytes, for the host's KV cache)
#   blob = block_params(wqkv, wo, quant)              the PARAMETER_CACHING blob (fc.py's layouts, six 288 x 288 matmuls)
#   o, k, v = attention_block_ref(x, cos, sin, K, V, wqkv, wo, quant)    the numpy bit model
# `python -m coral.codegen.attention_block` runs the acceptance test (test/test_codegen_attention_block.py), offline.
from __future__ import annotations
import numpy as np
from coral.isa import cdiv, round_up, f32, op as OP, wide_narrow as WN, ring_mesh as RM, scalar as SC, eltwise as EL
from coral.codegen import Emitter, caching_program, run_test, WIDE_TOP, WIDE_OUT_FIFO
from coral.codegen import attention as A, fc as FC
from coral.codegen.fc import rprod_output

NH, DH, DM, HW, ROWS, HEAD0 = A.NH, A.DH, A.DM, A.HW, A.ROWS, A.HEAD0
V1 = FC.FCGeom(DM, DM)          # every matmul of the block (q, swap(q), k, swap(k), v, wo) and every [288] vector: tile t = 0..4
T1 = V1.T                       # holds bytes [64t, 64t + 64) at base + 64t (edgetpu_compiler's 1-D layout, its FC output layout)
PARTS = [(1 << t, EL.tile_geometry([1, DM], t), 64 * t) for t in range(T1)]   # (tile mask, eltwise geometry, byte offset) per tile
SWAP = np.arange(DM) ^ 1        # RoPE's pairs (2j, 2j+1): swap(q)[2j] = q[2j+1], swap(q)[2j+1] = q[2j]
MATS = ("q", "sq", "k", "sk", "v", "wo")   # the six matmuls in blob / wide-memory order

# ***** quantization ((scale, zero point) per tensor) *****
DEFAULT_BLOCK = dict(x=(1/16, 128), wqkv=(1/64, 128), wo=(1/64, 128), qf=(1/16, 128), kf=(1/16, 128), v=(1/16, 128),
                     cos=(1/127, 128), sin=(1/127, 128), qc=(1/16, 128), kc=(1/16, 128), q=(1/16, 128), k=(1/16, 128),
                     qk=(1/16, 128), s=(1/4, 128), p=(1/256, 0), pv=(1/256, 128), att=(1/16, 128), o=(1/16, 128), beta=DH ** -0.5)

def block_quant(quant:dict|None=None) -> dict:
  """the full quantization of the block: DEFAULT_BLOCK, overridden by quant; qs / ks (the swap(q)*sin' and swap(k)*sin' products)
  default to qc / kc (then each ADD has the integer weights (1, 1): an exact sum, one rounding)"""
  q = DEFAULT_BLOCK | (quant or {})
  return q | dict(qs=q.get("qs", q["qc"]), ks=q.get("ks", q["kc"]))

def core_quant(q:dict) -> dict:
  """the attention core's quantization (attention.gen_attention's keys) inside the block"""
  return {k: q[k] for k in ("q", "k", "qk", "s", "p", "v", "pv", "beta")} | dict(o=q["att"])

def fc_quant(x_q:tuple, w_q:tuple, y_q:tuple) -> dict:
  """FULLY_CONNECTED op fields: mult f32(f32(s_x s_w) f32(1/s_y)), float32 clamps (codegen_fused.md 2.4)"""
  lo, hi = EL.out_clamps(y_q[0], y_q[1])
  return dict(w_zp=w_q[1], in_zp=x_q[1], out_zp=y_q[1], mult=EL.mul32(EL.mul32(x_q[0], w_q[0]), EL.recip32(y_q[0])), clamp_min=lo, clamp_max=hi)

def matmul_quant(q:dict) -> dict[str, dict]:
  """the FC op quantization of each of the six matmuls"""
  return dict(q=fc_quant(q["x"], q["wqkv"], q["qf"]), sq=fc_quant(q["x"], q["wqkv"], q["qf"]), k=fc_quant(q["x"], q["wqkv"], q["kf"]),
              sk=fc_quant(q["x"], q["wqkv"], q["kf"]), v=fc_quant(q["x"], q["wqkv"], q["v"]), wo=fc_quant(q["att"], q["wo"], q["o"]))

# ***** parameters: six FULLY_CONNECTED 288 -> 288 blobs (fc.py / fused.conv_blob layout), on tiles 0..4 one after the other *****
def block_matrices(wqkv:np.ndarray, wo:np.ndarray) -> dict[str, np.ndarray]:
  """wqkv [864, 288] (rows q | k | v) and wo [288, 288] uint8 -> the six weight matrices; swap(q) / swap(k) are q's / k's rows with
  every pair (2j, 2j+1) exchanged, so the FC computes them exactly (same weights, same quantization)"""
  wqkv, wo = np.asarray(wqkv, np.uint8), np.asarray(wo, np.uint8)
  assert wqkv.shape == (3 * DM, DM) and wo.shape == (DM, DM)
  Wq, Wk, Wv = wqkv[:DM], wqkv[DM:2 * DM], wqkv[2 * DM:]
  return dict(q=Wq, sq=Wq[SWAP], k=Wk, sk=Wk[SWAP], v=Wv, wo=wo)

def block_params(wqkv:np.ndarray, wo:np.ndarray, quant:dict|None=None) -> bytes:
  """the parameter blob of block_caching: conv_blob of q, swap(q), k, swap(k), v (weight zero point wqkv's) and wo, no bias"""
  from coral.codegen.fused import conv_blob
  q, W = block_quant(quant), block_matrices(wqkv, wo)
  return b"".join(conv_blob(W[m], q["wo" if m == "wo" else "wqkv"][1]) for m in MATS)

def param_offsets(param_offset:int=0) -> dict[str, int]:
  """wide-memory offset (64-byte units) of each matmul's region on tiles 0..4, from param_offset bytes (a multiple of 256)"""
  assert param_offset % 256 == 0 and param_offset >= 0, "param_offset must be a non-negative multiple of 256 bytes"
  per = V1.param_bytes // 64                       # 1460 units = 93440 bytes: 5 tiles x 292 units per matmul
  return {m: param_offset // 64 + i * (per // T1) for i, m in enumerate(MATS)}

def block_caching(param_offset:int=0) -> bytes:
  """PARAMETER_CACHING of the six blobs (as edgetpu_compiler caches several FCs: per blob its ringConsumers, then its infeeds)"""
  offs = param_offsets(param_offset)
  return caching_program(len(MATS) * V1.param_bytes, [FC.caching_part(V1, FC.Place(), offs[m], i * V1.param_bytes) for i, m in enumerate(MATS)])

def param_end(param_offset:int=0) -> int:
  """first wide unit above the block's parameters (on tiles 0..4)"""
  return param_offsets(param_offset)["wo"] + V1.param_bytes // 64 // T1

# ***** memory *****
def narrow_sizes(g:A.Geom) -> dict:
  """narrow buffers (bytes, every tile), in address order: the 1-D vectors take 320 bytes (tile t uses [64t, 64t + 64)), the attention
  core's buffers are attention.narrow_sizes'. Q holds q' | k' | v (on tile 0 and, after the broadcast, on every tile with positions);
  QC / KC hold the products q*cos | swap(q)*sin' (k the same) 288 bytes apart (the ADD's operand distance); O the attention output
  (tile 0 gathers all 288 bytes there)"""
  core = A.narrow_sizes(g)
  return dict(Stg=core["Stg"], C=16, X=320, R=64, Xc=320, Xs=320, Yq=320, Ysq=320, Yk=320, Ysk=320, Mr=256, QC=576, KC=576, Wm=64,
              Q=3 * DM, **{k: core[k] for k in ("XK", "XV", "Ms", "Ys", "S", "Pn", "Mv", "Yv")}, O=DM, Yo=320)

def block_alloc(g:A.Geom) -> dict:
  """our memory plan, nothing that is live at the same time shares memory. Narrow: narrow_sizes in order, 64-byte aligned. Wide (64-byte
  units, from the top): identity row, output FIFO, (2 rows unused), K / V input FIFO, MUL FIFO, SUM FIFO, scalar-memory FIFO, p's
  two FIFOs, p.V relay block, ADD weight rows, the q'|k'|v, o and x broadcast FIFOs. The three 1-D inputs (x, cos, sin) come first
  and use the ring FIFO at [8304, 8320) before the identity row is loaded there. Scalar memory: the score words [6][P], then p's
  words [6][P] at the top (attention.attention_alloc)."""
  P = g.P
  a, at = {}, 0
  for name, size in narrow_sizes(g).items():
    a[name] = at
    at += round_up(size, 64)
  a["narrow_end"] = at
  wt = WIDE_TOP
  for name, size in [("const", 4), ("outfifo", 4), ("unused", 8), ("in_kv", 8 * A.g4_of(P).c_in), ("mulfifo", 32), ("sumfifo", 16), ("smfifo", 8),
                     ("pfifo", 4), ("bfifo", 4), ("relay", 4 * 2 * cdiv(P, 4)), ("addw", 16), ("qkvfifo", 4), ("ofifo", 4), ("xfifo", 4)]:
    wt -= size
    a[name] = wt
  a["wide_bottom"] = wt
  a["M"] = P
  a["p_smem_w"] = A.SMEM_TOP // 4 - 6 * P
  a["s_smem_w"] = a["p_smem_w"] - 6 * P
  assert a["const"] == WIDE_TOP - 4 and a["outfifo"] == WIDE_OUT_FIFO and at <= 192 * 1024 and a["s_smem_w"] >= 0
  return a

# ***** stages *****
def x_input(e:Emitter, X:int, R:int, fifo:int):
  """x [288]: fc.input_block (host DMA, ring scatter to the input tiles 0..4, mesh gather to tile 0, ring broadcast to tiles 1..4)
  with the broadcast through our own wide FIFO"""
  g, pl = V1, FC.Place()
  e.sync(SC.input_head)
  e.sync(SC.sync_wn_fence)
  e.scalar(SC.input_dma(g.S))
  for t in range(g.T_in): e.tile(WN.w2n_input(g.K, t, X // 4, e.seq, pl.i(t)))
  for j in range(cdiv(g.T_in, 4)):
    ts = sum(pl.i(t) for t in range(4 * j, min(4 * j + 4, g.T_in)))
    e.tile(FC._rcons_input(g, j, ts, e.seq))
    e.scalar(SC.av_infeed(4 * j * g.b, min(4 * g.b, g.S - 4 * j * g.b), ts))
  e.sync(SC.sync_drain)
  e.sync(SC.sync_reset17)
  FC.gather(e, g, X, R, pl.i)
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_reset17)
  FC.broadcast(e, g, X, pl.i(0), pl.cspan(0, g.T) & ~pl.i(0), fifo)
  e.sync(SC.sync_reset17)

def scatter_input(e:Emitter, X:int, K:int=DM, top:int=WIDE_TOP):
  """a [K] input in edgetpu_compiler's 1-D layout: host DMA, ring multicast of 256-byte packets, each tile t < T_in takes its chunk
  t (64 bytes for K <= 1024) to narrow X + 64t; FULLY_CONNECTED's input block without its gather and broadcast. top: the ring FIFO
  ends there (the compiler's inputs after the identity row was loaded use WIDE_TOP - 4)"""
  g = FC.FCGeom(64, K)
  c = cdiv(g.S, 256)
  e.sync(SC.input_head)
  e.sync(SC.sync_wn_fence)
  e.scalar(SC.input_dma(g.S))
  for t in range(g.T_in):
    f = WN.decode_wide_to_narrow(WN.w2n_input(g.K, t, X // 4, e.seq, 1 << t))
    e.tile(WN.encode_wide_to_narrow(**(f | dict(wide_addr=f["wide_addr"] - (WIDE_TOP - top)))))
  for j in range(cdiv(g.T_in, 4)):
    ts = sum(1 << t for t in range(4 * j, min(4 * j + 4, g.T_in)))
    f = RM.decode_ringConsumer(FC._rcons_input(g, j, ts, e.seq))
    assert f["addr"] == WIDE_TOP - 8 * c
    e.tile(RM.encode_ringConsumer(**(f | dict(addr=top - 8 * c))))
    e.scalar(SC.av_infeed(4 * j * g.b, min(4 * g.b, g.S - 4 * j * g.b), ts))
  e.sync(SC.sync_drain)
  e.sync(SC.sync_reset17)

def mul_1d(e:Emitter, a:int, b:int, tmp:int, out:int, fifo:int, ident:int, q:dict, ident_narrow:int|None=None):
  """out = MUL(a, b) on [288] vectors in the 1-D layout, edgetpu_compiler's form (fused._mul_stage): per tile step 1 (a through the
  in TTU, b through the FIFO, into the tile-local intermediate tmp), the FIFO feeds, step 2 (pack into out). ident_narrow: load the
  identity row (narrow staging -> wide ident) between the feeds and step 2, as the compiler does in a program's first MUL (then the
  NARROW_TO_WIDE fence after it carries the count 1)"""
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_reset_n2w)
  e.sync(SC.sync_reset_par)
  def geo(tg): return (tg["w"], tg["cols"], tg["rows"])
  for m, tg, o in PARTS: e.tile(OP.encode_op(**EL.mul_op1_fields(*geo(tg), tg["R"], m, e.seq, a + o, tmp, fifo, **q)))
  for m, tg, o in PARTS: e.tile(WN.encode_narrow_to_wide(**EL.mul_feed_n2w_fields(e.seq, m, b + o, fifo, *geo(tg))))
  if ident_narrow is not None: e.tile(*EL.ident_prologue(e.seq, 0xffff, ident_narrow, ident))
  e.sync(SC.sync_wn_fence)
  for m, tg, o in PARTS: e.tile(OP.encode_op(**EL.mul_op2_fields(*geo(tg), tg["R"], m, e.seq, tmp, out + o, ident)))
  e.sync(SC.sync, counters=SC.tc("NARROW_TO_WIDE"), count=int(ident_narrow is not None))
  e.sync(SC.sync_reset_par)
  e.sync(SC.sync_reset17)

def add_1d(e:Emitter, a:int, b:int, out:int, wn:int, ww:int, q:dict):
  """out = ADD(a, b) on [288] vectors in the 1-D layout, edgetpu_compiler's form: two reset17, the 16-bit weight matrix (w1, w2) by
  10 mesh fills at narrow wn on tiles 0..4 and a narrowToWide to 4 wide rows at ww, one ADD op per tile (operand 1 at a, operand 2
  at b = a + 4 delta), reset17. q: eltwise.add_quant(a_q, b_q, out_q)"""
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_reset17)
  tiles = sum(m for m, _, _ in PARTS)
  e.tile(*EL.add_weight_fills(e.seq, tiles, wn, q["w1"], q["w2"]))
  e.tile(WN.encode_narrow_to_wide(**EL.add_weight_n2w_fields(e.seq, tiles, wn, ww)))
  assert (b - a) % 4 == 0
  for m, tg, o in PARTS:
    e.tile(OP.encode_op(**EL.add_op_fields(tg["w"], tg["cols"], tg["rows"], tg["R"], (b - a) // 4, tg["R"], m, e.seq, a + o, out + o, ww,
                                           q["offset"], q["mult"], q["out_zp"], q["clamp_min"], q["clamp_max"])))
  e.sync(SC.sync_reset17)

def gather_north(e:Emitter, O:int, R:int):
  """the attention output (tile row r: ROWS[r] heads x 48 bytes at narrow O on its column-0 tile) to tile 0 as o[288] at O: fc.gather's
  north phase with the head rows as pieces (tile 4r sends from O, tile 0 receives at O + 48 HEAD0[r], tiles 4.. relay through R)"""
  e.sync(SC.sync_reset17)
  cum, last = 0, 0xffff
  for k in (1, 2, 3):
    n = HW * ROWS[k]
    e.tile(FC._mesh(0x17, 1 << (4 * k), e.seq, out=(O, n), mode=0))
    relays = sum(1 << (4 * j) for j in range(1, k))
    if k >= 2: e.tile(FC.relay(0x17, relays, e.seq, R, n, cum))
    e.tile(FC._mesh(0x17, 1, e.seq, inn=(O + DH * HEAD0[k], n), mode=0))
    if k >= 2:
      cum += cdiv(n, 4)
      last = 0xffff & ~relays
      e.sync(SC.sync_mesh, 0x17, last, cum)
  e.sync(SC.sync_drain, last)
  e.sync(SC.sync_reset_mesh)

def slot_copies(e:Emitter, g:A.Geom, src:int, X:int):
  """the new row (6 heads x 48 bytes at narrow src, on every tile) into position P-1 of a [rows][cols][48] block at X: one reformat copy
  through the identity row on each tile of the last tile column that holds positions (attention.q_broadcast's copies)"""
  cl = max(c for c in range(4) if g.cols[c])
  j, s = g.P - 1 - g.p0(cl), g.cols[cl]
  per = [A.chain_copy(src + DH * HEAD0[t // 4], X + DH * j, HW, 1, ROWS[t // 4], [HW, HW, HW * ROWS[t // 4]], [HW, HW * s, HW * s * ROWS[t // 4]])
         if t % 4 == cl else None for t in range(16)]
  for m, f in A._group(per): e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))

def output_tile0(e:Emitter, Y:int, n:int):
  """n bytes at narrow Y of tile 0 to the host: one output DMA (fence, narrowToWide to the output FIFO, outfeed + ringProducer)"""
  e.sync(SC.signal_fence)
  e.scalar(SC.output_dma_head())
  e.tile(WN.n2w_output(n, 1, Y // 4, WIDE_OUT_FIFO, e.seq))
  e.scalar(SC.output_dma_tail(round_up(n, 8)))
  e.scalar(SC.outfeed(round_up(n, 8)))
  e.tile(rprod_output(n, 1, 0, e.seq))
  e.scalar(SC.output_wait(e.seq, 1), seqs=2)

# ***** the program *****
def gen_attention_block(P:int, quant:dict|None=None, param_offset:int=0) -> tuple[bytes, bytes, dict]:
  """(PARAMETER_CACHING, EXECUTION_ONLY, host contract block_io(P)) of the attention block of one layer for P = 1..256 positions (the
  new token at position P-1). quant: block_quant's keys. param_offset: bytes into wide memory of tiles 0..4 where the six blobs
  start (block_params; they end at param_end(param_offset), below block_alloc(Geom(P))["wide_bottom"])."""
  if not 1 <= P <= 256: raise NotImplementedError("P must be 1..256 (the scalar memory holds 2 x 6 x 256 words)")
  q, g = block_quant(quant), A.Geom(P)
  a, offs, mq = block_alloc(g), param_offsets(param_offset), matmul_quant(block_quant(quant))
  assert param_end(param_offset) <= a["wide_bottom"], "the parameters reach into the program's wide buffers"
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  # inputs: x (FULLY_CONNECTED's input block: gathered on tile 0, broadcast to tiles 1..4), cos and sin' (1-D layout on tiles 0..4),
  # then K and V (the attention core's input path, the identity row first); P = 1 has no cache rows, only the identity row
  x_input(e, a["X"], a["R"], a["xfifo"])
  scatter_input(e, a["Xc"])
  scatter_input(e, a["Xs"])
  if P > 1:
    A.input4d(e, P, a["Stg"], a["XK"], a["C"], a["const"], a["in_kv"], True)
    e.sync(SC.sync_reset17)
    A.input4d(e, P, a["Stg"], a["XV"], a["C"], a["const"], a["in_kv"], False)
    e.sync(SC.sync_reset17)
  else:
    e.tile(*EL.ident_prologue(e.seq, 0xffff, a["C"], a["const"]))
    e.sync(SC.sync_reset17)
  # q, swap(q), k, swap(k), v: five FULLY_CONNECTED 288 -> 288 on tiles 0..4 (outputs in the 1-D layout; v straight into Q + 576)
  for m, Y in (("q", a["Yq"]), ("sq", a["Ysq"]), ("k", a["Yk"]), ("sk", a["Ysk"]), ("v", a["Q"] + 2 * DM)):
    FC.ops_block(e, V1, FC.Place(), a["X"], Y, offs[m], mq[m], tail=False)
  # RoPE: q' = q*cos + swap(q)*sin' (into Q), k' = k*cos + swap(k)*sin' (into Q + 288), per tile on its 64-byte slices
  mul_1d(e, a["Yq"], a["Xc"], a["Mr"], a["QC"], a["mulfifo"], a["const"], EL.mul_quant(q["qf"], q["cos"], q["qc"]))
  mul_1d(e, a["Ysq"], a["Xs"], a["Mr"], a["QC"] + DM, a["mulfifo"], a["const"], EL.mul_quant(q["qf"], q["sin"], q["qs"]))
  add_1d(e, a["QC"], a["QC"] + DM, a["Q"], a["Wm"], a["addw"], EL.add_quant(q["qc"], q["qs"], q["q"]))
  mul_1d(e, a["Yk"], a["Xc"], a["Mr"], a["KC"], a["mulfifo"], a["const"], EL.mul_quant(q["kf"], q["cos"], q["kc"]))
  mul_1d(e, a["Ysk"], a["Xs"], a["Mr"], a["KC"] + DM, a["mulfifo"], a["const"], EL.mul_quant(q["kf"], q["sin"], q["ks"]))
  add_1d(e, a["KC"], a["KC"] + DM, a["Q"] + DM, a["Wm"], a["addw"], EL.add_quant(q["kc"], q["ks"], q["k"]))
  # q' | k' | v onto tile 0 (three FULLY_CONNECTED input gathers), then to every tile that holds positions (fc.py's ring broadcast)
  for k in range(3): FC.gather(e, V1, a["Q"] + DM * k, a["R"], lambda t: 1 << t)
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_reset17)
  if dest := sum(1 << t for t in range(1, 16) if g.cols[t % 4]): FC.broadcast(e, FC.FCGeom(64, 3 * DM), a["Q"], 1, dest, a["qkvfifo"])
  e.sync(SC.sync_reset17)
  # k' and v into row P-1 of the K and V blocks
  slot_copies(e, g, a["Q"] + DM, a["XK"])
  slot_copies(e, g, a["Q"] + 2 * DM, a["XV"])
  e.sync(SC.sync_reset17)
  # the attention core (attention.gen_attention from the scores on; q_h read from Q + 48 h on every tile)
  cq = core_quant(q)
  def q_bcast(t): return (a["Q"] + DH * HEAD0[t // 4], [1, 0, DH, DH * ROWS[t // 4]], [DH, g.cols[t % 4], ROWS[t // 4]])
  A.mul_stage(e, g, q_bcast, a["XK"], a["Ms"], a["Ys"], a["mulfifo"], a["const"], EL.mul_quant(cq["q"], cq["k"], cq["qk"]),
              lambda t: HW * g.cols[t % 4])
  A.chansum_stage(e, g, a["Ys"], a["S"], a["sumfifo"], A.sum_quant(cq["qk"], cq["s"]))
  A.tiles_to_smem(e, g, a["S"], a["s_smem_w"], a["M"], a["smfifo"])
  e.scalar(A.softmax_words(NH, P, a["s_smem_w"], a["M"], a["p_smem_w"], cq["s"][1], f32(cq["s"][0]), cq["beta"], cq["p"][0], cq["p"][1]))
  A.smem_to_tiles(e, 1, a["Pn"], a["p_smem_w"], P, NH, a["pfifo"], collapse=False)
  if dest:
    e.scalar(SC.scsync_nop())
    e.sync(SC.sync_reset17)
    FC.broadcast(e, FC.FCGeom(64, 24 * P), a["Pn"], 1, dest, a["bfifo"])
    e.sync(SC.sync_reset17)
  def p_bcast(t):
    r, c = divmod(t, 4)
    return a["Pn"] + 4 * (P * HEAD0[r] + g.p0(c)), [0, 4, 4 * P, 4 * P * ROWS[r]], [DH, g.cols[c], ROWS[r]]
  A.mul_stage(e, g, p_bcast, a["XV"], a["Mv"], a["Yv"], a["mulfifo"], a["const"], EL.mul_quant(cq["p"], cq["v"], cq["pv"]),
              lambda t: HW * (P - g.p0(t % 4)))
  A.possum_any(e, g, a["Yv"], a["O"], a["relay"], a["const"], A.sum_quant(cq["pv"], cq["o"]))
  e.sync(SC.sync_reset17)
  # o = wo(attention): the head rows onto tile 0, broadcast to tiles 1..4, FULLY_CONNECTED
  gather_north(e, a["O"], a["R"])
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_reset17)
  FC.broadcast(e, V1, a["O"], 1, (1 << T1) - 2, a["ofifo"])
  e.sync(SC.sync_reset17)
  FC.ops_block(e, V1, FC.Place(), a["O"], a["Yo"], offs["wo"], mq["wo"], tail=True)
  # outputs: o (tiles 0..4), then k' | v (576 bytes from tile 0) for the host's KV cache, as edgetpu_compiler emits several outputs
  FC.output_block(e, V1, FC.Place(), a["Yo"])
  output_tile0(e, a["Q"] + DM, 2 * DM)
  e.sync(SC.epilogue)
  return block_caching(param_offset), e.program(), block_io(P)

# ***** host contract *****
def block_io(P:int) -> dict:
  """inputs, in DMA order: x [288], cos [288], sin' [288] (rope_inputs), then for P >= 2 K and V as [6][P][48] uint8 head-major with
  row P-1 a placeholder the program overwrites (block_inputs builds them from the P-1 cache rows); outputs, in DMA order: o [288],
  kv = k' | v (576 bytes: this token's KV-cache rows)"""
  ins = [("x", DM), ("cos", DM), ("sin", DM)] + ([("k", DM * P), ("v", DM * P)] if P > 1 else [])
  return dict(inputs=ins, outputs=[("o", DM), ("kv", 2 * DM)], param_bytes=len(MATS) * V1.param_bytes)

def block_executables(caching:bytes, execution:bytes, P:int) -> tuple:
  """coral.executable.Executables (PARAMETER_CACHING, EXECUTION_ONLY) for coral.runtime.run_executable / tools.hw.run: the input
  hints point into ONE host buffer (block_inputs), the outputs come back as o | k' | v (block_outputs)"""
  from coral.executable import Executable, Bitstream, Hint
  io = block_io(P)
  dmas = [(t, n) for t, n in SC.dma_seqs(execution) if t in (SC.TAG_INPUT, SC.TAG_OUTPUT)]
  assert dmas == [(SC.TAG_INPUT, n) for _, n in io["inputs"]] + [(SC.TAG_OUTPUT, n) for _, n in io["outputs"]], dmas
  hints, off = [Hint("instruction", "INFEED", chunk=0)], 0
  for name, n in io["inputs"]:
    hints.append(Hint("dma", "INFEED", "INPUT", name, off, n))
    off += n
  hints += [Hint("dma", "OUTFEED", "OUTPUT", name, 0, n) for name, n in io["outputs"]] + [Hint("interrupt", "OUTFEED", interrupt=0)]
  def mk(kind, bs, hs): return Executable(None, kind, 1, 0, [Bitstream(bs, [])], b"", hs, True, [], [], "beagle", 0, 0, 0, 0)
  pc = mk("PARAMETER_CACHING", caching, [Hint("instruction", "INFEED", chunk=0), Hint("dma", "INFEED", "PARAMETER", "", 0, io["param_bytes"]),
                                         Hint("interrupt", "OUTFEED", interrupt=0)])
  return pc, mk("EXECUTION_ONLY", execution, hints)

def block_inputs(x:np.ndarray, cos:np.ndarray, sin:np.ndarray, K:np.ndarray|None=None, V:np.ndarray|None=None) -> bytes:
  """the host buffer of block_executables: x, cos, sin' [288] uint8; K, V = the cache rows 0..P-2 as [P-1][288] (None or empty at
  P = 1). Row P-1 of K and V is sent as zeros (the program writes k' and v there)."""
  buf = b"".join(np.asarray(t, np.uint8).reshape(DM).tobytes() for t in (x, cos, sin))
  if K is not None and len(K):
    for C in (K, V):
      buf += A.heads(np.concatenate([np.asarray(C, np.uint8).reshape(-1, DM), np.zeros((1, DM), np.uint8)])).tobytes()
  return buf

def block_outputs(buf:bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """the execution's output bytes -> (o, k', v), each [288] uint8"""
  b = np.frombuffer(buf, np.uint8)
  return b[:DM].copy(), b[DM:2 * DM].copy(), b[2 * DM:3 * DM].copy()

# ***** RoPE inputs (llama2.c's interleaved pairs: head dim i = 2j, 2j+1 rotate by pos * 10000^(-2j/48)) *****
def rope_angles(pos:int) -> np.ndarray:
  """[288] the rotation angle of every element (pairs share theirs)"""
  i = np.arange(DM) % DH // 2 * 2
  return pos * (1.0 / 10000.0 ** (i / DH))

def rope_inputs(pos:int, quant:dict|None=None) -> tuple[np.ndarray, np.ndarray]:
  """(cos, sin') [288] uint8 for position pos: cos of the pair's angle; sin' = -sin at the pair's first element, +sin at its second
  (the sign of the rotation, so that q' = q cos + swap(q) sin')"""
  q, ang = block_quant(quant), rope_angles(pos)
  sgn = np.where(np.arange(DM) % 2 == 0, -1.0, 1.0)
  def qz(v, k): return np.clip(np.rint(v / q[k][0]) + q[k][1], 0, 255).astype(np.uint8)
  return qz(np.cos(ang), "cos"), qz(sgn * np.sin(ang), "sin")

# ***** numpy bit models *****
def fc_ref(x:np.ndarray, W:np.ndarray, x_q:tuple, w_q:tuple, y_q:tuple) -> np.ndarray:
  """FULLY_CONNECTED without bias: sum (x - zx)(w - zw) exactly, f32(acc) * f32 multiplier, float32 clamps, half to even, + zy"""
  q = fc_quant(x_q, w_q, y_q)
  acc = (np.asarray(x).astype(np.int64) - q["in_zp"]) @ (np.asarray(W).astype(np.int64) - q["w_zp"]).T
  return A._requant(acc, q["mult"], q["clamp_min"], q["clamp_max"], q["out_zp"])

def add_ref(a:np.ndarray, b:np.ndarray, a_q:tuple, b_q:tuple, y_q:tuple) -> np.ndarray:
  """ADD (dp_mode 5): w1 a + w2 b + offset exactly (integer weights of the scale ratio, eltwise.add_quant), then requantized"""
  q = EL.add_quant(a_q, b_q, y_q)
  acc = q["w1"] * np.asarray(a).astype(np.int64) + q["w2"] * np.asarray(b).astype(np.int64) + q["offset"]
  return A._requant(acc, q["mult"], q["clamp_min"], q["clamp_max"], q["out_zp"])

def rope_ref(v:np.ndarray, cos:np.ndarray, sin:np.ndarray, v_q:tuple, q:dict, c:str, s:str, out:str) -> np.ndarray:
  """v' = ADD(MUL(v, cos), MUL(swap(v), sin')) with the products' quantizations q[c], q[s]"""
  vc = A.mul_ref(v, cos, v_q, q["cos"], q[c])
  vs = A.mul_ref(np.asarray(v)[SWAP], sin, v_q, q["sin"], q[s])
  return add_ref(vc, vs, q[c], q[s], q[out])

def attention_block_ref(x:np.ndarray, cos:np.ndarray, sin:np.ndarray, K:np.ndarray|None, V:np.ndarray|None, wqkv:np.ndarray,
                        wo:np.ndarray, quant:dict|None=None, parts:bool=False):
  """gen_attention_block's bit model: x, cos, sin' [288], K / V the cache rows [P-1][288] (None at P = 1), wqkv [864][288], wo [288][288]
  uint8 -> (o, k', v) [288] each. parts=True: also returns the intermediates (q, k before RoPE, q', the attention output, ...)"""
  q, W = block_quant(quant), block_matrices(wqkv, wo)
  x = np.asarray(x, np.uint8).reshape(DM)
  qf, kf = fc_ref(x, W["q"], q["x"], q["wqkv"], q["qf"]), fc_ref(x, W["k"], q["x"], q["wqkv"], q["kf"])
  v = fc_ref(x, W["v"], q["x"], q["wqkv"], q["v"])
  q2 = rope_ref(qf, cos, sin, q["qf"], q, "qc", "qs", "q")
  k2 = rope_ref(kf, cos, sin, q["kf"], q, "kc", "ks", "k")
  Kp = np.concatenate([np.asarray(K, np.uint8).reshape(-1, DM), k2[None]]) if K is not None and len(K) else k2[None]
  Vp = np.concatenate([np.asarray(V, np.uint8).reshape(-1, DM), v[None]]) if V is not None and len(V) else v[None]
  att, ap = A.attention_ref(q2, Kp, Vp, core_quant(q), parts=True)
  o = fc_ref(att, W["wo"], q["att"], q["wo"], q["o"])
  if parts: return (o, k2, v), dict(qf=qf, kf=kf, q=q2, k=k2, v=v, att=att, **ap)
  return o, k2, v

def attention_block_float(x:np.ndarray, pos:int, K:np.ndarray|None, V:np.ndarray|None, wqkv:np.ndarray, wo:np.ndarray,
                          quant:dict|None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """the same block in float64 from the dequantized x, weights and cache rows (exact RoPE, no intermediate quantization), outputs
  quantized to o's / k's / v's (scale, zero point)"""
  q = block_quant(quant)
  def dq(t, k): return (np.asarray(t, np.float64) - q[k][1]) * q[k][0]
  xf, Wf, Wof = dq(x, "x").reshape(DM), dq(wqkv, "wqkv"), dq(wo, "wo")
  h = Wf @ xf
  qv, kv, vv = h[:DM], h[DM:2 * DM], h[2 * DM:]
  ang = rope_angles(pos)
  def rope(t):
    out = np.empty_like(t)
    out[0::2] = t[0::2] * np.cos(ang[0::2]) - t[1::2] * np.sin(ang[0::2])
    out[1::2] = t[0::2] * np.sin(ang[0::2]) + t[1::2] * np.cos(ang[0::2])
    return out
  q2, k2 = rope(qv), rope(kv)
  Kf = np.concatenate([dq(K, "k").reshape(-1, DM), k2[None]]) if K is not None and len(K) else k2[None]
  Vf = np.concatenate([dq(V, "v").reshape(-1, DM), vv[None]]) if V is not None and len(V) else vv[None]
  sc = np.einsum("hc,phc->hp", q2.reshape(NH, DH), Kf.reshape(-1, NH, DH)) * q["beta"]
  pr = np.exp(sc - sc.max(1, keepdims=True))
  pr /= pr.sum(1, keepdims=True)
  att = np.einsum("hp,phc->hc", pr, Vf.reshape(-1, NH, DH)).reshape(DM)
  def qz(t, k): return np.clip(np.rint(t / q[k][0]) + q[k][1], 0, 255).astype(np.uint8)
  return qz(Wof @ att, "o"), qz(k2, "k"), qz(vv, "v")

if __name__ == "__main__": run_test("test_codegen_attention_block")
