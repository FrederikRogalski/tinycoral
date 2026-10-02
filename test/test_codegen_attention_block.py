# Acceptance test for coral/codegen/attention_block.py, offline (never touches the USB device; edgetpu_compiler runs in docker, its
# outputs are cached in .compile/):
#   1. the RoPE piece against edgetpu_compiler: x -> two FULLY_CONNECTED -> MUL(., cos), MUL(., sin) -> ADD, and the same MUL / MUL /
#      ADD on four inputs, rebuilt byte for byte from the block's builders (scatter_input, mul_1d, add_1d, fc.py's blocks)
#   2. the caching program and the parameter blob (six fc.py blobs at param_offsets)
#   3. gen_attention_block(P) for P = 1..256: every instruction decodes and re-encodes, sequence numbers, DMA descriptors = host
#      contract, every narrow range inside one buffer of block_alloc, every wide address above the parameters (or inside the right
#      matmul's region), the transfer sizes of the scalar-memory paths and broadcasts, and a data-flow check: on every tile, every
#      byte an instruction reads was written by an earlier instruction (program order), and the slot copies write exactly row P-1;
#      and that these checks catch deliberately broken programs (check_mutations)
#   4. the block's scalar-core softmax (interpreted, load latency 0..4) = eltops.softmax_ref
#   5. the bit model: RoPE against float rotation, the whole block against float math on a calibrated TinyStories-15M layer
#
#   python test/test_codegen_attention_block.py [--quick]        or        python -m coral.codegen.attention_block [--quick]
from __future__ import annotations
import sys, pathlib, argparse, struct, functools
ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "test"):
  if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import numpy as np, tflite
from tools.tflite_gen import Model, fc_options
from coral.isa import opcode, LAYOUTS, scalar as SC, scalar_core as S, eltwise as EL, ring_mesh as RM, op as OP
from coral.codegen import Emitter, attention as A, attention_block as B, fc as FC, eltops as E
from test_codegen_attention import Report, first_diff, narrow_accesses, _decode, _levels, _op_levels, _opt
BI = tflite.BuiltinOperator
MUL_OPT = _opt(tflite.MulOptionsStart, tflite.MulOptionsEnd, (tflite.MulOptionsAddFusedActivationFunction, 0))
ADD_OPT = _opt(tflite.AddOptionsStart, tflite.AddOptionsEnd, (tflite.AddOptionsAddFusedActivationFunction, 0))

# ============================== 1. the RoPE piece against edgetpu_compiler ==============================
ROPE_QUANTS = [dict(x=(1/32, 128), w=(1/64, 128), q=(1/8, 128), sq=(1/8, 128), c=(1/127, 128), s=(1/127, 128), qc=(1/8, 128), qs=(1/8, 128),
                    y=(1/8, 128)),
               dict(x=(0.047, 131), w=(0.0071, 125), q=(0.083, 125), sq=(0.061, 119), c=(1/127, 128), s=(0.0081, 120), qc=(0.07, 127),
                    qs=(0.05, 140), y=(0.094, 126)),
               dict(x=(1/16, 120), w=(1/50, 130), q=(1/4, 128), sq=(1/4, 128), c=(2/255, 128), s=(1/127, 128), qc=(1/8, 128), qs=(1/8, 128),
                    y=(1/4, 128))]   # (the last: y has twice qc's scale, so every odd sum is an exact .5 tie)

def rope_model(q:dict, with_fc:bool, seed:int=0) -> bytes:
  """with_fc: x [1,288] -> FC -> q, FC -> sq; else inputs q, sq. Then qc = MUL(q, c), qs = MUL(sq, s), y = ADD(qc, qs), c and s inputs"""
  rng = np.random.default_rng(seed)
  m = Model()
  if with_fc:
    x = m.tensor("x", [1, 288], np.uint8, *q["x"])
    def fc(name):
      w = m.tensor(f"{name}.w", [288, 288], np.uint8, *q["w"], data=rng.integers(0, 256, (288, 288)).astype(np.uint8))
      b = m.tensor(f"{name}.b", [288], np.int32, q["x"][0] * q["w"][0], 0, data=np.zeros(288, np.int32))
      y = m.tensor(f"{name}.y", [1, 288], np.uint8, *q[name])
      m.op(BI.FULLY_CONNECTED, [x, w, b], [y], tflite.BuiltinOptions.FullyConnectedOptions, fc_options(0))
      return y
    qt, sqt = fc("q"), fc("sq")
  else:
    qt, sqt = m.tensor("q", [1, 288], np.uint8, *q["q"]), m.tensor("sq", [1, 288], np.uint8, *q["sq"])
  c, s = m.tensor("c", [1, 288], np.uint8, *q["c"]), m.tensor("s", [1, 288], np.uint8, *q["s"])
  qc, qs = m.tensor("qc", [1, 288], np.uint8, *q["qc"]), m.tensor("qs", [1, 288], np.uint8, *q["qs"])
  m.op(BI.MUL, [qt, c], [qc], tflite.BuiltinOptions.MulOptions, MUL_OPT)
  m.op(BI.MUL, [sqt, s], [qs], tflite.BuiltinOptions.MulOptions, MUL_OPT)
  y = m.tensor("y", [1, 288], np.uint8, *q["y"])
  m.op(BI.ADD, [qc, qs], [y], tflite.BuiltinOptions.AddOptions, ADD_OPT)
  return m.build(([x] if with_fc else [qt, sqt]) + [c, s], [y])

def gen_rope_piece(q:dict, with_fc:bool) -> bytes:
  """edgetpu_compiler's program for rope_model, rebuilt from attention_block's builders and fc.py's blocks with the compiler's
  allocation and order (read from its programs; the same for every quantization):
    with_fc: x (gathered, broadcast) -> FC(q) at 0 (parameters at 292 units), x again, FC(sq) at 576 (parameters at 0); input s at
             288, MUL(sq, s) over s (the identity row loaded inside the MUL), reset17, input c at 576 (its ring FIFO below the identity
             row), MUL(q, c) over c, ADD(qs, qc) over c (weight rows at narrow 0 -> wide 8304), output y from 576
    else:    inputs q at 1444, c at 0, MUL(q, c) over c (identity inside), reset17, inputs sq at 1732, s at 288, MUL(sq, s) over s,
             ADD(qc, qs) over s (weight rows at narrow 576), output y from 288"""
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  def mul(a, b, tmp, ka, kb, ky, **kw): return B.mul_1d(e, a, b, tmp, b, 8284, 8316, EL.mul_quant(q[ka], q[kb], q[ky]), **kw)
  if with_fc:
    FC.input_block(e, B.V1, FC.Place(), 1024, 288)
    FC.ops_block(e, B.V1, FC.Place(), 1024, 0, 292, B.fc_quant(q["x"], q["w"], q["q"]), tail=False)
    e.scalar(SC.scsync_nop())
    e.sync(SC.sync_reset17)
    FC.broadcast(e, B.V1, 1024, 1, 0x1e)
    e.sync(SC.sync_reset17)
    FC.ops_block(e, B.V1, FC.Place(), 1024, 576, 0, B.fc_quant(q["x"], q["w"], q["sq"]), tail=False)
    B.scatter_input(e, 288)
    mul(576, 288, 864, "sq", "s", "qs", ident_narrow=2020)
    e.sync(SC.sync_reset17)
    B.scatter_input(e, 576, top=8316)
    mul(0, 576, 864, "q", "c", "qc")
    B.add_1d(e, 288, 576, 576, 0, 8304, EL.add_quant(q["qs"], q["qc"], q["y"]))
    out = 576
  else:
    B.scatter_input(e, 1444)
    B.scatter_input(e, 0)
    mul(1444, 0, 288, "q", "c", "qc", ident_narrow=1732)
    e.sync(SC.sync_reset17)
    B.scatter_input(e, 1732, top=8316)
    B.scatter_input(e, 288, top=8316)
    mul(1732, 288, 576, "sq", "s", "qs")
    B.add_1d(e, 0, 288, 288, 576, 8304, EL.add_quant(q["qc"], q["qs"], q["y"]))
    out = 288
  e.sync(SC.signal_fence)
  FC.output_block(e, B.V1, FC.Place(), out)
  e.sync(SC.epilogue)
  return e.program()

def compiled_exe(model:bytes):
  """the compiler's STAND_ALONE or EXECUTION_ONLY executable, or None"""
  from tools.compiler import compile_tflite
  try: exes, _ = compile_tflite(model)
  except RuntimeError: return None
  return next((e for e in exes if e.type in ("STAND_ALONE", "EXECUTION_ONLY")), None)

def check_rope(rep:Report):
  def one(c):
    qi, wf = c
    exe = compiled_exe(rope_model(ROPE_QUANTS[qi], wf))
    if exe is None: return c, "the compiler did not map it"
    ours, ref = gen_rope_piece(ROPE_QUANTS[qi], wf), exe.bitstreams[0].data
    return c, None if ours == ref else (first_diff(ours, ref) or "bytes differ")
  for wf, name in ((True, "x -> FC, FC -> MUL(., cos), MUL(., sin) -> ADD"), (False, "MUL(q, c), MUL(sq, s) -> ADD on four inputs")):
    rep.add(f"RoPE piece {name} vs edgetpu_compiler ({len(ROPE_QUANTS)} quantizations)", [one((qi, wf)) for qi in range(len(ROPE_QUANTS))])

# ============================== 2. parameters ==============================
def check_params(rep:Report) -> None:
  from coral.codegen.fused import conv_blob
  rng, res = np.random.default_rng(5), []
  wqkv, wo = rng.integers(0, 256, (864, 288), dtype=np.uint8), rng.integers(0, 256, (288, 288), dtype=np.uint8)
  q = dict(wqkv=(1/64, 117), wo=(1/50, 131))
  blob = B.block_params(wqkv, wo, q)
  W = B.block_matrices(wqkv, wo)
  ok = blob == b"".join(conv_blob(W[m], 131 if m == "wo" else 117) for m in B.MATS) and len(blob) == B.block_io(1)["param_bytes"]
  ok &= np.array_equal(W["sq"][0], wqkv[1]) and np.array_equal(W["sq"][1], wqkv[0]) and np.array_equal(W["sk"][47], wqkv[288 + 46])
  res.append(("blob = conv_blob of q, swap(q), k, swap(k), v, wo", None if ok else "differs"))
  for off in (0, 256 * 100):
    pc = B.block_caching(off)
    rc = [RM.decode_ringConsumer(ws) for _, ws in S.split(pc) if opcode(ws[0]) == 0x11]
    offs = B.param_offsets(off)
    want = [(1 << t, 2 + 4 + offs[m]) for m in B.MATS for t in range(5)]
    got = [(d["tile_mask"], d["addr"]) for d in rc]
    infeeds = [SC.INFEED.decode(ws) for _, ws in S.split(pc) if opcode(ws[0]) == 0x26]
    res.append((f"caching program at offset {off}: ringConsumers on tiles 0..4 at the regions, 30 infeeds",
                None if got == want and len(infeeds) == 30 else f"{got[:6]} vs {want[:6]}, {len(infeeds)} infeeds"))
  rep.add("parameter blob and caching program", res)

# ============================== 3. the composed program ==============================
def _walk(base:int, levels:list[tuple[int, int]], unit:int, elem:int) -> np.ndarray:
  """every byte a TTU walk touches: base + unit * (sum of i_k * stride_k), elem bytes per access"""
  a = np.array([base], np.int64)
  for s, c in levels:
    if c > 1: a = (a[None, :] + unit * s * np.arange(c, dtype=np.int64)[:, None]).reshape(-1)
  a = np.unique(a)
  return np.unique((a[:, None] + np.arange(elem)[None, :]).reshape(-1))

def _op_elem(levels:list[tuple[int, int]]) -> int:
  s, c = levels[0]
  return 4 if c == 1 or s == 0 or abs(s) >= 4 else abs(s)

def narrow_effects(prog:bytes):
  """per instruction: (index, opcode, tile mask, reads, writes, decoded fields); reads / writes: arrays of the narrow byte addresses
  the instruction touches on each tile of its mask (op in / out TTUs, narrowToWide reads, wideToNarrow writes except the mode-2 bias
  load, a mesh move's outbound half reads and inbound half writes)"""
  out = []
  for k, (_, ws) in enumerate(S.split(prog)):
    oc = opcode(ws[0])
    if oc not in (1, 2, 3, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18): continue
    d = _decode(ws)
    reads, writes = [], []
    if oc in (1, 2, 3):
      if d["out_tflags"] or d["out_base"]: writes.append(_walk(d["out_base"], _op_levels(d, "out"), 1, _op_elem(_op_levels(d, "out"))))
      if oc == 1 and (d["in_tflags"] or d["in_base"]): reads.append(_walk(d["in_base"], _op_levels(d, "in"), 1, _op_elem(_op_levels(d, "in"))))
    elif oc == 0x13:
      reads.append(_walk(4 * d["narrow_addr"], _levels(d, "n_", 6, "lim"), 4, 4))
    elif oc == 0x14:
      if d["mode"] != 2: writes.append(_walk(4 * d["narrow_addr"], _levels(d, "n_", 6, "lim"), 4, 4))
    else:
      if d["o_sdims"] or d["o_addr"]: reads.append(_walk(d["o_addr"], _levels(d, "o_", 4, "cnt"), 4, 4))
      if d["i_sdims"] or d["i_addr"] or d["fill_en"]: writes.append(_walk(d["i_addr"], _levels(d, "i_", 4, "cnt"), 4, 4))
    out.append((k, oc, d.get("tile_mask", 0), reads, writes, d))
  return out

def dataflow(prog:bytes, allow_overread=lambda k, d: False) -> tuple[str|None, dict]:
  """program-order data flow per tile: every byte read was written before on that tile. Mesh moves: the receivers' inbound half
  writes, the senders' outbound half reads; a relay reads and writes its buffer. -> (first problem, {(tile, addr): last writer})"""
  written = np.zeros((16, 192 * 1024), bool)
  last = np.full((16, 192 * 1024), -1, np.int64)
  for k, oc, mask, reads, writes, d in narrow_effects(prog):
    tiles = [t for t in range(16) if mask >> t & 1]
    if oc in (0x15, 0x16, 0x17, 0x18):         # a relay / forwarder sends on what its own inbound half received
      for w in writes:
        for t in tiles:
          written[t, w] = True
          last[t, w] = k
    for r in reads:
      for t in tiles:
        if not written[t, r].all() and not allow_overread(k, d):
          bad = r[~written[t, r]]
          return f"#{k} opcode {oc:#x} tile {t} reads unwritten bytes {bad.min()}..{bad.max()} ({len(bad)} bytes)", {}
    for w in writes:
      for t in tiles:
        written[t, w] = True
        last[t, w] = k
  return None, dict(last=last)

def check_block(P:int) -> str|None:
  pc, prog, io = B.gen_attention_block(P)
  g = A.Geom(P)
  a = B.block_alloc(g)
  ins = S.split(prog)
  seq = 0
  for k, (_, ws) in enumerate(ins):
    oc = opcode(ws[0])
    if oc in (0x20, 0x22, 0x23): continue
    d = _decode(ws)
    if d is None: return f"#{k}: opcode {oc:#x} does not decode"
    L = S.OP3 if oc == 3 else LAYOUTS[oc]
    if L.encode(**d) != ws: return f"#{k}: re-encoding differs"
    if oc == 0x1a or 0x01 <= oc <= 0x19:
      if d["seq"] != seq: return f"#{k}: seq {d['seq']} != {seq}"
      seq += 1
  try: B.block_executables(pc, prog, P)
  except AssertionError as ex: return f"host contract: {ex}"
  # narrow: inside one buffer
  bufs = sorted((a[n], a[n] + sz) for n, sz in B.narrow_sizes(g).items())
  for k, what, tiles, (lo, hi) in narrow_accesses(prog):
    if not any(b0 <= lo and hi <= b1 for b0, b1 in bufs): return f"#{k} {what} tiles {tiles:#06x}: [{lo}, {hi}) outside every buffer"
  if a["narrow_end"] > 192 * 1024: return "narrow memory"
  # wide: FC ops and bias loads inside their matmul's region, everything else above the parameters
  offs, pend, per = B.param_offsets(0), B.param_end(0), B.V1.param_bytes // 64 // B.T1
  regions = [(o, o + per) for o in offs.values()]
  for k, (_, ws) in enumerate(ins):
    oc = opcode(ws[0])
    d = _decode(ws) if oc in (0x01, 0x02, 0x10, 0x11, 0x12, 0x13, 0x14) else None
    if d is None: continue
    if oc == 0x01 and d["dp_mode"] == 3:
      if not any(lo + 4 <= d["par_base"] + 8192 * d["par_sel"] < hi for lo, hi in regions): return f"#{k}: FC op reads parameters at {d['par_base']}"
      continue
    if oc == 0x14 and d["mode"] == 2:
      if not any(lo == d["wide_addr"] for lo, _ in regions): return f"#{k}: bias load from {d['wide_addr']}"
      continue
    addr = d.get("wide_addr", d.get("addr")) if oc != 0x01 else d["par_base"] + 8192 * d["par_sel"]
    if oc in (0x01, 0x02) and d["par_base"] == 0 and d["par_sel"] == 0: continue
    if not pend <= addr < A.WIDE_TOP: return f"#{k} opcode {oc:#x}: wide address {addr} below the parameter end {pend}"
  if a["wide_bottom"] < pend: return "wide memory"
  # transfer sizes of the scalar-memory paths and the broadcasts
  n2w_w, out_w, inf, bc = [], [], [], []
  for k, (_, ws) in enumerate(ins):
    oc = opcode(ws[0])
    if oc == 0x27:
      d = SC.OUTFEED.decode(ws)
      if d["d3_stride"] == 3: out_w.append((d["d0_limit"] + 1) * (d["d1_limit"] + 1))
    elif oc == 0x13:
      d = _decode(ws)
      if d["wide_addr"] == a["smfifo"]: n2w_w.append(int(np.prod([c for _, c in _levels(d, "n_", 6, "lim")])))
      if d["wide_addr"] in (a["xfifo"], a["qkvfifo"], a["bfifo"], a["ofifo"]):
        bc.append(("n2w", d["wide_addr"], d["tile_mask"], int(np.prod([c for _, c in _levels(d, "n_", 6, "lim")]))))
    elif oc == 0x26:
      d = SC.INFEED.decode(ws)
      if d["rsv13"]: inf.append(d["d0_limit"] + 1)
    elif oc == 0x14:
      d = _decode(ws)
      if d["wide_addr"] == a["pfifo"]:
        if d["tile_mask"] != 1: return "p must come back to tile 0 only"
        inf.append(-int(np.prod([c for _, c in _levels(d, "n_", 6, "lim")])))
      if d["wide_addr"] in (a["xfifo"], a["qkvfifo"], a["bfifo"], a["ofifo"]):
        bc.append(("w2n", d["wide_addr"], d["tile_mask"], int(np.prod([c for _, c in _levels(d, "n_", 6, "lim")]))))
  if n2w_w != out_w or len(out_w) != sum(1 for t in range(16) if g.cols[t % 4]): return f"smem outfeeds {n2w_w} vs {out_w}"
  if len(inf) != 2 or sum(inf) != 0 or max(inf) != 6 * P: return f"smem infeed {inf}"
  dest = sum(1 << t for t in range(1, 16) if g.cols[t % 4])
  want = [("w2n", a["xfifo"], 0x1e, 72), ("n2w", a["xfifo"], 1, 72)] + \
         ([("w2n", a["qkvfifo"], dest, 216), ("n2w", a["qkvfifo"], 1, 216)] if dest else []) + \
         ([("w2n", a["bfifo"], dest, 6 * P), ("n2w", a["bfifo"], 1, 6 * P)] if dest else []) + [("w2n", a["ofifo"], 0x1e, 72),
                                                                                                ("n2w", a["ofifo"], 1, 72)]
  if bc != want: return f"broadcasts {bc} vs {want}"
  # data flow: every byte read was written before (the transposing relay of the position SUM over-reads the last quad: allowed)
  # and the copy ops of the K / V input path (conv2d's): issued before the input DMA that fills their staging, they wait on its counters
  stg = (a["Stg"], a["Stg"] + B.narrow_sizes(g)["Stg"])
  prob, st = dataflow(prog, allow_overread=lambda k, d: (opcode(ins[k][1][0]) == 0x13 and d.get("tail_f799") == 1) or
                      (opcode(ins[k][1][0]) == 1 and stg[0] <= d["in_base"] < stg[1]))
  if prob: return prob
  # the slot copies are the last writers of exactly row P-1 of K and V, on the last tile column with positions
  cl = max(c for c in range(4) if g.cols[c])
  j, s = P - 1 - g.p0(cl), g.cols[cl]
  copies = [k for k, (_, ws) in enumerate(ins) if opcode(ws[0]) == 1 and _decode(ws)["in_tflags"] == 0x15 and
            any(X <= _decode(ws)["out_base"] < X + 96 * g.smax for X in (a["XK"], a["XV"]))]
  for X in (a["XK"], a["XV"]):
    for t in range(16):
      r, c = divmod(t, 4)
      if not g.cols[c]: continue
      blk = np.arange(X, X + 48 * A.ROWS[r] * g.cols[c])
      by_copy = np.isin(st["last"][t, blk], copies)
      want = np.zeros(len(blk), bool)
      if c == cl:
        for h in range(A.ROWS[r]): want[(h * s + j) * 48:(h * s + j + 1) * 48] = True
      if not np.array_equal(by_copy, want): return f"tile {t}: the slot copies wrote {by_copy.sum()} bytes of the block at {X}, want {want.sum()}"
  return None

def check_mutations(rep:Report):
  """the checks of check_block catch broken programs: a q'|k'|v broadcast without tile 5 (data flow and broadcast check), slot copies
  into row P-2, no slot copies, a north gather that lands 4 bytes off"""
  import contextlib
  @contextlib.contextmanager
  def patched(mod, name, fn):
    old = getattr(mod, name)
    setattr(mod, name, fn)
    try: yield
    finally: setattr(mod, name, old)
  bc = FC.broadcast
  def no5(e, g, X, src, dest, fifo=FC.WIDE_FWD_FIFO): return bc(e, g, X, src, dest & ~(1 << 5) if g.K == 3 * B.DM else dest, fifo)
  def row_before(e, g, src, X):
    cl = max(c for c in range(4) if g.cols[c])
    j, s = g.P - 2 - g.p0(cl), g.cols[cl]
    per = [A.chain_copy(src + 48 * A.HEAD0[t // 4], X + 48 * j, 12, 1, A.ROWS[t // 4], [12, 12, 12 * A.ROWS[t // 4]],
                        [12, 12 * s, 12 * s * A.ROWS[t // 4]])
           if t % 4 == cl else None for t in range(16)]
    for m, f in A._group(per): e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))
  gn = B.gather_north
  res = []
  for name, mod, attr, fn, P in (("broadcast without tile 5", FC, "broadcast", no5, 64),
                                 ("slot copies into row P-2", B, "slot_copies", row_before, 64),
                                 ("no slot copies", B, "slot_copies", lambda e, g, src, X: None, 64),
                                 ("no slot copies", B, "slot_copies", lambda e, g, src, X: None, 1),
                                 ("north gather 4 bytes off", B, "gather_north", lambda e, O, R: gn(e, O + 4, R), 7)):
    with patched(mod, attr, fn): msg = check_block(P)
    res.append(((name, P), None if msg else "not caught"))
  rep.add("the program checks catch broken programs (mutations)", res)

def check_block_scalar(rng:np.random.Generator, Ps:list[int]) -> list[tuple[int, str|None]]:
  """the block's scalar-core section (first MOVI of the softmax .. its row branch) run by scalar_core.run on random score words"""
  res = []
  for P in Ps:
    _, prog, _ = B.gen_attention_block(P)
    a, q = B.block_alloc(A.Geom(P)), B.core_quant(B.block_quant())
    ins = S.split(prog)
    k0 = next(j for j, (_, ws) in enumerate(ins) if ws[0] == S.movi(4, a["s_smem_w"]))
    k1 = next(j for j, (_, ws) in enumerate(ins) if j > k0 and opcode(ws[0]) == 0x22)
    words = [ws[0] for _, ws in ins[k0:k1 + 1]]
    worst = 0
    for latency in (0, 2, 4):
      x = rng.integers(0, 256, (6, P), dtype=np.uint8)
      smem = bytearray(A.SMEM_TOP)
      wv = x.astype(np.uint32) | (rng.integers(0, 1 << 24, (6, P), dtype=np.uint32) << 8)
      smem[4 * a["s_smem_w"]: 4 * a["s_smem_w"] + 24 * P] = wv.astype("<u4").tobytes()
      S.run(words, smem, load_latency=latency)
      pw = np.frombuffer(bytes(smem[4 * a["p_smem_w"]:4 * a["p_smem_w"] + 24 * P]), "<u4").reshape(6, P)
      ref = E.softmax_ref(x, q["s"], q["p"], q["beta"]).astype(np.uint32)
      worst = max(worst, int(np.abs(pw.astype(np.int64) - (ref * 0x01010101).astype(np.int64)).max()))
    res.append((P, None if worst == 0 else f"{worst} LSB"))
  return res

# ============================== 5. a calibrated TinyStories-15M layer, the bit model ==============================
def load_checkpoint(path:pathlib.Path=ROOT / "models/stories15M.bin"):
  """karpathy's llama2.c checkpoint -> (config, float32 weights) (examples/stories.py's reader, without tinygrad)"""
  raw = path.read_bytes()
  dim, hidden, n_layers, n_heads, n_kv, vocab, seq_len = struct.unpack("7i", raw[:28])
  shared, vocab, hs = vocab > 0, abs(vocab), dim // n_heads
  arr, off = np.frombuffer(raw, np.float32, offset=28), 0
  def take(*shape):
    nonlocal off
    n = int(np.prod(shape))
    o = arr[off:off + n].reshape(shape)
    off += n
    return o
  w = dict(emb=take(vocab, dim), att_norm=take(n_layers, dim), wq=take(n_layers, dim, dim), wk=take(n_layers, n_kv * hs, dim),
           wv=take(n_layers, n_kv * hs, dim), wo=take(n_layers, dim, dim), ffn_norm=take(n_layers, dim), w1=take(n_layers, hidden, dim),
           w2=take(n_layers, dim, hidden), w3=take(n_layers, hidden, dim), norm=take(dim))
  take(seq_len, hs // 2)
  take(seq_len, hs // 2)
  w["out"] = w["emb"] if shared else take(vocab, dim)
  return dict(dim=dim, hidden=hidden, n_layers=n_layers, n_heads=n_heads, vocab=vocab, seq_len=seq_len), w

def _rope_f(x:np.ndarray, pos:int) -> np.ndarray:
  ang = B.rope_angles(pos)
  o = np.empty_like(x)
  o[0::2] = x[0::2] * np.cos(ang[0::2]) - x[1::2] * np.sin(ang[0::2])
  o[1::2] = x[0::2] * np.sin(ang[0::2]) + x[1::2] * np.cos(ang[0::2])
  return o

@functools.cache
def story_states(n:int=256, layer:int=3) -> dict:
  """TinyStories-15M in float64 numpy, greedy from BOS with a KV cache, n positions: per position of `layer` its attention-normed input
  xn, q / k before and after RoPE, v, the attention output att and wo's output o (all [n][288])"""
  cfg, w = load_checkpoint()
  D, H, L = cfg["dim"], cfg["n_heads"], cfg["n_layers"]
  def rms(x, g): return x / np.sqrt((x * x).mean() + 1e-5) * g
  Kc, Vc = [np.zeros((n, D)) for _ in range(L)], [np.zeros((n, D)) for _ in range(L)]
  rec = {k: np.zeros((n, D)) for k in ("xn", "q0", "k0", "q", "k", "v", "att", "o")}
  tok = 1
  for pos in range(n):
    x = w["emb"][tok].astype(np.float64)
    for l in range(L):
      xn = rms(x, w["att_norm"][l])
      q0, k0, v = w["wq"][l] @ xn, w["wk"][l] @ xn, w["wv"][l] @ xn
      q, k = _rope_f(q0, pos), _rope_f(k0, pos)
      Kc[l][pos], Vc[l][pos] = k, v
      sc = np.einsum("hc,phc->hp", q.reshape(H, -1), Kc[l][:pos + 1].reshape(pos + 1, H, -1)) / np.sqrt(D // H)
      p = np.exp(sc - sc.max(1, keepdims=True))
      p /= p.sum(1, keepdims=True)
      att = np.einsum("hp,phc->hc", p, Vc[l][:pos + 1].reshape(pos + 1, H, -1)).reshape(D)
      o = w["wo"][l] @ att
      if l == layer:
        for kk, vv in dict(xn=xn, q0=q0, k0=k0, q=q, k=k, v=v, att=att, o=o).items(): rec[kk][pos] = vv
      x = x + o
      h = rms(x, w["ffn_norm"][l])
      a, b = w["w1"][l] @ h, w["w3"][l] @ h
      x = x + w["w2"][l] @ (a / (1 + np.exp(-a)) * b)
    tok = int(np.argmax(w["out"] @ rms(x, w["norm"])))
  rec["W"] = dict(wqkv=np.concatenate([w["wq"][layer], w["wk"][layer], w["wv"][layer]]), wo=w["wo"][layer])
  return rec

def calibrated(layer:int=3, n:int=256) -> tuple[dict, np.ndarray, np.ndarray, dict]:
  """(quant, wqkv uint8, wo uint8, states) of a TinyStories-15M layer, every quantization point from its full range over the story
  (bench-style): x, weights, q / k before RoPE (qf, kf) and after (q, k), v, the RoPE products (their own ranges), the attention's
  q*k products, scores, p*v products and output (att), and wo's output (o)"""
  from coral.fused import quantize_params, quant as qz
  st = story_states(n, layer)
  def rng(*xs): return (float(min(x.min() for x in xs)), float(max(x.max() for x in xs)))
  ang = np.stack([B.rope_angles(p) for p in range(n)])
  sgn = np.where(np.arange(288) % 2 == 0, -1.0, 1.0)
  cs, sn = np.cos(ang), sgn * np.sin(ang)
  qc, qs, kc, ks = st["q0"] * cs, st["q0"][:, B.SWAP] * sn, st["k0"] * cs, st["k0"][:, B.SWAP] * sn
  qk, pv = [], []
  for pos in range(0, n, 7):
    qh, Kh, Vh = st["q"][pos].reshape(6, 48), st["k"][:pos + 1].reshape(-1, 6, 48), st["v"][:pos + 1].reshape(-1, 6, 48)
    qk.append((qh[None] * Kh).reshape(-1))
    sc = np.einsum("hc,phc->hp", qh, Kh) / np.sqrt(48)
    p = np.exp(sc - sc.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    pv.append((p.T[:, :, None] * Vh).reshape(-1))
  qk, pv = np.concatenate(qk), np.concatenate(pv)
  s = np.concatenate([np.einsum("hc,phc->hp", st["q"][p].reshape(6, 48), st["k"][:p + 1].reshape(-1, 6, 48)).reshape(-1) for p in range(0, n, 7)])
  W = st["W"]
  q = dict(x=rng(st["xn"]), qf=rng(st["q0"], st["q0"]), kf=rng(st["k0"]), v=rng(st["v"]), qc=rng(qc, qs), kc=rng(kc, ks),
           q=rng(st["q"]), k=rng(st["k"]), qk=rng(qk), s=rng(s), pv=rng(pv), att=rng(st["att"]), o=rng(st["o"]),
           wqkv=rng(W["wqkv"]), wo=rng(W["wo"]))
  quant = {k: quantize_params(*r) for k, r in q.items()}
  quant |= dict(p=(1/256, 0), beta=48 ** -0.5)
  return quant, qz(W["wqkv"], *quant["wqkv"]), qz(W["wo"], *quant["wo"]), st

def block_case(P:int, layer:int=3, quant=None, wqkv=None, wo=None):
  """the block's uint8 inputs at position P-1 of the story: x, cos, sin', the K / V cache rows 0..P-2 (the float k, v quantized)"""
  if quant is None: quant, wqkv, wo, st = calibrated(layer)
  else: st = story_states(256, layer)
  def qz(v, k): return np.clip(np.rint(v / quant[k][0]) + quant[k][1], 0, 255).astype(np.uint8)
  cos, sin = B.rope_inputs(P - 1, quant)
  return dict(x=qz(st["xn"][P - 1], "x"), cos=cos, sin=sin, K=qz(st["k"][:P - 1], "k"), V=qz(st["v"][:P - 1], "v")), quant, wqkv, wo, st

def check_models(rep:Report):
  res = []
  # RoPE: the uint8 rotation (MUL, MUL, ADD) against the float rotation of the same uint8 q, within a few LSB
  quant, wqkv, wo, st = calibrated()
  q = B.block_quant(quant)
  for pos in (0, 1, 17, 255):
    qu = np.clip(np.rint(st["q0"][pos] / q["qf"][0]) + q["qf"][1], 0, 255).astype(np.uint8)
    cos, sin = B.rope_inputs(pos, quant)
    got = B.rope_ref(qu, cos, sin, q["qf"], q, "qc", "qs", "q").astype(float)
    exact = np.clip(_rope_f((qu.astype(float) - q["qf"][1]) * q["qf"][0], pos) / q["q"][0] + q["q"][1], 0, 255)
    d = np.abs(got - exact).max()
    res.append((("rope", pos), None if d <= 2.5 else f"{d:.2f} LSB"))
  # the whole block against float math on the same uint8 inputs: relative L2 error of o (the elementwise attention rounds every
  # product, docs/isa/codegen_attention.md 7.2: about 0.1-0.25), k' and v within 1-3 LSB
  for P in (1, 7, 64, 256):
    c, quant, wqkv, wo, st = block_case(P)
    (o, k2, v), parts = B.attention_block_ref(c["x"], c["cos"], c["sin"], c["K"], c["V"], wqkv, wo, quant, parts=True)
    of, kf, vf = B.attention_block_float(c["x"], P - 1, c["K"], c["V"], wqkv, wo, quant)
    qq = B.block_quant(quant)
    def dq(t, k): return (t.astype(float) - qq[k][1]) * qq[k][0]
    rel = np.linalg.norm(dq(o, "o") - dq(of, "o")) / np.linalg.norm(dq(of, "o"))
    dk, dv = np.abs(k2.astype(int) - kf.astype(int)).max(), np.abs(v.astype(int) - vf.astype(int)).max()
    res.append((("block", P), None if rel < 0.35 and dk <= 3 and dv <= 1 else f"rel {rel:.3f}, k' {dk} LSB, v {dv} LSB"))
    print(f"    P={P:3d}: o rel. L2 error vs float {rel:.3f}; k' max {dk} LSB, v max {dv} LSB; p max {parts['p'].max()}")
  rep.add("bit models: RoPE within 2.5 LSB of float rotation, the block near float math (calibrated layer 3)", res)

# ============================== main ==============================
def main(argv:list[str]|None=None) -> int:
  ap = argparse.ArgumentParser(description="coral/codegen/attention_block.py (offline)")
  ap.add_argument("--quick", action="store_true", help="a sample of every check")
  args = ap.parse_args(argv)
  rep, rng = Report(), np.random.default_rng(0)
  check_rope(rep)
  check_params(rep)
  sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 31, 33, 64, 65, 100, 127, 128, 129, 200, 255, 256]
  Ps = sample if args.quick else list(range(1, 257))
  rep.add("gen_attention_block(P): decodes, seq, host contract, buffers, wide, transfers, data flow, slot copies", [(P, check_block(P)) for P in Ps])
  check_mutations(rep)
  rep.add("the block's scalar-core softmax (interpreted, load latency 0, 2, 4) == eltops.softmax_ref", check_block_scalar(rng, [1, 2, 7, 64, 256]))
  check_models(rep)
  print("PASS" if rep.ok else "FAIL")
  return 0 if rep.ok else 1

def test_codegen_attention_block(): assert main(["--quick"]) == 0        # pytest entry point

if __name__ == "__main__": sys.exit(main())
