# Acceptance test for coral/codegen/fused.py (offline only: never touches the USB device; edgetpu_compiler runs in docker,
# its outputs are cached in .compile/):
#   1. FFN: coral.fused.ffn_block compiled alone for Mp in {1, 16, 32, 64, 128, 256} x quantization sets (random weights,
#      asymmetric activation ranges): caching + execution program, the parameter blob (ffn_params), the DMA hints, the host
#      sizes and the output layout, all byte-exact / equal.
#   2. argmax: coral.fused.argmax_block for Mp in {16, 32, 64, 128, 256} x quantization sets: caching program, every
#      execution bitstream, the token blob (argmax_params), the hints (DMA interleaving and instruction-chunk splits), the
#      output layout.
#   3. the co-compiled LLM set as coral.fused builds it (per layer conv(864,288), conv(288,288), ffn; x6; + the classifier for
#      Mp > 1) for every Mp: every program regenerated from the compiler's (tile, offset) per program; then our own plan():
#      no overlaps, every program's wide buffers above the plan's limit (scanned from the generated programs), the whole set
#      generated without the compiler.
#
#   python test/test_codegen_fused.py [--quick]        or        python -m coral.codegen.fused [--quick]
from __future__ import annotations
import sys, pathlib, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
from coral.codegen import fused as G
from coral.isa import split, opcode, LAYOUTS

MPS, AM_MPS = (1, 16, 32, 64, 128, 256), (16, 32, 64, 128, 256)

# ============================== reporting ==============================
def decode(ws:list[int]) -> dict: return LAYOUTS[op].decode(ws) if (op:=opcode(ws[0])) in LAYOUTS else {"word": ws[0]}

def first_diff(ours:bytes, ref:bytes) -> str|None:
  sa, sb = split(ours), split(ref)
  for k, ((ia, wa), (ib, wb)) in enumerate(zip(sa, sb)):
    if wa == wb: continue
    if opcode(wa[0]) != opcode(wb[0]) or len(wa) != len(wb): return f"instruction #{k}: opcode {opcode(wa[0]):#x} vs {opcode(wb[0]):#x}"
    da, db = decode(wa), decode(wb)
    return f"instruction #{k} (word {ia}) {opcode(wa[0]):#x}: (ours, compiler) {({n: (da[n], db[n]) for n in da if da[n] != db.get(n)})}"
  return None if len(sa) == len(sb) else f"{len(sa)} vs {len(sb)} instructions"

def compare_programs(name:str, ours:list[bytes], ref:list[bytes]) -> str|None:
  if len(ours) != len(ref): return f"{name}: {len(ours)} bitstreams vs {len(ref)}"
  for k, (a, b) in enumerate(zip(ours, ref)):
    if a != b: return f"{name} bitstream {k}: {first_diff(a, b)}"
  return None

def hints_equal(a, b) -> bool: return [repr(h) for h in a] == [repr(h) for h in b]

class Report:
  def __init__(self): self.rows: list[tuple[str, int, int]] = []
  def add(self, name:str, results:list[tuple[object, str|None]]):
    fails = [(k, m) for k, m in results if m is not None]
    self.rows.append((name, len(results) - len(fails), len(results)))
    print(f"{name:78s} {len(results) - len(fails):4d}/{len(results)}")
    for k, m in fails[:4]: print(f"    FAIL {k}: {m}")
    sys.stdout.flush()
  @property
  def ok(self) -> bool: return all(p == t for _, p, t in self.rows)

# ============================== quantization sets (as coral.fused derives them from ranges) ==============================
def qp(lo:float, hi:float) -> tuple[float, int]:
  from coral.fused import quantize_params
  return quantize_params(lo, hi)

def ffn_case(seed:int):
  """random weights (random spread) and asymmetric activation ranges; seed 0 is the real model's kind of ranges"""
  rng = np.random.default_rng(seed)
  def sd(): return float(rng.uniform(0.01, 0.1))
  W1, W3, W2 = (rng.normal(0, sd(), (768, 288)).astype(np.float32), rng.normal(0, sd(), (768, 288)).astype(np.float32),
                rng.normal(0, sd(), (288, 768)).astype(np.float32))
  if seed == 0: r = dict(x=(-4.0, 5.0), h1=(-6.0, 4.0), h3=(-5.0, 5.5), m=(-3.0, 3.5), y=(-2.0, 2.5))
  else: r = {k: (-float(rng.uniform(0.3, 9)), float(rng.uniform(0.3, 9))) for k in ("x", "h1", "h3", "m", "y")}
  return W1, W3, W2, r

def ffn_quant(W1, W3, W2, r) -> dict:
  from coral.fused import silu_range
  q = {k: qp(*v) for k, v in r.items()}
  q.update(a=qp(*silu_range(*r["h1"])), g=(1 / 256, 0))
  for k, W in dict(w1=W1, w3=W3, w2=W2).items(): q[k] = qp(float(W.min()), float(W.max()))
  return q

def ffn_blob(W1, W3, W2, q) -> bytes:
  from coral.fused import quant
  return G.ffn_params(*(quant(W, *q[k]) for k, W in (("w1", W1), ("w3", W3), ("w2", W2))), zps=(q["w1"][1], q["w3"][1], q["w2"][1]))

# ============================== 1. FFN ==============================
def check_ffn(rep:Report, seeds:list[int], mps=MPS):
  from coral import fused
  from tools.compiler import compile_tflite
  res = []
  for seed in seeds:
    W1, W3, W2, r = ffn_case(seed)
    blk = fused.ffn_block(f"_test_ffn{seed}", W1, W3, W2, r["x"], r["h1"], r["h3"], r["m"], r["y"])
    fused.REGISTRY.pop(blk.name, None)
    q = ffn_quant(W1, W3, W2, r)
    for Mp in mps:
      ex = {e.type: e for e in compile_tflite(__import__('tools.oracle', fromlist=['build']).build(blk, Mp))[0]}
      pc, eo = G.gen_ffn(Mp, quant=q)
      msgs = [m for m in (compare_programs("caching", [pc], [b.data for b in ex["PARAMETER_CACHING"].bitstreams]),
                          compare_programs("execution", [eo], [b.data for b in ex["EXECUTION_ONLY"].bitstreams])) if m]
      if ffn_blob(W1, W3, W2, q) != ex["PARAMETER_CACHING"].parameters: msgs.append("parameter blob differs")
      if not hints_equal(G.program_hints([eo]), ex["EXECUTION_ONLY"].hints): msgs.append("execution hints differ")
      if not hints_equal(G.caching_hints(len(ex["PARAMETER_CACHING"].parameters)), ex["PARAMETER_CACHING"].hints): msgs.append("caching hints differ")
      io, ei = G.block_io(G.BlockSpec("ffn", "ffn", 768, 288), Mp), ex["EXECUTION_ONLY"]
      sizes = tuple(sum(h.size for h in ei.hints if h.kind == "dma" and h.desc == d) for d in ("INPUT", "OUTPUT"))
      if sizes != (io["input_bytes"], io["output_bytes"]): msgs.append(f"io sizes {sizes} != {(io['input_bytes'], io['output_bytes'])}")
      if Mp > 1 and {k: list(v) for k, v in ei.outputs[0].output_layout.items()} != io["output_layout"]: msgs.append("output layout differs")
      res.append(((Mp, seed), "; ".join(msgs) or None))
  rep.add(f"1. FFN alone, Mp in {mps} x {len(seeds)} quantizations (programs, blob, hints, io)", res)

def ffn_model_with_bias(Mp:int, wq:dict, biases:dict, q:dict) -> bytes:
  """coral.fused.ffn_block's TFLite structure (conv form for Mp > 1, FULLY_CONNECTED for Mp = 1) with given int32 biases"""
  import tflite
  from tools.tflite_gen import Model, conv_options, fc_options
  from tools.oracle import _mul_options
  m = Model()
  H, W = G.GRIDS[Mp] if Mp > 1 else (1, 1)
  def t(n, c, k): return m.tensor(n, [1, H, W, c] if Mp > 1 else [1, c], np.uint8, *q[k])
  def lin(x, k, xk, cin, cout, out):
    w = m.tensor(k, [cout, 1, 1, cin] if Mp > 1 else [cout, cin], np.uint8, *q[k], data=wq[k].reshape([cout, 1, 1, cin] if Mp > 1 else [cout, cin]))
    b = m.tensor(k + ".b", [cout], np.int32, q[xk][0] * q[k][0], 0, data=biases[k])
    if Mp > 1: m.op(tflite.BuiltinOperator.CONV_2D, [x, w, b], [out], tflite.BuiltinOptions.Conv2DOptions, conv_options(1, 1, 0))
    else: m.op(tflite.BuiltinOperator.FULLY_CONNECTED, [x, w, b], [out], tflite.BuiltinOptions.FullyConnectedOptions, fc_options(0))
  x, h1, h3, g, a, mm, y = (t("x", 288, "x"), t("h1", 768, "h1"), t("h3", 768, "h3"), t("g", 768, "g"), t("a", 768, "a"), t("m", 768, "m"),
                            t("y", 288, "y"))
  lin(x, "w1", "x", 288, 768, h1)
  lin(x, "w3", "x", 288, 768, h3)
  m.op(tflite.BuiltinOperator.LOGISTIC, [h1], [g])
  m.op(tflite.BuiltinOperator.MUL, [h1, g], [a], tflite.BuiltinOptions.MulOptions, _mul_options)
  m.op(tflite.BuiltinOperator.MUL, [a, h3], [mm], tflite.BuiltinOptions.MulOptions, _mul_options)
  lin(mm, "w2", "m", 768, 288, y)
  return m.build([x], [y])

def check_ffn_bias(rep:Report, mps=(1, 16, 256)):
  """random int32 biases: the bias rows of the blob (the coral.fused models have zero biases) and the programs"""
  from tools.compiler import compile_tflite
  from coral.fused import quant
  W1, W3, W2, r = ffn_case(7)
  q = ffn_quant(W1, W3, W2, r)
  wq = {k: quant(W, *q[k]) for k, W in (("w1", W1), ("w3", W3), ("w2", W2))}
  rng = np.random.default_rng(7)
  biases = {k: rng.integers(-3000, 3000, n).astype(np.int32) for k, n in (("w1", 768), ("w3", 768), ("w2", 288))}
  res = []
  for Mp in mps:
    ex = {e.type: e for e in compile_tflite(ffn_model_with_bias(Mp, wq, biases, q))[0]}
    pc, eo = G.gen_ffn(Mp, quant=q)
    msgs = [m for m in (compare_programs("caching", [pc], [b.data for b in ex["PARAMETER_CACHING"].bitstreams]),
                        compare_programs("execution", [eo], [b.data for b in ex["EXECUTION_ONLY"].bitstreams])) if m]
    blob = G.ffn_params(wq["w1"], wq["w3"], wq["w2"], biases["w1"], biases["w3"], biases["w2"], zps=(q["w1"][1], q["w3"][1], q["w2"][1]))
    if blob != ex["PARAMETER_CACHING"].parameters: msgs.append("parameter blob with biases differs")
    res.append(((Mp, "bias"), "; ".join(msgs) or None))
  rep.add(f"1b. FFN with random int32 biases, Mp in {mps} (programs, blob with bias rows)", res)

# ============================== 2. argmax ==============================
def argmax_case(seed:int):
  rng = np.random.default_rng(100 + seed)
  Wv = rng.normal(0, float(rng.uniform(0.02, 0.1)), (32000, 288)).astype(np.float32)
  x_range = (-4.0, 5.0) if seed == 0 else (-float(rng.uniform(1, 8)), float(rng.uniform(1, 8)))
  l_range = (-10.0, 12.0) if seed == 0 else (-float(rng.uniform(3, 20)), float(rng.uniform(3, 20)))
  return Wv, x_range, l_range

def check_argmax(rep:Report, seeds:list[int], mps=AM_MPS):
  from coral import fused
  from tools.compiler import compile_tflite
  res = []
  for seed in seeds:
    Wv, xr, lr = argmax_case(seed)
    blk = fused.argmax_block(f"_test_cls{seed}", Wv, xr, lr)
    fused.REGISTRY.pop(blk.name, None)
    q = dict(vocab=blk.quant["vocab"], tokens=blk.in_q, logits=blk.out_q)
    for Mp in mps:
      ex = {e.type: e for e in compile_tflite(__import__('tools.oracle', fromlist=['build']).build(blk, Mp))[0]}
      pc, eo = G.gen_argmax(Mp, quant=q)
      ei = ex["EXECUTION_ONLY"]
      msgs = [m for m in (compare_programs("caching", [pc], [b.data for b in ex["PARAMETER_CACHING"].bitstreams]),
                          compare_programs("execution", eo, [b.data for b in ei.bitstreams])) if m]
      tokens = np.random.default_rng(0).integers(0, 256, (Mp, 1, 1, 288), dtype=np.uint8).reshape(Mp, 288)   # argmax_block's data
      if G.argmax_params(tokens, Mp, q["tokens"][1]) != ex["PARAMETER_CACHING"].parameters: msgs.append("token blob differs")
      if not hints_equal(G.argmax_hints(Mp, eo), ei.hints): msgs.append("execution hints differ")
      if not hints_equal(G.caching_hints(len(ex["PARAMETER_CACHING"].parameters)), ex["PARAMETER_CACHING"].hints): msgs.append("caching hints differ")
      if {k: list(v) for k, v in ei.outputs[0].output_layout.items()} != G.argmax_output_layout(Mp): msgs.append("output layout differs")
      io = G.argmax_io(Mp)
      if (ei.inputs[0].size_bytes, ei.outputs[0].size_bytes) != (io["input_bytes"], 20 * 25 * Mp): msgs.append("io sizes differ")
      res.append(((Mp, seed), "; ".join(msgs) or None))
  rep.add(f"2. argmax alone, Mp in {mps} x {len(seeds)} quantizations (3 bitstreams, blob, hints, layout)", res)

# ============================== 3. the co-compiled set, and our own plan ==============================
def build_registry(seed:int=1) -> dict[str, dict]:
  """register the LLM set in coral.fused (random weights, asymmetric ranges) -> quantization per block"""
  from coral import fused
  rng, Q = np.random.default_rng(seed), {}
  fused.REGISTRY.clear()
  for l in range(6):
    for name, (N, K), xr, yr in ((f"L{l}.qkv", (864, 288), (-4.0, 5.0), (-3.0, 3.5)), (f"L{l}.wo", (288, 288), (-2.0, 2.0), (-1.0, 1.2))):
      W = rng.normal(0, 0.05, (N, K)).astype(np.float32)
      fused.conv_block(name, W, xr, yr)
      Q[name] = dict(x=qp(*xr), w=qp(float(W.min()), float(W.max())), y=qp(*yr))
    W1, W3, W2, r = ffn_case(10 + l)
    fused.ffn_block(f"L{l}.ffn", W1, W3, W2, r["x"], r["h1"], r["h3"], r["m"], r["y"])
    Q[f"L{l}.ffn"] = ffn_quant(W1, W3, W2, r)
  Wv, xr, lr = argmax_case(0)
  blk = fused.argmax_block("cls", Wv, xr, lr)
  Q["cls"] = dict(vocab=blk.quant["vocab"], tokens=blk.in_q, logits=blk.out_q)
  return Q

def regenerate(name:str, spec:G.BlockSpec, Mp:int, ex:dict, q:dict) -> tuple[bytes, list[bytes]]:
  """our program for one co-compiled block, from the compiler's placement only"""
  pc0 = ex["PARAMETER_CACHING"].bitstreams[0].data
  rcs = [G.RM.decode_ringConsumer(ws) for _, ws in split(pc0) if opcode(ws[0]) == 0x11]
  ps = [(d["tile_mask"].bit_length() - 1, d["addr"] - 2, d["cnt0"] + 1) for d in rcs]
  if spec.kind == "argmax": return G.gen_argmax(Mp, quant=q, placement=ps)
  if spec.kind == "ffn":
    if Mp == 1 and len(rcs) > 3:      # spread: per matmul its tiles at one offset (aux_addr0 = the bias rows)
      T = G.FC.FCGeom(768, 288).T
      pc, eo = G.gen_ffn(1, quant=q, param_offset=tuple(64 * rcs[i]["aux_addr0"] for i in (0, T, 2 * T)))
      return pc, [eo]
    blocks, out, i = list(G.FFNGeom(max(Mp, 16)).blocks), [], 0
    for n in blocks:
      cur, tot = [], 0
      while tot < n:
        cur.append(ps[i])
        tot += ps[i][2]
        i += 1
      out.append(cur)
    pc, eo = G.gen_ffn(Mp, quant=q, pieces=out)
    return pc, [eo]
  cq = G.conv_quant(q["x"], q["w"], q["y"])
  if Mp == 1:
    if len(rcs) == 1: pc, eo = G.gen_fc_streamed(spec.N, spec.K, ps[0][0], 64 * ps[0][1], cq)
    else: pc, eo = G.FC.gen_fc(spec.N, spec.K, param_offset=64 * rcs[0]["aux_addr0"], quant=cq)
  else: pc, eo = G.gen_conv_pieces(Mp, spec.N, spec.K, ps, cq)
  return pc, [eo]

def wide_floor_violations(bitstreams:list[bytes], own:list[tuple[int, int, int]], limit:int) -> list[str]:
  """every wide address an execution program uses below `limit` must lie in its own parameter regions [(tile, a, z)]"""
  addrs = []
  for bs in bitstreams:
    for _, ws in split(bs):
      op = opcode(ws[0])
      if op in (0x01, 0x02):
        d = G.OP.decode_op(ws)
        if d["par_tflags"] or d["par_base"] or d["par_sel"]: addrs.append(("op.par", d["par_base"] | d["par_sel"] << 13))
        if d["psum_base"] or d["psum_sel"]: addrs.append(("op.psum", d["psum_base"] | d["psum_sel"] << 13))
      elif op in (0x13, 0x14): addrs.append(("n2w" if op == 0x13 else "w2n", decode(ws)["wide_addr"]))
      elif op in (0x11, 0x12):
        d = G.RM.decode_ringConsumer(ws)
        addrs.append(("rcons", d["addr"]))
        if d["aux_en0"]: addrs += [("rcons.aux", d["aux_addr0"]), ("rcons.aux", d["aux_addr1"])]
      elif op == 0x10: addrs.append(("rprod", G.RM.decode_ringProducer(ws)["addr"]))
  return [f"{k} @{a}" for k, a in addrs if a < limit and not any(lo <= a < hi for _, lo, hi in own)]

def check_set(rep:Report, mps=MPS):
  from coral import fused
  Q = build_registry()
  specs = G.llm_specs()
  for Mp in mps:
    exes = __import__('tools.oracle', fromlist=['compiled']).compiled(list(fused.REGISTRY.values()), Mp)
    res = []
    for spec in specs:
      if spec.name not in exes: continue
      ex = exes[spec.name]
      pc, eo = regenerate(spec.name, spec, Mp, ex, Q[spec.name])
      msgs = [m for m in (compare_programs("caching", [pc], [b.data for b in ex["PARAMETER_CACHING"].bitstreams]),
                          compare_programs("execution", eo, [b.data for b in ex["EXECUTION_ONLY"].bitstreams])) if m]
      names = ("vocab", "blockmax") if spec.kind == "argmax" else ("x", "y")
      if not hints_equal(G.program_hints(eo, *names), ex["EXECUTION_ONLY"].hints): msgs.append("hints differ")
      res.append(((Mp, spec.name), "; ".join(msgs) or None))
    rep.add(f"3a. co-compiled set Mp={Mp}: {len(res)} programs from the compiler's (tile, offset)", res)
  for Mp in mps:
    pl = G.plan(specs, Mp)
    L = G.set_limit(specs, Mp)
    progs = G.gen_set(specs, Mp, pl, Q)
    res = [(("plan", Mp), "; ".join(G.check_plan(specs, Mp, pl)) or None)]
    for spec in specs:
      if spec.name not in progs: continue
      eo = [b.data for b in progs[spec.name]["EXECUTION_ONLY"].bitstreams]
      bad = wide_floor_violations(eo, [(0, lo, hi) for _, lo, hi in G.regions(spec, Mp, pl[spec.name])], L)
      res.append(((Mp, spec.name), f"wide addresses below the limit {L} outside its parameters: {bad[:4]}" if bad else None))
    rep.add(f"3b. our plan Mp={Mp}: no overlap, {len(res) - 1} programs generated, buffers above the limit {L}", res)
  big = [Mp for Mp in mps if Mp > 1]
  if len(big) > 1:
    pls = {Mp: G.plan(specs, Mp) for Mp in big}
    res = [((Mp, n), f"{pls[Mp][n]} != {pls[big[0]][n]}" if pls[Mp][n] != pls[big[0]][n] else None)
           for Mp in big[1:] for n in pls[Mp] if n != "cls"]
    rep.add(f"3c. our plan: the layer placements are the same for Mp in {big} (no re-caching when Mp changes)", res)

def main(argv:list[str]|None=None) -> int:
  ap = argparse.ArgumentParser(description="coral/codegen/fused.py vs edgetpu_compiler (offline)")
  ap.add_argument("--quick", action="store_true", help="one quantization set per block, no fresh co-compiled sets beyond Mp in (1, 16)")
  ap.add_argument("--extra", action="store_true", help="+ 12 FFN and 6 argmax quantization sets (Mp 1, 16, 64, 256 / 16, 256)")
  args = ap.parse_args(argv)
  rep = Report()
  check_ffn(rep, [0] if args.quick else [0, 1, 2, 3])
  check_ffn_bias(rep, (16,) if args.quick else (1, 16, 256))
  check_argmax(rep, [0] if args.quick else [0, 1, 2])
  check_set(rep, (1, 16) if args.quick else MPS)
  if args.extra:
    check_ffn(rep, list(range(4, 16)), (1, 16, 64, 256))
    check_argmax(rep, list(range(3, 9)), (16, 256))
  print("PASS" if rep.ok else "FAIL")
  return 0 if rep.ok else 1

def test_codegen_fused(): assert main(["--quick"]) == 0        # pytest entry point

if __name__ == "__main__": sys.exit(main())
