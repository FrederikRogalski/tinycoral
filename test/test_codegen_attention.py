# Acceptance test for coral/codegen/attention.py, offline (never touches the USB device; edgetpu_compiler runs in docker, its outputs
# are cached in .compile/):
#   1. the scores piece, SUM_axis3(MUL(q [1,6,1,48], K [1,6,P,48])), byte for byte against edgetpu_compiler (+ hints, host sizes)
#   2. the p.V piece with p expanded, SUM_axis2(MUL(p [1,6,P,48], V [1,6,P,48])), the same
#   3. SOFTMAX over [1,6,P] (eltops.gen_softmax), the same, at the P the device test uses
#   4. the scalar-memory -> tile path of the composed program (smem_to_tiles, one tile) against edgetpu_compiler's SOFTMAX -> RELU
#   5. the composed program gen_attention(P) for P = 1..256: decodes, sequence numbers, DMA descriptors = host contract, every narrow
#      address an instruction touches inside one buffer of attention_alloc, every transfer's producer = consumer size
#   6. its scalar-core softmax (run by scalar_core.run on random scores) = eltops.softmax_ref; the bit models against float math
#
#   python test/test_codegen_attention.py [--quick]        or        python -m coral.codegen.attention [--quick]
from __future__ import annotations
import sys, pathlib, argparse
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "test"):                       # test/: test_codegen_eltops, also when run by `python -m coral.codegen.attention`
  if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import numpy as np, tflite
from tools.tflite_gen import Model
from coral.isa import opcode, LAYOUTS, round_up, scalar as SC, scalar_core as S, wide_narrow as WN, ring_mesh as RM
from coral.codegen import attention as A, eltops as E
B = tflite.BuiltinOperator

# ============================== models ==============================
def _opt(start, end, *adds):
  def fn(b):
    start(b)
    for f, v in adds: f(b, v)
    return end(b)
  return fn
MUL_OPT = _opt(tflite.MulOptionsStart, tflite.MulOptionsEnd, (tflite.MulOptionsAddFusedActivationFunction, 0))
SUM_OPT = _opt(tflite.ReducerOptionsStart, tflite.ReducerOptionsEnd, (tflite.ReducerOptionsAddKeepDims, False))

def _mul_sum(m:Model, a:int, b:int, pshape:list[int], axis:int, prod_q:tuple, out_shape:list[int], out_q:tuple) -> int:
  pr = m.tensor("prod", pshape, np.uint8, *prod_q)
  m.op(B.MUL, [a, b], [pr], tflite.BuiltinOptions.MulOptions, MUL_OPT)
  ax = m.tensor("axes", [1], np.int32, data=np.array([axis], np.int32))
  y = m.tensor("y", out_shape, np.uint8, *out_q)
  m.op(B.SUM, [pr, ax], [y], tflite.BuiltinOptions.ReducerOptions, SUM_OPT)
  return y

def scores_model(P:int, quant:dict|None=None) -> bytes:
  """s [1,6,P] = SUM over axis 3 of MUL(q [1,6,1,48], K [1,6,P,48]) (q broadcast over the positions)"""
  q = A.DEFAULT_SCORES | (quant or {})
  m = Model()
  qt = m.tensor("q", [1, 6, 1, 48], np.uint8, *q["q"])
  kt = m.tensor("k", [1, 6, P, 48], np.uint8, *q["k"])
  y = _mul_sum(m, qt, kt, [1, 6, P, 48], 3, q["qk"], [1, 6, P], q["s"])
  return m.build([qt, kt], [y])

def pv_model(P:int, quant:dict|None=None) -> bytes:
  """o [1,6,48] = SUM over axis 2 of MUL(p [1,6,P,48], V [1,6,P,48]) (p expanded over the channels on the host)"""
  q = A.DEFAULT_PV | (quant or {})
  m = Model()
  pt = m.tensor("p", [1, 6, P, 48], np.uint8, *q["p"])
  vt = m.tensor("v", [1, 6, P, 48], np.uint8, *q["v"])
  y = _mul_sum(m, pt, vt, [1, 6, P, 48], 2, q["pv"], [1, 6, 48], q["o"])
  return m.build([pt, vt], [y])

def softmax_relu_model(P:int) -> bytes:
  """SOFTMAX [1,6,P] -> RELU: edgetpu_compiler brings the scalar core's result back to tile 0 (the path smem_to_tiles reproduces)"""
  m = Model()
  x = m.tensor("x", [1, 6, P], np.uint8, 1/16, 128)
  p = m.tensor("p", [1, 6, P], np.uint8, 1/256, 0)
  m.op(B.SOFTMAX, [x], [p], tflite.BuiltinOptions.SoftmaxOptions, _opt(tflite.SoftmaxOptionsStart, tflite.SoftmaxOptionsEnd,
                                                                      (tflite.SoftmaxOptionsAddBeta, 1.0)))
  y = m.tensor("y", [1, 6, P], np.uint8, 1/128, 0)
  m.op(B.RELU, [p], [y])
  return m.build([x], [y])

def compiled(model:bytes):
  """the compiler's STAND_ALONE executable, or None (not mapped / failed)"""
  from tools.compiler import compile_tflite
  try: exes, _ = compile_tflite(model)
  except RuntimeError: return None
  return next((e for e in exes if e.type == "STAND_ALONE"), None)

# ============================== comparison ==============================
def _decode(ws:list[int]) -> dict|None:
  oc = opcode(ws[0])
  L = S.OP3 if oc == 3 else LAYOUTS.get(oc)
  return L.decode(ws) if L is not None and len(ws) == L.nwords else None

def first_diff(ours:bytes, ref:bytes) -> str|None:
  sa, sb = S.split(ours), S.split(ref)
  for k, ((_, wa), (ib, wb)) in enumerate(zip(sa, sb)):
    if wa == wb: continue
    oa, ob = opcode(wa[0]), opcode(wb[0])
    if oa != ob or len(wa) != len(wb): return f"instruction #{k} (word {ib}): opcode {oa:#x} vs {ob:#x}"
    da, db = _decode(wa), _decode(wb)
    if da is None: return f"instruction #{k} (word {ib}) {oa:#x}: {wa[0]:#x} vs {wb[0]:#x}"
    return f"instruction #{k} (word {ib}) {oa:#x}: (ours, compiler) {({n: (da[n], db[n]) for n in da if da[n] != db[n]})}"
  return None if len(sa) == len(sb) else f"{len(sa)} vs {len(sb)} instructions"

def check_piece(ours:bytes, exe, in_sizes:list[int], out_bytes:int) -> str|None:
  if exe is None: return "the compiler did not map it"
  if (d := first_diff(ours, exe.bitstreams[0].data)) or ours != exe.bitstreams[0].data: return d or "bytes differ"
  dmas = [(t, n) for t, n in S.dma_seqs(ours) if t in (SC.TAG_INPUT, SC.TAG_OUTPUT)]
  want = [(SC.TAG_INPUT, h.size) for h in exe.hints if h.kind == "dma" and h.direction == "INFEED"] + \
         [(SC.TAG_OUTPUT, h.size) for h in exe.hints if h.kind == "dma" and h.direction == "OUTFEED"]
  if dmas != want: return f"DMA descriptors {dmas} vs hints {want}"
  if [l.size_bytes for l in exe.inputs] != in_sizes or exe.outputs[0].size_bytes != out_bytes:
    return f"host sizes {in_sizes, out_bytes} vs {[l.size_bytes for l in exe.inputs], exe.outputs[0].size_bytes}"
  return None

class Report:
  def __init__(self): self.rows: list[tuple[str, int, int, int]] = []
  def add(self, name:str, results:list[tuple[object, str|None]], skipped:int=0):
    fails = [(k, m) for k, m in results if m is not None]
    self.rows.append((name, len(results) - len(fails), len(results), skipped))
    print(f"{name:84s} {len(results) - len(fails):4d}/{len(results)}" + (f"   (+{skipped} refused: NotImplementedError)" if skipped else ""))
    for k, m in fails[:4]: print(f"    FAIL {k}: {m}")
    sys.stdout.flush()
  @property
  def ok(self) -> bool: return all(p == t for _, p, t, _ in self.rows)

def _par(fn, items, workers:int=6) -> list:
  with ThreadPoolExecutor(workers) as ex: return list(ex.map(fn, items))

# ============================== 1-4: the pieces against edgetpu_compiler ==============================
SC_QUANTS = [None, dict(q=(0.0798, 133), k=(0.0684, 129), qk=(0.5165, 174), s=(0.8298, 185)),
             dict(q=(1/8, 0), k=(0.3, 7), qk=(0.05, 250), s=(1/2, 3))]
PV_QUANTS = [None, dict(p=(1/256, 0), v=(0.0142, 134), pv=(0.0087, 98), o=(0.0090, 95)),
             dict(p=(1/256, 10), v=(1/2, 255), pv=(0.3, 7), o=(0.17, 128))]

def check_scores(rep:Report, Ps:list[int]):
  def one(c):
    P, qi = c
    return c, check_piece(A.gen_scores(P, SC_QUANTS[qi]), compiled(scores_model(P, SC_QUANTS[qi])), [288, 288 * P], A.scores_io(P)["out_bytes"])
  rep.add("scores: SUM_axis3 MUL(q [1,6,1,48], K [1,6,P,48]) vs edgetpu_compiler", _par(one, [(P, 0) for P in Ps]))
  rep.add("scores, 2 more quantizations", _par(one, [(P, qi) for P in (1, 7, 64, 256) for qi in (1, 2)]))

def check_pv(rep:Report, Ps:list[int]):
  def one(c):
    P, qi = c
    try: ours = A.gen_pv(P, PV_QUANTS[qi])
    except NotImplementedError: return c, "skip"
    return c, check_piece(ours, compiled(pv_model(P, PV_QUANTS[qi])), [288 * P] * 2, 288)
  res = _par(one, [(P, 0) for P in Ps])
  rep.add("p.V: SUM_axis2 MUL(p [1,6,P,48], V [1,6,P,48]) vs edgetpu_compiler", [(c, r) for c, r in res if r != "skip"],
          sum(r == "skip" for _, r in res))
  rep.add("p.V, 2 more quantizations", _par(one, [(P, qi) for P in (4, 7, 64, 256) for qi in (1, 2)]))

def check_softmax(rep:Report, Ps:list[int]):
  from test_codegen_eltops import softmax_model, check_program
  def one(P):
    io = E.softmax_io(6, P)
    return P, check_program(E.gen_softmax(6, P, (0.8298, 185), (1/256, 0), A.DH ** -0.5), compiled(softmax_model(6, P, (0.8298, 185), (1/256, 0),
                                                                                                                   A.DH ** -0.5)), io["in_bytes"],
                                                                                                                   io["out_bytes"])
  rep.add("SOFTMAX [1,6,P], beta 1/sqrt(48) (eltops.gen_softmax) vs edgetpu_compiler", _par(one, Ps))

def check_return_path(rep:Report, Ps:list[int]):
  """the three instructions that bring the softmax result back to tile 0 in edgetpu_compiler's SOFTMAX -> RELU, rebuilt by
  smem_to_tiles (tile 0, the compiler's FIFO and narrow address, the scalar-memory word of the [6][m] block)"""
  from coral.codegen import Emitter
  def one(P):
    exe = compiled(softmax_relu_model(P))
    if exe is None: return P, "not mapped"
    ins = S.split(exe.bitstreams[0].data)
    k = next(i for i, (_, ws) in enumerate(ins) if opcode(ws[0]) == 0x22)        # the softmax row loop's branch
    ref = [ws for _, ws in ins[k + 1:k + 4]]
    w2n, rc = WN.decode_wide_to_narrow(ref[0]), RM.decode_ringConsumer(ref[1])
    e = Emitter()
    e.seq = w2n["seq"]
    m = round_up(P, 4)
    A.smem_to_tiles(e, 1, 4 * w2n["narrow_addr"], SC.INFEED.decode(ref[2])["buf_off"], m // 4, 6, rc["addr"])
    ours = [ws for _, ws in S.split(e.program())[1:4]]
    return P, None if ours == ref else first_diff(b"".join(w.to_bytes(16, "little") for ws in ours for w in ws),
                                                b"".join(w.to_bytes(16, "little") for ws in ref for w in ws))
  rep.add("scalar memory -> tile 0 (smem_to_tiles) vs edgetpu_compiler's SOFTMAX -> RELU", _par(one, Ps))

# ============================== 5: the composed program ==============================
def _levels(d:dict, p:str, n:int, cnt:str) -> list[tuple[int, int]]:
  """logical (stride, count) levels of a DMA / mesh TTU (decoded fields {p}inc{k} / {p}{cnt}{k})"""
  out, acc = [], 0
  for k in range(n):
    inc, c = d.get(f"{p}inc{k}", 0), d.get(f"{p}{cnt}{k}", 0) + 1
    s = inc + acc
    out.append((s, c))
    acc += s * (c - 1)
  return out

def _op_levels(d:dict, p:str) -> list[tuple[int, int]]:
  out, acc = [], 0
  for k in range(8):
    lo2 = d.get(f"{p}_hmode", 0) if k == 0 else d.get(f"{p}_mode{k-1}", 0)
    s = 4 * d.get(f"{p}_inc{k}", 0) + lo2 + acc
    c = d.get(f"{p}_cnt{k}", 0) + 1
    out.append((s, c))
    acc += s * (c - 1)
  return out

def _extent(base:int, levels:list[tuple[int, int]], unit:int, elem:int) -> tuple[int, int]:
  """[lo, hi) of a TTU walk from base; elem = bytes per access (an op TTU whose innermost level moves by less than a word reads bytes)"""
  if unit == 1 and levels and levels[0][1] > 1 and abs(levels[0][0]) < 4: elem = 1
  lo = base + unit * sum(min(0, s * (c - 1)) for s, c in levels)
  hi = base + unit * sum(max(0, s * (c - 1)) for s, c in levels) + elem
  return lo, hi

def narrow_accesses(prog:bytes) -> list[tuple[int, str, int, tuple[int, int]]]:
  """(instruction index, what, tile mask, [lo, hi) bytes) of every narrow-memory range the tile instructions read or write"""
  out = []
  for k, (_, ws) in enumerate(S.split(prog)):
    oc = opcode(ws[0])
    if oc in (1, 2, 3):
      d = _decode(ws)
      if d["out_tflags"] or d["out_base"]: out.append((k, "op.out", d["tile_mask"], _extent(d["out_base"], _op_levels(d, "out"), 1, 4)))
      if oc != 3 and (d["in_tflags"] or d["in_base"]): out.append((k, "op.in", d["tile_mask"], _extent(d["in_base"], _op_levels(d, "in"), 1, 4)))
    elif oc in (0x13, 0x14):
      d = _decode(ws)
      if oc == 0x14 and d["mode"] == 2: continue                       # the bias store, not narrow memory
      out.append((k, "n2w" if oc == 0x13 else "w2n", d["tile_mask"], _extent(4 * d["narrow_addr"], _levels(d, "n_", 6, "lim"), 4, 4)))
    elif oc in (0x15, 0x16, 0x17, 0x18):
      d = _decode(ws)
      for p in ("o_", "i_"):
        if d[f"{p}sdims"] or d[f"{p}addr"] or (p == "i_" and d["fill_en"]):
          out.append((k, "mesh." + p[0], d["tile_mask"], _extent(d[f"{p}addr"], _levels(d, p, 4, "cnt"), 4, 4)))
  return out

def check_composed(P:int) -> str|None:
  prog, io = A.gen_attention(P)
  a = A.attention_alloc(A.Geom(P))
  ins = S.split(prog)
  # every instruction decodes and re-encodes; the sequence numbers count the tile instructions
  seq = 0
  for k, (_, ws) in enumerate(ins):
    oc = opcode(ws[0])
    if oc in (0x20, 0x22, 0x23): continue
    d = _decode(ws)
    if d is None: return f"#{k}: opcode {oc:#x} does not decode"
    L = S.OP3 if oc == 3 else LAYOUTS[oc]
    if L.encode(**d) != ws: return f"#{k}: re-encoding differs"
    if oc in (0x1a,) or 0x01 <= oc <= 0x19:
      if d["seq"] != seq: return f"#{k}: seq {d['seq']} != {seq}"
      seq += 1
  # host contract
  exe = A.attention_executable(prog, P)
  if [h.size for h in exe.hints if h.kind == "dma"] != [n for _, n in io["inputs"]] + [io["out_bytes"]]: return "hints"
  # narrow addresses: inside one buffer (the identity row's mesh fills write C .. C+16)
  bufs = sorted((a[n], a[n] + sz) for n, sz in A.narrow_sizes(A.Geom(P)).items())
  for k, what, tiles, (lo, hi) in narrow_accesses(prog):
    if not any(b0 <= lo and hi <= b1 for b0, b1 in bufs): return f"#{k} {what} tiles {tiles:#06x}: [{lo}, {hi}) outside every buffer {bufs}"
  if a["narrow_end"] > 192 * 1024 or a["wide_bottom"] < 0: return "memory"
  # transfer sizes: scores -> scalar memory (n2w words = outfeed words per tile), scalar memory -> tiles (infeed words = w2n words)
  n2w_w, out_w, inf, bc = [], [], [], []
  for k, (_, ws) in enumerate(ins):
    oc = opcode(ws[0])
    if oc == 0x27:
      d = SC.OUTFEED.decode(ws)
      if d["d3_stride"] == 3: out_w.append((d["d0_limit"] + 1) * (d["d1_limit"] + 1))
    elif oc == 0x13:
      d = _decode(ws)
      if d["wide_addr"] == a["smfifo"]: n2w_w.append(np.prod([c for _, c in _levels(d, "n_", 6, "lim")]))
    elif oc == 0x26:
      d = SC.INFEED.decode(ws)
      if d["rsv13"]: inf.append(d["d0_limit"] + 1)
    elif oc == 0x14:
      d = _decode(ws)
      if d["wide_addr"] == a["pfifo"]:
        if d["tile_mask"] != 1: return "p must come back to tile 0 only (as edgetpu_compiler does it)"
        inf.append(-int(np.prod([c for _, c in _levels(d, "n_", 6, "lim")])))
      if d["wide_addr"] == a["bfifo"]: bc.append(("w2n", d["tile_mask"], int(np.prod([c for _, c in _levels(d, "n_", 6, "lim")]))))
    elif oc == 0x10:
      d = _decode(ws)
      if d["addr"] == a["bfifo"]: bc.append(("rprod", d["tile_mask"], d["dest"]))
  if [int(x) for x in n2w_w] != out_w or len(out_w) != sum(1 for t in range(16) if A.Geom(P).cols[t % 4]): return f"smem outfeeds {n2w_w} vs {out_w}"
  if len(inf) != 2 or sum(inf) != 0 or max(inf) != 6 * P: return f"smem infeed {inf}"
  dest = sum(1 << t for t in range(1, 16) if A.Geom(P).cols[t % 4])
  if dest and bc != [("w2n", dest, 6 * P), ("rprod", 1, dest)]: return f"broadcast of p {bc}"
  return None

def check_composed_scalar(rng:np.random.Generator, Ps:list[int]) -> list[tuple[int, str|None]]:
  """the scalar-core section of gen_attention (from the first MOVI of the softmax to its row branch) run by scalar_core.run on
  random score words (random upper bytes) with load latencies 0..4: p words = eltops.softmax_ref * 0x01010101, the scores untouched"""
  res = []
  for P in Ps:
    prog, _ = A.gen_attention(P)
    a, q = A.attention_alloc(A.Geom(P)), A.DEFAULT_ATTN
    ins = S.split(prog)
    k0 = next(j for j, (_, ws) in enumerate(ins) if ws[0] == S.movi(4, a["s_smem_w"]))
    k1 = next(j for j, (_, ws) in enumerate(ins) if j > k0 and opcode(ws[0]) == 0x22)
    words = [ws[0] for _, ws in ins[k0:k1 + 1]]
    worst = 0
    for latency in (0, 1, 2, 3, 4):
      x = rng.integers(0, 256, (6, P), dtype=np.uint8)
      smem = bytearray(A.SMEM_TOP)
      wv = x.astype(np.uint32) | (rng.integers(0, 1 << 24, (6, P), dtype=np.uint32) << 8)
      smem[4 * a["s_smem_w"]: 4 * a["s_smem_w"] + 24 * P] = wv.astype("<u4").tobytes()
      S.run(words, smem, load_latency=latency)
      pw = np.frombuffer(bytes(smem[4 * a["p_smem_w"]:4 * a["p_smem_w"] + 24 * P]), "<u4").reshape(6, P)
      ref = E.softmax_ref(x, q["s"], q["p"], q["beta"]).astype(np.uint32)
      worst = max(worst, int(np.abs(pw.astype(np.int64) - (ref * 0x01010101).astype(np.int64)).max()))
      if not np.array_equal(np.frombuffer(bytes(smem[4 * a["s_smem_w"]: 4 * a["s_smem_w"] + 24 * P]), "<u4").reshape(6, P), wv):
        worst = max(worst, 999)
    res.append((P, None if worst == 0 else f"{worst} LSB"))
  return res

def check_models(rng:np.random.Generator) -> list[tuple[object, str|None]]:
  """the bit models against float math on data inside the quantization ranges: scores and p.V within the rounding of their two
  requantizations, attention within the intermediate quantization (loose bound: a sanity check, the device test is the real one)"""
  res = []
  for P in (1, 7, 64, 256):
    q = A.DEFAULT_SCORES
    qu = rng.integers(112, 145, (6, 48)).astype(np.uint8)
    Ku = rng.integers(112, 145, (6, P, 48)).astype(np.uint8)
    def dq(x, k, qq): return (x.astype(np.float64) - qq[k][1]) * qq[k][0]
    exact = (dq(qu, "q", q)[:, None, :] * dq(Ku, "k", q)).sum(2) / q["s"][0] + q["s"][1]
    d = np.abs(A.scores_ref(qu, Ku).astype(float) - np.clip(exact, 0, 255)).max()
    # each product is off by <= 0.5 LSB of qk (48 of them), then 0.5 LSB of s
    res.append((("scores", P), None if d <= 48 * 0.5 * q["qk"][0] / q["s"][0] + 0.5 + 1e-6 else f"{d:.2f} LSB"))
    q = A.DEFAULT_PV
    pu = rng.integers(0, 256, (6, P, 48)).astype(np.uint8)
    Vu = rng.integers(96, 161, (6, P, 48)).astype(np.uint8)
    exact = (dq(pu, "p", q) * dq(Vu, "v", q)).sum(1) / q["o"][0] + q["o"][1]
    d = np.abs(A.pv_ref(pu, Vu).astype(float) - np.clip(exact, 0, 255)).max()
    res.append((("p.V", P), None if d <= P * 0.5 * q["pv"][0] / q["o"][0] + 0.5 + 1e-6 else f"{d:.2f} LSB"))
  return res

# ============================== main ==============================
def main(argv:list[str]|None=None) -> int:
  ap = argparse.ArgumentParser(description="coral/codegen/attention.py vs edgetpu_compiler (offline)")
  ap.add_argument("--quick", action="store_true", help="a sample of every check")
  args = ap.parse_args(argv)
  rep, rng = Report(), np.random.default_rng(0)
  sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 31, 33, 64, 65, 100, 127, 128, 129, 200, 255, 256]
  check_scores(rep, sample if args.quick else list(range(1, 257)))
  check_pv(rep, sample if args.quick else list(range(1, 257)))
  check_softmax(rep, [4, 7, 64, 256] if args.quick else [4, 5, 6, 7, 8, 13, 31, 64, 100, 129, 255, 256])
  check_return_path(rep, [2, 7, 40, 64, 256] if args.quick else [2, 3, 4, 7, 9, 40, 41, 64, 100, 170, 256])   # P = 1: not mapped
  Ps = sample if args.quick else list(range(1, 257))
  rep.add("gen_attention(P): decodes, seq, host contract, narrow buffers, transfer sizes", [(P, check_composed(P)) for P in Ps])
  rep.add("gen_attention's scalar-core softmax (interpreted, load latency 0..4) == eltops.softmax_ref",
          check_composed_scalar(rng, [1, 2, 7, 64, 255, 256]))
  rep.add("bit models (scores_ref, pv_ref) within rounding of float math", check_models(rng))
  print("PASS" if rep.ok else "FAIL")
  return 0 if rep.ok else 1

def test_codegen_attention(): assert main(["--quick"]) == 0        # pytest entry point

if __name__ == "__main__": sys.exit(main())
