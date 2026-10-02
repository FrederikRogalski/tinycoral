# Acceptance test for coral/codegen/eltops.py (offline only: never touches the USB device; edgetpu_compiler runs in docker, its
# outputs are cached in .compile/). Every generated program is compared with the compiler's byte for byte, plus its DMA hints
# and host sizes:
#   1. SOFTMAX, one row: n = 1..256 and 13 longer rows up to 8192 (the compiler leaves 16384 on the CPU)
#   2. SOFTMAX, 6 rows: n = 4..256
#   3. SOFTMAX, 3..32 rows at sample n (configurations the generator refuses are counted separately: NotImplementedError)
#   4. SOFTMAX quantizations: input scales 1/2..0.0123 and zero points 0..255, output (1/256, 0 / 10 / 255), (1/128, 0),
#      (0.01, 3), beta 0.5 / 1 / 2
#   5. L2_NORMALIZATION: n = 4..2048 (n % 4 == 0) and input quantizations; the NLU table and multiplier for 40 input scales
#   6. SUM / MEAN over the last axis: n = 4..2048 and quantizations
#   7. the arithmetic: the scalar-core interpreter running each generated softmax program equals softmax_ref; the gen_scalar_probe
#      loops do what they say; l2norm_ref / reduce_ref / softmax_ref within 1 LSB of float math (offline consistency only)
#
#   python test/test_codegen_eltops.py [--quick]        or        python -m coral.codegen.eltops [--quick]
from __future__ import annotations
import sys, pathlib, argparse
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np, tflite
from tools.tflite_gen import Model
from coral.isa import opcode, LAYOUTS, scalar_core as S
from coral.codegen import eltops as E
B = tflite.BuiltinOperator

# ============================== models ==============================
def _opt(start, end, *adds):
  def fn(b):
    start(b)
    for f, v in adds: f(b, v)
    return end(b)
  return fn

def softmax_model(rows:int, n:int, in_q=(1/16, 128), out_q=(1/256, 0), beta:float=1.0) -> bytes:
  shape = [1, n] if rows == 1 else [1, rows, n]
  m = Model()
  x = m.tensor("x", shape, np.uint8, *in_q)
  y = m.tensor("y", shape, np.uint8, *out_q)
  m.op(B.SOFTMAX, [x], [y], tflite.BuiltinOptions.SoftmaxOptions,
       _opt(tflite.SoftmaxOptionsStart, tflite.SoftmaxOptionsEnd, (tflite.SoftmaxOptionsAddBeta, beta)))
  return m.build([x], [y])

def l2norm_model(n:int, in_q=(1/16, 128)) -> bytes:
  m = Model()
  x = m.tensor("x", [1, 1, n], np.uint8, *in_q)
  y = m.tensor("y", [1, 1, n], np.uint8, 1/128, 128)
  m.op(B.L2_NORMALIZATION, [x], [y], tflite.BuiltinOptions.L2NormOptions,
       _opt(tflite.L2NormOptionsStart, tflite.L2NormOptionsEnd, (tflite.L2NormOptionsAddFusedActivationFunction, 0)))
  return m.build([x], [y])

def reduce_model(kind:str, n:int, in_q=(1/16, 128), out_q=(1/16, 128)) -> bytes:
  m = Model()
  x = m.tensor("x", [1, 1, n], np.uint8, *in_q)
  a = m.tensor("axes", [1], np.int32, data=np.array([2], np.int32))
  y = m.tensor("y", [1, 1, 1], np.uint8, *out_q)
  m.op({"mean": B.MEAN, "sum": B.SUM}[kind], [x, a], [y], tflite.BuiltinOptions.ReducerOptions,
       _opt(tflite.ReducerOptionsStart, tflite.ReducerOptionsEnd, (tflite.ReducerOptionsAddKeepDims, True)))
  return m.build([x], [y])

def compiled(model:bytes):
  """the compiler's STAND_ALONE executable, or None (not mapped / failed)"""
  from tools.compiler import compile_tflite
  try: exes, _ = compile_tflite(model)
  except RuntimeError: return None
  return next((e for e in exes if e.type == "STAND_ALONE"), None)

# ============================== comparison ==============================
def first_diff(ours:bytes, ref:bytes) -> str|None:
  sa, sb = S.split(ours), S.split(ref)
  for k, ((ia, wa), (ib, wb)) in enumerate(zip(sa, sb)):
    if wa == wb: continue
    oa, ob = opcode(wa[0]), opcode(wb[0])
    if oa != ob or len(wa) != len(wb): return f"instruction #{k} (word {ib}): opcode {oa:#x} vs {ob:#x}"
    L = S.OP3 if oa == 3 else LAYOUTS.get(oa)
    if L is None or len(wa) != L.nwords: return f"instruction #{k} (word {ib}) {oa:#x}: {wa[0]:#x} vs {wb[0]:#x}"
    da, db = L.decode(wa), L.decode(wb)
    return f"instruction #{k} (word {ib}) {oa:#x}: (ours, compiler) {({n: (da[n], db[n]) for n in da if da[n] != db[n]})}"
  return None if len(sa) == len(sb) else f"{len(sa)} vs {len(sb)} instructions"

def check_program(ours:bytes, exe, in_bytes:int, out_bytes:int) -> str|None:
  if exe is None: return "the compiler did not map it"
  if (d := first_diff(ours, exe.bitstreams[0].data)) or ours != exe.bitstreams[0].data: return d or "bytes differ"
  hints = E.executable(ours, in_bytes, out_bytes, exe.inputs[0].name, exe.outputs[0].name).hints
  if [repr(h) for h in hints] != [repr(h) for h in exe.hints]: return f"hints {hints} vs {exe.hints}"
  if (exe.inputs[0].size_bytes, exe.outputs[0].size_bytes) != (in_bytes, out_bytes):
    return f"host sizes {(in_bytes, out_bytes)} vs {(exe.inputs[0].size_bytes, exe.outputs[0].size_bytes)}"
  return None

class Report:
  def __init__(self): self.rows: list[tuple[str, int, int, int]] = []
  def add(self, name:str, results:list[tuple[object, str|None]], skipped:int=0):
    fails = [(k, m) for k, m in results if m is not None]
    self.rows.append((name, len(results) - len(fails), len(results), skipped))
    print(f"{name:72s} {len(results) - len(fails):4d}/{len(results)}" + (f"   (+{skipped} refused: NotImplementedError)" if skipped else ""))
    for k, m in fails[:4]: print(f"    FAIL {k}: {m}")
    sys.stdout.flush()
  @property
  def ok(self) -> bool: return all(p == t for _, p, t, _ in self.rows)

def _par(fn, items, workers:int=6) -> list:
  with ThreadPoolExecutor(workers) as ex: return list(ex.map(fn, items))

# ============================== checks ==============================
def check_softmax(rep:Report, name:str, cases:list[tuple]):
  """cases: (rows, n, in_q, out_q, beta)"""
  def one(c):
    rows, n, iq, oq, beta = c
    try: ours = E.gen_softmax(rows, n, iq, oq, beta)
    except NotImplementedError: return c, "skip"
    io = E.softmax_io(rows, n)
    return c, check_program(ours, compiled(softmax_model(rows, n, iq, oq, beta)), io["in_bytes"], io["out_bytes"])
  res = _par(one, cases)
  rep.add(name, [(c, r) for c, r in res if r != "skip"], sum(r == "skip" for _, r in res))

def check_l2norm(rep:Report, cases:list[tuple]):
  def one(c):
    n, iq = c
    io = E.l2norm_io(n)
    return c, check_program(E.gen_l2norm(n, iq), compiled(l2norm_model(n, iq)), io["in_bytes"], io["out_bytes"])
  rep.add("L2_NORMALIZATION [1, 1, n], n % 4 == 0, x quantizations", _par(one, cases))

def check_rsqrt_table(rep:Report, scales:list[float]):
  def one(s):
    exe = compiled(l2norm_model(288, (s, 128)))
    if exe is None: return s, "not mapped"
    ins = S.split(exe.bitstreams[0].data)
    nlu = next(ws for _, ws in ins if opcode(ws[0]) == 0x19)
    from coral.isa.eltwise import decode_nlu
    from coral.isa.op import decode_op
    d = decode_nlu(nlu)
    ops = [decode_op(ws) for _, ws in ins if opcode(ws[0]) == 1]
    slots_ok = [d[f"slot{k}"] for k in range(50)] == E.rsqrt_slots(s)
    q = E.l2norm_quant(s)
    from coral.isa import f32_bits
    mult_ok = ops[1]["mult_bits"] == f32_bits(q["mult"]) and ops[0]["mult_bits"] == f32_bits(q["sq"])
    return s, None if slots_ok and mult_ok else f"slots {slots_ok}, multipliers {mult_ok}"
  rep.add(f"L2_NORMALIZATION NLU table + multipliers, {len(scales)} input scales", _par(one, scales))

def check_reduce(rep:Report, cases:list[tuple]):
  def one(c):
    kind, n, iq, oq = c
    io = E.reduce_io(n)
    return c, check_program(E.gen_reduce(kind, n, iq, oq), compiled(reduce_model(kind, n, iq, oq)), io["in_bytes"], io["out_bytes"])
  rep.add("SUM / MEAN over the last axis of [1, 1, n], n % 4 == 0, x quantizations", _par(one, cases))

def _scalar_section(prog:bytes, start_word:int) -> list[int]:
  """the scalar-core words of a generated softmax / probe program from the instruction `start_word` to the row loop's branch"""
  ins = S.split(prog)
  k = next(j for j, (_, ws) in enumerate(ins) if ws[0] == start_word)
  end = next(j for j, (_, ws) in enumerate(ins) if j > k and opcode(ws[0]) == 0x22)
  return [ws[0] for _, ws in ins[k:end + 1]]

def check_models(rep:Report, rng:np.random.Generator):
  res = []
  for rows, n, iq, oq, beta in [(1, 256, (1/16, 128), (1/256, 0), 1.0), (1, 13, (0.07, 200), (1/256, 0), 1.0), (6, 100, (0.3, 7), (1/256, 0), 1.0),
                                (6, 37, (1/8, 100), (1/256, 10), 0.5), (6, 256, (1/2, 128), (1/256, 0), 1.0), (6, 130, (1/8, 0), (1/256, 0), 2.0)]:
    m = (n + 3) // 4 * 4 if rows > 1 else n
    base = E.SMEM_TOP - rows * ((n + 3) // 4 * 4)
    words = _scalar_section(E.gen_softmax(rows, n, iq, oq, beta), S.movi(4, base))
    worst = 0
    for _ in range(4):
      xq = rng.integers(0, 256, (rows, n), dtype=np.uint8)
      smem = bytearray(E.SMEM_TOP)
      for r in range(rows): smem[base + r * m: base + r * m + n] = xq[r].tobytes()
      S.run(words, smem)
      got = np.array([list(smem[base + r * m: base + r * m + n]) for r in range(rows)], np.uint8)
      worst = max(worst, int(np.abs(got.astype(int) - E.softmax_ref(xq, iq, oq, beta).astype(int)).max()))
    res.append(((rows, n, iq, oq, beta), None if worst == 0 else f"interpreter vs softmax_ref: {worst} LSB"))
  rep.add("softmax: generated scalar program (interpreted) == softmax_ref", res)
  res = []
  for kind, fn in [("exp2", lambda v: np.exp2(v.astype(np.float64)).astype(np.float32)), ("recip", lambda v: np.float32(1) / v),
                   ("f2i", lambda v: np.rint(v).astype(np.int32)), ("fmul", lambda v: v * np.float32(0.3))]:
    w = 64
    prog = E.gen_scalar_probe(kind, w, 0.3)
    base = E.SMEM_TOP - 4 * w
    ins = S.split(prog)
    k = next(j for j, (_, ws) in enumerate(ins) if ws[0] == S.movi(6, base // 4))
    words = [ws[0] for _, ws in ins[k:k + 2 + (ins[k + 1][1][0] >> 28)]]
    vals = (np.linspace(1, 300, w) if kind == "recip" else np.linspace(-20, 20, w)).astype(np.float32)
    smem = bytearray(E.SMEM_TOP)
    smem[base:] = vals.tobytes()
    S.run(words, smem)
    got = np.frombuffer(bytes(smem[base:]), np.int32 if kind == "f2i" else np.float32)
    res.append((kind, None if np.array_equal(got, fn(vals)) else "probe loop result differs"))
  rep.add("gen_scalar_probe loops (interpreted)", res)
  res = []
  for s, zp in [(1/16, 128), (0.1, 100), (0.0123, 3)]:
    x = rng.integers(0, 256, (100, 288))
    xr = (x - zp) * s
    ref = np.clip(xr / np.sqrt(np.maximum((xr ** 2).sum(1, keepdims=True), s * s)) * 128, -128, 127)
    d = np.abs(E.l2norm_ref(x, (s, zp)).astype(int) - 128 - ref).max()
    res.append((("l2norm", s, zp), None if d <= 1.0 else f"{d:.3f} LSB from float"))
  for kind, iq, oq in [("sum", (1/16, 128), (1/4, 100)), ("mean", (1/16, 128), (1/16, 128)), ("mean", (0.1, 3), (0.05, 200))]:
    x = rng.integers(0, 256, (100, 288))
    xr = (x - iq[1]) * iq[0]
    ref = np.clip((xr.sum(1) if kind == "sum" else xr.mean(1)) / oq[0] + oq[1], 0, 255)
    d = np.abs(E.reduce_ref(kind, x, iq, oq).astype(int) - ref).max()
    res.append(((kind, iq, oq), None if d <= 0.5 + 1e-6 else f"{d:.3f} LSB from float"))
  for rows, n, iq in [(6, 256, (1/16, 128)), (6, 37, (0.3, 7)), (1, 100, (1/2, 128))]:
    x = rng.integers(0, 256, (rows, n))
    xr = (x - iq[1]) * iq[0]
    p = np.exp(xr - xr.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    d = np.abs(E.softmax_ref(x, iq).astype(int) - np.clip(256 * p, 0, 255)).max()
    res.append((("softmax", rows, n, iq), None if d <= 0.5 + 1e-3 else f"{d:.3f} LSB from float"))
  rep.add("numpy models within rounding of float math", res)

# ============================== main ==============================
SM_QUANT = [(iq, oq, beta) for iq in [(1/16, 128), (1/2, 128), (0.3, 7), (0.0123, 250), (1/8, 0), (0.17, 255)]
            for oq, beta in [((1/256, 0), 1.0), ((1/256, 10), 1.0), ((1/256, 0), 0.5), ((1/256, 0), 2.0), ((1/256, 255), 1.0), ((0.01, 3), 1.0),
                             ((1/128, 0), 1.0)]]
SIZES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 100, 128, 160, 192, 256, 288, 320, 384, 512, 768, 1024, 2048]
QUANTS = [(1/32, 100), (1/8, 0), (0.5, 255), (0.1, 3), (0.3, 77)]

def main(argv:list[str]|None=None) -> int:
  ap = argparse.ArgumentParser(description="coral/codegen/eltops.py vs edgetpu_compiler (offline)")
  ap.add_argument("--quick", action="store_true", help="a sample of every check")
  args = ap.parse_args(argv)
  rep, rng, d = Report(), np.random.default_rng(0), (1/16, 128)
  def q(ns): return [n for n in ns if n % 7 == 0 or n in (1, 2, 3, 4, 32, 64, 100, 128, 255, 256)] if args.quick else list(ns)
  check_softmax(rep, "SOFTMAX one row, n = 1..256", [(1, n, d, (1/256, 0), 1.0) for n in q(range(1, 257))])
  long_rows = (300, 1000, 4096) if args.quick else (257, 300, 333, 512, 777, 1000, 1024, 1025, 2000, 2048, 3001, 4096, 8192)
  check_softmax(rep, "SOFTMAX one row, n = 257..8192", [(1, n, d, (1/256, 0), 1.0) for n in long_rows])
  check_softmax(rep, "SOFTMAX 6 rows, n = 4..256", [(6, n, d, (1/256, 0), 1.0) for n in q(range(4, 257))])
  other = [(r, n) for r in (3, 4, 5, 7, 8, 12, 16, 32) for n in (4, 5, 8, 13, 32, 33, 34, 35, 36, 44, 48, 64, 65, 66, 68, 85, 86, 88, 100, 101, 128,
           130, 132, 192, 193, 200, 255, 256) if r * ((n + 3) // 4 * 4) <= 16384]
  check_softmax(rep, "SOFTMAX 3..32 rows, sample n", [(r, n, d, (1/256, 0), 1.0) for r, n in (other[::5] if args.quick else other)])
  shapes = [(1, 32), (6, 101)] if args.quick else [(1, 32), (1, 100), (6, 64), (6, 101), (6, 256)]
  check_softmax(rep, "SOFTMAX quantizations x output quantizations x beta", [(r, n, *c) for r, n in shapes for c in SM_QUANT])
  sizes = [16, 48, 288, 1024] if args.quick else SIZES
  check_l2norm(rep, [(n, d) for n in sizes] + [(288, iq) for iq in QUANTS])
  check_rsqrt_table(rep, [float(np.float32(s)) for s in np.random.default_rng(3).uniform(0.002, 1.0, 10 if args.quick else 40)])
  check_reduce(rep, [(k, n, d, oq) for k, oq in (("sum", (1/4, 100)), ("mean", (1/16, 128))) for n in sizes] +
               [(k, 288, iq, oq) for k, oq in (("sum", (0.7, 13)), ("mean", (0.05, 200))) for iq in QUANTS])
  check_models(rep, rng)
  print("PASS" if rep.ok else "FAIL")
  return 0 if rep.ok else 1

def test_codegen_eltops(): assert main(["--quick"]) == 0        # pytest entry point

if __name__ == "__main__": sys.exit(main())
