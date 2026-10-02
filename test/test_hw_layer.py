# hardware: coral/codegen/layer.py's programs against their numpy bit models, through tools/hw.py (one stage per run; stop at the
# first failure, never retry a failed transfer). Every stage is ONE device session (tools.hw.run_jobs) whose jobs are reported one by
# one as they finish, so a hang shows how far the session got.
#   logistic                  edgetpu_compiler's own LOGISTIC programs ([1, 256], every uint8 input, 18 input quantizations) vs
#                             layer.logistic_ref (byte-identical compiler programs: no new=True needed)
#   bisect P [layers] [diag] [from=stop] [control]
#                             the prefix programs of the last layer (gen_model(stop=(L-1, s)) for s in layer.STOPS: each ends after
#                             one more stage and returns that stage's vector), then the whole program; layers 1 (layer 3's weights) or 6.
#                             control: edgetpu_compiler's L2 -> FC -> LOGISTIC first; diag: first the qc prefix without / with the
#                             identity reload after the L2 norm
#   layer P[,P..] [n]         gen_model for ONE layer (layer 3's real weights, calibrated; P = 1 with the BOS quantization), 3 inputs per
#                             P; n more calls of the first input for timing
#   stream P[,P..]            the same with q, w1 and w2 streamed from other tiles (plan(stream=...))
#   model P[,P..] [n]         gen_model for all 6 layers (layer.plan(6): some matmuls streamed)
#   replay FILE L s,s,..      a call saved by examples/stories_chip.py --check: the 6-layer prefix programs stopping at stages s of layer L
#   .venv/bin/python test/test_hw_layer.py bisect 1
# 'local' anywhere in the arguments: the tile-local layout (layer.gen_model(local=True)).
# The generated programs have no compiler equivalent: they run with tools/hw.run_jobs(new=True). $LAYER_SAVE: raw outputs (npz).
import sys, pathlib, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))
import numpy as np
from coral.codegen import layer as LY
from tools import hw

SAVE = os.environ.get("LAYER_SAVE")

def report(name:str, got:np.ndarray, ref:np.ndarray) -> int:
  d = np.abs(got.astype(int) - ref.astype(int))
  print(f"{name}: max diff {d.max()}, exact {np.mean(d == 0) * 100:.1f}% of {d.size}", flush=True)
  return int(d.max())

# ============================== LOGISTIC ==============================
def logistic_quants() -> list[tuple[float, int]]:
  from test_codegen_layer import calibrated
  w, qm, qb, Ws, _ = calibrated()
  extra = [(1/16, 128), (0.02, 100), (0.3, 77), (0.0473, 140), (0.12, 200), (1/8, 0)]
  return sorted({tuple(q[l]["h1"]) for q in (qm, qb) for l in range(6)}) + extra

def stage_logistic() -> int:
  from test_eltwise import logistic
  from tools.compiler import compile_tflite
  qs, jobs = logistic_quants(), []
  x = np.arange(256, dtype=np.uint8)
  for in_q in qs:
    exes, _ = compile_tflite(logistic([1, 256], in_q))
    ex = {e.type: e for e in exes}
    jobs.append(dict(caching=None, execution=ex["STAND_ALONE"], inputs=[x.tobytes()], oracle=ex))
  outs = hw.run_jobs(jobs)
  got = np.stack([np.frombuffer(o[0], np.uint8)[:256] for o in outs])
  if SAVE: np.savez(f"{SAVE}/logistic.npz", quants=np.array(qs), out=got)
  worst = 0
  for in_q, y in zip(qs, got): worst = max(worst, report(f"LOGISTIC in_q ({in_q[0]:.6g}, {in_q[1]})", y, LY.logistic_ref(x, in_q)))
  return worst

def compiler_l2_fc_logistic() -> tuple[dict, callable]:
  """edgetpu_compiler's own program for x [1,288] -> L2_NORMALIZATION -> FULLY_CONNECTED 768 -> LOGISTIC (byte-identical, no new=True
  needed) as a job, checked against layer.l2norm_ref / fc_ref / logistic_ref: does the device run an L2 norm followed by a LOGISTIC?"""
  import tflite
  from tools.tflite_gen import Model, fc_options
  from tools.compiler import compile_tflite
  from test_codegen_eltops import _opt
  from coral.codegen import attention_block as B
  BI = tflite.BuiltinOperator
  rng = np.random.default_rng(0)
  W = rng.integers(0, 256, (768, 288)).astype(np.uint8)
  m = Model()
  x = m.tensor("x", [1, 288], np.uint8, 1/16, 128)
  xn = m.tensor("xn", [1, 288], np.uint8, 1/128, 128)
  m.op(BI.L2_NORMALIZATION, [x], [xn], tflite.BuiltinOptions.L2NormOptions,
       _opt(tflite.L2NormOptionsStart, tflite.L2NormOptionsEnd, (tflite.L2NormOptionsAddFusedActivationFunction, 0)))
  w = m.tensor("w", [768, 288], np.uint8, 1/64, 128, data=W)
  b = m.tensor("b", [768], np.int32, (1/128) / 64, 0, data=np.zeros(768, np.int32))
  h = m.tensor("h", [1, 768], np.uint8, 1/8, 128)
  m.op(BI.FULLY_CONNECTED, [xn, w, b], [h], tflite.BuiltinOptions.FullyConnectedOptions, fc_options(0))
  g = m.tensor("g", [1, 768], np.uint8, 1/256, 0)
  m.op(BI.LOGISTIC, [h], [g])
  exes, _ = compile_tflite(m.build([x], [g]))
  ex = {e.type: e for e in exes}
  xs = [np.clip(np.rint(np.random.default_rng(i).normal(128, 20, 288)), 0, 255).astype(np.uint8) for i in range(3)]
  def check(outs):
    print("edgetpu_compiler's L2_NORMALIZATION -> FULLY_CONNECTED 768 -> LOGISTIC", flush=True)
    worst = 0
    for i, (o, xx) in enumerate(zip(outs, xs)):
      ref = LY.logistic_ref(B.fc_ref(LY.l2norm_ref(xx, (1/16, 128)), W, (1/128, 128), (1/64, 128), (1/8, 128)), (1/8, 128))
      worst = max(worst, report(f"  #{i} g", np.frombuffer(o, np.uint8)[:768], ref))
    return worst
  return dict(caching=ex["PARAMETER_CACHING"], execution=ex["EXECUTION_ONLY"], inputs=[xx.tobytes() for xx in xs], oracle=ex), check

# ============================== layers ==============================
def story_states(w:dict, story:list[int]):
  """the float model over a story: per position and layer the residual input (res), and the KV cache (k after RoPE, v)"""
  class Rec(LY.Ranges):
    def __init__(self):
      super().__init__()
      self.res = {}
      self.pos = 0
    def __call__(self, key, x):
      if key.endswith(".res"): self.res[(self.pos, int(key.split(".")[0]))] = np.asarray(x, np.float64).copy()
  rec = Rec()
  fm = LY.FloatModel(w, rec)
  for p, t in enumerate(story):
    rec.pos = p
    fm.hidden(t, p)
  return rec.res, fm.K, fm.V

def case_inputs(P:int, layers:list[int], quants:list[dict], res:dict, K:np.ndarray, V:np.ndarray, rng) -> list[tuple]:
  """three inputs at P (the first layer of `layers` takes x): the story's residual at position P-1 with cos / sin' of P-1; the residual
  of another position with cos / sin' of another; a Gaussian x; every one with the story's K / V cache rows 0..P-2 of each layer (k in
  the order SIGMA, as the program writes them)"""
  def qz(v, q): return np.clip(np.rint(v / q[0]) + q[1], 0, 255).astype(np.uint8)
  n = max(p for p, _ in res) + 1
  Ks = [qz(K[l, :P - 1][:, LY.SIGMA], quants[i]["k"]) if P > 1 else None for i, l in enumerate(layers)]
  Vs = [qz(V[l, :P - 1], quants[i]["v"]) if P > 1 else None for i, l in enumerate(layers)]
  x0 = qz(res[(P - 1, layers[0])], quants[0]["res"])
  x1 = qz(res[((P - 1 + 23) % n, layers[0])], quants[0]["res"])
  x2 = np.clip(np.rint(rng.normal(quants[0]["res"][1], x0.astype(float).std(), 288)), 0, 255).astype(np.uint8)
  return [(x, *LY.rope_inputs((P - 1 + d) % 256, quants[0]), Ks, Vs) for x, d in ((x0, 0), (x1, 37), (x2, 74))]

LOCAL = "local" in sys.argv          # the tile-local layout of the 1-D vectors (layer.gen_model(local=True))

def job(P:int, layers:list[int], timing:int=0, stop:tuple|None=None, stream:set|None=None, reload_ident:bool=True) -> tuple[dict, callable]:
  """one program pair with its inputs (a tools.hw.run_jobs job) and the function that checks its outputs (-> worst diff)"""
  from test_codegen_layer import calibrated
  w, qm, qb, Wall, stories = calibrated()
  q = [dict((qb if P == 1 else qm)[l]) for l in layers]
  Ws = [Wall[l] for l in layers]
  story, seed = list(stories[0]), 11
  while len(story) < max(P, 30):                 # longer contexts: more float stories appended (without their BOS)
    story += LY.float_stories(w, 1, 128, seed=seed)[0][1:]
    seed += 1
  res, K, V = story_states(w, story[:max(P, 30)])
  L = len(layers)
  pl = LY.plan(L, stream=stream)
  pc, bss, io = LY.gen_model(P, q, pl, stop, reload_ident, LOCAL)
  cex, eex = LY.model_executables(pc, bss, io)
  blob = LY.model_params(Ws, q)
  xs = case_inputs(P, layers, q, res, K, V, np.random.default_rng(20 + P))
  name = (f"{L} layer(s) P={P}" + (f" stop {stop}" if stop else "") + (" streamed" if stream else "") + ("" if reload_ident else " no-reload") +
          (" local" if LOCAL else ""))
  info = (f"{name}: bitstreams {[len(b) // 16 for b in bss]} words, inputs {sum(n for _, n in io['inputs'])} B, outputs "
          f"{[n for _, n in io['outputs']]} B, streamed {sum(1 for d in pl for p in d.values() if p[0] == 'stream')} matmuls")
  times: list = []
  nin = sum(n for _, n in io["inputs"])
  inputs = [LY.model_inputs(*x)[:nin] for x in xs] + [LY.model_inputs(*xs[0])[:nin]] * timing
  def check(outs:list[bytes]) -> int:
    print(info, flush=True)
    worst = 0
    for i, (o, x) in enumerate(zip(outs, xs)):
      rx, rkv, parts = LY.model_ref(*x, Ws, q, parts=True)
      if stop is not None: worst = max(worst, report(f"  #{i} {stop[1]}", LY.model_outputs(o, L, stop), LY.stop_ref(parts[stop[0]], stop[1])))
      else:
        gx, gkv = LY.model_outputs(o, L)
        worst = max(worst, report(f"  #{i} x_out", gx, rx), report(f"  #{i} k'|v", gkv, rkv))
    same = all(o == outs[0] for o in outs[3:])
    if timing: print(f"  wall time per call (USB included), {timing} calls: median {np.median(times[3:]) * 1e3:.3f} ms, "
                     f"min {min(times[3:]) * 1e3:.3f} ms; repeated calls identical: {same}", flush=True)
    if SAVE: np.savez(f"{SAVE}/{name.replace(' ', '_').replace(',', '')}.npz", out=np.frombuffer(b"".join(outs[:3]), np.uint8), times=np.array(times))
    return worst if same else max(worst, 999)
  return dict(caching=cex, execution=eex, inputs=inputs, params=blob, times=times), check

def replay_jobs(path:str, layer, stops:list[str]) -> list:
  """a call saved by examples/stories_chip.py --check (its inputs, the example's calibration): the 6-layer prefix programs that stop
  inside `layer` on those inputs, each compared with the bit model's vector there (where the device first leaves the model)"""
  sys.path.insert(0, str(ROOT / "examples"))
  import stories_chip as SCH
  d = np.load(path)
  cfg, w = LY.load_checkpoint()
  qm, qb = SCH.calibration(w)
  pos = d["Kh"].shape[2]
  P = pos + 1
  q = qb if P == 1 else qm
  Ws = LY.quant_weights(w, qm)
  def rows(C): return [C[l].transpose(1, 0, 2).reshape(pos, 288) if pos else None for l in range(6)]
  Ks, Vs = rows(d["Kh"]), rows(d["Vh"])
  rx, rkv, parts = LY.model_ref(d["x"], d["cos"], d["sin"], Ks, Vs, Ws, q, parts=True)
  pl, blob, out = LY.plan(6), LY.model_params(Ws, q), []
  for spec in stops:                             # "st" (in `layer`) or "L:st"
    lay, st = (int(spec.split(":")[0]), spec.split(":")[1]) if ":" in spec else (layer, spec)
    pc, bss, io = LY.gen_model(P, q, pl, (lay, st), local=LOCAL)
    cex, eex = LY.model_executables(pc, bss, io)
    nin = sum(n for _, n in io["inputs"])
    buf = LY.model_inputs(d["x"], d["cos"], d["sin"], Ks, Vs)[:nin]
    def check(outs, st=st, lay=lay):
      got, ref = LY.model_outputs(outs[0], 6, (lay, st)), LY.stop_ref(parts[lay], st)
      diff = np.nonzero(got != ref)[0]
      print(f"replay P={P} layer {lay} stop {st}: {len(diff)} of {len(ref)} differ" +
            (f" at {diff[:8]}: device {got[diff[:8]]} model {ref[diff[:8]]}" if len(diff) else ""), flush=True)
      return int(np.abs(got.astype(int) - ref.astype(int)).max())
    out.append((dict(caching=cex, execution=eex, inputs=[buf], params=blob), check))
  return out

def probe_jobs(path:str|None=None) -> list:
  """the scalar core's exp2 and reciprocal (eltops.gen_scalar_probe: the softmax's data path, the unit op on every word) on the softmax
  arguments of a saved call's 6 layers (every exp2 argument, every row sum) and on grids; saves the measured values ($LAYER_SAVE)"""
  from coral.codegen import eltops as E, attention_block as B
  F = np.float32
  ts, sums = [], []
  if path is not None:
    sys.path.insert(0, str(ROOT / "examples"))
    import stories_chip as SCH
    d = np.load(path)
    cfg, w = LY.load_checkpoint()
    qm, qb = SCH.calibration(w)
    Ws = LY.quant_weights(w, qm)
    pos = d["Kh"].shape[2]
    def rows(C): return [C[l].transpose(1, 0, 2).reshape(pos, 288) for l in range(6)]
    _, _, parts = LY.model_ref(d["x"], d["cos"], d["sin"], rows(d["Kh"]), rows(d["Vh"]), Ws, qm if pos else qb, parts=True)
    for l in range(6):
      cq = B.core_quant(B.block_quant((qm if pos else qb)[l]))
      x = (parts[l]["s"].astype(np.int64) - cq["s"][1]).astype(F) * F(cq["s"][0])
      t = ((x - x.max(1, keepdims=True)) * F(E.exp2_scale(cq["beta"]))).astype(F)
      ts.append(t.reshape(-1))
      sums.append(np.add.accumulate((2.0 ** t.astype(np.float64)).astype(F), axis=1, dtype=F)[:, -1])
  t_all = np.concatenate(ts + [np.linspace(-30, 0, 2048 - sum(len(t) for t in ts), dtype=F)])
  s_all = np.concatenate(sums + [np.linspace(1, 256, 2048 - sum(len(s) for s in sums), dtype=F)])
  jobs = []
  for kind, vals in (("exp2", t_all), ("recip", s_all)):
    prog = E.gen_scalar_probe(kind, 2048)
    def check(outs, kind=kind, vals=vals):
      got = np.frombuffer(outs[0], F)[:2048]
      want = (2.0 ** vals.astype(np.float64)).astype(F) if kind == "exp2" else (F(1) / vals).astype(F)
      ulp = np.abs(got.view(np.int32).astype(np.int64) - want.view(np.int32).astype(np.int64))
      print(f"scalar core {kind}: {np.mean(ulp == 0) * 100:.1f}% correctly rounded, max {ulp.max()} ulp; the saved call's values: "
            f"max {ulp[:len(np.concatenate(ts if kind == 'exp2' else sums)) if ts else 0].max(initial=0)} ulp", flush=True)
      if SAVE: np.savez(f"{SAVE}/probe_{kind}.npz", x=vals, device=got)
      return 0
    jobs.append((dict(caching=None, execution=E.executable(prog, 8192, 8192), inputs=[vals.astype(F).tobytes()]), check))
  return jobs

def run(jobs_checks:list) -> int:
  worst = [0]
  def done(i, outs): worst[0] = max(worst[0], jobs_checks[i][1](outs))
  hw.run_jobs([j for j, _ in jobs_checks], new=True, on_done=done)
  return worst[0]

def control_jobs() -> list:
  """the compiler's L2 -> FC -> LOGISTIC program first ('control' in the arguments)"""
  return [compiler_l2_fc_logistic()] if "control" in sys.argv[4:] else []

if __name__ == "__main__":
  st = sys.argv[1]
  Ps = [int(p) for p in sys.argv[2].split(",")] if len(sys.argv) > 2 and st != "replay" else []
  n = int(sys.argv[3]) if len(sys.argv) > 3 and st not in ("bisect", "replay") else 0
  if st == "logistic": worst = stage_logistic()
  elif st == "bisect":
    layers = [3] if (int(sys.argv[3]) if len(sys.argv) > 3 else 1) == 1 else list(range(6))
    diag = [job(Ps[0], layers, 0, (len(layers) - 1, "qc"), reload_ident=r) for r in (False, True)] if "diag" in sys.argv[4:] else []
    first = next((a[5:] for a in sys.argv[4:] if a.startswith("from=")), LY.STOPS[0])     # from=<stop>: skip the earlier prefixes
    stops = LY.STOPS[LY.STOPS.index(first):]
    worst = run(control_jobs() + diag + [job(Ps[0], layers, 0, (len(layers) - 1, s)) for s in stops] + [job(Ps[0], layers)])
  elif st == "layer": worst = run([job(P, [3], n) for P in Ps])
  elif st == "stream": worst = run([job(P, [3], n, stream={(0, "q"), (0, "w1"), (0, "w2")}) for P in Ps])
  elif st == "layers":            # one session: the layer at every P, then q / w1 / w2 streamed at P = 1 and 64
    worst = run([job(P, [3], n) for P in Ps] + [job(P, [3], 0, stream={(0, "q"), (0, "w1"), (0, "w2")}) for P in (1, 64)])
  elif st == "stream-bisect":     # q alone streamed: the prefixes up to qkv, the whole layer; then w1 and w2 streamed, the whole layer
    worst = run([job(Ps[0], [3], 0, (0, s), stream={(0, "q")}) for s in ("q", "k", "qc", "qs", "qkv")] +
                [job(Ps[0], [3], 0, stream={(0, "q")}), job(Ps[0], [3], 0, stream={(0, "w1"), (0, "w2")})])
  elif st == "model": worst = run([job(P, list(range(6)), n) for P in Ps])
  elif st == "replay":            # replay FILE LAYER stop,stop,.. [probe]  (a call saved by examples/stories_chip.py --check)
    worst = run(replay_jobs(sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",")) + (probe_jobs(sys.argv[2]) if "probe" in sys.argv[5:] else []))
  else: raise SystemExit(f"unknown stage {st!r}: logistic | bisect P [layers] | layer P,.. [n] | stream P,.. | model P,.. [n]")
  print(f"{'PASS' if worst == 0 else 'DIFF'} worst={worst}")
