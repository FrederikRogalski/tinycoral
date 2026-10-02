# hardware: the attention pieces and the composed attention program (coral/codegen/attention.py) against their numpy bit models,
# through tools/hw.py. One stage per run, in this order (stop at the first failure):
#   scores P     gen_scores (byte-identical to edgetpu_compiler's program)          vs scores_ref
#   pv P         gen_pv (byte-identical)                                             vs pv_ref
#   smmul        edgetpu_compiler's own SOFTMAX -> MUL [1,6,64] (the scalar memory -> tile 0 path the composition uses)
#   attn P       gen_attention (no compiler equivalent: tools/hw.run(new=True))
#   .venv/bin/python test/test_hw_attention.py scores 64
# Inputs: Gaussian uint8 with the spread of TinyStories-15M's real q / k / v under a calibrated quantization (layer 3), plus fixed
# quantizations of the pieces' tests.
import sys, pathlib, dataclasses
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))
import numpy as np
from coral.codegen import attention as A, eltops as E
from coral.executable import Bitstream
from tools import hw

# layer 3 of TinyStories-15M calibrated on a greedy story (bench-style ranges): every intermediate inside its range
REAL_Q = dict(q=(0.08273, 125), k=(0.07579, 121), qk=(0.55815, 173), s=(0.72165, 177), p=(1/256, 0), v=(0.02456, 144), pv=(0.01219, 117),
              o=(0.01303, 124), beta=48 ** -0.5)

def gauss(rng, zp:int, sd:float, shape) -> np.ndarray: return np.clip(np.rint(rng.normal(zp, sd, shape)), 0, 255).astype(np.uint8)

def report(name:str, got:np.ndarray, ref:np.ndarray) -> int:
  d = np.abs(got.astype(int) - ref.astype(int))
  print(f"{name}: max diff {d.max()}, exact {np.mean(d == 0) * 100:.1f}% of {d.size}", flush=True)
  return int(d.max())

def stage_scores(P:int) -> int:
  from test_codegen_attention import scores_model, compiled
  q = {k: REAL_Q[k] for k in ("q", "k", "qk", "s")}
  exe = compiled(scores_model(P, q))
  ours = dataclasses.replace(exe, bitstreams=[Bitstream(A.gen_scores(P, q), [])])
  rng = np.random.default_rng(1)
  xs = [(gauss(rng, 125, 17, (6, 48)), gauss(rng, 121, 20, (6, P, 48))) for _ in range(3)]
  outs = hw.run(None, ours, [dict(q=a.tobytes(), k=b.tobytes()) for a, b in xs], oracle={"STAND_ALONE": exe})
  return max(report(f"scores P={P} #{i}", A.scores_unpack(o, P), A.scores_ref(a, b, q)) for i, (o, (a, b)) in enumerate(zip(outs, xs)))

def stage_pv(P:int) -> int:
  from test_codegen_attention import pv_model, compiled
  q = {k: REAL_Q[k] for k in ("p", "v", "pv", "o")}
  exe = compiled(pv_model(P, q))
  ours = dataclasses.replace(exe, bitstreams=[Bitstream(A.gen_pv(P, q), [])])
  rng = np.random.default_rng(2)
  xs = []
  for _ in range(3):
    p = rng.dirichlet(np.full(P, 0.3), 6) * 256                    # a probability row per head, expanded over the 48 channels
    xs.append((np.repeat(np.clip(np.rint(p), 0, 255).astype(np.uint8)[:, :, None], 48, 2), gauss(rng, 144, 17.5, (6, P, 48))))
  outs = hw.run(None, ours, [dict(p=a.tobytes(), v=b.tobytes()) for a, b in xs], oracle={"STAND_ALONE": exe})
  return max(report(f"p.V P={P} #{i}", np.frombuffer(o, np.uint8)[:288].reshape(6, 48), A.pv_ref(a, b, q))
             for i, (o, (a, b)) in enumerate(zip(outs, xs)))

def stage_smmul() -> int:
  """edgetpu_compiler's SOFTMAX [1,6,64] -> MUL with v [1,6,64]: everything on tile 0, the softmax result back from scalar memory"""
  import tflite
  from tools.tflite_gen import Model
  from test_codegen_attention import compiled, _opt, MUL_OPT
  B = tflite.BuiltinOperator
  m = Model()
  x = m.tensor("x", [1, 6, 64], np.uint8, 1/16, 128)
  p = m.tensor("p", [1, 6, 64], np.uint8, 1/256, 0)
  m.op(B.SOFTMAX, [x], [p], tflite.BuiltinOptions.SoftmaxOptions, _opt(tflite.SoftmaxOptionsStart, tflite.SoftmaxOptionsEnd,
                                                                      (tflite.SoftmaxOptionsAddBeta, 1.0)))
  v = m.tensor("v", [1, 6, 64], np.uint8, 1/16, 128)
  y = m.tensor("y", [1, 6, 64], np.uint8, 1/64, 128)
  m.op(B.MUL, [p, v], [y], tflite.BuiltinOptions.MulOptions, MUL_OPT)
  exe = compiled(m.build([x, v], [y]))
  rng = np.random.default_rng(3)
  xs = [(rng.integers(0, 256, (6, 64), dtype=np.uint8), rng.integers(0, 256, (6, 64), dtype=np.uint8)) for _ in range(3)]
  outs = hw.run(None, exe, [dict(x=a.tobytes(), v=b.tobytes()) for a, b in xs], oracle={"STAND_ALONE": exe})
  def ref(a, b): return A.mul_ref(E.softmax_ref(a, (1/16, 128), (1/256, 0)), b, (1/256, 0), (1/16, 128), (1/64, 128))
  return max(report(f"softmax->mul #{i}", np.frombuffer(o, np.uint8)[:384].reshape(6, 64), ref(a, b)) for i, (o, (a, b)) in enumerate(zip(outs, xs)))

def stage_attn(P:int, n:int=3, timing:int=0) -> int:
  prog, io = A.gen_attention(P, REAL_Q)
  exe = A.attention_executable(prog, P)
  rng = np.random.default_rng(4 + P)
  xs = [(gauss(rng, 125, 17, 288), gauss(rng, 121, 20, (P, 288)), gauss(rng, 144, 17.5, (P, 288))) for _ in range(n)]
  times = []
  outs = hw.run(None, exe, [A.attention_inputs(*x) for x in xs] + [A.attention_inputs(*xs[0])] * timing, new=True, times=times)
  if (d := __import__("os").environ.get("ATTN_SAVE")):        # raw device outputs and inputs, for offline analysis
    np.savez(f"{d}/attn_P{P}.npz", out=np.stack([np.frombuffer(o, np.uint8) for o in outs[:n]]), q=np.stack([x[0] for x in xs]),
             K=np.stack([x[1] for x in xs]), V=np.stack([x[2] for x in xs]), times=np.array(times))
  worst = 0
  for i, (o, x) in enumerate(zip(outs, xs)):
    ref, parts = A.attention_ref(*x, REAL_Q, parts=True)
    worst = max(worst, report(f"attention P={P} #{i} (p max {parts['p'].max()})", np.frombuffer(o, np.uint8)[:288], ref))
  same = all(o == outs[0] for o in outs[n:])
  if timing:
    print(f"wall time per call (USB included), {timing} calls: median {np.median(times[n:]) * 1e3:.3f} ms, min {min(times[n:]) * 1e3:.3f} ms;"
                   f" repeated calls identical: {same}")
  return worst

def stage_attn_smem(P:int, n:int=3) -> int:
  """debug variant: the composed program up to the softmax, then the scalar memory's score words and p words to the host"""
  prog, io = A.gen_attention(P, REAL_Q, debug_smem=True)
  exe = A.attention_executable(prog, P, io)
  rng = np.random.default_rng(4 + P)
  xs = [(gauss(rng, 125, 17, 288), gauss(rng, 121, 20, (P, 288)), gauss(rng, 144, 17.5, (P, 288))) for _ in range(n)]
  outs = hw.run(None, exe, [A.attention_inputs(*x) for x in xs], new=True)
  if (d := __import__("os").environ.get("ATTN_SAVE")):
    np.savez(f"{d}/attn_smem_P{P}.npz", out=np.stack([np.frombuffer(o, np.uint8) for o in outs]), q=np.stack([x[0] for x in xs]),
             K=np.stack([x[1] for x in xs]), V=np.stack([x[2] for x in xs]))
  worst = 0
  for i, (o, x) in enumerate(zip(outs, xs)):
    w = np.frombuffer(o, "<u4")[:12 * P]
    sw, pw = w[:6 * P].reshape(6, P), w[6 * P:].reshape(6, P)
    _, parts = A.attention_ref(*x, REAL_Q, parts=True)
    print(f"#{i} score words, bytes 1..3 of the first row: {[hex(int(v) >> 8) for v in sw[0]]}")
    worst = max(worst, report(f"  scores (byte 0) #{i}", (sw & 0xff).astype(np.uint8), parts["s"]))
    worst = max(worst, report(f"  p words #{i}", (pw & 0xff).astype(np.uint8), parts["p"]))
    if not np.all(pw == (pw & 0xff) * 0x01010101):
      print("  p words are not 4 copies of their byte")
      worst = max(worst, 999)
  return worst

if __name__ == "__main__":
  st, P = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64
  worst = {"scores": lambda: stage_scores(P), "pv": lambda: stage_pv(P), "smmul": stage_smmul,
           "attn": lambda: stage_attn(P, timing=int(sys.argv[3]) if len(sys.argv) > 3 else 0), "attn_smem": lambda: stage_attn_smem(P)}[st]()
  print(f"{'PASS' if worst == 0 else 'DIFF'} worst={worst}")
