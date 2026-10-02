# hardware: the attention block (coral/codegen/attention_block.py) and its RoPE piece against their numpy bit models, through
# tools/hw.py. One stage per run, in this order (stop at the first failure, never retry a failed transfer):
#   rope Q        edgetpu_compiler's program for x -> FC, FC -> MUL(., cos), MUL(., sin) -> ADD (quantization set Q of
#                 test_codegen_attention_block.ROPE_QUANTS), byte-identical to our gen_rope_piece      vs fc_ref, mul_ref, add_ref
#   block P [n]   gen_attention_block(P) with layer 3 of TinyStories-15M (calibrated, real weights; no compiler equivalent:
#                 tools/hw.run(new=True)), 3 inputs; n more calls of the first input for timing
#   .venv/bin/python test/test_hw_attention_block.py block 64 30
# Inputs of the block: x, cos, sin' and the K / V cache rows of a greedy story at position P-1 (the float model's states quantized), plus
# x from another position and a Gaussian x with the same cache (any input is valid for the bit model).
import sys, pathlib, dataclasses, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))
import numpy as np
from coral.codegen import attention_block as B
from coral.executable import Bitstream
from tools import hw

def report(name:str, got:np.ndarray, ref:np.ndarray) -> int:
  d = np.abs(got.astype(int) - ref.astype(int))
  print(f"{name}: max diff {d.max()}, exact {np.mean(d == 0) * 100:.1f}% of {d.size}", flush=True)
  return int(d.max())

def stage_rope(qi:int) -> int:
  from test_codegen_attention_block import rope_model, gen_rope_piece, ROPE_QUANTS
  from tools.compiler import compile_tflite
  q = ROPE_QUANTS[qi]
  exes, _ = compile_tflite(rope_model(q, True))
  ex = {e.type: e for e in exes}
  ours = dataclasses.replace(ex["EXECUTION_ONLY"], bitstreams=[Bitstream(gen_rope_piece(q, True), [])])
  rng = np.random.default_rng(10 + qi)
  wr = np.random.default_rng(0)                             # the model's weights: rope_model draws q's, then sq's, from seed 0
  Wq, Wsq = wr.integers(0, 256, (288, 288)).astype(np.uint8), wr.integers(0, 256, (288, 288)).astype(np.uint8)
  xs = []
  for _ in range(3):
    pos = int(rng.integers(0, 256))
    c, s = B.rope_inputs(pos, dict(cos=q["c"], sin=q["s"]))
    xs.append((np.clip(np.rint(rng.normal(q["x"][1], 6, 288)), 0, 255).astype(np.uint8), c, s))
  outs = hw.run(ex["PARAMETER_CACHING"], ours, [dict(x=x.tobytes(), c=c.tobytes(), s=s.tobytes()) for x, c, s in xs], oracle=ex)
  worst = 0
  for i, (o, (x, c, s)) in enumerate(zip(outs, xs)):
    qv, sqv = B.fc_ref(x, Wq, q["x"], q["w"], q["q"]), B.fc_ref(x, Wsq, q["x"], q["w"], q["sq"])
    qc, qs = B.A.mul_ref(qv, c, q["q"], q["c"], q["qc"]), B.A.mul_ref(sqv, s, q["sq"], q["s"], q["qs"])
    ref = B.add_ref(qs, qc, q["qs"], q["qc"], q["y"])          # the compiler's ADD: operand 1 = qs (see gen_rope_piece)
    worst = max(worst, report(f"rope set {qi} #{i} (q in [{qv.min()}, {qv.max()}])", np.frombuffer(o, np.uint8)[:288], ref))
  return worst

def block_inputs(P:int, quant, st, rng) -> list[tuple]:
  """three inputs at P with the story's cache rows 0..P-2: the story's x and cos / sin' at position P-1; x of position P+49 with cos / sin'
  of position P+36; a Gaussian x with cos / sin' of position P+73 (the program takes any cos / sin', so P = 1 rotates too)"""
  def qz(v, k): return np.clip(np.rint(v / quant[k][0]) + quant[k][1], 0, 255).astype(np.uint8)
  K, V = qz(st["k"][:P - 1], "k"), qz(st["v"][:P - 1], "v")
  x0 = qz(st["xn"][P - 1], "x")
  x1 = qz(st["xn"][(P - 1 + 50) % 256], "x")
  x2 = np.clip(np.rint(rng.normal(quant["x"][1], x0.astype(float).std(), 288)), 0, 255).astype(np.uint8)
  return [(x, *B.rope_inputs((P - 1 + d) % 256, quant), K, V) for x, d in ((x0, 0), (x1, 37), (x2, 74))]

def stage_block(P:int, timing:int=0, layer:int=3) -> int:
  from test_codegen_attention_block import calibrated
  quant, wqkv, wo, st = calibrated(layer)
  pc, prog, io = B.gen_attention_block(P, quant)
  cex, eex = B.block_executables(pc, prog, P)
  blob = B.block_params(wqkv, wo, quant)
  xs = block_inputs(P, quant, st, np.random.default_rng(20 + P))
  print(f"block P={P}: program {len(prog)} bytes, inputs {sum(n for _, n in io['inputs'])} bytes, outputs {sum(n for _, n in io['outputs'])} bytes, "
        f"blob {len(blob)} bytes", flush=True)
  times = []
  outs = hw.run(cex, eex, [B.block_inputs(*x) for x in xs] + [B.block_inputs(*xs[0])] * timing, params=blob, new=True, times=times)
  if (d := os.environ.get("BLOCK_SAVE")):        # raw device outputs and inputs, for offline analysis
    np.savez(f"{d}/block_P{P}.npz", out=np.stack([np.frombuffer(o, np.uint8) for o in outs[:len(xs)]]), times=np.array(times),
             **{f"in{i}_{k}": v for i, x in enumerate(xs) for k, v in zip(("x", "cos", "sin", "K", "V"), x)})
  worst = 0
  for i, (o, x) in enumerate(zip(outs, xs)):
    (ro, rk, rv), parts = B.attention_block_ref(*x, wqkv, wo, quant, parts=True)
    go, gk, gv = B.block_outputs(o)
    worst = max(worst, report(f"P={P} #{i} k'", gk, rk), report(f"P={P} #{i} v ", gv, rv),
                report(f"P={P} #{i} o  (p max {parts['p'].max()})", go, ro))
    of, _, _ = B.attention_block_float(x[0], (P - 1 + (0, 37, 74)[i]) % 256, x[3], x[4], wqkv, wo, quant)
    def dq(t): return (t.astype(float) - quant["o"][1]) * quant["o"][0]
    print(f"        o vs float math: relative L2 error {np.linalg.norm(dq(go) - dq(of)) / np.linalg.norm(dq(of)):.3f}")
  same = all(o == outs[0] for o in outs[len(xs):])
  if timing: print(f"wall time per call (USB included), {timing} calls: median {np.median(times[len(xs):]) * 1e3:.3f} ms, "
                   f"min {min(times[len(xs):]) * 1e3:.3f} ms; repeated calls identical: {same}")
  return worst if same else max(worst, 999)

if __name__ == "__main__":
  st = sys.argv[1]
  if st == "rope": worst = stage_rope(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
  elif st == "block": worst = stage_block(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
  else: raise SystemExit(f"unknown stage {st!r}: rope Q | block P [timed calls]")
  print(f"{'PASS' if worst == 0 else 'DIFF'} worst={worst}")
