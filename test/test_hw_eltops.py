# hardware: SOFTMAX (scalar core, float32), L2_NORMALIZATION and SUM/MEAN programs from coral/codegen/eltops.py (byte-identical to
# edgetpu_compiler's) against their numpy bit models; for softmax also which float-to-int rounding the device uses
#   .venv/bin/python test/test_hw_eltops.py
import numpy as np
from coral.codegen import eltops as E
from coral.device import EdgeTPU
from coral.runtime import run_executable

def run(tpu, prog:bytes, x:np.ndarray, io:dict) -> np.ndarray:
  buf = np.zeros(io["in_bytes"], np.uint8)
  buf[:x.size] = x.reshape(-1)
  return np.frombuffer(run_executable(tpu, E.executable(prog, io["in_bytes"], io["out_bytes"]), buf.tobytes()), np.uint8)

if __name__ == "__main__":
  rng, tpu, worst = np.random.default_rng(0), EdgeTPU(), 0
  for rows, n, in_q in [(1, 64, (1/16, 128)), (6, 256, (1/8, 128)), (6, 100, (1/4, 100)), (1, 256, (1/2, 200))]:
    io = E.softmax_io(rows, n)
    x = rng.integers(0, 256, (rows, n), dtype=np.uint8)
    y = run(tpu, E.gen_softmax(rows, n, in_q), x, io)[:rows * io["out_row_stride"]].reshape(rows, io["out_row_stride"])[:, :n]
    res = {f2i: np.abs(y.astype(int) - E.softmax_ref(x, in_q, f2i=f2i).astype(int)) for f2i in ("rne", "trunc", "rna")}
    best = min(res, key=lambda k: (res[k].max(), res[k].sum()))
    worst = max(worst, res[best].max())
    stats = ", ".join(f"{k}: max {v.max()} exact {np.mean(v == 0)*100:5.1f}%" for k, v in res.items())
    print(f"softmax {rows}x{n} in_q {in_q}: {stats}  -> {best}")
  for n, in_q in [(288, (1/16, 128)), (48, (1/8, 100)), (1024, (1/32, 128))]:
    io = E.l2norm_io(n)
    x = rng.integers(0, 256, (1, n), dtype=np.uint8)
    d = np.abs(run(tpu, E.gen_l2norm(n, in_q), x, io)[:n].astype(int) - E.l2norm_ref(x, in_q).reshape(-1).astype(int))
    worst = max(worst, d.max())
    print(f"l2norm {n} in_q {in_q}: max diff {d.max()}, exact {np.mean(d == 0)*100:5.1f}%")
  for kind in ("sum", "mean"):
    for n, in_q, out_q in [(288, (1/16, 128), (1/2, 128)), (48, (1/8, 100), (1/4, 120))]:
      io = E.reduce_io(n)
      x = rng.integers(100, 160, (1, n), dtype=np.uint8)
      y, ref = run(tpu, E.gen_reduce(kind, n, in_q, out_q), x, io)[0], int(E.reduce_ref(kind, x, in_q, out_q).reshape(-1)[0])
      worst = max(worst, abs(int(y) - ref))
      print(f"{kind} {n}: device {y}, model {ref}")
  print(f"{'PASS' if worst == 0 else 'DIFF'} worst={worst}")
