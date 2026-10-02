# plain tinygrad on hardware: the same qlinear code on DEV=CPU and on the Coral (tinygrad's compiler + our kernel selection +
# our generated programs) must give bit-identical results
import time, numpy as np
from tinygrad import Tensor
import coral.tpu as ct
from coral.qops import qlinear

if __name__ == "__main__":
  rng, ok, n = np.random.default_rng(5), True, 0
  calls = {"tpu": 0}
  orig = ct.run_fc
  def counted(*a, **k):
    calls["tpu"] += 1
    return orig(*a, **k)
  ct.run_fc = counted
  for M, N, K, tie in [(1, 288, 288, 0), (1, 864, 288, 0), (1, 100, 300, 0), (16, 288, 288, 0), (128, 1536, 288, 0), (256, 288, 768, 0),
                       (256, 864, 288, 0), (64, 500, 333, 0), (1, 288, 288, 1), (128, 288, 288, 1)]:
    x, w = rng.integers(0, 256, (M, K), dtype=np.uint8), rng.integers(0, 256, (N, K), dtype=np.uint8)
    b = rng.integers(-20000, 20000, N).astype(np.int32)
    in_zp, w_zp, out_zp = (int(v) for v in rng.integers(0, 256, 3))
    acc = (x[:4].astype(np.int64) - in_zp) @ (w.astype(np.int64) - w_zp).T + b
    mult = float(np.float32(rng.uniform(60, 140) / max(1, np.abs(acc).max())))
    if tie:                                    # exact .5 ties everywhere: small values around the zero points, multiplier 0.5
      x, w = rng.integers(120, 137, (M, K), dtype=np.uint8), rng.integers(120, 137, (N, K), dtype=np.uint8)
      b, in_zp, w_zp, out_zp, mult = rng.integers(-50, 50, N).astype(np.int32), 128, 128, 128, 0.5
    outs = {}
    for dev in ("CPU", "CORAL"):
      args = [Tensor(a, device=dev) for a in (x, w, b)]
      st = time.perf_counter()
      outs[dev] = qlinear(*args, in_zp, w_zp, out_zp, mult).numpy()
      outs[dev + "_t"] = time.perf_counter() - st
    same = np.array_equal(outs["CPU"], outs["CORAL"])
    ok &= same
    n += 1
    print(f"M={M:3d} N={N:4d} K={K:3d}{' ties' if tie else '     '}: CPU == CORAL {same}  (TPU programs so far: {calls['tpu']})", flush=True)
  print("PASS" if ok else "FAIL", f"{n} shapes, {calls['tpu']} TPU programs")
