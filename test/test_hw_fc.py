# hardware sweep: FC programs from our code generator (no compiler) vs the numpy reference, random shapes/weights/quantization
import sys, time, numpy as np
from coral.tpu import FCSpec, fc_reference, TPURunner

if __name__ == "__main__":
  rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
  r = TPURunner()
  shapes = [(1, 1), (7, 5), (64, 64), (65, 63), (100, 300), (288, 288), (768, 288), (288, 768), (1024, 1024), (1500, 700), (333, 2000), (16, 4096),
            (1024, 4096)]
  shapes += [(int(rng.integers(1, 1100)), int(rng.integers(1, 4097))) for _ in range(12)]
  worst, st = 0, time.perf_counter()
  for N, K in shapes:
    x, w = rng.integers(0, 256, (1, K), dtype=np.uint8), rng.integers(0, 256, (N, K), dtype=np.uint8)
    b = rng.integers(-20000, 20000, N).astype(np.int32)
    in_zp, w_zp, out_zp = (int(v) for v in rng.integers(0, 256, 3))
    acc = (x.astype(np.int64) - in_zp) @ (w.astype(np.int64) - w_zp).T + b
    spec = FCSpec(1, N, K, in_zp, w_zp, out_zp, float(np.float32(rng.uniform(60, 140) / max(1, np.abs(acc).max()))))
    y = r.fc(spec, x, w, b, wkey=(N, K))
    d = np.abs(y.astype(int) - fc_reference(spec, x, w, b).astype(int))
    worst = max(worst, d.max())
    print(f"N={N:5d} K={K:5d}: max diff {d.max()} exact {np.mean(d == 0)*100:5.1f}%")
  print(f"{'PASS' if worst <= 1 else 'FAIL'} worst={worst} in {time.perf_counter()-st:.1f}s, stats {r.stats}")
