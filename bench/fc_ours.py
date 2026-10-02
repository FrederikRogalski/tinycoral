# FC latency with our stack: a program from our code generator on our driver (weights resident on chip), then the same layer
# through tinygrad (coral.qops.qlinear and kernel selection)
#   .venv/bin/python bench/fc_ours.py <max|high> N K [iters]
import sys, os, time, json, numpy as np
os.environ["CORAL_CLOCK"] = sys.argv[1]
from coral.tpu import FCSpec, runner
N, K = int(sys.argv[2]), int(sys.argv[3])
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 200
rng = np.random.default_rng(0)
w, b = rng.integers(0, 256, (N, K), dtype=np.uint8), np.zeros(N, np.int32)
x = rng.integers(0, 256, (1, K), dtype=np.uint8)
r = runner()
spec = FCSpec(1, N, K, 128, 128, 128, float(np.float32((1/32)*(1/64)/(1/4))))
for _ in range(10): y = r.fc(spec, x, w, b, wkey=0)
ts = []
for _ in range(iters):
  st = time.perf_counter()
  y = r.fc(spec, x, w, b, wkey=0)
  ts.append(time.perf_counter() - st)
# tinygrad-level call: plain tinygrad qlinear, compiled to a TPU program by the CORAL backend's kernel selection
from tinygrad import Tensor
from coral.qops import qlinear
xt, wt, bt = Tensor(x, device="CORAL"), Tensor(w, device="CORAL").realize(), Tensor(b, device="CORAL").realize()
for _ in range(5): qlinear(xt, wt, bt, spec.in_zp, spec.w_zp, spec.out_zp, spec.mult).numpy()
tt = []
for _ in range(iters // 2):
  st = time.perf_counter()
  qlinear(xt, wt, bt, spec.in_zp, spec.w_zp, spec.out_zp, spec.mult).numpy()
  tt.append(time.perf_counter() - st)
print(json.dumps({"min_ms": min(ts)*1e3, "median_ms": float(np.median(ts))*1e3, "tinygrad_median_ms": float(np.median(tt))*1e3,
                  "uploads": r.stats["param_uploads"]}))
