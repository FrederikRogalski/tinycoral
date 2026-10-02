# our runtime (the pure Python USB driver) running an edgetpu_compiler model, e.g. MobileNet, on the same input as bench/native.py
#   .venv/bin/python bench/ours.py <max|high> model_edgetpu.tflite [iters]
import sys, os, time, json, hashlib, numpy as np
os.environ["CORAL_CLOCK"] = sys.argv[1]
from coral.device import EdgeTPU
from coral.executable import load_edgetpu_tflite
from coral.runtime import Model
exes = load_edgetpu_tflite(sys.argv[2])
inf = [e for e in exes if e.type != "PARAMETER_CACHING"][0]
shape = (1, inf.inputs[0].y, inf.inputs[0].x, inf.inputs[0].z) if inf.inputs[0].y > 1 else (1, inf.inputs[0].z)
# same input bytes as bench/native.py (rng(0).integers(0, 256, shape)) for the common model input shapes
x = np.random.default_rng(0).integers(0, 256, shape).astype(np.uint8)
m = Model(EdgeTPU(), exes)
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 200
for _ in range(10): out = m(x.tobytes())
ts = []
for _ in range(iters):
  st = time.perf_counter()
  out = m(x.tobytes())
  ts.append(time.perf_counter() - st)
n_out = inf.outputs[0].y * inf.outputs[0].x * inf.outputs[0].z
print(json.dumps({"min_ms": min(ts)*1e3, "median_ms": float(np.median(ts))*1e3, "out_sha": hashlib.sha1(out[:n_out]).hexdigest()[:12],
                  "x_sha": hashlib.sha1(x.tobytes()).hexdigest()[:12]}))
