# Google native stack: edgetpu_compiler output + libedgetpu (feranick arm64 build of google-coral/libedgetpu) + tflite_runtime
#   .venv-native/bin/python bench/native.py <max|std> model_edgetpu.tflite [iters]
import sys, time, json, pathlib, hashlib, numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
ROOT = pathlib.Path(__file__).resolve().parent.parent
lib = str(ROOT / ".native/lib" / sys.argv[1] / "libedgetpu.1.dylib")
it = Interpreter(model_path=sys.argv[2], experimental_delegates=[load_delegate(lib)])
it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]
x = np.random.default_rng(0).integers(0, 256, inp["shape"]).astype(inp["dtype"])
it.set_tensor(inp["index"], x)
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 200
for _ in range(10): it.invoke()
ts = []
for _ in range(iters):
  st = time.perf_counter()
  it.invoke()
  ts.append(time.perf_counter() - st)
y = it.get_tensor(out["index"])
print(json.dumps({"min_ms": min(ts)*1e3, "median_ms": float(np.median(ts))*1e3, "out_sha": hashlib.sha1(y.tobytes()).hexdigest()[:12],
                  "x_sha": hashlib.sha1(x.tobytes()).hexdigest()[:12]}))
