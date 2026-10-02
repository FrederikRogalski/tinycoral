# FC latency at the same clock: Google's stack (edgetpu_compiler + libedgetpu + tflite_runtime, bench/native.py) vs ours (our
# generated program and driver, bench/fc_ours.py). Each measurement runs in its own process: the device takes one at a time.
#   .venv/bin/python bench/fc.py [--max]        std clock (250 MHz) by default; needs Docker, .venv-native and .native/lib (README)
import sys, json, subprocess, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.tflite_gen import fc_model
from tools.compiler import compiled_path

SHAPES = [(64, 64), (288, 288), (768, 288), (288, 768), (1024, 1024), (2048, 1024), (1024, 4096)]
def run(cmd:list[str]) -> dict: return json.loads(subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip().splitlines()[-1])

if __name__ == "__main__":
  clock_native, clock_ours = ("max", "max") if "--max" in sys.argv else ("std", "high")
  for N, K in SHAPES:
    w = np.random.default_rng(0).integers(0, 256, (N, K), dtype=np.uint8)
    model = compiled_path(fc_model(w, np.zeros(N, np.int32), in_q=(1/32, 128), w_q=(1/64, 128), out_q=(1/4, 128)))
    nat = run([".venv-native/bin/python", "bench/native.py", clock_native, str(model), "300"])
    ours = run([".venv/bin/python", "bench/fc_ours.py", clock_ours, str(N), str(K), "300"])
    print(f"FC {N:5d}x{K:5d}: native median {nat['median_ms']:.3f} ms | ours runner {ours['median_ms']:.3f} ms, "
          f"via tinygrad {ours['tinygrad_median_ms']:.3f} ms", flush=True)
