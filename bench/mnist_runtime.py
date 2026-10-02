# our USB runtime (coral/runtime.py) running edgetpu_compiler's whole-network MNIST programs, the tall-image batches that
# bench/mnist_google.py builds: the same programs libedgetpu runs there, so the difference is the runtime alone.
#   CORAL_CLOCK=max .venv/bin/python bench/mnist_runtime.py .compile/<hash>_edgetpu.tflite      (after bench/mnist_google.py build B)
import sys, time, pathlib, numpy as np
from coral.device import EdgeTPU
from coral.executable import load_edgetpu_tflite
from coral.runtime import run_executable
from coral.fused import relayout
ROOT = pathlib.Path(__file__).resolve().parent.parent

if __name__ == "__main__":
  exes = {e.type: e for e in load_edgetpu_tflite(sys.argv[1])}
  exe, tpu = exes["EXECUTION_ONLY"], EdgeTPU()
  run_executable(tpu, exes["PARAMETER_CACHING"])
  d = np.load(ROOT / "models" / "mnist_test.npz")
  X, Y, ours = d["x"], d["y"], d["ours"]
  B = max(h.offset + h.size for h in exe.hints if h.kind == "dma" and h.desc == "INPUT") // X[0].size   # images per invoke
  rows = 7 * B - 6 if B > 1 else 1                         # the output rows (image b's logits in row 7b), see tools/export.py
  n = len(X) // B * B
  logits, st = [], time.perf_counter()
  for i in range(0, n, B):
    out = run_executable(tpu, exe, X[i:i+B].tobytes())     # NCHW with one channel is the tall NHWC image, byte for byte
    logits.append(relayout(exe, out, rows, 1, 10)[::7] if B > 1 else np.frombuffer(out, np.uint8)[None, :10])
  dt, logits = time.perf_counter() - st, np.concatenate(logits)
  print(f"clock {tpu.clock}: {B} images per invoke, {n / dt:.0f} images/s, {dt / (n // B) * 1e3:.2f} ms per invoke, "
        f"accuracy {(logits.argmax(1) == Y[:n]).mean() * 100:.2f}%, logits identical to tinygrad's: {(logits == ours[:n]).all(1).mean() * 100:.1f}%")
