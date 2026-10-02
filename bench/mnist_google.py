# the same MNIST CNN (examples/mnist.py, trained weights in models/mnist.safetensors) through Google's stack: exactly the uint8
# model coral.quantize makes (batch norms folded, quantizations shared across ReLU / max pool), exported by tools/export.py
#   DEV=CPU .venv/bin/python bench/mnist_google.py build [B]         quantize, export, compile with edgetpu_compiler; saves the test
#                                                                   set and our model's uint8 logits (computed on the host).
#                                                                   B > 1: B images stacked into one tall image (tools/export.py)
#   .venv-native/bin/python bench/mnist_google.py run <max|std> <model_edgetpu.tflite>      the test set through libedgetpu
import sys, time, json, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST = ROOT / "models" / "mnist_test.npz"

def build(tall:int=1) -> bytes:
  from tinygrad import nn
  from tinygrad.nn.datasets import mnist
  from mnist import Model, WEIGHTS
  from coral.quantize import quantize
  from tools.export import tflite_chain
  X_train, _, X_test, Y_test = mnist()
  model = Model()
  nn.state.load_state_dict(model, nn.state.safe_load(str(WEIGHTS)))
  chain = quantize(model, X_train[:256].float() / 255.0)["chain"]
  last = chain[-1][0]
  ours = np.concatenate([model(X_test[i:i+1000].float() / 255.0).numpy() for i in range(0, 10000, 1000)])
  ours = np.round(ours / last.ys + last.yz).astype(np.uint8)                        # back to the uint8 logits (exact)
  np.savez(TEST, x=X_test.numpy().astype(np.uint8).reshape(-1, 28, 28, 1), y=Y_test.numpy().astype(np.int64), ours=ours)
  print(f"ours on the host: {(ours.argmax(1) == Y_test.numpy()).mean() * 100:.2f}%")
  return tflite_chain(chain, tall)

def run(clock:str, path:str):
  from tflite_runtime.interpreter import Interpreter, load_delegate
  it = Interpreter(model_path=path, experimental_delegates=[load_delegate(str(ROOT / ".native/lib" / clock / "libedgetpu.1.dylib"))])
  it.allocate_tensors()
  inp, out = it.get_input_details()[0], it.get_output_details()[0]
  X, Y, ours = (np.load(TEST)[k] for k in ("x", "y", "ours"))                 # (an npz decompresses on every access)
  B = inp["shape"][1] // X.shape[1]                                             # images per invoke (stacked into a tall image)
  n = len(X) // B * B
  X, Y, ours = X[:n].reshape(n // B, B * X.shape[1], *X.shape[2:]), Y[:n], ours[:n]
  stride = (out["shape"][1] + 6) // B if B > 1 else 0                           # output rows per image (7 for this CNN)
  for _ in range(10):
    it.set_tensor(inp["index"], X[:1])
    it.invoke()
  st, logits = time.perf_counter(), []
  for i in range(len(X)):
    it.set_tensor(inp["index"], X[i:i+1])
    it.invoke()
    y = it.get_tensor(out["index"])[0]
    logits.append(y[::stride, 0] if B > 1 else y[None])
  dt, logits = time.perf_counter() - st, np.concatenate(logits)
  print(json.dumps({"clock": clock, "images_per_invoke": int(B), "images": len(logits), "accuracy": float((logits.argmax(1) == Y).mean() * 100),
                    "images_per_s": len(logits) / dt, "ms_per_invoke": dt / len(X) * 1e3,
                    "logits_identical_to_ours": float((logits == ours).all(1).mean() * 100)}))

if __name__ == "__main__":
  if sys.argv[1] == "build":
    sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
    from tools.compiler import compile_tflite, compiled_path
    model = build(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    exes, log = compile_tflite(model)
    print(log[log.find("Edge TPU Compiler"):].strip())
    print(compiled_path(model))
  else: run(sys.argv[2], sys.argv[3])
