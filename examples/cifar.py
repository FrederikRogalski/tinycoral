# tinygrad's CIFAR-10 speedrun model (SpeedyConvNet of tinygrad's examples/beautiful_cifar.py) on the Edge TPU:
#   DEV=METAL python examples/cifar.py train     runs tinygrad's own training script, unchanged, and saves models/cifar.safetensors
#   DEV=CORAL python examples/cifar.py           freezes the batch norms' statistics (the script normalizes with the batch's own,
#                                                which a deployed model can't), quantizes with coral.quantize and runs the test set:
#                                                the convolutions and the linear layer on the TPU, everything else on the host
import sys, time, runpy, pathlib, numpy as np
from tinygrad import Tensor, TinyJit, nn, Device
from tinygrad.nn.datasets import cifar
from tinygrad.helpers import getenv
ROOT = pathlib.Path(__file__).resolve().parent.parent
TINYGRAD = ROOT / "tinygrad-src"                                  # the script imports tinygrad's extra/ (not part of the package)
SCRIPT = TINYGRAD / "examples" / "beautiful_cifar.py"
WEIGHTS = ROOT / "models" / "cifar.safetensors"

if __name__ == "__main__" and sys.argv[1:2] == ["train"]:
  sys.path.insert(0, str(TINYGRAD))
  g = runpy.run_path(str(SCRIPT), run_name="__main__")           # tinygrad's script as it is: trains, evaluates every epoch
  state = {k: v.float() for k, v in nn.state.get_state_dict(g["model"]).items()}
  state["cifar10_mean"], state["cifar10_std"] = g["cifar10_mean"].float(), g["cifar10_std"].float()
  WEIGHTS.parent.mkdir(exist_ok=True)
  nn.state.safe_save(state, str(WEIGHTS))
  print(f"saved {WEIGHTS}")

def load():
  """SpeedyConvNet with the trained weights, and the input normalization"""
  from tinygrad.helpers import DEFAULT_FLOAT
  sys.path.insert(0, str(TINYGRAD))
  from examples.beautiful_cifar import SpeedyConvNet
  DEFAULT_FLOAT.value = "float32"                                 # (importing the script switches tinygrad's default float to half)
  model, state = SpeedyConvNet(), nn.state.safe_load(str(WEIGHTS))
  mean, std = state.pop("cifar10_mean").to(Device.DEFAULT), state.pop("cifar10_std").to(Device.DEFAULT)
  nn.state.load_state_dict(model, state)
  return model, lambda X: (X.float() - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)

def freeze_batchnorms(model, X:Tensor, bs:int=2500):
  """the script's batch norms normalize with the statistics of whatever batch they see. Fix them: the mean and variance of each
  one's input over X (in batches of bs, the script's eval batch size), as running statistics"""
  from coral.quantize import _slots
  bns, acc = [l for _, _, l in _slots(model) if isinstance(l, nn.BatchNorm)], {}
  calls = {id(b): b.__class__.__call__ for b in bns}
  def record(b, x):
    v = x.numpy().astype(np.float64)
    n, s1, s2 = acc.get(id(b), (0, 0, 0))
    acc[id(b)] = (n + v.size // v.shape[1], s1 + v.sum((0, 2, 3)), s2 + (v * v).sum((0, 2, 3)))
    return calls[id(b)](b, x)
  nn.BatchNorm.__call__ = record
  try:
    for i in range(0, X.shape[0], bs): model(X[i:i+bs]).realize()
  finally: nn.BatchNorm.__call__ = calls[id(bns[0])]
  for b in bns:
    n, s1, s2 = acc[id(b)]
    b.running_mean, b.running_var = Tensor((s1 / n).astype(np.float32)), Tensor((s2 / n - (s1 / n) ** 2).astype(np.float32))
    b.track_running_stats = True

def accuracy(model, X:Tensor, Y:Tensor, bs:int) -> tuple[float, float]:
  """on the first len(X) // bs * bs images, bs per TinyJit call (on CORAL the JIT also fuses TPU layers, coral/chain.py)"""
  run, n = TinyJit(lambda x: model(x).argmax(axis=1).realize()), X.shape[0] // bs * bs
  for i in range(2): run(X[i*bs:(i+1)*bs].clone().realize())                      # (a slice of a realized tensor is a view)
  st, preds = time.perf_counter(), []
  for i in range(0, n, bs): preds.append(run(X[i:i+bs].clone().realize()).numpy())
  return float((np.concatenate(preds) == Y[:n].numpy()).mean() * 100), time.perf_counter() - st

if __name__ == "__main__" and sys.argv[1:2] != ["train"]:
  X_train, Y_train, X_test, Y_test = cifar()
  model, preprocess = load()
  X_train, X_test = preprocess(X_train[:10000]).realize(), preprocess(X_test).realize()
  freeze_batchnorms(model, X_train)
  n, bs = getenv("N", 10000), getenv("BS", 100)
  acc_f, t_f = accuracy(model, X_test[:n], Y_test[:n], bs)
  print(f"float on {Device.DEFAULT} (batch norms frozen): {acc_f:.2f}%  ({n // bs * bs / t_f:6.0f} images/s)", flush=True)
  if Device.DEFAULT == "CORAL":
    from coral.quantize import quantize
    from coral.tpu import runner
    rep = quantize(model, X_train[:256])
    print(f"quantized: {rep['folded']} batch norms folded; between the layers: {[how for _, how in rep['chain'][:-1]]}")
    acc_q, t_q = accuracy(model, X_test[:n], Y_test[:n], bs)
    print(f"uint8, convs and the linear layer on the TPU: {acc_q:.2f}%  ({n // bs * bs / t_q:6.0f} images/s)   tpu {r.stats if (r:=runner()) else ''}")
