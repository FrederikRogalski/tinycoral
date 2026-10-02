# an ordinary tinygrad CNN (the model of tinygrad's examples/beautiful_mnist.py) on the Edge TPU:
#   python examples/mnist.py train              trains it in float with tinygrad (any device, e.g. DEV=METAL), saves models/mnist.safetensors
#   DEV=CORAL python examples/mnist.py          quantizes it with coral.quantize and runs the test set: the convolutions and
#                                               the linear layer as Edge TPU programs, everything else on the host
#   (DEV=CPU runs the same uint8 model with ordinary kernels: the results are bit-identical)
import sys, time, pathlib, numpy as np
from tinygrad import Tensor, TinyJit, nn, Device, Context
from tinygrad.nn.datasets import mnist
from tinygrad.helpers import getenv
WEIGHTS = pathlib.Path(__file__).resolve().parent.parent / "models" / "mnist.safetensors"

class Model:
  def __init__(self):
    self.layers = [nn.Conv2d(1, 32, 5), Tensor.relu, nn.Conv2d(32, 32, 5), Tensor.relu, nn.BatchNorm(32), Tensor.max_pool2d,
                   nn.Conv2d(32, 64, 3), Tensor.relu, nn.Conv2d(64, 64, 3), Tensor.relu, nn.BatchNorm(64), Tensor.max_pool2d,
                   lambda x: x.flatten(1), nn.Linear(576, 10)]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

def accuracy(model, X:Tensor, Y:Tensor, bs:int) -> tuple[float, float]:
  """on the first len(X) // bs * bs images, bs per TinyJit call (on CORAL the JIT also fuses the TPU layers, coral/chain.py)"""
  run, n = TinyJit(lambda x: model(x).argmax(axis=1).realize()), X.shape[0] // bs * bs
  for i in range(2): run(X[i*bs:(i+1)*bs].contiguous().realize())                  # warm up: capture the JIT
  st, preds = time.perf_counter(), []
  for i in range(0, n, bs): preds.append(run(X[i:i+bs].contiguous().realize()).numpy())
  return float((np.concatenate(preds) == Y[:n].numpy()).mean() * 100), time.perf_counter() - st

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()
  X_train, X_test = X_train.float() / 255.0, X_test.float() / 255.0
  model = Model()
  if sys.argv[1:2] == ["train"]:
    opt = nn.optim.Adam(nn.state.get_parameters(model))
    @TinyJit
    @Context(TRAINING=1)
    def step() -> Tensor:
      opt.zero_grad()
      s = Tensor.randint(512, high=X_train.shape[0])
      loss = model(X_train[s]).sparse_categorical_crossentropy(Y_train[s]).backward()
      return loss.realize(*opt.schedule_step())
    for i in range(getenv("STEPS", 400)):
      loss = step()
      if i % 100 == 99: print(f"step {i+1}: loss {loss.item():.3f}", flush=True)
    WEIGHTS.parent.mkdir(exist_ok=True)
    nn.state.safe_save(nn.state.get_state_dict(model), str(WEIGHTS))
    acc, _ = accuracy(model, X_test, Y_test, 1000)
    print(f"float test accuracy on {Device.DEFAULT}: {acc:.2f}% -> {WEIGHTS}")
  else:
    nn.state.load_state_dict(model, nn.state.safe_load(str(WEIGHTS)))
    n, bs = getenv("N", 10000), getenv("BS", 100)
    acc_f, t_f = accuracy(model, X_test[:n], Y_test[:n], bs)
    print(f"float on the host:         {acc_f:.2f}%  ({n // bs * bs / t_f:6.0f} images/s)", flush=True)
    if Device.DEFAULT == "CORAL":
      from coral.quantize import quantize
      from coral.tpu import runner
      rep = quantize(model, X_train[:256])                # calibrate on 256 training images (folds the batch norms first)
      print(f"quantized: {rep['folded']} batch norms folded; between the layers: {[how for _, how in rep['chain'][:-1]]}")
      acc_q, t_q = accuracy(model, X_test[:n], Y_test[:n], bs)
      print(f"uint8, convs and the linear layer on the TPU: {acc_q:.2f}%  ({n // bs * bs / t_q:6.0f} images/s)   tpu {r.stats if (r:=runner()) else ''}")
