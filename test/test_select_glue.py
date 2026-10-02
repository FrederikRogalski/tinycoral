#!/usr/bin/env python
# The glue between two layers (coral/select.py select_glue, for coral/chain.py): in a model quantized by coral.quantize the
# kernels between two selected kernels map uint8 to uint8 (dequantize, ReLU, max pool, quantize). select_glue recognizes the
# exact copies (CopySpec, with a clamp) and exact max pools (PoolSpec); everything else, and select() itself, leaves them alone.
# MOCKCORAL=1 (the default here): the TPU is the bit-exact numpy model.
#   .venv/bin/python test/test_select_glue.py [TestGlue[.test_near_misses] ...]
import os
os.environ.setdefault("MOCKCORAL", "1")       # never touch the USB device unless asked to explicitly (MOCKCORAL= ...)
import pathlib, sys, unittest
import numpy as np
from tinygrad import Tensor, dtypes, nn, TinyJit
from tinygrad.helpers import DEV
from tinygrad.uop.ops import UOp
from tinygrad.runtime.ops_coral import CoralProgram
from coral.quantize import quantize
from coral.select import select, select_glue, why_not_glue, glue_reference, CopySpec, PoolSpec, CopyMatch, PoolMatch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
from mnist import Model                       # noqa: E402  examples/mnist.py: tinygrad's MNIST CNN

F32, U8 = dtypes.float32, dtypes.uint8
DEV.value = "CPU"                             # the glue kernels below are built on CPU (the model is moved to CORAL)
COUNT = {"recognized": 0, "rejected": 0, "checked": 0}

def kernel_asts(t:Tensor) -> list[UOp]: return [c.without_after.body for c in t.schedule_linear().src]
def glue(t:Tensor):
  """select_glue of the kernel that computes t"""
  m = select_glue(kernel_asts(t)[-1])
  COUNT["recognized" if m is not None else "rejected"] += 1
  return m
def dq(y:Tensor, s:float, z:int) -> Tensor: return (y.cast(F32) - z) * s                           # as QLayer dequantizes
def q(x:Tensor, s:float, z:int) -> Tensor: return (x / s + z).round().clip(0, 255).cast(U8)        # as QLayer quantizes

# ***** the quantized MNIST CNN *****

def quantized_mnist(seed:int=0):
  """examples/mnist.py's model with random weights and batch norms as after training, quantized by coral.quantize on DEV=CORAL"""
  Tensor.manual_seed(seed)
  rng, model = np.random.default_rng(seed), Model()
  for bn in (l for l in model.layers if isinstance(l, nn.BatchNorm)):
    C = bn.weight.shape[0]
    bn.weight, bn.bias = Tensor(rng.uniform(0.5, 1.5, C).astype(np.float32)), Tensor(rng.normal(0, 0.1, C).astype(np.float32))
    bn.running_mean, bn.running_var = Tensor(rng.normal(0, 0.2, C).astype(np.float32)), Tensor(rng.uniform(0.5, 2, C).astype(np.float32))
  for p in nn.state.get_state_dict(model).values(): p.replace(p.to("CORAL").realize())
  rep = quantize(model, Tensor(rng.uniform(0, 1, (64, 1, 28, 28)).astype(np.float32), device="CORAL"))
  return model, rep

class TestMNIST(unittest.TestCase):
  @classmethod
  def setUpClass(cls): cls.model, cls.rep = quantized_mnist()

  def test_quantization_is_linked(self):
    self.assertEqual(self.rep["folded"], 2)
    self.assertEqual([how for _, how in self.rep["chain"]], ["relu", "relu,maxpool2", "relu", "relu,maxpool2", None])

  def test_glue_recognized(self):
    """the 11 kernels of a forward: quantize, conv, copy, conv, pool, conv, copy, conv, pool, fc, dequantize. select() takes
    the 4 convs and the fc, select_glue the 2 copies and the 2 pools (one of them stored flattened, as the fc reads it)"""
    for B in (1, 7):
      with self.subTest(B=B):
        x = Tensor(np.random.default_rng(B).uniform(0, 1, (B, 1, 28, 28)).astype(np.float32), device="CORAL").realize()
        calls = [c.without_after for c in self.model(x).schedule_linear().src]
        asts = [c.body for c in calls]
        kinds = [type(select(a)).__name__ if select(a) else type(select_glue(a)).__name__ if select_glue(a) else "clang" for a in asts]
        self.assertEqual(kinds, ["clang", "ConvMatch", "CopyMatch", "ConvMatch", "PoolMatch", "ConvMatch", "CopyMatch", "ConvMatch",
                                 "PoolMatch", "FCMatch", "clang"])
        self.assertEqual([select_glue(a).spec for a in asts if select_glue(a)],
                         [CopySpec(B*32*24*24), PoolSpec(B, 32, 20, 20, 2, 2), CopySpec(B*64*8*8), PoolSpec(B, 64, 6, 6, 2, 2)])
        COUNT["recognized"] += 4
        def buf(i, p): return calls[i].src[1 + p.arg.slot]         # a PARAM's buffer: the call's argument at its slot
        for i, a in enumerate(asts):
          if (g:=select_glue(a)) is not None:
            self.assertIsNone(select(a))                            # alone, a glue kernel stays clang
            self.assertIs(buf(i, g.params[1]), buf(i - 1, select(asts[i - 1]).params[0]))   # in: the layer before's out
            self.assertIs(buf(i, g.params[0]), buf(i + 1, select(asts[i + 1]).params[1]))   # out: the layer after's x

  def test_jit_chain(self):
    """under TinyJit, coral/chain.py fuses the 9 TPU-able kernels (4 convs, 2 copies, 2 pools, the fc) into one chain call:
    bit-identical to the forward without fusion"""
    calls, call = {"tpu": 0, "clang": 0}, CoralProgram.__call__
    def count(prg, *args, **kwargs):
      calls["tpu" if prg.tpu else "clang"] += 1
      return call(prg, *args, **kwargs)
    fwd = TinyJit(lambda x: self.model(x).realize())
    rng = np.random.default_rng(3)
    CoralProgram.__call__ = count
    try:
      for i in range(4):
        x = Tensor(rng.uniform(0, 1, (7, 1, 28, 28)).astype(np.float32), device="CORAL").realize()
        calls.update(tpu=0, clang=0)
        got = fwd(x).numpy()
        jit_calls = dict(calls)
        np.testing.assert_array_equal(got, self.model(x).numpy())
        if i >= 2: self.assertEqual(jit_calls, {"tpu": 1, "clang": 2}, f"step {i}: one chain between the quantize and the dequantize")
    finally: CoralProgram.__call__ = call

# ***** glue kernels written as QLayer writes them *****

S, N, C, H, W = 0.0237, 2, 3, 8, 6
def y8(rng, *shape): return Tensor(rng.integers(0, 256, shape or (N, C, H, W), dtype=np.uint8), device="CPU").realize()
H6 = int(np.round(np.float32(6) * np.float32(1 / S)))           # relu6 behind a linked quantization: the clamp at round(6/s)
RECOGNIZED = {                                                  # name: (glue, its spec)
  "relu, zero point 0":      (lambda y: q(dq(y, S, 0).relu(), S, 0), CopySpec(N*C*H*W)),
  "no relu, zero point 37":  (lambda y: q(dq(y, S, 37), S, 37), CopySpec(N*C*H*W)),
  "relu, zero point 37":     (lambda y: q(dq(y, S, 37).relu(), S, 37), CopySpec(N*C*H*W, 37, 255)),
  "relu6, linked":           (lambda y: q(dq(y, S, 0).relu6(), S, 0), CopySpec(N*C*H*W, 0, H6)),
  "relu, scale 1/s":         (lambda y: (dq(y, 1 / 7, 0).relu() * 7).round().clip(0, 255).cast(U8), CopySpec(N*C*H*W)),
  "max pool 2x2":            (lambda y: q(dq(y, S, 0).relu().max_pool2d(2), S, 0), PoolSpec(N, C, H, W, 2, 2)),
  "max pool, zero point 37": (lambda y: q(dq(y, S, 37).max_pool2d(2), S, 37), PoolSpec(N, C, H, W, 2, 2)),
  "max pool 3x3 stride 2":   (lambda y: q(dq(y, S, 0).relu().max_pool2d(3, stride=2), S, 0), PoolSpec(N, C, H, W, 3, 2)),
  "max pool 2x2 stride 1":   (lambda y: q(dq(y, S, 0).relu().max_pool2d(2, stride=1), S, 0), PoolSpec(N, C, H, W, 2, 1)),
  "max pool, flattened":     (lambda y: q(dq(y, S, 0).relu().max_pool2d(2).flatten(1), S, 0), PoolSpec(N, C, H, W, 2, 2)),
  "max pool on uint8":       (lambda y: y.max_pool2d(2), PoolSpec(N, C, H, W, 2, 2)),
}
NOT_GLUE = {
  "unlinked scales": lambda y: q(dq(y, S, 0).relu(), 1.3 * S, 0),
  "unlinked zero points": lambda y: q(dq(y, S, 3), S, 9),
  "relu6, as quantized": lambda y: q(dq(y, S, 128).relu6(), 6 / 255, 0),       # its own scale for [0, 6]: a rescale
  "relu6, then max pool": lambda y: q(dq(y, S, 0).relu6().max_pool2d(2), S, 0),  # a clamped pool: no PoolSpec
  "avg pool": lambda y: q(dq(y, S, 0).relu().avg_pool2d(2), S, 0),
  "max pool, padding 1": lambda y: q(dq(y, S, 0).relu().max_pool2d(2, padding=1), S, 0),
  "min pool": lambda y: q(-(-dq(y, S, 0)).max_pool2d(2), S, 0),      # max of a decreasing map
  "max pool 2x1": lambda y: q(dq(y, S, 0).relu().max_pool2d((2, 1)), S, 0),
  "max pool, dilation 2": lambda y: q(dq(y, S, 0).relu().max_pool2d(2, dilation=2), S, 0),
  "transposed": lambda y: q(dq(y, S, 0).relu().permute(0, 1, 3, 2), S, 0),
  "dequantize (float out)": lambda y: dq(y, S, 0).relu(),
  "cast to int8 (wraps)": lambda y: q(dq(y.cast(dtypes.int8), S, 0).relu(), S, 0),
  "int8 buffer in": lambda y: q(dq(y.cast(dtypes.int8).contiguous(), S, 0).relu(), S, 0),
  "doubled": lambda y: q(dq(y, S, 0).relu() * 2, S, 0),
  "by position": lambda y: q(dq(y, S, 0) + Tensor.arange(W).cast(F32) * S, S, 0),
}

class TestGlue(unittest.TestCase):
  def test_recognized(self):
    """each glue is recognized with its spec, and does what the spec says on inputs that hold every uint8 value"""
    rng = np.random.default_rng(0)
    for name, (f, spec) in RECOGNIZED.items():
      with self.subTest(name):
        m = glue(f(y8(rng)))
        self.assertIsInstance(m, CopyMatch if isinstance(spec, CopySpec) else PoolMatch, why_not_glue(kernel_asts(f(y8(rng)))[-1]))
        self.assertEqual(m.spec, spec)
        self.assertEqual([p.arg.dtype for p in m.params], [U8, U8])
        for x in (np.arange(N*C*H*W, dtype=np.uint8).reshape(N, C, H, W), rng.integers(0, 256, (N, C, H, W), dtype=np.uint8)):
          np.testing.assert_array_equal(f(Tensor(x, device="CPU").realize()).numpy().reshape(-1), glue_reference(spec, x).reshape(-1))

  def test_near_misses(self):
    rng = np.random.default_rng(1)
    for name, f in NOT_GLUE.items():
      with self.subTest(name):
        self.assertIsNone(glue(f(y8(rng))))
        self.assertIsNotNone(why_not_glue(kernel_asts(f(y8(rng)))[-1]))

  def test_selection_is_exact(self):
    """random glue-like kernels (scales and zero points linked or not, ReLU, ReLU6, max or avg pools of any window, stride and
    padding, negated maps): whatever select_glue takes one for reproduces its output on inputs that hold every uint8 value"""
    rng, recognized = np.random.default_rng(2), 0
    for i in range(200):
      s1 = float(rng.choice([0.0237, 0.0051, 0.31, 1 / 3, 2.0]))
      s2, z1 = (s1 if rng.random() < 0.6 else s1 * float(rng.uniform(0.5, 2))), int(rng.choice([0, 0, 37, 128]))
      z2 = z1 if rng.random() < 0.8 else int(rng.integers(0, 256))
      act, pool = rng.choice(["", "relu", "relu6", "neg"]), rng.choice(["", "", "max", "avg"])
      k, st, pad = int(rng.integers(2, 4)), int(rng.integers(1, 4)), int(rng.integers(0, 2)) if rng.random() < 0.3 else 0
      def f(y):
        x = dq(y, s1, z1)
        x = {"relu": x.relu, "relu6": x.relu6, "neg": lambda: -x}.get(act, lambda: x)()
        if pool: x = (x.max_pool2d if pool == "max" else x.avg_pool2d)(k, stride=st, padding=pad)
        return q(-x if act == "neg" else x, s2, z2)
      try: m = glue(f(y8(rng)))
      except (AssertionError, ValueError, RuntimeError): continue                  # not a valid pool shape
      if m is None: continue
      recognized += 1
      for x in (np.arange(N*C*H*W, dtype=np.uint8).reshape(N, C, H, W), rng.integers(0, 256, (N, C, H, W), dtype=np.uint8)):
        with self.subTest(i=i, spec=m.spec):
          np.testing.assert_array_equal(f(Tensor(x, device="CPU").realize()).numpy().reshape(-1), glue_reference(m.spec, x).reshape(-1))
      COUNT["checked"] += 1
    self.assertGreater(recognized, 40)

if __name__ == "__main__":
  res = unittest.main(exit=False, verbosity=2).result
  print(f"\n{res.testsRun - len(res.failures) - len(res.errors)}/{res.testsRun} tests passed; select_glue recognized {COUNT['recognized']} "
        f"kernels and rejected {COUNT['rejected']}; {COUNT['checked']} recognized random kernels checked against their spec")
  sys.exit(not res.wasSuccessful())
