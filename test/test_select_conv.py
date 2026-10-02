#!/usr/bin/env python
# Kernel selection of the quantized CONV_2D (coral/qops.py qconv2d, coral/select.py match_conv): the plain tinygrad qconv2d
# schedules as ONE kernel, which the selector maps to a ConvSpec; CPU and CORAL are bit-identical, and equal to the Edge TPU's
# numpy model (coral.select.conv_reference: im2col + coral.tpu.fc_reference). Every other kernel stays a clang kernel. The TPU
# is the bit-exact numpy model (MOCKCORAL=1, the default here); ops_coral runs a selected conv with coral.tpu.run_conv.
#   .venv/bin/python test/test_select_conv.py [TestMatcher[.test_variants] ...]
import os
os.environ.setdefault("MOCKCORAL", "1")       # never touch the USB device unless asked to explicitly (MOCKCORAL= ...)
import dataclasses, itertools, json, sys, unittest
import numpy as np
from tinygrad import Tensor, dtypes, TinyJit, Device
from tinygrad.codegen import to_program
from tinygrad.uop.ops import UOp, Ops
import coral.tpu as tpu
from coral.qops import qconv2d, qlinear
from coral.select import ConvSpec, ConvMatch, FCMatch, select, select_conv, why_not, conv_reference, im2col
import tinygrad.runtime.ops_coral as ops_coral
from tinygrad.runtime.ops_coral import CoralProgram, CORAL_SRC

I32, F32, U8, I16 = dtypes.int32, dtypes.float32, dtypes.uint8, dtypes.int16
TOTAL = {"tpu": 0, "clang": 0, "run_conv": 0, "run_fc": 0, "cases": 0}

# ***** helpers *****

class Count:
  """counts the programs CORAL runs: Edge TPU programs (and the coral.tpu.run_conv / run_fc calls they make) vs clang programs"""
  def __enter__(self):
    self.tpu = self.clang = self.run_conv = self.run_fc = 0
    self._call, self._run_conv, self._run_fc = CoralProgram.__call__, tpu.run_conv, tpu.run_fc
    def call(prg, *args, **kwargs):
      if prg.tpu: self.tpu += 1
      else: self.clang += 1
      return self._call(prg, *args, **kwargs)
    def run_conv(*args, **kwargs):
      self.run_conv += 1
      return self._run_conv(*args, **kwargs)
    def run_fc(*args, **kwargs):
      self.run_fc += 1
      return self._run_fc(*args, **kwargs)
    CoralProgram.__call__, tpu.run_conv, tpu.run_fc = call, run_conv, run_fc
    return self
  def __exit__(self, *exc):
    CoralProgram.__call__, tpu.run_conv, tpu.run_fc = self._call, self._run_conv, self._run_fc
    for k in ("tpu", "clang", "run_conv", "run_fc"): TOTAL[k] += getattr(self, k)

def on(dev:str, *arrays): return [None if a is None else Tensor(a, device=dev).realize() for a in arrays]
def run_both(f, *arrays):
  """f on realized inputs on CPU and on CORAL: (cpu result, coral result, Count of the CORAL run)"""
  cpu = f(*on("CPU", *arrays)).numpy()
  ins = on("CORAL", *arrays)
  with Count() as c: coral = f(*ins).numpy()
  TOTAL["cases"] += 1
  return cpu, coral, c
def kernel_asts(t:Tensor) -> list[UOp]: return [c.without_after.body for c in t.schedule_linear().src]
def kernel_ast(t:Tensor) -> UOp: return kernel_asts(t)[-1]   # the kernel that computes t

def u8(rng, *shape): return rng.integers(0, 256, shape, dtype=np.uint8)
def i32(rng, n, lim): return rng.integers(-lim, lim, n).astype(np.int32)
def R(y:Tensor) -> Tensor: return y.round()             # round half to even, the Edge TPU's rounding
def half_away(y:Tensor) -> Tensor: return y.sign() * (y.abs() + 0.5).floor()    # NOT the TPU's rounding

def qc(spec:ConvSpec):
  return lambda x, w, b: qconv2d(x, w, b, spec.in_zp, spec.w_zp, spec.out_zp, spec.mult, spec.lo, spec.hi, spec.stride, spec.padding)

def random_case(rng, N, Cin, H, W, Cout, kh, kw, s, p, bias=True):
  """random data and a random quantization whose outputs spread over [lo, hi] (some saturate); a quarter of the multipliers
  are powers of two, which put exact .5 ties into the rounding"""
  x, w, b = u8(rng, N, Cin, H, W), u8(rng, Cout, Cin, kh, kw), i32(rng, Cout, 2**16) if bias else None
  in_zp, w_zp, out_zp = (int(v) for v in rng.integers(0, 256, 3))
  spec = ConvSpec(N, Cin, H, W, Cout, kh, kw, s, p, in_zp, w_zp, out_zp, 1.0)
  acc = (im2col(spec, x).astype(np.float64) - in_zp) @ (w.reshape(Cout, -1).astype(np.float64) - w_zp).T + (0 if b is None else b)
  mult = float(np.float32(rng.uniform(20, 150) / max(float(acc.std()), 1.0)))
  if rng.random() < 0.25: mult = float(2.0 ** np.round(np.log2(mult)))
  lo, hi = (0, 255) if rng.random() < 0.5 else tuple(sorted(int(v) for v in rng.integers(0, 256, 2)))
  if rng.random() < 0.2: lo, hi = out_zp, 255                    # fused relu
  return (x, w, b), dataclasses.replace(spec, mult=mult, lo=lo, hi=hi)

def random_shape(rng):
  """a conv in which every output position and every kernel tap reads the image (padding < kernel), output at least 2x2"""
  while True:
    kh, kw, s = (int(v) for v in rng.integers(1, [6, 6, 4]))
    p, N, Cin, Cout = int(rng.integers(0, min(kh, kw))), int(rng.integers(1, 4)), int(rng.integers(1, 9)), int(rng.integers(1, 17))
    H, W = int(rng.integers(kh + 1, 15)), int(rng.integers(kw + 1, 15))
    if Cin * kh * kw >= 2 and (H + 2*p - kh) // s >= 1 and (W + 2*p - kw) // s >= 1: return N, Cin, H, W, Cout, kh, kw, s, p

# the convs of tinygrad's MNIST CNN (examples/beautiful_mnist.py): (Cin, H, W, Cout, k)
MNIST = [(1, 28, 28, 32, 5), (32, 24, 24, 32, 5), (32, 10, 10, 64, 3), (64, 8, 8, 64, 3)]

# ***** qconv2d: bit-exact and one selected kernel *****

class TestQConv2d(unittest.TestCase):
  def check(self, arrays, spec:ConvSpec):
    cpu, coral, c = run_both(qc(spec), *arrays)
    expect = conv_reference(spec, *arrays)
    np.testing.assert_array_equal(cpu, expect, err_msg=f"CPU vs im2col + fc_reference {spec}")
    np.testing.assert_array_equal(coral, expect, err_msg=f"CORAL vs im2col + fc_reference {spec}")
    self.assertEqual((c.tpu, c.run_conv, c.clang), (1, 1, 0), f"{spec}: want one selected kernel and no clang kernel")
    m = select(kernel_ast(qc(spec)(*on("CPU", *arrays))))
    self.assertIsInstance(m, ConvMatch)
    self.assertEqual(m.spec, spec)

  def test_mnist(self):
    """the 4 convs of the MNIST CNN, one image and a batch of 3"""
    rng = np.random.default_rng(0)
    for (Cin, H, W, Cout, k), N in itertools.product(MNIST, (1, 3)):
      with self.subTest(N=N, Cin=Cin, H=H, Cout=Cout, k=k): self.check(*random_case(rng, N, Cin, H, W, Cout, k, k, 1, 0))

  def test_random_shapes(self):
    """random shapes (non-square images and kernels, 1x1 and 1xk kernels), strides 1-3, paddings, zero points, with and
    without bias"""
    rng = np.random.default_rng(1)
    for i in range(40):
      shape = random_shape(rng)
      with self.subTest(i=i, shape=shape): self.check(*random_case(rng, *shape, bias=bool(i % 4)))

  def test_strides_and_paddings(self):
    """3x3 and 5x5 kernels with every padding < kernel at strides 1-3 (SAME and VALID ones among them)"""
    rng = np.random.default_rng(2)
    for k, s in itertools.product((3, 5), (1, 2, 3)):
      for p in range(k):
        with self.subTest(k=k, s=s, p=p): self.check(*random_case(rng, 2, 3, 11, 9, 5, k, k, s, p))

  def test_ties(self):
    """exact .5 ties round half to even (the device's rounding), with the padding inside the windows"""
    rng = np.random.default_rng(3)
    x, w, b = rng.integers(112, 129, (2, 3, 6, 6), dtype=np.uint8), rng.integers(122, 139, (8, 3, 3, 3), dtype=np.uint8), i32(rng, 8, 9)
    for mult, out_zp, lo, hi in [(0.5, 128, 0, 255), (1.5, 128, 0, 255), (2.5, 128, 20, 240), (0.25, 128, 0, 255), (0.5, 128, 128, 255)]:
      spec = ConvSpec(2, 3, 6, 6, 8, 3, 3, 1, 1, 120, 130, out_zp, mult, lo, hi)
      acc = (im2col(spec, x).astype(np.int64) - 120) @ (w.reshape(8, -1).astype(np.int64) - 130).T + b
      y = np.clip(acc * mult, lo - out_zp, hi - out_zp)
      self.assertTrue(np.any((y % 1 == 0.5) & (np.floor(y) % 2 == 0)) and np.any((y % 1 == 0.5) & (np.floor(y) % 2 == 1)), f"ties missing for {spec}")
      self.check((x, w, b), spec)

# ***** the schedule and the CORAL runtime *****

class TestSchedule(unittest.TestCase):
  def test_one_kernel_one_program(self):
    """qconv2d schedules ONE kernel (reduce + epilogue); compiled for CORAL it is an Edge TPU program with the ConvSpec as its
    source, launched with the buffers in TPU order (out, x, w, b) whatever slots the scheduler gave them"""
    (x, w, b), spec = random_case(np.random.default_rng(4), 2, 3, 9, 7, 5, 3, 3, 2, 1)
    xt, wt, bt = on("CORAL", x, w, b)
    xz, wz = (xt.pad(((0, 0), (0, 0), (1, 1), (1, 1)), value=spec.in_zp).cast(I32) - spec.in_zp), wt.cast(I32) - spec.w_zp
    bias_first = (R(((bt.reshape(1, -1, 1, 1) + xz.conv2d(wz, stride=2)).cast(F32) * spec.mult).clip(spec.lo - spec.out_zp, spec.hi - spec.out_zp))
                  + spec.out_zp).cast(U8)
    for out in (qc(spec)(xt, wt, bt), bias_first):
      calls = [c.without_after for c in out.schedule_linear().src]
      self.assertEqual([c.body.op for c in calls], [Ops.SINK])
      prg = to_program(calls[0].body, Device["CORAL"].renderer)
      self.assertTrue(prg.src[2].arg.startswith(CORAL_SRC))
      self.assertEqual(ConvSpec.loads(prg.src[2].arg[len(CORAL_SRC):]), spec)
      args = calls[0].src[1:]
      self.assertEqual([args[s].dtype for s in prg.arg.globals], [U8, U8, U8, I32])
      self.assertEqual([args[s].buffer for s in prg.arg.globals[1:]], [t.uop.base.buffer for t in (xt, wt, bt)])
      self.assertEqual(prg.arg.globals, tuple(p.arg.slot for p in select_conv(calls[0].body).params))
    self.assertNotEqual(prg.arg.globals, tuple(sorted(prg.arg.globals)))   # bias first: the scheduler's slots are not TPU order

  def test_mnist_network(self):
    """the MNIST CNN quantized: quantize, conv relu, conv relu, max pool, conv relu, conv relu, max pool, linear. Every
    qconv2d (and the qlinear) is one Edge TPU program, the rest clang kernels; CPU, CORAL and the numpy model bit-identical"""
    rng, N = np.random.default_rng(5), 3
    xf = rng.normal(size=(N, 1, 28, 28)).astype(np.float32)
    ws = [u8(rng, Cout, Cin, k, k) for Cin, _, _, Cout, k in MNIST] + [u8(rng, 10, 576)]
    bs = [i32(rng, w.shape[0], 3000) for w in ws]
    # the numpy model; each multiplier from its accumulators, so that every layer's outputs spread (relu: lo = out_zp = 20)
    specs, h, zp = [], np.clip(np.round(xf / np.float32(0.02)) + np.float32(128), 0, 255).astype(np.uint8), 128
    for i, ((Cin, H, W, Cout, k), w, b) in enumerate(zip(MNIST, ws, bs)):
      s0 = ConvSpec(N, Cin, H, W, Cout, k, k, 1, 0, zp, 128, 20, 1.0, 20)
      acc = (im2col(s0, h).astype(np.float64) - zp) @ (w.reshape(Cout, -1).astype(np.float64) - 128).T + b
      specs.append(dataclasses.replace(s0, mult=float(np.float32(60 / acc.std()))))
      h, zp = conv_reference(specs[-1], h, w, b), 20
      if i in (1, 3): h = h.reshape(N, Cout, h.shape[2] // 2, 2, h.shape[3] // 2, 2).max(axis=(3, 5))
    acc = (h.reshape(N, 576).astype(np.float64) - 20) @ (ws[4].astype(np.float64) - 128).T + bs[4]
    fc = tpu.FCSpec(N, 10, 576, 20, 128, 128, float(np.float32(50 / acc.std())))
    expect = tpu.fc_reference(fc, h.reshape(N, 576), ws[4], bs[4])
    def f(xf, *wb):
      h = (xf / 0.02 + 128).round().clip(0, 255).cast(U8)
      for i, s in enumerate(specs):
        h = qc(s)(h, wb[2*i], wb[2*i+1])
        if i in (1, 3): h = h.max_pool2d(2)
      return qlinear(h.reshape(N, 576), wb[8], wb[9], fc.in_zp, fc.w_zp, fc.out_zp, fc.mult)
    cpu, coral, c = run_both(f, xf, *[a for wb in zip(ws, bs) for a in wb])
    np.testing.assert_array_equal(coral, expect)
    np.testing.assert_array_equal(cpu, coral)
    self.assertEqual((c.tpu, c.run_conv, c.run_fc), (5, 4, 1))     # 4 qconv2d + 1 qlinear
    self.assertEqual(c.clang, 3)                    # the quantize and the two max pools
    self.assertGreater(len(np.unique(expect)), 10)

  def test_lazy_input_and_consumer(self):
    """quantize -> qconv2d -> dequantize: the quantize and the dequantize are clang kernels, the layer one TPU program"""
    rng = np.random.default_rng(6)
    xf, w, b = rng.normal(size=(2, 3, 10, 8)).astype(np.float32), u8(rng, 6, 3, 3, 3), i32(rng, 6, 999)
    spec = ConvSpec(2, 3, 10, 8, 6, 3, 3, 1, 1, 128, 127, 100, 0.002)
    def f(xf, w, b): return (qc(spec)((xf / 0.02 + 128).round().clip(0, 255).cast(U8), w, b).cast(F32) - 100) * 0.1
    cpu, coral, c = run_both(f, xf, w, b)
    np.testing.assert_array_equal(cpu, coral)
    self.assertEqual((c.tpu, c.clang), (1, 2))

  def test_jit(self):
    rng = np.random.default_rng(7)
    w, b = on("CORAL", u8(rng, 8, 4, 3, 3), i32(rng, 8, 3000))
    spec = ConvSpec(1, 4, 9, 9, 8, 3, 3, 2, 1, 128, 130, 110, 0.0031, 3, 250)
    @TinyJit
    def step(xf:Tensor) -> Tensor: return ((qc(spec)((xf / 0.02 + 128).round().clip(0, 255).cast(U8), w, b).cast(F32) - 110) * 0.05).realize()
    for i in range(4):
      xf = rng.normal(size=(1, 4, 9, 9)).astype(np.float32)
      with Count() as c: got = step(Tensor(xf, device="CORAL").realize()).numpy()
      xq = np.clip(np.round(xf / np.float32(0.02) + np.float32(128)), 0, 255).astype(np.uint8)
      np.testing.assert_array_equal(got, (conv_reference(spec, xq, w.numpy(), b.numpy()).astype(np.float32) - 110) * np.float32(0.05))
      self.assertEqual((c.tpu, c.clang), (1, 2), f"step {i}")

  def test_select_off(self):
    (x, w, b), spec = random_case(np.random.default_rng(8), 1, 3, 7, 7, 4, 3, 3, 1, 1)
    old, ops_coral.SELECT = ops_coral.SELECT, 0
    try: cpu, coral, c = run_both(qc(spec), x, w, b)
    finally: ops_coral.SELECT = old
    np.testing.assert_array_equal(cpu, coral)
    self.assertEqual((c.tpu, c.clang), (0, 1))

# ***** the matcher: the forms tinygrad may give the conv, and near misses *****

N_, CIN, H_, W_, COUT, K, S, P = 2, 3, 7, 6, 5, 3, 2, 1
SPEC = ConvSpec(N_, CIN, H_, W_, COUT, K, K, S, P, 120, 130, 110, 0.5, 3, 250)    # narrow data: .5 ties at odd accumulators
RELU = dataclasses.replace(SPEC, lo=110)
PADS = ((0, 0), (0, 0), (P, P), (P, P))
def xz(x, zp=120): return x.pad(PADS, value=zp).cast(I32) - zp                     # padded with the zero point, as qconv2d
def wz(w): return w.cast(I32) - 130
def conv(xp, w, **kw): return xp.conv2d(w, stride=S, **kw)
def acc(x, w, b): return conv(xz(x), wz(w)) + (0 if b is None else b.reshape(1, -1, 1, 1))
def y_(a): return a.cast(F32) * 0.5
def out(y): return (R(y.clip(-107, 140)) + 110).cast(U8)                               # requant to SPEC from the float y
def windows(xp): return xp._pool((K, K), S).reshape(N_, 1, CIN, (H_ + 2*P - K) // S + 1, (W_ + 2*P - K) // S + 1, K, K)
def wk(w): return wz(w).reshape(1, COUT, CIN, 1, 1, K, K)
VARIANTS = {
  "qconv2d":                   (lambda x,w,b: qc(SPEC)(x, w, b), SPEC),
  "pad after the zero point":  (lambda x,w,b: out(y_(conv(x.cast(I32) - 120, wz(w), padding=P) + b.reshape(1, -1, 1, 1))), SPEC),
  "bias first":                (lambda x,w,b: out(y_(b.reshape(1, -1, 1, 1) + conv(xz(x), wz(w)))), SPEC),
  "windows by hand":           (lambda x,w,b: out(y_((windows(xz(x)) * wk(w)).sum((2, 5, 6)) + b.reshape(1, -1, 1, 1))), SPEC),
  "w times the windows":       (lambda x,w,b: out(y_((wk(w) * windows(xz(x))).sum((2, 5, 6)) + b.reshape(1, -1, 1, 1))), SPEC),
  "+ -zp":                     (lambda x,w,b: out(y_(conv(x.pad(PADS, value=120).cast(I32) + (-120), w.cast(I32) + (-130)) + b.reshape(1, -1, 1, 1))),
                                SPEC),
  "widening casts":            (lambda x,w,b: out(y_(conv(x.pad(PADS, value=120).cast(I16).cast(I32) - 120,
                                                          w.cast(dtypes.uint16).cast(I32) - 130) + b.reshape(1, -1, 1, 1))), SPEC),
  "padding and zp in int16":   (lambda x,w,b: out(y_(conv((x.cast(I16).pad(PADS, value=120) - 120).cast(I32), wz(w)) + b.reshape(1, -1, 1, 1))),
                                SPEC),
  "zp 0, mult 1 folded":       (lambda x,w,b: R((conv(x.cast(I32), w.cast(I32), padding=P) + b.reshape(1, -1, 1, 1)).cast(F32).mul(1.0)
                                                .clip(0, 255)).cast(U8),
                                dataclasses.replace(SPEC, in_zp=0, w_zp=0, out_zp=0, mult=1.0, lo=0, hi=255)),
  "no bias":                   (lambda x,w,b: out(y_(acc(x, w, None))), SPEC),
  "clip after rounding":       (lambda x,w,b: (R(y_(acc(x, w, b))) + 110).clip(3, 250).cast(U8), SPEC),
  "relu().minimum()":          (lambda x,w,b: (R(y_(acc(x, w, b)).relu().minimum(140)) + 110).cast(U8), RELU),
}
NO_MATCH = {
  "half away from zero": lambda x,w,b: (half_away(y_(acc(x, w, b)).clip(-107, 140)) + 110).cast(U8),
  "half up: floor(y+.5)": lambda x,w,b: ((y_(acc(x, w, b)).clip(-107, 140) + 0.5).floor() + 110).cast(U8),
  "padding 0, in_zp 120": lambda x,w,b: out(y_(conv(xz(x, 0) - 120, wz(w)) + b.reshape(1, -1, 1, 1))),
  "padding in_zp + 1": lambda x,w,b: out(y_(conv(x.pad(PADS, value=121).cast(I32) - 120, wz(w)) + b.reshape(1, -1, 1, 1))),
  "padding 1 after the zp": lambda x,w,b: out(y_(conv((x.cast(I32) - 120).pad(PADS, value=1), wz(w)) + b.reshape(1, -1, 1, 1))),
  "padding on top only": lambda x,w,b: out(y_(conv(x.pad(((0, 0), (0, 0), (P, 0), (P, P)), value=120).cast(I32) - 120, wz(w)))),
  "padding of the rows only": lambda x,w,b: out(y_(conv(x.pad(((0, 0), (0, 0), (P, P), (0, 0)), value=120).cast(I32) - 120, wz(w)))),
  "groups 3 (depthwise)": lambda x,w,b: out(y_(conv(xz(x), wz(w.reshape(-1)[:CIN*K*K].reshape(CIN, 1, K, K)), groups=CIN))),
  "groups 3, 2 outputs each": lambda x,w,b: out(y_(conv(xz(x), wz(w.reshape(-1)[:2*CIN*K*K].reshape(2*CIN, 1, K, K)), groups=CIN))),
  "dilation 2": lambda x,w,b: out(y_(xz(x).conv2d(wz(w), stride=S, dilation=2))),
  "w read [o, c, kx, ky]": lambda x,w,b: out(y_(conv(xz(x), wz(w).permute(0, 1, 3, 2)))),
  "w flipped": lambda x,w,b: out(y_(conv(xz(x), wz(w).flip((2, 3))))),
  "x read NHWC": lambda x,w,b: out(y_(conv(x.reshape(N_, H_, W_, CIN).permute(0, 3, 1, 2).pad(PADS, value=120).cast(I32) - 120, wz(w)))),
  "max over the window": lambda x,w,b: out((windows(xz(x)) * wk(w)).max((2, 5, 6)).cast(F32)),
  "float accumulation": lambda x,w,b: out(conv(x.pad(PADS, value=120).cast(F32) - 120, w.cast(F32) - 130) * 0.5),
  "per-channel multiplier": lambda x,w,b: out(acc(x, w, b).cast(F32) * ((b.cast(F32).abs() + 1) * 1e-6).reshape(1, -1, 1, 1)),
  "bias indexed by row": lambda x,w,b: out(y_(acc(x, w, None) + b[:4].reshape(1, 1, -1, 1))),
  "input clamped": lambda x,w,b: out(y_(conv(x.maximum(115).pad(PADS, value=120).cast(I32) - 120, wz(w)))),
  "x squared": lambda x,w,b: out(y_(conv(xz(x) * xz(x), wz(w)))),
}

def near_conv(rng):
  """a random quantized conv, perturbed: (f(x, w, b), (x, w, b))"""
  N, Cin, Cout, kh, kw, H, W = (int(v) for v in rng.integers(1, [3, 4, 5, 4, 4, 9, 9]))
  x, w, b = u8(rng, N, Cin, H, W), u8(rng, Cout, Cin, kh, kw), i32(rng, Cout, 500)
  zx, zw, zo = (int(v) for v in rng.integers(0, 256, 3))
  pads = (int(rng.integers(0, 3)),) * 4 if rng.random() < 0.5 else tuple(int(v) for v in rng.integers(0, 3, 4))
  fill = zx if rng.random() < 0.7 else int(rng.choice([0, (zx + 1) % 256, (zx + 255) % 256]))
  stride = int(rng.integers(1, 4)) if rng.random() < 0.7 else tuple(int(v) for v in rng.integers(1, 4, 2))
  dilation = 1 if rng.random() < 0.8 else 2
  xform = "" if rng.random() < 0.5 else rng.choice(["slice", "step", "T", "flip", "pad after zp"])
  wform = "" if rng.random() < 0.7 else rng.choice(["T", "flip"])
  bias, mult = rng.random() < 0.7, float(np.float32(rng.uniform(1e-4, 1e-2)))
  def f(x, w, b):
    x = {"slice": lambda: x[:, :, 1:], "step": lambda: x[:, :, :, ::2], "T": lambda: x.permute(0, 1, 3, 2),
         "flip": lambda: x.flip(2)}.get(xform, lambda: x)()
    w = {"T": lambda: w.permute(0, 1, 3, 2), "flip": lambda: w.flip((2, 3))}.get(wform, lambda: w)()
    pad = ((0, 0), (0, 0), pads[:2], pads[2:])
    xp = (x.cast(I32) - zx).pad(pad, value=fill - zx) if xform == "pad after zp" else x.pad(pad, value=fill).cast(I32) - zx
    acc = xp.conv2d(w.cast(I32) - zw, stride=stride, dilation=dilation) + (b.reshape(1, -1, 1, 1) if bias else 0)
    return (R((acc.cast(F32) * mult).clip(-zo, 255 - zo)) + zo).cast(U8)
  return f, (x, w, b)

class TestMatcher(unittest.TestCase):
  rng = np.random.default_rng(10)
  x, w, b = rng.integers(112, 129, (N_, CIN, H_, W_), dtype=np.uint8), rng.integers(122, 139, (COUT, CIN, K, K), dtype=np.uint8), i32(rng, COUT, 60)

  def test_variants(self):
    """the same conv written in other forms: one selected kernel with exactly these constants"""
    for name, (f, spec) in VARIANTS.items():
      with self.subTest(name):
        cpu, coral, c = run_both(f, self.x, self.w, self.b)
        np.testing.assert_array_equal(coral, conv_reference(spec, self.x, self.w, None if name == "no bias" else self.b))
        np.testing.assert_array_equal(cpu, coral)
        self.assertEqual((c.tpu, c.run_conv, c.clang), (1, 1, 0))
        self.assertEqual(select(kernel_ast(f(*on("CPU", self.x, self.w, self.b)))).spec, spec)

  def test_near_misses_stay_clang(self):
    for name, f in NO_MATCH.items():
      with self.subTest(name):
        cpu, coral, c = run_both(f, self.x, self.w, self.b)
        np.testing.assert_array_equal(cpu, coral)
        self.assertEqual(c.tpu, 0)
        self.assertGreaterEqual(c.clang, 1)
        asts = kernel_asts(f(*on("CPU", self.x, self.w, self.b)))
        self.assertEqual([select(a) for a in asts], [None] * len(asts))
        self.assertIsNotNone(why_not(asts[-1]))

  def test_selection_is_exact(self):
    """random near-convs (paddings of any size and value, per-axis strides, dilation, sliced, strided, transposed or flipped
    inputs and weights): whatever select() takes one for reproduces its output, run on the kernel's own buffers"""
    rng, selected = np.random.default_rng(13), 0
    for i in range(300):
      f, arrays = near_conv(rng)
      try: call = f(*on("CPU", *arrays)).schedule_linear().src[-1].without_after
      except (AssertionError, ValueError, RuntimeError, IndexError): continue     # not a valid conv shape
      if (m:=select(call.body)) is None: continue
      bufs = [np.frombuffer(call.src[1 + p.arg.slot].buffer.as_memoryview(), np.int32 if p is m.b else np.uint8) for p in m.params[1:]]
      s, (x, w), b = m.spec, bufs[:2], bufs[2] if m.b is not None else None
      if isinstance(m, ConvMatch): got = conv_reference(s, x[:s.N*s.Cin*s.H*s.W], w[:s.Cout*s.Cin*s.kh*s.kw], None if b is None else b[:s.Cout])
      else: got = tpu.fc_reference(s, x[:s.M*s.K].reshape(s.M, s.K), w[:s.N*s.K].reshape(s.N, s.K), np.zeros(s.N, np.int32) if b is None else b[:s.N])
      with self.subTest(i=i, spec=s): np.testing.assert_array_equal(got.reshape(-1), f(*on("CPU", *arrays)).numpy().reshape(-1))
      selected += 1
    self.assertGreater(selected, 20)

  def test_fc_with_row_bias_is_a_conv(self):
    """out[m, n] = requant(x[m] . w[n] + b[m]) is no FC (its bias is per row) but exactly a conv: w's rows as an N x K image,
    x's rows as M filters of 1 x K, b per filter. It is selected as that conv and runs as one, bit-identical"""
    rng = np.random.default_rng(12)
    x, w, b = u8(rng, 6, 7), u8(rng, 12, 7), i32(rng, 6, 999)
    def f(x, w, b):
      acc = (x.cast(I32) - 120) @ (w.cast(I32) - 130).T + b.reshape(6, 1)
      return (R((acc.cast(F32) * 0.01).clip(-110, 145)) + 110).cast(U8)
    cpu, coral, c = run_both(f, x, w, b)
    spec = ConvSpec(1, 1, 12, 7, 6, 1, 7, 1, 0, 130, 120, 110, float(np.float32(0.01)))
    self.assertEqual(select(kernel_ast(f(*on("CPU", x, w, b)))).spec, spec)
    np.testing.assert_array_equal(coral, conv_reference(spec, w, x, b).reshape(6, 12))
    np.testing.assert_array_equal(cpu, coral)
    self.assertEqual((c.tpu, c.clang), (1, 0))

  def test_fc_is_still_fc(self):
    """select() gives the FC of a qlinear kernel, and the conv of a qconv2d kernel"""
    rng = np.random.default_rng(11)
    x, w, b = on("CPU", u8(rng, 4, 20), u8(rng, 7, 20), i32(rng, 7, 99))
    self.assertIsInstance(select(kernel_ast(qlinear(x, w, b, 1, 2, 3, 0.01))), FCMatch)
    self.assertIsInstance(select(kernel_ast(qc(SPEC)(*on("CPU", self.x, self.w, self.b)))), ConvMatch)

  def test_spec_json(self):
    s = ConvSpec(1, 1, 28, 28, 32, 5, 5, 1, 0, 128, 127, 20, 0.0031, 20, 255)
    self.assertEqual(json.loads(s.dumps()), {"conv2d": {"N": 1, "Cin": 1, "H": 28, "W": 28, "Cout": 32, "kh": 5, "kw": 5, "stride": 1,
                     "padding": 0, "in_zp": 128, "w_zp": 127, "out_zp": 20, "mult": 0.0031, "lo": 20, "hi": 255}})
    self.assertEqual(ConvSpec.loads(s.dumps()), s)
    self.assertEqual((s.OH, s.OW, s.fc().M, s.fc().N, s.fc().K), (24, 24, 576, 32, 25))
    def pad(H, k, st, p): return ConvSpec(1, 1, H, H, 1, k, k, st, p, 0, 0, 0, 1.0).tflite_padding
    self.assertEqual([pad(28, 5, 1, 0), pad(8, 3, 1, 1), pad(9, 5, 1, 2), pad(7, 3, 2, 1), pad(8, 3, 2, 1), pad(8, 3, 1, 2)],
                     ["VALID", "SAME", "SAME", "SAME", None, None])   # stride 2, even size: SAME pads the bottom only

if __name__ == "__main__":
  res = unittest.main(exit=False, verbosity=2).result
  print(f"\n{res.testsRun - len(res.failures) - len(res.errors)}/{res.testsRun} tests passed; {TOTAL['cases']} CPU-vs-CORAL runs, "
        f"CORAL ran {TOTAL['tpu']} Edge TPU programs ({TOTAL['run_conv']} run_conv, {TOTAL['run_fc']} run_fc calls) "
        f"and {TOTAL['clang']} clang programs")
  sys.exit(not res.wasSuccessful())
