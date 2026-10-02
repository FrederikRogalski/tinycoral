#!/usr/bin/env python
# Kernel selection on DEV=CORAL (tinygrad/runtime/ops_coral.py + coral/select.py): the plain tinygrad qlinear of coral/qops.py
# goes through tinygrad's own scheduler and compile pipeline and becomes ONE Edge TPU program, bit-identical to DEV=CPU;
# every other kernel stays a clang kernel. The TPU is the bit-exact numpy model (MOCKCORAL=1, the default here).
#   .venv/bin/python test/test_select.py [TestMatcher[.test_variants] ...]
import os
os.environ.setdefault("MOCKCORAL", "1")       # never touch the USB device unless asked to explicitly (MOCKCORAL= ...)
import itertools, json, sys, unittest
import numpy as np
from tinygrad import Tensor, dtypes, TinyJit, Device
from tinygrad.codegen import to_program
from tinygrad.uop.ops import UOp, Ops, KernelInfo
import coral.tpu as tpu
from coral.tpu import FCSpec, fc_reference
from coral.qops import qlinear
from coral.select import match_fc, why_not, select, select_fc, ConvMatch
import tinygrad.runtime.ops_coral as ops_coral
from tinygrad.runtime.ops_coral import CoralProgram, CORAL_SRC

I32, F32, U8, I16 = dtypes.int32, dtypes.float32, dtypes.uint8, dtypes.int16
TOTAL = {"tpu": 0, "clang": 0, "run_fc": 0, "run_conv": 0, "cases": 0}
def R(y:Tensor) -> Tensor: return y.round()             # round half to even, the Edge TPU's rounding
def half_away(y:Tensor) -> Tensor: return y.sign() * (y.abs() + 0.5).floor()    # NOT the TPU's rounding

class Count:
  """counts the programs CORAL runs: Edge TPU programs (and coral.tpu.run_fc / run_conv calls) vs clang programs"""
  def __enter__(self):
    self.tpu = self.clang = self.run_fc = self.run_conv = 0
    self._call, self._run_fc, self._run_conv = CoralProgram.__call__, tpu.run_fc, tpu.run_conv
    def call(prg, *args, **kwargs):
      if prg.tpu: self.tpu += 1
      else: self.clang += 1
      return self._call(prg, *args, **kwargs)
    def run_fc(*args, **kwargs):
      self.run_fc += 1
      return self._run_fc(*args, **kwargs)
    def run_conv(*args, **kwargs):
      self.run_conv += 1
      return self._run_conv(*args, **kwargs)
    CoralProgram.__call__, tpu.run_fc, tpu.run_conv = call, run_fc, run_conv
    return self
  def __exit__(self, *exc):
    CoralProgram.__call__, tpu.run_fc, tpu.run_conv = self._call, self._run_fc, self._run_conv
    for k in ("tpu", "clang", "run_fc", "run_conv"): TOTAL[k] += getattr(self, k)

def tpu_fc(x:Tensor, w:Tensor, b:Tensor, spec:FCSpec) -> Tensor:
  """an Edge TPU FC as a custom kernel: a PROGRAM whose SOURCE is the spec"""
  def fxn(o:UOp, xi:UOp, wi:UOp, bi:UOp) -> UOp:
    sink = UOp.sink(o, xi, wi, bi, arg=KernelInfo(name=f"tpu_fc_{spec.M}x{spec.K}x{spec.N}"))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=CORAL_SRC + spec.dumps())))
  return Tensor.custom_kernel(Tensor.empty(spec.M, spec.N, dtype=U8, device=x.device), x, w, b, fxn=fxn)[0]

def on(dev:str, *arrays): return [None if a is None else Tensor(a, device=dev).realize() for a in arrays]
def run_both(f, *arrays):
  """f on realized inputs on CPU and on CORAL: (cpu result, coral result, Count of the CORAL run)"""
  cpu = f(*on("CPU", *arrays)).numpy()
  ins = on("CORAL", *arrays)
  with Count() as c: coral = f(*ins).numpy()
  TOTAL["cases"] += 1
  return cpu, coral, c
def kernel_ast(t:Tensor): return t.schedule_linear().src[0].without_after.body

def u8(rng, *shape): return rng.integers(0, 256, shape, dtype=np.uint8)
def i32(rng, n, lim): return rng.integers(-lim, lim, n).astype(np.int32)

def random_case(rng, M:int, N:int, K:int, bias=True):
  """random data and a random quantization whose outputs spread over [lo, hi] (some saturate); a quarter of the multipliers
  are powers of two, which put exact .5 ties into the rounding"""
  x, w, b = u8(rng, M, K), u8(rng, N, K), i32(rng, N, 2**16) if bias else None
  in_zp, w_zp, out_zp = (int(v) for v in rng.integers(0, 256, 3))
  acc = (x.astype(np.float64) - in_zp) @ (w.astype(np.float64) - w_zp).T + (0 if b is None else b)
  mult = float(np.float32(rng.uniform(20, 150) / max(float(acc.std()), 1.0)))
  if rng.random() < 0.25: mult = float(2.0 ** np.round(np.log2(mult)))
  lo, hi = (0, 255) if rng.random() < 0.5 else tuple(sorted(int(v) for v in rng.integers(0, 256, 2)))
  if rng.random() < 0.2: lo, hi = out_zp, 255                    # fused relu
  return (x, w, b), FCSpec(M, N, K, in_zp, w_zp, out_zp, mult, lo, hi)

def ref(spec:FCSpec, x, w, b): return fc_reference(spec, x, w, np.zeros(spec.N, np.int32) if b is None else b)
def ql(spec:FCSpec): return lambda x, w, b: qlinear(x, w, b, spec.in_zp, spec.w_zp, spec.out_zp, spec.mult, spec.lo, spec.hi)

class TestQLinear(unittest.TestCase):
  def check(self, arrays, spec):
    cpu, coral, c = run_both(ql(spec), *arrays)
    expect = ref(spec, *arrays)
    np.testing.assert_array_equal(cpu, expect, err_msg=f"CPU vs fc_reference {spec}")
    np.testing.assert_array_equal(coral, expect, err_msg=f"CORAL vs fc_reference {spec}")
    self.assertEqual((c.tpu, c.run_fc, c.clang), (1, 1, 0), f"{spec}: want one Edge TPU program and no clang program")

  def test_grid(self):
    """the same qlinear code on DEV=CPU and DEV=CORAL: bit-identical (and equal to fc_reference) on every shape"""
    rng = np.random.default_rng(0)
    for M, N, K in itertools.product([1, 4, 128], [1, 10, 288, 864, 1536], [5, 288, 768]):
      with self.subTest(M=M, N=N, K=K): self.check(*random_case(rng, M, N, K))
    for N, K in itertools.product([1, 10, 288, 864, 1536], [5, 288, 768]):       # without bias: a 3 buffer program
      with self.subTest(M=4, N=N, K=K, bias=False): self.check(*random_case(rng, 4, N, K, bias=False))

  def test_random_quantizations(self):
    rng = np.random.default_rng(1)
    for i in range(40):
      M, N, K = int(rng.integers(1, 40)), int(rng.integers(1, 300)), int(rng.integers(2, 300))
      with self.subTest(i=i, M=M, N=N, K=K): self.check(*random_case(rng, M, N, K, bias=bool(i % 4)))

  def test_ties(self):
    """exact .5 ties round half to even (the device's rounding): odd accumulators times 0.5 (or 1.5, 2.5, 0.25 at acc = 2
    mod 4), positive and negative, below even and odd integers; the result must differ from half away from zero"""
    rng = np.random.default_rng(2)
    x, w, b = rng.integers(112, 129, (16, 7), dtype=np.uint8), rng.integers(122, 139, (64, 7), dtype=np.uint8), i32(rng, 64, 9)
    acc = (x.astype(np.int64) - 120) @ (w.astype(np.int64) - 130).T + b
    for mult, out_zp, lo, hi in [(0.5, 128, 0, 255), (1.5, 128, 0, 255), (2.5, 128, 20, 240), (0.25, 128, 0, 255), (0.5, 0, 0, 255),
                                 (0.5, 128, 128, 255)]:
      spec = FCSpec(16, 64, 7, 120, 130, out_zp, mult, lo, hi)
      y = np.clip(acc * mult, lo - out_zp, hi - out_zp)
      ties = (y % 1 == 0.5)
      classes = [y > 0, np.floor(y) % 2 == 0, np.floor(y) % 2 == 1] + ([y < 0] if lo < out_zp else [])
      self.assertTrue(all(np.any(ties & cls) for cls in classes), f"ties missing for {spec}")
      expect = ref(spec, x, w, b)
      np.testing.assert_array_equal(expect, (np.round(y) + out_zp).astype(np.uint8))             # half even ...
      self.assertTrue(np.any(expect != (np.sign(y) * np.floor(np.abs(y) + 0.5) + out_zp).astype(np.uint8)))   # ... not half away
      self.check((x, w, b), spec)
    # near ties: acc * mult = 0.4999999776 rounds to the float32 0.49999997, which rounds to 0
    mult, x1, w1 = 0.026315787807106972, np.array([[19, 0]], np.uint8), np.array([[1, 1]], np.uint8)   # acc = 19
    self.check((x1, w1, None), FCSpec(1, 1, 2, 0, 0, 10, mult, 0, 255))
    self.assertEqual(int(ref(FCSpec(1, 1, 2, 0, 0, 10, mult, 0, 255), x1, w1, None)[0, 0]), 10)

class TestSchedule(unittest.TestCase):
  def test_one_kernel_one_program(self):
    """qlinear schedules ONE kernel (reduce + epilogue); compiled for CORAL it is an Edge TPU program with the FCSpec as its
    source, launched with the buffers in TPU order (out, x, w, b) whatever slots the scheduler gave them"""
    (x, w, b), spec = random_case(np.random.default_rng(3), 4, 10, 5)
    xt, wt, bt = on("CORAL", x, w, b)
    bias_first = (R(((bt + (xt.cast(I32) - spec.in_zp) @ (wt.cast(I32) - spec.w_zp).T).cast(F32) * spec.mult)
                    .clip(spec.lo - spec.out_zp, spec.hi - spec.out_zp)) + spec.out_zp).cast(U8)
    for out in (ql(spec)(xt, wt, bt), bias_first):
      calls = [c.without_after for c in out.schedule_linear().src]
      self.assertEqual([c.body.op for c in calls], [Ops.SINK])
      prg = to_program(calls[0].body, Device["CORAL"].renderer)
      self.assertTrue(prg.src[2].arg.startswith(CORAL_SRC))
      self.assertEqual(FCSpec(**json.loads(prg.src[2].arg[len(CORAL_SRC):])), spec)
      args = calls[0].src[1:]
      self.assertEqual([args[s].dtype for s in prg.arg.globals], [U8, U8, U8, I32])
      self.assertEqual([args[s].buffer for s in prg.arg.globals[1:]], [t.uop.base.buffer for t in (xt, wt, bt)])
      self.assertEqual(prg.arg.globals, tuple(p.arg.slot for p in match_fc(calls[0].body).params))
    self.assertNotEqual(prg.arg.globals, tuple(sorted(prg.arg.globals)))   # bias first: the scheduler's slots are (out, b, x, w)

  def test_lazy_input_and_consumer(self):
    """quantize -> qlinear -> dequantize: the quantize and the dequantize are clang kernels, the layer one TPU program"""
    rng = np.random.default_rng(4)
    xf, w, b = rng.normal(size=(8, 33)).astype(np.float32), u8(rng, 40, 33), i32(rng, 40, 999)
    spec = FCSpec(8, 40, 33, 128, 127, 100, 0.002, 0, 255)
    def f(xf, w, b): return (ql(spec)((xf / 0.02 + 128).round().clip(0, 255).cast(U8), w, b).cast(F32) - 100) * 0.1
    cpu, coral, c = run_both(f, xf, w, b)
    np.testing.assert_array_equal(cpu, coral)
    self.assertEqual((c.tpu, c.clang), (1, 2))

  def test_chain_and_parallel_compile(self):
    """several layers in one schedule: compiled by tinygrad's worker pool (spawned processes), all selected"""
    rng = np.random.default_rng(5)
    x, w1, w2, w3, b1, b2 = u8(rng, 6, 17), u8(rng, 23, 17), u8(rng, 19, 23), u8(rng, 11, 17), i32(rng, 23, 999), i32(rng, 19, 999)
    s1, s2, s3 = FCSpec(6, 23, 17, 120, 128, 110, 0.003, 3, 250), FCSpec(6, 19, 23, 110, 128, 90, 0.004, 90, 255), FCSpec(6, 11, 17, 7, 9, 11, 0.0021)
    def f(x, w1, b1, w2, b2, w3): return ql(s2)(ql(s1)(x, w1, b1), w2, b2).cat(ql(s3)(x, w3, None), dim=1)
    cpu, coral, c = run_both(f, x, w1, b1, w2, b2, w3)
    np.testing.assert_array_equal(coral, np.concatenate([ref(s2, ref(s1, x, w1, b1), w2, b2), ref(s3, x, w3, None)], axis=1))
    np.testing.assert_array_equal(cpu, coral)
    self.assertEqual((c.tpu, c.clang), (3, 1))   # the cat is a clang kernel

  def test_jit(self):
    rng = np.random.default_rng(6)
    w, b = on("CORAL", u8(rng, 24, 16), i32(rng, 24, 3000))
    spec = FCSpec(4, 24, 16, 128, 130, 110, 0.0031, 3, 250)
    @TinyJit
    def step(xf:Tensor) -> Tensor: return ((ql(spec)((xf / 0.02 + 128).round().clip(0, 255).cast(U8), w, b).cast(F32) - 110) * 0.05).realize()
    for i in range(4):
      xf = rng.normal(size=(4, 16)).astype(np.float32)
      with Count() as c: got = step(Tensor(xf, device="CORAL").realize()).numpy()
      xq = np.clip(np.round(xf / np.float32(0.02) + np.float32(128)), 0, 255).astype(np.uint8)
      np.testing.assert_array_equal(got, (ref(spec, xq, w.numpy(), b.numpy()).astype(np.float32) - 110) * np.float32(0.05))
      self.assertEqual((c.tpu, c.clang), (1, 2), f"step {i}")

  def test_views_and_aliases(self):
    rng = np.random.default_rng(7)
    big, w, b = u8(rng, 10, 7), u8(rng, 12, 7), i32(rng, 12, 3000)
    spec = FCSpec(6, 12, 7, 120, 130, 110, 0.0031, 3, 250)
    cpu, coral, c = run_both(lambda big, w, b: ql(spec)(big[2:8], w, b), big, w, b)     # x is a view at an offset
    np.testing.assert_array_equal(coral, ref(spec, big[2:8], w, b))
    np.testing.assert_array_equal(cpu, coral)
    self.assertEqual((c.tpu, c.clang), (1, 0))
    spec = FCSpec(10, 10, 7, 120, 130, 110, 0.0031, 3, 250)
    cpu, coral, c = run_both(lambda x, b: ql(spec)(x, x, b), big, b[:10])               # x is w: one buffer, passed twice
    np.testing.assert_array_equal(coral, ref(spec, big, big, b[:10]))
    np.testing.assert_array_equal(cpu, coral)
    self.assertEqual((c.tpu, c.clang), (1, 0))

  def test_custom_kernel_still_runs(self):
    """the custom kernel path (an Edge TPU FC as a precompiled PROGRAM, the way coral.nn runs fused blocks) still works"""
    rng = np.random.default_rng(8)
    x, w, b = u8(rng, 1, 50), u8(rng, 30, 50), i32(rng, 30, 999)
    spec = FCSpec(1, 30, 50, 120, 130, 110, 3.1e-4)
    with Count() as c: y = tpu_fc(*on("CORAL", x, w, b), spec).numpy()
    np.testing.assert_array_equal(y, fc_reference(spec, x, w, b))
    self.assertEqual((c.tpu, c.clang), (1, 0))

  def test_select_off(self):
    (x, w, b), spec = random_case(np.random.default_rng(9), 3, 9, 13)
    old, ops_coral.SELECT = ops_coral.SELECT, 0
    try: cpu, coral, c = run_both(ql(spec), x, w, b)
    finally: ops_coral.SELECT = old
    np.testing.assert_array_equal(cpu, coral)
    self.assertEqual((c.tpu, c.clang), (0, 1))

# how tinygrad may canonicalize the expression: every variant must become one TPU program with exactly these constants
M_, N_, K_ = 6, 12, 7
MULT = 0.5                                # with narrow data: an exact .5 tie at every odd accumulator
SPEC, RELU = FCSpec(M_, N_, K_, 120, 130, 110, MULT, 3, 250), FCSpec(M_, N_, K_, 120, 130, 110, MULT, 110, 250)
def xz(x): return x.cast(I32) - 120
def wz(w): return w.cast(I32) - 130
def acc(x, w, b): return xz(x) @ wz(w).T + (0 if b is None else b)
def y_(a): return a.cast(F32) * MULT
def out(y): return (R(y.clip(-107, 140)) + 110).cast(U8)                               # requant to SPEC from the float y
def rnd_with(f): return lambda y: (f(y.clip(-107, 140)) + 110).cast(U8)                  # ... with another rounding
def half_odd(y):                                                                         # Tensor.round with the branches swapped
  return ((y > 0) == ((t:=y.trunc() / 2.0).trunc() == t)).where((y + 0.5).floor(), (y - 0.5).ceil())
VARIANTS = {
  "bias first":            (lambda x,w,b: out(y_(b + xz(x) @ wz(w).T)), SPEC),
  "w @ x.T, transposed":   (lambda x,w,b: out(y_((wz(w) @ xz(x).T).T + b)), SPEC),
  "mul + sum":             (lambda x,w,b: out(y_((xz(x).reshape(M_, 1, K_) * wz(w).reshape(1, N_, K_)).sum(-1) + b)), SPEC),
  "+ -zp":                 (lambda x,w,b: out(y_((x.cast(I32) + (-120)) @ (w.cast(I32) + (-130)).T + b)), SPEC),
  "widening casts":        (lambda x,w,b: out(y_((x.cast(I16).cast(I32) - 120) @ (w.cast(dtypes.uint16).cast(I32) - 130).T + b)), SPEC),
  "zp 0, mult 1 folded":   (lambda x,w,b: R(((x.cast(I32) - 0) @ (w.cast(I32) - 0).T + b).cast(F32).mul(1.0).clip(0, 255)).cast(U8),
                            FCSpec(M_, N_, K_, 0, 0, 0, 1.0)),
  "no bias":               (lambda x,w,b: out(y_(acc(x, w, None))), SPEC),
  "clip after rounding":   (lambda x,w,b: (R(y_(acc(x, w, b))) + 110).clip(3, 250).cast(U8), SPEC),
  "relu().minimum()":      (lambda x,w,b: (R(y_(acc(x, w, b)).relu().minimum(140)) + 110).cast(U8), RELU),
  "clip(0, ...) relu":     (lambda x,w,b: (R(y_(acc(x, w, b)).clip(0, 140)) + 110).cast(U8), RELU),
  "maximum().minimum()":   (lambda x,w,b: (R(y_(acc(x, w, b)).maximum(-107).minimum(140)) + 110).cast(U8), SPEC),
  "minimum().maximum()":   (lambda x,w,b: (R(y_(acc(x, w, b)).minimum(140).maximum(-107)) + 110).cast(U8), SPEC),
  "fractional bounds":     (lambda x,w,b: (R(y_(acc(x, w, b)).clip(-107.3, 139.6)) + 110).cast(U8), SPEC),
  "two clips":             (lambda x,w,b: (R(y_(acc(x, w, b)).clip(-120, 140)).clip(-107, 200) + 110).cast(U8), SPEC),
  "int zero point":        (lambda x,w,b: (R(y_(acc(x, w, b)).clip(-107, 140)).cast(I32) + 110).cast(U8), SPEC),
  "int clip":              (lambda x,w,b: (R(y_(acc(x, w, b))).clip(-1000, 1000).cast(I32) + 110).clip(3, 250).cast(U8), SPEC),
}
# near misses: each must stay a clang kernel (and of course give the CPU result)
NO_MATCH = {
  "half away from zero": lambda x,w,b: rnd_with(half_away)(y_(acc(x, w, b))),
  "half away, trunc": lambda x,w,b: rnd_with(lambda y: y.sign() * (y.abs() + 0.5).trunc())(y_(acc(x, w, b))),
  "half away, relu clip": lambda x,w,b: (half_away(y_(acc(x, w, b)).clip(0, 140)) + 110).cast(U8),
  "half up: floor(y+.5)": lambda x,w,b: rnd_with(lambda y: (y + 0.5).floor())(y_(acc(x, w, b))),
  "half up, relu clip": lambda x,w,b: ((y_(acc(x, w, b)).clip(0, 140) + 0.5).floor() + 110).cast(U8),
  "half down: ceil(y-.5)": lambda x,w,b: rnd_with(lambda y: (y - 0.5).ceil())(y_(acc(x, w, b))),
  "half to odd": lambda x,w,b: rnd_with(half_odd)(y_(acc(x, w, b))),
  "trunc": lambda x,w,b: rnd_with(lambda y: y.trunc())(y_(acc(x, w, b))),
  "zero point before rounding": lambda x,w,b: R(y_(acc(x, w, b)).clip(-107, 140) + 110).cast(U8),
  "non-integer zero point": lambda x,w,b: (R(y_(acc(x, w, b)).clip(-107, 140)) + 110.5).cast(U8),
  "no lower clamp": lambda x,w,b: (R(y_(acc(x, w, b)).minimum(140)) + 110).cast(U8),
  "clamp below 0 (lo < 0)": lambda x,w,b: (R(y_(acc(x, w, b)).clip(-200, 140)) + 110).cast(U8),
  "float32 output": lambda x,w,b: R(y_(acc(x, w, b)).clip(-107, 140)) + 110,
  "bias added in float": lambda x,w,b: out((acc(x, w, None).cast(F32) + b.cast(F32)) * MULT),
  "bias + constant": lambda x,w,b: out(y_(acc(x, w, b) + 5)),
  "bias indexed by row": lambda x,w,b: out(y_(acc(x, w, None) + b[:M_].reshape(M_, 1))),
  "scale before the int cast": lambda x,w,b: out((acc(x, w, b) * 3).cast(F32)),
  "per-channel multiplier": lambda x,w,b: out(acc(x, w, b).cast(F32) * ((b.cast(F32).abs() + 1) * 1e-6)),
  "clamp before the multiply": lambda x,w,b: (R(acc(x, w, b).cast(F32).clip(-30000, 30000) * 0.0031) + 110).clip(3, 250).cast(U8),
  "int16 products": lambda x,w,b: out(((x.cast(I16) - 120) @ (w.cast(I16) - 130).T).cast(F32)),
  "float accumulation": lambda x,w,b: out((x.cast(F32) - 120) @ (w.cast(F32) - 130).T * MULT),
  "zero point 300": lambda x,w,b: out(y_((x.cast(I32) - 300) @ wz(w).T + b)),
  "(zp - x) * (w - zp)": lambda x,w,b: out(y_((120 - x.cast(I32)) @ wz(w).T + b)),
  "x stored [K, M]": lambda x,w,b: out(y_(xz(x.T.contiguous().T) @ wz(w).T + b)),
  "w stored [K, N]": lambda x,w,b: out(y_(xz(x) @ wz(w.T.contiguous().T).T + b)),
  "max reduce": lambda x,w,b: out((xz(x).reshape(M_, 1, K_) * wz(w).reshape(1, N_, K_)).max(-1).cast(F32)),
}
# near misses of the FC that are exactly a conv (coral.select.match_conv): with the bias indexed by row, out[m, n] is w's rows
# (an N x K image) convolved with x's rows as M filters of 1 x K, b per filter. select() gives that conv (test_select_conv runs it)
NO_MATCH_CONVS = {"bias indexed by row"}

class TestMatcher(unittest.TestCase):
  rng = np.random.default_rng(10)
  x, w, b = rng.integers(112, 129, (M_, K_), dtype=np.uint8), rng.integers(122, 139, (N_, K_), dtype=np.uint8), i32(rng, N_, 60)
  acc_np = (x.astype(np.int64) - 120) @ (w.astype(np.int64) - 130).T + b

  def test_variants(self):
    y = self.acc_np * MULT
    self.assertTrue(np.any((y % 1 == 0.5) & (np.floor(y) % 2 == 0)) and np.any((y % 1 == 0.5) & (np.floor(y) % 2 == 1)))   # ties both ways
    for name, (f, spec) in VARIANTS.items():
      with self.subTest(name):
        cpu, coral, c = run_both(f, self.x, self.w, self.b)
        np.testing.assert_array_equal(coral, ref(spec, self.x, self.w, None if name == "no bias" else self.b))
        np.testing.assert_array_equal(cpu, coral)
        self.assertEqual((c.tpu, c.clang), (1, 0))
        self.assertEqual(match_fc(kernel_ast(f(*on("CPU", self.x, self.w, self.b)))).spec, spec)

  def test_batched_rows(self):
    """x[B, T, K]: two output ranges that flatten row-major into M = B*T rows"""
    x3, spec = np.random.default_rng(11).integers(112, 129, (3, 5, K_), dtype=np.uint8), FCSpec(15, N_, K_, 120, 130, 110, MULT, 3, 250)
    cpu, coral, c = run_both(ql(spec), x3, self.w, self.b)
    np.testing.assert_array_equal(coral.reshape(15, N_), ref(spec, x3.reshape(15, K_), self.w, self.b))
    np.testing.assert_array_equal(cpu, coral)
    self.assertEqual((c.tpu, c.clang), (1, 0))

  def test_near_misses_stay_clang(self):
    for name, f in NO_MATCH.items():
      with self.subTest(name):
        cpu, coral, c = run_both(f, self.x, self.w, self.b)
        np.testing.assert_array_equal(cpu, coral)
        ast = kernel_ast(f(*on("CPU", self.x, self.w, self.b)))
        self.assertIsNone(select_fc(ast))
        if name in NO_MATCH_CONVS:                 # one conv (ops_coral dispatches a ConvSpec)
          self.assertIsInstance(select(ast), ConvMatch)
          self.assertEqual((c.tpu, c.run_conv, c.clang), (1, 1, 0))
          continue
        self.assertEqual(c.tpu, 0)
        self.assertGreaterEqual(c.clang, 1)
        self.assertIsNone(select(ast))
        self.assertIsNotNone(why_not(ast))

  def test_other_kernels_stay_clang(self):
    rng = np.random.default_rng(12)
    a, b, fa, fb = u8(rng, 64), u8(rng, 64), rng.normal(size=(32, 48)).astype(np.float32), rng.normal(size=(40, 48)).astype(np.float32)
    for name, f, arrays in [("elementwise add", lambda a, b: a + b, (a, b)), ("float matmul", lambda a, b: a @ b.T, (fa, fb)),
                            ("float linear + relu", lambda a, b: (a @ b.T + 1).relu(), (fa, fb)), ("sum", lambda a, b: (a * b).sum(), (fa, fa))]:
      with self.subTest(name):
        cpu, coral, c = run_both(f, *arrays)
        np.testing.assert_array_equal(cpu, coral)
        self.assertEqual(c.tpu, 0)
        self.assertGreaterEqual(c.clang, 1)

if __name__ == "__main__":
  res = unittest.main(exit=False, verbosity=2).result
  print(f"\n{res.testsRun - len(res.failures) - len(res.errors)}/{res.testsRun} tests passed; {TOTAL['cases']} CPU-vs-CORAL runs, "
        f"CORAL ran {TOTAL['tpu']} Edge TPU programs ({TOTAL['run_fc']} run_fc calls) and {TOTAL['clang']} clang programs")
  sys.exit(not res.wasSuccessful())
