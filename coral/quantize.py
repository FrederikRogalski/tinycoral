# post-training quantization of a tinygrad model for the Edge TPU:
#   quantize(model, calibration_inputs)
# swaps every nn.Linear / nn.Conv2d of the model, in place, for a uint8 layer written in plain tinygrad (coral.qops). On DEV=CORAL
# the backend's kernel selection compiles those into Edge TPU programs; on every other device they run as ordinary kernels with
# the same result. Like TFLite's converter it first rewrites the float model, and checks every rewrite on the calibration data:
#   fold:  a batch norm right behind a layer (or behind its max pool, scale > 0) goes into that layer's weights. Behind a ReLU
#          (scale > 0 commutes with it), its scale goes into the layer before and its shift into the bias of the layer after
#          (through max pools and flattens).
#   link:  when a layer's output reaches the next layer only through ReLU, max pool or flatten, both share one quantization. The
#          layer's requantization then clamps at 0 (that is the ReLU) and what runs in between is an exact max pool on uint8.
# A convolution becomes coral.qops.qconv2d (one kernel, an Edge TPU CONV_2D); dilated ones or ones with uneven strides or
# paddings become im2col + qlinear. Everything else (activations, norms that don't fold, pooling) stays as written.
from __future__ import annotations
import numpy as np
from tinygrad import Tensor, dtypes, nn
from coral.fused import quantize_params
from coral.qops import qlinear, qconv2d

LAYERS = (nn.Linear, nn.Conv2d, nn.BatchNorm)

class _Record:
  """stands in for a layer during calibration: logs (layer, input, output) in call order"""
  def __init__(self, layer, log:list): self.layer, self.log = layer, log
  def __call__(self, x:Tensor) -> Tensor:
    y = self.layer(x)
    self.log.append((self.layer, x.numpy(), y.numpy()))
    return y

def _slots(obj, seen=None):
  """(container, key, layer) of every nn.Linear / nn.Conv2d / nn.BatchNorm reachable from obj through attributes, lists and dicts"""
  seen = set() if seen is None else seen
  if id(obj) in seen: return
  seen.add(id(obj))
  items = obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, list) else vars(obj).items() if hasattr(obj, "__dict__") else []
  for k, v in list(items):
    if isinstance(v, LAYERS): yield obj, k, v
    elif isinstance(v, (list, dict)) or (hasattr(v, "__dict__") and not isinstance(v, (Tensor, type))): yield from _slots(v, seen)

def _set(c, k, v):
  if isinstance(c, (dict, list)): c[k] = v
  else: setattr(c, k, v)

def calibrate(model, calib:Tensor, slots) -> tuple[list, np.ndarray]:
  """run the float model on calib -> ([(layer, input, output)] in call order, the model's output)"""
  log: list = []
  for c, k, layer in slots: _set(c, k, _Record(layer, log))
  try: out = model(calib).numpy()
  finally:
    for c, k, layer in slots: _set(c, k, layer)
  return log, out

# *** fold batch norms ***

def _identity(x:Tensor) -> Tensor: return x
def _np(t:Tensor|None) -> np.ndarray|None: return None if t is None else t.numpy().astype(np.float64)
def _assign(layer, w:np.ndarray, b:np.ndarray):
  layer.weight = Tensor(w.astype(np.float32), device=layer.weight.device).realize()
  layer.bias = Tensor(b.astype(np.float32), device=layer.weight.device).realize()

def _fold(log, i:int) -> list[tuple]|None:
  """the weight updates that fold the batch norm log[i] away, or None: [(layer, new weight, new bias)]"""
  bn, x, _ = log[i]
  if i == 0 or not isinstance(prev:=log[i-1][0], (nn.Linear, nn.Conv2d)) or bn.running_mean is None: return None
  a = _np(bn.weight) / np.sqrt(_np(bn.running_var) + bn.eps) if bn.weight is not None else 1 / np.sqrt(_np(bn.running_var) + bn.eps)
  s = (_np(bn.bias) if bn.bias is not None else 0) - a * _np(bn.running_mean)
  w, b, y = _np(prev.weight), _np(prev.bias) if prev.bias is not None else np.zeros(prev.weight.shape[0]), log[i-1][2]
  ax = (slice(None),) + (None,) * (w.ndim - 1)
  if np.array_equal(x, y): return [(prev, w * a[ax], b * a + s)]              # right behind the layer
  if (a > 0).all() and (m:=_maxpool(y, 2, 2)) is not None and m.shape == x.shape and np.array_equal(x, m):
    return [(prev, w * a[ax], b * a + s)]                                     # behind a max pool: a > 0 and the shift commute with it
  if not (np.array_equal(x, np.maximum(y, 0)) and (a > 0).all() and i + 1 < len(log)): return None
  if not isinstance(nxt:=log[i+1][0], (nn.Linear, nn.Conv2d)): return None  # behind its ReLU: the shift goes to the next layer
  wn, bn_= _np(nxt.weight), _np(nxt.bias) if nxt.bias is not None else np.zeros(nxt.weight.shape[0])
  if isinstance(nxt, nn.Conv2d) and nxt.groups == 1 and wn.shape[1] == len(s) and not any(np.atleast_1d(nxt.padding)):
    bn_ = bn_ + np.einsum("ochw,c->o", wn, s)
  elif isinstance(nxt, nn.Linear) and wn.shape[1] % len(s) == 0: bn_ = bn_ + wn @ np.repeat(s, wn.shape[1] // len(s))   # NCHW flatten
  else: return None
  return [(prev, w * a[ax], b * a), (nxt, wn, bn_)]

def fold_batchnorms(model, calib:Tensor) -> int:
  """fold every batch norm that folds exactly (the model's output on calib stays the same) -> how many"""
  slots, folded = list(_slots(model)), 0
  log, ref = calibrate(model, calib, slots)
  for c, k, bn in [s for s in slots if isinstance(s[2], nn.BatchNorm)]:
    i = next(j for j, (l, _, _) in enumerate(log) if l is bn)
    if (updates:=_fold(log, i)) is None: continue
    saved = [(l, l.weight, l.bias) for l, _, _ in updates]
    for l, w, b in updates: _assign(l, w, b)
    _set(c, k, _identity)
    out = model(calib).numpy()
    if np.abs(out - ref).max() <= 1e-4 * max(np.abs(ref).max(), 1e-6) + 1e-6: folded += 1
    else:                                                                   # not exact (something between doesn't commute): undo
      for l, w, b in saved: l.weight, l.bias = w, b
      _set(c, k, bn)
    log, _ = calibrate(model, calib, list(_slots(model)))
  return folded

# *** link quantizations across ReLU / max pool / flatten ***

def _maxpool(v:np.ndarray, k:int, s:int) -> np.ndarray|None:
  if v.ndim != 4 or v.shape[2] < k or v.shape[3] < k: return None
  w = np.lib.stride_tricks.sliding_window_view(v, (k, k), axis=(2, 3))[:, :, ::s, ::s]
  return w.max((4, 5))

def link(y:np.ndarray, x:np.ndarray) -> str|None:
  """how the next layer's input x was computed from this layer's output y, if it is ReLU and/or a max pool and/or a flatten"""
  for name, f in (("", lambda v: v), ("maxpool2", lambda v: _maxpool(v, 2, 2)), ("maxpool3s2", lambda v: _maxpool(v, 3, 2))):
    for relu in (False, True):
      if (v:=f(np.maximum(y, 0) if relu else y)) is not None and v.size == x.size and np.array_equal(v.reshape(x.shape), x):
        return ",".join(n for n in (("relu" if relu else ""), name) if n) or "identity"
  return None

class QLayer:
  """a calibrated nn.Linear or nn.Conv2d as uint8 weights (per tensor) and uint8 activations"""
  def __init__(self, layer, in_range:tuple[float, float], out_range:tuple[float, float]):
    W, self.conv = layer.weight.numpy().astype(np.float32), isinstance(layer, nn.Conv2d)
    if self.conv:
      assert layer.groups == 1, "grouped convolutions are not supported"
      self.kernel, self.stride, self.dilation, self.padding = tuple(W.shape[2:]), layer.stride, layer.dilation, layer.padding
      def one(v): return (set(np.atleast_1d(v).tolist()) or {0}).pop() if len(set(np.atleast_1d(v).tolist())) <= 1 else None
      self.native = one(self.dilation) == 1 and one(self.stride) is not None and one(self.padding) is not None   # else im2col
    W2 = W.reshape(W.shape[0], -1)                                    # [out, in * kh * kw]: the order the windows are cut in
    def f32(q): return (float(np.float32(q[0])), q[1])                  # scales as a TFLite file stores them
    (self.ws, self.wz), (self.xs, self.xz), (self.ys, self.yz) = map(f32, (quantize_params(float(W2.min()), float(W2.max())),
                                                                         quantize_params(*in_range), quantize_params(*out_range)))
    b = layer.bias.numpy() if layer.bias is not None else np.zeros(W.shape[0], np.float32)
    self.wq = Tensor(np.clip(np.round(W / self.ws) + self.wz, 0, 255).astype(np.uint8), device=layer.weight.device).realize()   # W's shape
    self.bq = Tensor(np.round(b / (self.xs * self.ws)).astype(np.int32), device=layer.weight.device).realize()
    self.mult = float(np.float32(self.xs * self.ws / self.ys))
  def __call__(self, x:Tensor) -> Tensor:
    xq = (x / self.xs + self.xz).round().clip(0, 255).cast(dtypes.uint8)
    if not self.conv:
      y = qlinear(xq.reshape(-1, xq.shape[-1]), self.wq, self.bq, self.xz, self.wz, self.yz, self.mult).reshape(*x.shape[:-1], -1)
      return (y.cast(dtypes.float32) - self.yz) * self.ys
    if self.native:                     # one kernel: the backend selects it as an Edge TPU CONV_2D
      s, p = (int(np.atleast_1d(v)[0]) for v in (self.stride, self.padding))
      y = qconv2d(xq, self.wq, self.bq, self.xz, self.wz, self.yz, self.mult, stride=s, padding=p)
      return (y.cast(dtypes.float32) - self.yz) * self.ys
    # anything else (dilation, uneven strides or paddings) as im2col: pad with the input zero point (it contributes nothing), cut
    # out the windows, one row per output position, one quantized matmul
    bs, c = x.shape[:2]
    p = self.padding if isinstance(self.padding, (tuple, list)) else [self.padding] * 2
    pads = ((0, 0), (0, 0), (p[0], p[0]), (p[-1], p[-1])) if len(p) == 2 else ((0, 0), (0, 0), (p[2], p[3]), (p[0], p[1]))
    win = xq.pad(pads, value=self.xz)._pool(self.kernel, self.stride, self.dilation)          # [bs, c, oy, ox, kh, kw]
    oy, ox = win.shape[2:4]
    rows = win.permute(0, 2, 3, 1, 4, 5).reshape(bs * oy * ox, c * self.kernel[0] * self.kernel[1])
    y = qlinear(rows, self.wq.reshape(self.wq.shape[0], -1), self.bq, self.xz, self.wz, self.yz, self.mult)   # [bs * oy * ox, out]
    return (y.reshape(bs, oy, ox, -1).permute(0, 3, 1, 2).cast(dtypes.float32) - self.yz) * self.ys

def quantize(model, calib:Tensor, fold:bool=True, share:bool=True) -> dict:
  """fold the batch norms (fold), calibrate on `calib` in float, share quantizations across ReLU / max pool / flatten (share),
  then swap every Linear / Conv2d of `model` for its uint8 version, in place -> {"folded": n, "chain": [(QLayer, link)]}: the
  quantized layers in call order, each with how its output reaches the next one ("relu,maxpool2", ...; None: unknown)"""
  folded = fold_batchnorms(model, calib) if fold else 0
  slots = [s for s in _slots(model) if isinstance(s[2], (nn.Linear, nn.Conv2d))]
  log, _ = calibrate(model, calib, slots)
  def rng(v): return (float(v.min()), float(v.max()))
  ranges, links = {id(l): [rng(x), rng(y)] for l, x, y in log}, [None] * len(log)
  for i, ((a, _, y), (b, x, _)) in enumerate(zip(log, log[1:])):
    if not share or (how:=link(y, x)) is None: continue
    ranges[id(a)][1] = ranges[id(b)][0] = (0.0, float(y.max())) if "relu" in how else rng(y)
    links[i] = how
  q = {id(layer): QLayer(layer, *ranges[id(layer)]) for _, _, layer in slots}
  for l, x, y in log: q[id(l)].shapes = (x.shape[1:], y.shape[1:])         # per sample: what an exporter needs to know
  for c, k, layer in slots: _set(c, k, q[id(layer)])
  return {"folded": folded, "chain": [(q[id(l)], how) for (l, _, _), how in zip(log, links)]}
