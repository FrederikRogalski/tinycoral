# tinygrad layers for the Edge TPU
#   QLinear:     a quantized linear layer in plain tinygrad ops (coral.qops.qlinear). On DEV=CORAL tinygrad's scheduler fuses it
#                into one kernel and the CORAL backend's kernel selector (coral/select.py) turns that kernel into an Edge TPU
#                program; on any other device it runs as ordinary kernels, with the same bit-exact result.
#   FusedBlock / FusedArgmax: a coral.fused block (several layers in one TPU program) as an explicit custom kernel
from __future__ import annotations
import json, numpy as np
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import Ops, KernelInfo
from coral.fused import quantize_params

class QLinear:
  """y = x @ W.T + b with uint8 weights (per tensor); activations quantized with calibrated ranges"""
  def __init__(self, W:np.ndarray, b:np.ndarray|None, in_range:tuple[float, float], out_range:tuple[float, float], relu:bool=False, device="CORAL"):
    N, K = W.shape
    self.w_scale, self.w_zp = quantize_params(float(W.min()), float(W.max()))
    self.in_scale, self.in_zp = quantize_params(*in_range)
    self.out_scale, self.out_zp = quantize_params(0.0 if relu else out_range[0], out_range[1])
    wq = np.clip(np.round(W / self.w_scale) + self.w_zp, 0, 255).astype(np.uint8)
    bq = np.round((b if b is not None else np.zeros(N)) / (self.in_scale * self.w_scale)).astype(np.int32)
    self.wq, self.bq = Tensor(wq, device=device).realize(), Tensor(bq, device=device).realize()
    self.N, self.K, self.lo = N, K, self.out_zp if relu else 0
    self.mult = float(np.float32(self.in_scale * self.w_scale / self.out_scale))
  def __call__(self, x:Tensor) -> Tensor:
    from coral.qops import qlinear
    lead = x.shape[:-1]
    xq = (x / self.in_scale + self.in_zp).round().clip(0, 255).cast(dtypes.uint8).reshape(-1, self.K)
    y = qlinear(xq, self.wq, self.bq, self.in_zp, self.w_zp, self.out_zp, self.mult, self.lo)
    return ((y.cast(dtypes.float32) - self.out_zp) * self.out_scale).reshape(*lead, self.N)

def _kernel(name:str, spec:dict, out:Tensor, *ins:Tensor) -> Tensor:
  from tinygrad.runtime.ops_coral import CORAL_SRC
  def fxn(o:UOp, *xs:UOp) -> UOp:
    sink = UOp.sink(o, *xs, arg=KernelInfo(name=name))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=CORAL_SRC + json.dumps(spec))))
  return Tensor.custom_kernel(out, *[t.contiguous() for t in ins], fxn=fxn)[0]

class FusedBlock:
  """a coral.fused block as a tinygrad layer: quantize, one TPU program, dequantize"""
  def __init__(self, name:str):
    from coral.fused import REGISTRY
    self.blk = REGISTRY[name]
  def __call__(self, x:Tensor) -> Tensor:
    b, lead = self.blk, x.shape[:-1]
    M = int(np.prod(lead)) if lead else 1
    (si, zi), (so, zo) = b.in_q, b.out_q
    xq = (x / si + zi).round().clip(0, 255).cast(dtypes.uint8).reshape(M, b.K)
    y = _kernel(f"tpu_{b.name.replace('.', '_')}_{M}", {"block": b.name, "M": M, "K": b.K, "N": b.N},
                Tensor.empty(M, b.N, dtype=dtypes.uint8, device=x.device), xq)
    return ((y.cast(dtypes.float32) - zo) * so).reshape(*lead, b.N)

class FusedArgmax:
  """the classifier as a coral.fused argmax block: token ids [...] instead of logits (block maxima on the TPU, refined on
  the host with the float activations)"""
  def __init__(self, name:str):
    from coral.fused import REGISTRY
    self.blk = REGISTRY[name]
  def __call__(self, x:Tensor) -> Tensor:
    b, lead = self.blk, x.shape[:-1]
    M = int(np.prod(lead)) if lead else 1
    si, zi = b.in_q
    xq = (x / si + zi).round().clip(0, 255).cast(dtypes.uint8).reshape(M, b.K)
    ids = _kernel(f"tpu_{b.name}_argmax_{M}", {"block": b.name, "M": M, "K": b.K, "argmax": True},
                  Tensor.empty(M, dtype=dtypes.int32, device=x.device),
                  xq, x.reshape(M, b.K).cast(dtypes.float32))
    return ids.reshape(*lead)
