# quantized ops written in plain tinygrad Tensor ops. They run on every device; on DEV=CORAL the kernel selector
# (coral/select.py, hooked in by tinygrad/runtime/ops_coral.py) compiles the qlinear kernel into one Edge TPU program and
# recognizes the qconv2d kernel as one Edge TPU CONV_2D (coral.select.ConvSpec).
from __future__ import annotations
import numpy as np
from tinygrad import Tensor, dtypes

def qlinear(x_u8:Tensor, w_u8:Tensor, b_i32:Tensor|None, in_zp:int, w_zp:int, out_zp:int, mult:float, lo:int=0, hi:int=255) -> Tensor:
  """y[..., N] (uint8) = requant(x[..., K] (uint8) @ w[N, K].T (uint8) + b[N] (int32)), bit-exact with the Edge TPU (coral.tpu.fc_reference):
    acc = sum_k (x - in_zp) * (w - w_zp) + b          int32
    y   = clip(float32(acc) * float32(mult), lo - out_zp, hi - out_zp)
    out = uint8(round_half_to_even(y) + out_zp)       Tensor.round(), exact here since |y| <= 255
  The layer is a kernel of its own: the inputs are contiguous (a lazy x, e.g. a quantize, is a separate kernel instead of being
  recomputed inside the matmul) and the result is assigned to an output buffer, so a consumer (a dequantize) is not fused into it.
  Like a custom kernel's output that buffer outlives the schedule: a consumer realized later (llama's attention realizes the
  KV cache first) reads it instead of recomputing the layer, which .contiguous() would do."""
  assert x_u8.dtype == dtypes.uint8 and w_u8.dtype == dtypes.uint8, (x_u8.dtype, w_u8.dtype)
  assert b_i32 is None or b_i32.dtype == dtypes.int32, b_i32.dtype
  assert all(0 <= int(v) <= 255 for v in (in_zp, w_zp, out_zp)) and 0 <= lo <= hi <= 255, (in_zp, w_zp, out_zp, lo, hi)
  acc = (x_u8.contiguous().cast(dtypes.int32) - int(in_zp)).matmul((w_u8.contiguous().cast(dtypes.int32) - int(w_zp)).T)
  if b_i32 is not None: acc = acc + b_i32.contiguous()
  y = (acc.cast(dtypes.float32) * float(np.float32(mult))).clip(int(lo) - int(out_zp), int(hi) - int(out_zp))
  out = (y.round() + int(out_zp)).cast(dtypes.uint8)
  return Tensor.empty(*out.shape, dtype=dtypes.uint8, device=out.device).assign(out)

def qconv2d(x_u8:Tensor, w_u8:Tensor, b_i32:Tensor|None, in_zp:int, w_zp:int, out_zp:int, mult:float, lo:int=0, hi:int=255,
            stride:int=1, padding:int=0) -> Tensor:
  """y[N, Cout, OH, OW] (uint8) = requant(conv2d(x[N, Cin, H, W] (uint8), w[Cout, Cin, kh, kw] (uint8)) + b[Cout] (int32)), bit-exact
  with the Edge TPU (coral.select.conv_reference: im2col + coral.tpu.fc_reference):
    acc = sum_{c,ky,kx} (xp[n, c, oy*stride + ky, ox*stride + kx] - in_zp) * (w[o, c, ky, kx] - w_zp) + b[o]     int32
    out = qlinear's requant of acc
  with xp = x padded by `padding` on every side with in_zp, a real 0: the padding contributes nothing. tinygrad's conv2d (the
  windows of the padded input, times w, summed over cin, kh, kw) and the requant schedule as ONE kernel; inputs and output are
  buffers of their own as in qlinear. groups = 1, dilation = 1, the same stride and padding on both axes."""
  assert x_u8.dtype == dtypes.uint8 and w_u8.dtype == dtypes.uint8, (x_u8.dtype, w_u8.dtype)
  assert x_u8.ndim == 4 and w_u8.ndim == 4 and x_u8.shape[1] == w_u8.shape[1], (
    f"want x[N, Cin, H, W], w[Cout, Cin, kh, kw]: {x_u8.shape}, {w_u8.shape}")
  assert b_i32 is None or (b_i32.dtype == dtypes.int32 and b_i32.shape == w_u8.shape[:1]), (b_i32.dtype, b_i32.shape)
  assert all(0 <= int(v) <= 255 for v in (in_zp, w_zp, out_zp)) and 0 <= lo <= hi <= 255, (in_zp, w_zp, out_zp, lo, hi)
  stride, padding = int(stride), int(padding)
  assert stride >= 1 and padding >= 0, (stride, padding)
  x = x_u8.contiguous()
  if padding: x = x.pad(((0, 0), (0, 0), (padding, padding), (padding, padding)), value=int(in_zp))
  acc = (x.cast(dtypes.int32) - int(in_zp)).conv2d(w_u8.contiguous().cast(dtypes.int32) - int(w_zp), stride=stride)
  if b_i32 is not None: acc = acc + b_i32.contiguous().reshape(1, -1, 1, 1)
  y = (acc.cast(dtypes.float32) * float(np.float32(mult))).clip(int(lo) - int(out_zp), int(hi) - int(out_zp))
  out = (y.round() + int(out_zp)).cast(dtypes.uint8)
  return Tensor.empty(*out.shape, dtype=dtypes.uint8, device=out.device).assign(out)
