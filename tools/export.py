# a model quantized by coral.quantize as a TFLite file, so Google's stack (edgetpu_compiler + libedgetpu) runs exactly the same
# uint8 model: an oracle for our programs, and a fair baseline. Covers chains of convolutions and linear layers joined by ReLU,
# max pools and flattens (coral.quantize's links); convolutions with VALID padding or stride-1 SAME padding (odd kernels).
from __future__ import annotations
import numpy as np, tflite
from tools.tflite_gen import Model, conv_options, fc_options

def _pool(k:int, s:int):
  def fn(b):
    tflite.Pool2DOptionsStart(b)
    tflite.Pool2DOptionsAddPadding(b, 1)
    tflite.Pool2DOptionsAddStrideW(b, s)
    tflite.Pool2DOptionsAddStrideH(b, s)
    tflite.Pool2DOptionsAddFilterWidth(b, k)
    tflite.Pool2DOptionsAddFilterHeight(b, k)
    tflite.Pool2DOptionsAddFusedActivationFunction(b, 0)
    return tflite.Pool2DOptionsEnd(b)
  return fn
POOLS = {"maxpool2": (2, 2), "maxpool3s2": (3, 2)}

def tflite_chain(chain:list, tall:int=1) -> bytes:
  """chain: coral.quantize(...)["chain"], [(QLayer, link to the next layer)]. NCHW tensors become NHWC.
  tall > 1: `tall` images stacked into one tall image. VALID convolutions and pools never mix the rows of two images, so each
  image's rows come out where they belong; a final linear layer on the flattened maps becomes a convolution over the map, whose
  output row r * (rows per image there) is image r's result. To edgetpu_compiler it is an ordinary single-image model."""
  m = Model()
  first, _ = chain[0]
  shape = [1, first.shapes[0][1] * tall, first.shapes[0][2], first.shapes[0][0]] if first.conv else [1, first.shapes[0][0]]
  t = inp = m.tensor("input", shape, np.uint8, first.xs, first.xz)
  for i, (q, how) in enumerate(chain):
    assert i == len(chain) - 1 or how is not None, f"layer {i}: unknown ops between it and the next layer"
    act, O = int(how is not None and "relu" in how), q.wq.shape[0]
    bt = m.tensor(f"b{i}", [O], np.int32, q.xs * q.ws, 0, data=q.bq.numpy())
    if q.conv:
      (kh, kw), C = q.kernel, q.shapes[0][0]
      st, pad = (q.stride if isinstance(q.stride, int) else q.stride[0]), np.atleast_1d(q.padding)
      assert q.dilation in (1, (1, 1)), "dilated convolutions"
      if not pad.any(): padding = 1                                                     # VALID
      elif st == 1 and kh == kw and kh % 2 == 1 and set(pad) == {kh // 2} and tall == 1: padding = 0   # SAME
      else: raise NotImplementedError(f"padding {q.padding} with stride {q.stride}" + (" in a tall image" if tall > 1 else ""))
      w = q.wq.numpy().reshape(O, C, kh, kw).transpose(0, 2, 3, 1)               # [O, kh, kw, C]
      shape = [1, (shape[1] - kh) // st + 1, (shape[2] - kw) // st + 1, O] if padding == 1 else [1, *q.shapes[1][1:], O]
    else:
      w, K, padding, st = q.wq.numpy(), q.wq.shape[1], 1, 1
      if len(shape) == 4:                                                             # flattened NCHW maps: weights to NHWC order
        C, wd = shape[3], shape[2]
        h = K // C // wd
        w = w.reshape(O, C, h, wd).transpose(0, 2, 3, 1)
        if tall == 1: w, shape = w.reshape(O, K), [1, O]
        else: shape = [1, shape[1] - h + 1, 1, O]                                      # a convolution over each image's map
      else: shape = [1, O]
    wt = m.tensor(f"w{i}", w.shape, np.uint8, q.ws, q.wz, data=w)
    o = m.tensor(f"y{i}", shape, np.uint8, q.ys, q.yz)
    if w.ndim == 4: m.op(tflite.BuiltinOperator.CONV_2D, [t, wt, bt], [o], tflite.BuiltinOptions.Conv2DOptions, conv_options(st, padding, act))
    else: m.op(tflite.BuiltinOperator.FULLY_CONNECTED, [t, wt, bt], [o], tflite.BuiltinOptions.FullyConnectedOptions, fc_options(act))
    t = o
    for name, (k, s) in POOLS.items():
      if how is not None and name in how.split(","):
        shape = [1, (shape[1] - k) // s + 1, (shape[2] - k) // s + 1, shape[3]]
        o = m.tensor(f"pool{i}", shape, np.uint8, q.ys, q.yz)
        m.op(tflite.BuiltinOperator.MAX_POOL_2D, [t], [o], tflite.BuiltinOptions.Pool2DOptions, _pool(k, s))
        t = o
  return m.build([inp], [t])
