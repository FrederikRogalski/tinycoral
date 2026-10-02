# tiny TFLite model writer (uint8/int8 quantized) used to drive edgetpu_compiler as an oracle
from __future__ import annotations
import numpy as np, flatbuffers, tflite

NP2TFL = {np.uint8: tflite.TensorType.UINT8, np.int8: tflite.TensorType.INT8, np.int32: tflite.TensorType.INT32,
          np.float32: tflite.TensorType.FLOAT32}

class Model:
  def __init__(self):
    self.tensors: list[dict] = []
    self.buffers: list[bytes] = [b""]
    self.opcodes: list[tuple[int, int]] = []
    self.ops: list[tuple] = []

  def tensor(self, name:str, shape, dtype, scale=None, zero_point=None, data:np.ndarray|None=None, qdim:int=0) -> int:
    buf = 0
    if data is not None:
      self.buffers.append(np.ascontiguousarray(data, dtype=dtype).tobytes())
      buf = len(self.buffers) - 1
    else:
      self.buffers.append(b"")
      buf = len(self.buffers) - 1
    q = None if scale is None else (np.atleast_1d(np.asarray(scale, np.float32)), np.atleast_1d(np.asarray(zero_point, np.int64)), qdim)
    self.tensors.append(dict(name=name, shape=list(shape), type=NP2TFL[np.dtype(dtype).type], buffer=buf, q=q))
    return len(self.tensors) - 1

  def op(self, builtin:int, inputs:list[int], outputs:list[int], options_type:int=0, options=None, version:int=1):
    if (builtin, version) not in self.opcodes: self.opcodes.append((builtin, version))
    self.ops.append((self.opcodes.index((builtin, version)), inputs, outputs, options_type, options))

  def build(self, inputs:list[int], outputs:list[int]) -> bytes:
    b = flatbuffers.Builder(1024)
    def ivec(start, xs, prepend="PrependInt32"):
      start(b, len(xs))
      for x in reversed(xs): getattr(b, prepend)(x)
      return b.EndVector()
    buffers = []
    for data in self.buffers:
      dv = b.CreateNumpyVector(np.frombuffer(data, np.uint8)) if data else None
      tflite.BufferStart(b)
      if dv is not None: tflite.BufferAddData(b, dv)
      buffers.append(tflite.BufferEnd(b))
    tensors = []
    for t in self.tensors:
      name = b.CreateString(t["name"])
      shape = ivec(tflite.TensorStartShapeVector, t["shape"])
      qo = None
      if t["q"] is not None:
        scale, zp, qdim = t["q"]
        sv = b.CreateNumpyVector(scale.astype(np.float32))
        zv = b.CreateNumpyVector(zp.astype(np.int64))
        tflite.QuantizationParametersStart(b)
        tflite.QuantizationParametersAddScale(b, sv)
        tflite.QuantizationParametersAddZeroPoint(b, zv)
        tflite.QuantizationParametersAddQuantizedDimension(b, qdim)
        qo = tflite.QuantizationParametersEnd(b)
      tflite.TensorStart(b)
      tflite.TensorAddShape(b, shape)
      tflite.TensorAddType(b, t["type"])
      tflite.TensorAddBuffer(b, t["buffer"])
      tflite.TensorAddName(b, name)
      if qo is not None: tflite.TensorAddQuantization(b, qo)
      tensors.append(tflite.TensorEnd(b))
    ops = []
    for opcode_idx, ins, outs, opt_type, opt_fn in self.ops:
      opt = opt_fn(b) if opt_fn is not None else None
      iv, ov = ivec(tflite.OperatorStartInputsVector, ins), ivec(tflite.OperatorStartOutputsVector, outs)
      tflite.OperatorStart(b)
      tflite.OperatorAddOpcodeIndex(b, opcode_idx)
      tflite.OperatorAddInputs(b, iv)
      tflite.OperatorAddOutputs(b, ov)
      if opt is not None:
        tflite.OperatorAddBuiltinOptionsType(b, opt_type)
        tflite.OperatorAddBuiltinOptions(b, opt)
      ops.append(tflite.OperatorEnd(b))
    tv = ivec(tflite.SubGraphStartTensorsVector, tensors, "PrependUOffsetTRelative")
    opv = ivec(tflite.SubGraphStartOperatorsVector, ops, "PrependUOffsetTRelative")
    inv, outv = ivec(tflite.SubGraphStartInputsVector, inputs), ivec(tflite.SubGraphStartOutputsVector, outputs)
    sgname = b.CreateString("main")
    tflite.SubGraphStart(b)
    tflite.SubGraphAddTensors(b, tv)
    tflite.SubGraphAddInputs(b, inv)
    tflite.SubGraphAddOutputs(b, outv)
    tflite.SubGraphAddOperators(b, opv)
    tflite.SubGraphAddName(b, sgname)
    sg = tflite.SubGraphEnd(b)
    codes = []
    for builtin, version in self.opcodes:
      tflite.OperatorCodeStart(b)
      tflite.OperatorCodeAddDeprecatedBuiltinCode(b, min(builtin, 127))
      tflite.OperatorCodeAddBuiltinCode(b, builtin)
      tflite.OperatorCodeAddVersion(b, version)
      codes.append(tflite.OperatorCodeEnd(b))
    cv = ivec(tflite.ModelStartOperatorCodesVector, codes, "PrependUOffsetTRelative")
    sgv = ivec(tflite.ModelStartSubgraphsVector, [sg], "PrependUOffsetTRelative")
    bv = ivec(tflite.ModelStartBuffersVector, buffers, "PrependUOffsetTRelative")
    desc = b.CreateString("tinycoral")
    tflite.ModelStart(b)
    tflite.ModelAddVersion(b, 3)
    tflite.ModelAddOperatorCodes(b, cv)
    tflite.ModelAddSubgraphs(b, sgv)
    tflite.ModelAddDescription(b, desc)
    tflite.ModelAddBuffers(b, bv)
    b.Finish(tflite.ModelEnd(b), file_identifier=b"TFL3")
    return bytes(b.Output())

def fc_options(act:int=0):
  def fn(b):
    tflite.FullyConnectedOptionsStart(b)
    tflite.FullyConnectedOptionsAddFusedActivationFunction(b, act)
    return tflite.FullyConnectedOptionsEnd(b)
  return fn

def fc_model(w:np.ndarray, bias:np.ndarray|None=None, in_q=(1/128, 128), w_q=(1/128, 128), out_q=(1/16, 128), batch:int=1, act:int=0) -> bytes:
  """uint8 FULLY_CONNECTED: y[batch, N] = x[batch, K] @ w[N, K].T + bias"""
  N, K = w.shape
  m = Model()
  x = m.tensor("x", [batch, K], np.uint8, *in_q)
  wt = m.tensor("w", [N, K], np.uint8, *w_q, data=w.astype(np.uint8))
  bias = np.zeros(N, np.int32) if bias is None else bias
  bt = m.tensor("b", [N], np.int32, in_q[0]*w_q[0], 0, data=bias.astype(np.int32))
  y = m.tensor("y", [batch, N], np.uint8, *out_q)
  m.op(tflite.BuiltinOperator.FULLY_CONNECTED, [x, wt, bt], [y], tflite.BuiltinOptions.FullyConnectedOptions, fc_options(act))
  return m.build([x], [y])

def conv_options(stride:int=1, padding:int=1, act:int=0):  # padding: 0 SAME, 1 VALID
  def fn(b):
    tflite.Conv2DOptionsStart(b)
    tflite.Conv2DOptionsAddPadding(b, padding)
    tflite.Conv2DOptionsAddStrideW(b, stride)
    tflite.Conv2DOptionsAddStrideH(b, stride)
    tflite.Conv2DOptionsAddFusedActivationFunction(b, act)
    tflite.Conv2DOptionsAddDilationWFactor(b, 1)
    tflite.Conv2DOptionsAddDilationHFactor(b, 1)
    return tflite.Conv2DOptionsEnd(b)
  return fn

def conv_model(w:np.ndarray, bias:np.ndarray|None=None, H:int=1, W:int=1, in_q=(1/128, 128), w_q=(1/128, 128), out_q=(1/16, 128),
               stride:int=1, padding:int=1, act:int=0) -> bytes:
  """uint8 CONV_2D: w is [Cout, kh, kw, Cin], input [1, H, W, Cin] -> output [1, Ho, Wo, Cout]"""
  Cout, kh, kw, Cin = w.shape
  Ho, Wo = ((H - kh) // stride + 1, (W - kw) // stride + 1) if padding == 1 else (-(-H // stride), -(-W // stride))
  m = Model()
  x = m.tensor("x", [1, H, W, Cin], np.uint8, *in_q)
  wt = m.tensor("w", [Cout, kh, kw, Cin], np.uint8, *w_q, data=w.astype(np.uint8))
  bias = np.zeros(Cout, np.int32) if bias is None else bias
  bt = m.tensor("b", [Cout], np.int32, in_q[0]*w_q[0], 0, data=bias.astype(np.int32))
  y = m.tensor("y", [1, Ho, Wo, Cout], np.uint8, *out_q)
  m.op(tflite.BuiltinOperator.CONV_2D, [x, wt, bt], [y], tflite.BuiltinOptions.Conv2DOptions, conv_options(stride, padding, act))
  return m.build([x], [y])
