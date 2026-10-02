# edgetpu_compiler as an oracle for coral.fused blocks: the TFLite models our code generator must match, compiled alone or
# together (co-compiled, which is how the compiler keeps several models resident). Offline only.
from __future__ import annotations
import numpy as np, tflite
from tools.tflite_gen import Model, conv_options, conv_model, fc_model, fc_options
from coral.codegen.conv import GRIDS

def _mul_options(b):
  tflite.MulOptionsStart(b)
  tflite.MulOptionsAddFusedActivationFunction(b, 0)
  return tflite.MulOptionsEnd(b)
def _pool_options(k:int):
  def fn(b):
    tflite.Pool2DOptionsStart(b)
    tflite.Pool2DOptionsAddPadding(b, 1)
    tflite.Pool2DOptionsAddStrideW(b, k)
    tflite.Pool2DOptionsAddStrideH(b, k)
    tflite.Pool2DOptionsAddFilterWidth(b, k)
    tflite.Pool2DOptionsAddFilterHeight(b, k)
    tflite.Pool2DOptionsAddFusedActivationFunction(b, 0)
    return tflite.Pool2DOptionsEnd(b)
  return fn

def build(b, Mp:int) -> bytes:
  """the TFLite model of block b (coral.fused.Block) for Mp rows: conv (Mp=1: FULLY_CONNECTED), the FFN chain, or the
  turned-around classifier with its max pool"""
  q, wt = b.quant, b.weights
  if b.kind == "conv":
    if Mp == 1: return fc_model(wt["w"], np.zeros(b.N, np.int32), in_q=q["x"], w_q=q["w"], out_q=q["y"])
    H, W = GRIDS[Mp]
    return conv_model(wt["w"][:, None, None, :], np.zeros(b.N, np.int32), H=H, W=W, in_q=q["x"], w_q=q["w"], out_q=q["y"])
  m = Model()
  if b.kind == "argmax":
    from coral.fused import VOCAB_GRID, POOL
    (Hv, Wg), K = VOCAB_GRID, b.K
    V = m.tensor("vocab", [1, Hv, Wg, K], np.uint8, *q["vocab"])
    x = m.tensor("tokens", [Mp, 1, 1, K], np.uint8, *q["tokens"], data=np.random.default_rng(0).integers(0, 256, (Mp, 1, 1, K), dtype=np.uint8))
    bt = m.tensor("b", [Mp], np.int32, q["vocab"][0] * q["tokens"][0], 0, data=np.zeros(Mp, np.int32))
    y = m.tensor("logits", [1, Hv, Wg, Mp], np.uint8, *q["logits"])
    m.op(tflite.BuiltinOperator.CONV_2D, [V, x, bt], [y], tflite.BuiltinOptions.Conv2DOptions, conv_options(1, 1, 0))
    z = m.tensor("blockmax", [1, Hv // POOL, Wg // POOL, Mp], np.uint8, *q["logits"])
    m.op(tflite.BuiltinOperator.MAX_POOL_2D, [y], [z], tflite.BuiltinOptions.Pool2DOptions, _pool_options(POOL))
    return m.build([V], [z])
  D, Hd = b.K, wt["w1"].shape[0]
  shape = (lambda c: [1, c]) if Mp == 1 else (lambda c: [1, *GRIDS[Mp], c])
  t = {k: m.tensor(k, shape(D if k in ("x", "y") else Hd), np.uint8, *q[k]) for k in ("x", "h1", "h3", "g", "a", "m", "y")}
  def mm(x, k, xq, out):
    ws, wz = q[k]
    w = m.tensor(k, [wt[k].shape[0], wt[k].shape[1]] if Mp == 1 else [wt[k].shape[0], 1, 1, wt[k].shape[1]], np.uint8, ws, wz,
                 data=wt[k] if Mp == 1 else wt[k][:, None, None, :])
    bb = m.tensor(k + ".b", [wt[k].shape[0]], np.int32, xq[0] * ws, 0, data=np.zeros(wt[k].shape[0], np.int32))
    if Mp == 1: m.op(tflite.BuiltinOperator.FULLY_CONNECTED, [x, w, bb], [out], tflite.BuiltinOptions.FullyConnectedOptions, fc_options(0))
    else: m.op(tflite.BuiltinOperator.CONV_2D, [x, w, bb], [out], tflite.BuiltinOptions.Conv2DOptions, conv_options(1, 1, 0))
  mm(t["x"], "w1", q["x"], t["h1"])
  mm(t["x"], "w3", q["x"], t["h3"])
  m.op(tflite.BuiltinOperator.LOGISTIC, [t["h1"]], [t["g"]])
  m.op(tflite.BuiltinOperator.MUL, [t["h1"], t["g"]], [t["a"]], tflite.BuiltinOptions.MulOptions, _mul_options)
  m.op(tflite.BuiltinOperator.MUL, [t["a"], t["h3"]], [t["m"]], tflite.BuiltinOptions.MulOptions, _mul_options)
  mm(t["m"], "w2", q["m"], t["y"])
  return m.build([t["x"]], [t["y"]])

def compiled(blocks, Mp:int) -> dict[str, dict]:
  """co-compile the blocks for Mp rows in one edgetpu_compiler call -> {name: {executable type: executable}}"""
  from tools.reloc import cocompile
  blocks = [b for b in blocks if Mp > 1 or b.kind != "argmax"]
  return {b.name: e for b, e in zip(blocks, cocompile([build(b, Mp) for b in blocks]))}

# *** single matmuls: the compiler's 1x1-conv programs and the co-compiled TinyStories sets (oracles for coral/codegen/conv.py) ***
REF = dict(in_q=(1/32, 128), w_q=(1/64, 128), out_q=(1/4, 128))
LLM_LAYERS = [(864, 288), (288, 288), (1536, 288), (288, 768)] * 6

def compile_conv(Mp:int, N:int, K:int) -> dict:
  from tools.compiler import compile_tflite
  H, W = GRIDS[Mp]
  return {e.type: e for e in compile_tflite(conv_model(np.full((N, 1, 1, K), 128, np.uint8), np.zeros(N, np.int32), H=H, W=W, **REF))[0]}

def coset(Mp:int) -> list[tuple[tuple[int, int], dict]]:
  """the 24 TinyStories matmuls (plus the streamed classifier) co-compiled for Mp positions -> [((N, K), exes)]"""
  from tools.reloc import cocompile
  shapes = LLM_LAYERS + ([] if Mp == 1 else [(32000 if Mp <= 64 else 16000 if Mp == 128 else 8000, 288)])
  def rng(i): return np.random.default_rng(100 + i)   # distinct weights, so nothing gets deduplicated
  if Mp == 1: models = [fc_model(rng(i).integers(0, 256, (N, K), dtype=np.uint8), np.zeros(N, np.int32), **REF) for i, (N, K) in enumerate(shapes)]
  else: models = [conv_model(rng(i).integers(0, 256, (N, 1, 1, K), dtype=np.uint8), np.zeros(N, np.int32), H=GRIDS[Mp][0], W=GRIDS[Mp][1], **REF)
                  for i, (N, K) in enumerate(shapes)]
  return list(zip(shapes, cocompile(models)))
