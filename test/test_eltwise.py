# Corpus test for coral/isa/eltwise.py: rebuild every LOGISTIC / RELU-type / MUL / ADD / MAX_POOL_2D op instruction (and the
# 0x19 NLU loads and data-flow instructions around them) of 198 programs compiled by edgetpu_compiler, byte for byte.
#
#   .venv/bin/python test/test_eltwise.py           compile (docker `etpc`, cached in .compile/) and check
#   .venv/bin/python test/test_eltwise.py --quick   without the three pooled-conv programs (33k words, ~10 s compile each)
#
# Offline only: never touches the USB device.
#
# Two checks per instruction:
#   "model":     geometry from the TFLite tensor shapes (compiler tile split rules), quantization fields from the TFLite
#                scales/zero points (float32 formulas of eltwise.md), allocation from the instruction (tile mask, seq, narrow
#                bases, wide addresses, ADD operand distance). The narrow layout strides (R) follow the two placement rules
#                (global: tile t's slice at base + 64*P*t with stride ceil(n/4); tile-local: every tile at the same address
#                with stride = its own chunk, as written by an FC op broadcast to several tiles); anything else would be
#                counted separately ("layout from insn").
#   "roundtrip": every parameter read back from the instruction (op_params below), rebuilt with the same builders.
from __future__ import annotations
import sys, pathlib, struct, argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, tflite
from tools.tflite_gen import Model, conv_options, fc_options
from tools.compiler import compile_tflite

from coral.isa import cdiv, split, bits_f32, op as OP, wide_narrow as WN, ring_mesh as RM, eltwise as EL
B = tflite.BuiltinOperator

# ============================== models ==============================
def _opt(start, end, *adds):
  def fn(b):
    start(b)
    for f, v in adds: f(b, v)
    return end(b)
  return fn
def _mul_opt(act=0): return _opt(tflite.MulOptionsStart, tflite.MulOptionsEnd, (tflite.MulOptionsAddFusedActivationFunction, act))
def _add_opt(act=0): return _opt(tflite.AddOptionsStart, tflite.AddOptionsEnd, (tflite.AddOptionsAddFusedActivationFunction, act))
def _pool_opt(kh, kw, sh, sw, padding=1):
  def fn(b):
    tflite.Pool2DOptionsStart(b)
    tflite.Pool2DOptionsAddPadding(b, padding)
    tflite.Pool2DOptionsAddStrideW(b, sw)
    tflite.Pool2DOptionsAddStrideH(b, sh)
    tflite.Pool2DOptionsAddFilterWidth(b, kw)
    tflite.Pool2DOptionsAddFilterHeight(b, kh)
    tflite.Pool2DOptionsAddFusedActivationFunction(b, 0)
    return tflite.Pool2DOptionsEnd(b)
  return fn

def unary(op, shape, in_q=(1/16, 128), out_q=(1/16, 128)):
  m = Model()
  x = m.tensor("x", shape, np.uint8, *in_q)
  y = m.tensor("y", shape, np.uint8, *out_q)
  m.op(op, [x], [y])
  return m.build([x], [y])

def logistic(shape, in_q=(1/16, 128), out_q=(1/256, 0)): return unary(B.LOGISTIC, shape, in_q, out_q)

def binary(kind, shape, a_q=(1/16, 128), b_q=(1/32, 100), out_q=(1/8, 120), act=0):
  m = Model()
  a = m.tensor("a", shape, np.uint8, *a_q)
  b = m.tensor("b", shape, np.uint8, *b_q)
  y = m.tensor("y", shape, np.uint8, *out_q)
  if kind == "mul": m.op(B.MUL, [a, b], [y], tflite.BuiltinOptions.MulOptions, _mul_opt(act))
  else: m.op(B.ADD, [a, b], [y], tflite.BuiltinOptions.AddOptions, _add_opt(act))
  return m.build([a, b], [y])

def maxpool(shape, k=2, s=2, q=(1/16, 128)):
  N, H, W, C = shape
  m = Model()
  x = m.tensor("x", shape, np.uint8, *q)
  y = m.tensor("y", [N, (H - k) // s + 1, (W - k) // s + 1, C], np.uint8, *q)
  m.op(B.MAX_POOL_2D, [x], [y], tflite.BuiltinOptions.Pool2DOptions, _pool_opt(k, k, s, s))
  return m.build([x], [y])

def ffn(H, W, C=288, F=768, form="conv", seed=0):
  """the FFN block: h1 = x W1, h3 = x W3, g = LOGISTIC(h1), a = h1*g, m = a*h3, y = m W2 (CONV_2D 1x1 or FULLY_CONNECTED)"""
  rng = np.random.default_rng(seed)
  q = dict(x=(1/32, 128), w=(1/64, 128), h=(1/8, 128), a=(1/16, 20), m=(1/16, 128), y=(1/4, 128))
  m = Model()
  xs, hs = ([1, H, W, C], [1, H, W, F]) if form == "conv" else ([1, C], [1, F])
  x = m.tensor("x", xs, np.uint8, *q["x"])
  def lin(name, inp, n_out, n_in, out_shape, out_q):
    wsh = [n_out, 1, 1, n_in] if form == "conv" else [n_out, n_in]
    w = m.tensor(f"{name}.w", wsh, np.uint8, *q["w"], data=rng.integers(0, 256, wsh).astype(np.uint8))
    b = m.tensor(f"{name}.b", [n_out], np.int32, float(m.tensors[inp]["q"][0][0]) * q["w"][0], 0,
                 data=rng.integers(-500, 500, n_out).astype(np.int32))
    y = m.tensor(f"{name}.y", out_shape, np.uint8, *out_q)
    if form == "conv": m.op(B.CONV_2D, [inp, w, b], [y], tflite.BuiltinOptions.Conv2DOptions, conv_options(1, 1, 0))
    else: m.op(B.FULLY_CONNECTED, [inp, w, b], [y], tflite.BuiltinOptions.FullyConnectedOptions, fc_options(0))
    return y
  h1 = lin("w1", x, F, C, hs, q["h"])
  h3 = lin("w3", x, F, C, hs, q["h"])
  g = m.tensor("g", hs, np.uint8, 1/256, 0)
  m.op(B.LOGISTIC, [h1], [g])
  a = m.tensor("a", hs, np.uint8, *q["a"])
  m.op(B.MUL, [h1, g], [a], tflite.BuiltinOptions.MulOptions, _mul_opt())
  mm = m.tensor("m", hs, np.uint8, *q["m"])
  m.op(B.MUL, [a, h3], [mm], tflite.BuiltinOptions.MulOptions, _mul_opt())
  y = lin("w2", mm, C, F, xs, q["y"])
  return m.build([x], [y])

def pooled_conv(M, H=160, W=200, C=288, k=8, seed=0):
  """CONV_2D [1,160,200,288] x [M,1,1,288] then MAX_POOL_2D 8x8/8 VALID -> [1,20,25,M]"""
  rng = np.random.default_rng(seed)
  m = Model()
  x = m.tensor("x", [1, H, W, C], np.uint8, 1/64, 128)
  w = m.tensor("w", [M, 1, 1, C], np.uint8, 1/32, 128, data=rng.integers(0, 256, (M, 1, 1, C)).astype(np.uint8))
  b = m.tensor("b", [M], np.int32, 1/64/32, 0, data=rng.integers(-500, 500, M).astype(np.int32))
  y = m.tensor("y", [1, H, W, M], np.uint8, 1/4, 128)
  m.op(B.CONV_2D, [x, w, b], [y], tflite.BuiltinOptions.Conv2DOptions, conv_options(1, 1, 0))
  z = m.tensor("z", [1, H // k, W // k, M], np.uint8, 1/4, 128)
  m.op(B.MAX_POOL_2D, [y], [z], tflite.BuiltinOptions.Pool2DOptions, _pool_opt(k, k, k, k))
  return m.build([x], [z])

def fc_chain(K, Ns, ops=(), seed=0):
  """FC(K->N0) -> FC(N0->N1) -> ... -> optional LOGISTIC / RELU / x*x"""
  rng = np.random.default_rng(seed)
  m = Model()
  cur = m.tensor("x", [1, K], np.uint8, 1/32, 128)
  ck, cs = K, 1/32
  for i, N in enumerate(Ns):
    w = m.tensor(f"w{i}", [N, ck], np.uint8, 1/64, 128, data=rng.integers(0, 256, (N, ck)).astype(np.uint8))
    b = m.tensor(f"b{i}", [N], np.int32, cs / 64, 0, data=rng.integers(-500, 500, N).astype(np.int32))
    y = m.tensor(f"y{i}", [1, N], np.uint8, 1/8, 128)
    m.op(B.FULLY_CONNECTED, [cur, w, b], [y], tflite.BuiltinOptions.FullyConnectedOptions, fc_options(0))
    cur, ck, cs = y, N, 1/8
  for j, o in enumerate(ops):
    if o == "logistic":
      y = m.tensor(f"l{j}", [1, ck], np.uint8, 1/256, 0)
      m.op(B.LOGISTIC, [cur], [y])
    elif o == "relu":
      y = m.tensor(f"r{j}", [1, ck], np.uint8, cs, 128)
      m.op(B.RELU, [cur], [y])
    elif o == "square":
      y = m.tensor(f"s{j}", [1, ck], np.uint8, 1/4, 10)
      m.op(B.MUL, [cur, cur], [y], tflite.BuiltinOptions.MulOptions, _mul_opt())
    cur = y
  return m.build([0], [cur])

def fc_pair(K, N, kind="mul", seed=0):
  """two FCs of the same x, then MUL or ADD of the two"""
  rng = np.random.default_rng(seed)
  m = Model()
  x = m.tensor("x", [1, K], np.uint8, 1/32, 128)
  outs = []
  for i in range(2):
    w = m.tensor(f"w{i}", [N, K], np.uint8, 1/64, 128, data=rng.integers(0, 256, (N, K)).astype(np.uint8))
    b = m.tensor(f"b{i}", [N], np.int32, 1/32/64, 0, data=rng.integers(-500, 500, N).astype(np.int32))
    y = m.tensor(f"y{i}", [1, N], np.uint8, 1/8 if i == 0 else 1/16, 128 if i == 0 else 100)
    m.op(B.FULLY_CONNECTED, [x, w, b], [y], tflite.BuiltinOptions.FullyConnectedOptions, fc_options(0))
    outs.append(y)
  z = m.tensor("z", [1, N], np.uint8, 1/4, 30)
  if kind == "add": m.op(B.ADD, outs, [z], tflite.BuiltinOptions.AddOptions, _add_opt())
  else: m.op(B.MUL, outs, [z], tflite.BuiltinOptions.MulOptions, _mul_opt())
  return m.build([x], [z])

def jobs(quick:bool=False) -> list[tuple[str, callable]]:
  J = []
  # LOGISTIC: 1-D sizes, 4-D shapes (dense, odd channels, uneven tile split), quantizations
  for n in [4, 16, 64, 100, 256, 768, 1000, 1024, 2048, 4096, 8192, 16384]: J.append((f"logistic_n{n}", lambda n=n: logistic([1, n])))
  for iq, oz in [((0.1, 50), 0), ((0.05, 0), 10), ((0.3, 255), 0)]:
    for n in [64, 1000]: J.append((f"logistic_n{n}_{iq}_{oz}", lambda n=n, iq=iq, oz=oz: logistic([1, n], iq, (1/256, oz))))
  for sh in [[1,4,4,16], [1,8,8,16], [1,4,4,288], [1,4,4,768], [1,16,8,768], [1,7,7,5], [1,2,2,64], [1,8,8,64], [1,1,1,1000], [1,4,4,5], [1,8,8,4]]:
    J.append((f"logistic_{'x'.join(map(str, sh))}", lambda sh=sh: logistic(sh)))
  # RELU-type clamps
  for name, op in [("relu", B.RELU), ("relu6", B.RELU6), ("relun1", B.RELU_N1_TO_1)]:
    for sh in [[1, 64], [1, 1000], [1, 4, 4, 16], [1, 7, 7, 5]]:
      for iq, oq in [((1/16, 128), (1/16, 128)), ((0.05, 100), (0.03, 7)), ((0.1, 0), (0.02, 250))]:
        J.append((f"{name}_{'x'.join(map(str, sh))}_{iq}_{oq}", lambda op=op, sh=sh, iq=iq, oq=oq: unary(op, sh, iq, oq)))
  # MUL: per-tile sizes 1..64 words (FIFO depth 1..4, partial last rounds), 4-D, quantizations and fused activations
  for n in [1, 4, 8, 12, 16, 20, 28, 36, 40, 44, 64, 100, 256, 1000, 1024, 4096]: J.append((f"mul_n{n}", lambda n=n: binary("mul", [1, n])))
  for qa, qb, qo in [((0.1, 3), (0.2, 250), (0.7, 7)), ((1/16, 0), (1/16, 0), (1/256, 0)), ((0.02, 128), (0.03, 128), (0.0001, 128)),
                     ((0.05, 100), (0.07, 90), (0.03, 7))]:
    for act in [0, 1, 2, 3]:
      J.append((f"mul_n64_{qa}{qb}{qo}_act{act}", lambda qa=qa, qb=qb, qo=qo, act=act: binary("mul", [1, 64], qa, qb, qo, act)))
  for sh in [[1,4,4,16], [1,4,4,768], [1,16,8,768], [1,8,8,64], [1,7,7,5], [1,2,2,64], [1,4,4,36], [1,8,8,20], [1,8,8,44]]:
    J.append((f"mul_{'x'.join(map(str, sh))}", lambda sh=sh: binary("mul", sh)))
  # ADD: sizes, 4-D, weight-ratio sweep (integer weights up to 32767), quantizations and activations
  for n in [16, 64, 100, 256, 1000, 1024, 4096]: J.append((f"add_n{n}", lambda n=n: binary("add", [1, n])))
  import math
  for r in [0.3, 0.37, 1/3, 1/7, 0.123, 0.999, 2.5, 3.7, 10.1, 100, 300, 1000, math.pi / 10, 0.0039, 1.001, 0.6180339, 7/13, 0.9]:
    bs = struct.unpack("<f", struct.pack("<f", r / 16))[0]
    J.append((f"add_ratio{r:.6g}", lambda bs=bs: binary("add", [1, 64], (1/16, 128), (bs, 100), (1/8, 120))))
  for qa, qo in [((0.05, 100), (0.03, 7)), ((0.013, 3), (0.017, 200)), ((0.3, 30), (0.07, 60))]:
    for act in [0, 1, 2, 3]: J.append((f"add_n64_{qa}{qo}_act{act}", lambda qa=qa, qo=qo, act=act: binary("add", [1, 64], qa, (0.07, 90), qo, act)))
  for sh in [[1,4,4,16], [1,8,8,16], [1,4,4,768], [1,16,8,768], [1,7,7,5]]:
    J.append((f"add_{'x'.join(map(str, sh))}", lambda sh=sh: binary("add", sh)))
  # MAX_POOL_2D: windows 2..8, strides, channel groups (C > 64), zero points
  for sh, k, s in [([1,8,8,16],2,2), ([1,8,8,16],4,4), ([1,16,16,16],2,2), ([1,8,8,64],2,2), ([1,16,16,32],8,8), ([1,4,4,16],2,2),
                   ([1,8,8,16],3,1), ([1,8,8,16],2,1), ([1,8,8,3],2,2), ([1,32,32,16],8,8), ([1,16,16,64],4,4), ([1,8,8,32],8,8),
                   ([1,16,16,16],8,8), ([1,12,12,16],3,3), ([1,20,20,16],5,5), ([1,24,24,16],6,6), ([1,28,28,16],7,7), ([1,64,64,16],8,8),
                   ([1,16,16,128],4,4), ([1,16,16,256],4,4), ([1,16,16,100],4,4), ([1,16,16,72],4,4), ([1,16,16,16],3,2), ([1,16,16,16],5,3)]:
    J.append((f"maxpool_{'x'.join(map(str, sh))}_k{k}s{s}", lambda sh=sh, k=k, s=s: maxpool(sh, k, s)))
  for q in [(0.1, 100), (0.03, 0), (0.0123, 255)]: J.append((f"maxpool_8x8x16_q{q}", lambda q=q: maxpool([1, 8, 8, 16], 2, 2, q)))
  # chains (data flow between ops)
  for K, Ns, ops in [(288, [768], ("logistic",)), (64, [64], ("logistic",)), (256, [256, 128], ()), (128, [128], ("relu",)),
                     (128, [100], ("logistic",)), (256, [1000], ("logistic",)), (64, [64], ("logistic", "square"))]:
    J.append((f"chain_fc{K}_{'_'.join(map(str, Ns))}_{'_'.join(ops)}", lambda K=K, Ns=Ns, ops=ops: fc_chain(K, Ns, ops)))
  for K, N in [(64, 64), (288, 768), (128, 100)]:
    for kind in ("mul", "add"): J.append((f"fc{kind}_{K}_{N}", lambda K=K, N=N, kind=kind: fc_pair(K, N, kind)))
  # the target models
  for H, W in [(4, 4), (16, 8), (16, 16), (2, 2), (8, 8)]: J.append((f"ffn_conv_{H}x{W}", lambda H=H, W=W: ffn(H, W)))
  J.append(("ffn_conv_4x4_C64_F128", lambda: ffn(4, 4, C=64, F=128)))
  J.append(("ffn_fc", lambda: ffn(1, 1, form="fc")))
  if not quick: J += [(f"pooled_conv_M{M}", lambda M=M: pooled_conv(M)) for M in (16, 64, 256)]
  return J

# ============================== TFLite introspection ==============================
ACT_OF = {B.RELU: 1, B.RELU_N1_TO_1: 2, B.RELU6: 3}
def tflite_ops(buf:bytes) -> list[dict]:
  mdl = tflite.Model.GetRootAsModel(buf, 0)
  sg = mdl.Subgraphs(0)
  def tq(i):
    t = sg.Tensors(i)
    q = t.Quantization()
    return dict(shape=[int(v) for v in t.ShapeAsNumpy()],
                q=(float(q.ScaleAsNumpy()[0]), int(q.ZeroPointAsNumpy()[0])) if q and q.ScaleLength() else None)
  out = []
  for i in range(sg.OperatorsLength()):
    o = sg.Operators(i)
    code = mdl.OperatorCodes(o.OpcodeIndex()).BuiltinCode()
    d = dict(code=code, ins=[tq(o.Inputs(j)) for j in range(o.InputsLength())], out=tq(o.Outputs(0)), act=ACT_OF.get(code, 0))
    opt = o.BuiltinOptions()
    if code in (B.MUL, B.ADD) and opt is not None:
      t = (tflite.MulOptions if code == B.MUL else tflite.AddOptions)()
      t.Init(opt.Bytes, opt.Pos)
      d["act"] = t.FusedActivationFunction()
    if code == B.MAX_POOL_2D:
      t = tflite.Pool2DOptions()
      t.Init(opt.Bytes, opt.Pos)
      d["pool"] = (t.FilterHeight(), t.FilterWidth(), t.StrideH(), t.StrideW())
    out.append(d)
  return out

hwc = EL.hwc

def geometry(shape:list[int], tile_mask:int) -> dict:
  """per-tile block of a tensor from the compiler's placement rules (eltwise.tile_geometry), global and tile-local layouts"""
  t = (tile_mask & -tile_mask).bit_length() - 1
  g = EL.tile_geometry(shape, t)
  g["Rp"] = g["R"][0]
  g["Rloc"] = EL.tile_geometry(shape, t, local=True)["R"]
  return g

# ============================== expected instructions from the model ==============================
def model_candidates(kind:str, ops:list[dict], words:list[int], pooled:bool) -> list[tuple[str, list[int]]]:
  """all (label, words) the builders produce for this instruction from the TFLite ops of the matching type; label 'rule' =
  layout strides from the placement rules, 'layout' = strides read from the instruction (chained producers)"""
  _, a = op_params(words)          # allocation (and, for 'layout', the strides) read back from the instruction
  out = []
  if kind == "nlu":
    if any(o["code"] == B.LOGISTIC for o in ops): out.append(("rule", EL.encode_nlu(**EL.nlu_fields(a["tile_mask"], a["seq"]))))
    return out
  for o in ops:
    if kind in ("logistic", "requant") and o["code"] in ((B.LOGISTIC,) if kind == "logistic" else (B.RELU, B.RELU6, B.RELU_N1_TO_1)):
      g = geometry(o["ins"][0]["shape"], a["tile_mask"])
      for lab, Rin, Rout in (("rule", g["Rp"], g["Rp"]), ("rule", g["Rloc"][0], g["Rp"]), ("layout", a["R_in"], a["R_out"])):
        common = dict(P=g["P"], Cw=g["w"], R_in=Rin, R_out=Rout, tile_mask=a["tile_mask"], seq=a["seq"], in_base=a["in_base"],
                      out_base=a["out_base"], ident_addr=a["ident_addr"], flat=g["flat"])
        if kind == "logistic":
          f = EL.logistic_op_fields(**common, **EL.logistic_quant(o["ins"][0]["q"], o["out"]["q"][1]))
        else:
          q = EL.requant_quant(o["ins"][0]["q"], o["out"]["q"], o["act"])
          f = EL.requant_op_fields(**common, **q)
        out.append((lab, OP.encode_op(**f)))
    elif kind in ("mul1", "mul2") and o["code"] == B.MUL:
      g = geometry(o["out"]["shape"], a["tile_mask"])
      for x, y in ((0, 1), (1, 0)):                                      # operand order is an allocation choice
        q = EL.mul_quant(o["ins"][x]["q"], o["ins"][y]["q"], o["out"]["q"], o["act"])
        for lab, R in (("rule", g["R"]), ("rule", g["Rloc"]), ("layout", a.get("R", a.get("R_out")))):
          if kind == "mul1":
            f = EL.mul_op1_fields(g["w"], g["cols"], g["rows"], R, a["tile_mask"], a["seq"], a["in_base"], a["out_base"], a["fifo_addr"], **q)
          else:
            f = EL.mul_op2_fields(g["w"], g["cols"], g["rows"], R, a["tile_mask"], a["seq"], a["in_base"], a["out_base"], a["ident_addr"])
          out.append((lab, OP.encode_op(**f)))
    elif kind == "add" and o["code"] == B.ADD:
      g = geometry(o["out"]["shape"], a["tile_mask"])
      for x, y in ((0, 1), (1, 0)):
        q = EL.add_quant(o["ins"][x]["q"], o["ins"][y]["q"], o["out"]["q"], o["act"])
        q.pop("w1")
        q.pop("w2")
        for lab, Rin, Rout in (("rule", g["R"], g["R"]), ("rule", g["Rloc"], g["R"]), ("layout", a["R_in"], a["R_out"])):
          f = EL.add_op_fields(g["w"], g["cols"], g["rows"], Rin, a["delta"], Rout, a["tile_mask"], a["seq"], a["in_base"], a["out_base"],
                               a["wmat_addr"], **q)
          out.append((lab, OP.encode_op(**f)))
    elif kind == "maxpool" and o["code"] == B.MAX_POOL_2D:
      kh, kw, sh, sw = o["pool"]
      H, W, C = hwc(o["out"]["shape"])
      t = (a["tile_mask"] & -a["tile_mask"]).bit_length() - 1
      OH, OW = (1, 1) if pooled else EL.tile_block(H, W, t)            # the streamed pooled-conv schedule: 1 output per op
      Wo = cdiv(C, 4)
      q = EL.maxpool_quant(*o["out"]["q"])
      f = EL.maxpool_op_fields(kh, kw, sh, sw, OH, OW, C, (Wo, OW * Wo), a["tile_mask"], a["seq"], a["out_base"], a["src_addr"], **q)
      out.append(("rule", OP.encode_op(**f)))
  return out

def check_dataflow(ins:list[tuple[int, list[int]]], ops:list[dict], st:Counter, pooled:bool=False):
  """identity prologue, MUL FIFO feed, ADD weight fills + transfer, MAX_POOL relay, against the builders"""
  for k, (_, ws) in enumerate(ins):
    if (ws[0] >> 6) & 0x3F != 0x13: continue
    d = WN.decode_narrow_to_wide(ws)
    if d["sync_id"] == 5 and k >= 4:
      f0 = RM.decode_mesh(ins[k - 4][1])
      st[("ident_prologue", EL.ident_prologue(f0["seq"], d["tile_mask"], f0["i_addr"], d["wide_addr"]) == [w for _, w in ins[k - 4:k + 1]])] += 1
    elif d["sync_id"] == 1:
      mul = [o for o in ops if o["code"] == B.MUL]
      ok = False
      for o in mul:
        g = geometry(o["out"]["shape"], d["tile_mask"])
        f = EL.mul_feed_n2w_fields(d["seq"], d["tile_mask"], d["narrow_addr"] * 4, d["wide_addr"], g["w"], g["cols"], g["rows"])
        ok |= WN.encode_narrow_to_wide(**f) == ws
      st[("mul_fifo_feed", ok)] += 1
    elif d["sync_id"] == 3 and k >= 10:
      f0 = RM.decode_mesh(ins[k - 10][1])
      ok = False
      for o in [o for o in ops if o["code"] == B.ADD]:
        for x, y in ((0, 1), (1, 0)):
          w1, w2 = EL.add_weights(o["ins"][x]["q"][0], o["ins"][y]["q"][0])
          got = EL.add_weight_fills(f0["seq"], d["tile_mask"], f0["i_addr"], w1, w2)
          got.append(WN.encode_narrow_to_wide(**EL.add_weight_n2w_fields(d["seq"], d["tile_mask"], d["narrow_addr"] * 4, d["wide_addr"])))
          ok |= got == [w for _, w in ins[k - 10:k + 1]]
      st[("add_weights", ok)] += 1
    elif d["tail_f799"] and not pooled:      # standalone MAX_POOL relay (the pooled conv gathers its windows over the mesh first)
      o = next(o for o in ops if o["code"] == B.MAX_POOL_2D)
      kh, kw, sh, sw = o["pool"]
      H, W, C = hwc(o["out"]["shape"])
      t = (d["tile_mask"] & -d["tile_mask"]).bit_length() - 1
      OH, OW = EL.tile_block(H, W, t)
      Hb, Wb = (OH - 1) * sh + kh, (OW - 1) * sw + kw
      S, adv = [], 0                                                       # source block layout (written by copy/gather ops)
      for j in range(6):
        S.append(d[f"n_inc{j}"] + adv)
        adv += d[f"n_lim{j}"] * S[-1]
      levels = d["narrow_lvl_mask"].bit_length()
      Rr = S[levels - 1 - (C > 64)] if Hb > 1 else 0
      f = EL.maxpool_relay_n2w_fields(d["seq"], d["tile_mask"], d["narrow_addr"] * 4, d["wide_addr"], S[0], Rr, Wb, Hb, C)
      st[("maxpool_relay", WN.encode_narrow_to_wide(**f) == ws)] += 1

# ============================== reading op instructions back: the inverse of the builders ==============================
def ttu_strides(d:dict, p:str) -> list[int]:
  """logical strides of op TTU p in address units: increment_k = 4*inc_k + the 2 low bits in {p}_hmode (k = 0) / {p}_mode{k-1},
  S_k = increment_k + sum_{e<k} cnt_e * S_e"""
  lo2, S, adv = [d[f"{p}_hmode"]] + [d[f"{p}_mode{k}"] for k in range(7)], [], 0
  for k in range(8):
    S.append(4 * d[f"{p}_inc{k}"] + lo2[k] + adv)
    adv += d[f"{p}_cnt{k}"] * S[-1]
  return S

def wide_addr(d:dict, p:str="par") -> int: return d[f"{p}_base"] | d[f"{p}_sel"] << 13

def classify(words:list[int]) -> str|None:
  """op kind of an instruction: 'nlu', 'logistic', 'requant', 'mul1', 'mul2', 'add', 'maxpool', 'copy', 'fc', 'conv' or None"""
  op = (words[0] >> 6) & 0x3F
  if op == 0x19: return "nlu"
  if op == 2: return "maxpool" if OP.decode_op(words)["cfg0"] == 0x127 else None
  if op != 1: return None
  d = OP.decode_op(words)
  dp, c0 = d["dp_mode"], d["cfg0"]
  if dp == 1: return "logistic" if d["rsv1946"] else "requant"
  if dp == 2: return "mul2"
  if dp == 5: return "add"
  if dp == 4 and c0 == 0x60: return "mul1"
  if dp == 3 and c0 == 0xA5: return "fc"
  if dp == 4 and c0 in (0xE5, 0xE7): return "conv"
  if c0 in (0x7, 0x9): return "copy"
  return None

def op_params(words:list[int]) -> tuple[str, dict]|None:
  """(kind, kwargs) such that the kind's builder reproduces `words`: shape, layout, allocation and quantization read back"""
  kind = classify(words)
  if kind is None or kind in ("copy", "fc", "conv"): return None
  if kind == "nlu":
    d = EL.decode_nlu(words)
    return kind, dict(tile_mask=d["tile_mask"], seq=d["seq"], slots=[d[f"slot{k}"] for k in range(50)])
  d = OP.decode_op(words)
  alloc = dict(tile_mask=d["tile_mask"], seq=d["seq"])
  def quant():
    return dict(mult=bits_f32(d["mult_bits"]), out_zp=d["out_zp"], clamp_min=bits_f32(d["clamp_min_bits"]), clamp_max=bits_f32(d["clamp_max_bits"]))
  if kind in ("logistic", "requant"):
    kw = dict(P=d["loop0"] + 1, Cw=d["loop1"] + 1, R_in=d["in_inc1"], R_out=d["out_inc1"], in_base=d["in_base"], out_base=d["out_base"],
              ident_addr=wide_addr(d), in_zp=d["in_zp"], out_zp=d["out_zp"], mult=bits_f32(d["mult_bits"]), flat=d["cfg1"] >> 8 & 1)
    return kind, alloc | kw | (dict(clamp_min=bits_f32(d["clamp_min_bits"]), clamp_max=bits_f32(d["clamp_max_bits"])) if kind == "requant" else {})
  if kind == "mul1":
    S = ttu_strides(d, "in")
    return kind, alloc | dict(w=(d["in_cnt0"] + 1) // 4, cols=d["loop2"] + 1, rows=d["loop3"] + 1, R=tuple(s // 4 for s in S[1:4]),
                              in_base=d["in_base"], out_base=d["out_base"], fifo_addr=wide_addr(d), a_zp=d["in_zp"], b_zp=d["w_zp"]) | quant()
  if kind == "mul2":
    S = ttu_strides(d, "out")
    return kind, alloc | dict(w=d["loop1"] + 1, cols=d["loop2"] + 1, rows=d["loop3"] + 1, R_out=tuple(s // 4 for s in S[1:4]),
                              in_base=d["in_base"], out_base=d["out_base"], ident_addr=wide_addr(d))
  if kind == "add":
    S, So = ttu_strides(d, "in"), ttu_strides(d, "out")
    return kind, alloc | dict(w=d["loop5"] + 1, cols=d["loop2"] + 1, rows=d["loop3"] + 1, R_in=tuple(s // 4 for s in S[2:5]), delta=S[1] // 4,
                              R_out=tuple(s // 4 for s in So[1:4]), in_base=d["in_base"], out_base=d["out_base"], wmat_addr=wide_addr(d),
                              offset=d["offset"]) | quant()
  S, So = ttu_strides(d, "par"), ttu_strides(d, "out")           # maxpool
  return kind, alloc | dict(kh=d["loop1"] + 1, kw=d["loop0"] + 1, sh=S[3] // S[1], sw=S[2], OH=d["loop3"] + 1, OW=d["loop2"] + 1,
                            C=64 * d["loop4"] + d["out_ch_last"] + 1, R_out=(So[1] // 4, So[2] // 4), out_base=d["out_base"], src_addr=wide_addr(d),
                            zp=d["out_zp"], clamp_min=bits_f32(d["clamp_min_bits"]), clamp_max=bits_f32(d["clamp_max_bits"]), Sy=S[1], blk=S[4])

BUILDERS = {"logistic": EL.logistic_op_fields, "requant": EL.requant_op_fields, "mul1": EL.mul_op1_fields, "mul2": EL.mul_op2_fields,
            "add": EL.add_op_fields, "maxpool": EL.maxpool_op_fields, "nlu": EL.nlu_fields}

def rebuild(words:list[int]) -> list[int]|None:
  """decode -> builder -> encode; equals `words` for every compiled instance of the covered kinds"""
  if (p:=op_params(words)) is None: return None
  f = BUILDERS[p[0]](**p[1])
  return EL.encode_nlu(**f) if p[0] == "nlu" else OP.encode_op(**f)

# ============================== driver ==============================

def run(quick:bool=False, workers:int=4, verbose:bool=True) -> dict:
  J = jobs(quick)
  def comp(j):
    name, fn = j
    model = fn()
    try: exes, _ = compile_tflite(model)
    except Exception: return name, model, None
    return name, model, [b.data for e in exes for b in e.bitstreams]
  with ThreadPoolExecutor(workers) as ex: progs = list(ex.map(comp, J))
  failed = [n for n, _, b in progs if b is None]
  model_st, rt_st, flow_st, layout_used, misses = Counter(), Counter(), Counter(), Counter(), defaultdict(list)
  for name, model, bss in progs:
    if bss is None: continue
    ops = tflite_ops(model)
    for bs in bss:
      ins = split(bs)
      for _, ws in ins:
        kind = classify(ws)
        if kind in (None, "copy", "fc", "conv"): continue
        rt_st[(kind, rebuild(ws) == ws)] += 1
        cands = model_candidates(kind, ops, ws, name.startswith("pooled"))
        hit = [lab for lab, w in cands if w == ws]
        model_st[(kind, bool(hit))] += 1
        if hit and "rule" not in hit: layout_used[kind] += 1
        if not hit: misses[kind].append(name)
      check_dataflow(ins, ops, flow_st, name.startswith("pooled"))
  res = dict(model=model_st, roundtrip=rt_st, dataflow=flow_st, layout=layout_used, misses=misses, failed=failed, programs=len(progs) - len(failed))
  if verbose:
    print(f"{res['programs']} programs compiled ({len(failed)} failed to compile: {failed})")
    print(f"{'kind':16s} {'model':>13s} {'(layout from insn)':>19s} {'roundtrip':>13s}")
    for kind in ("nlu", "logistic", "requant", "mul1", "mul2", "add", "maxpool"):
      n = model_st[(kind, True)] + model_st[(kind, False)]
      r = rt_st[(kind, True)] + rt_st[(kind, False)]
      print(f"{kind:16s} {model_st[(kind, True)]:6d}/{n:<6d} {layout_used[kind]:19d} {rt_st[(kind, True)]:6d}/{r:<6d}")
    print("data flow:", {k: f"{flow_st[(k, True)]}/{flow_st[(k, True)] + flow_st[(k, False)]}" for k in sorted({k for k, _ in flow_st})})
    for kind, names in misses.items(): print(f"  misses {kind}: {len(names)} in {sorted(set(names))[:6]}")
  return res

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--quick", action="store_true")
  args = ap.parse_args()
  r = run(quick=args.quick)
  bad = sum(v for (k, ok), v in list(r["model"].items()) + list(r["roundtrip"].items()) + list(r["dataflow"].items()) if not ok)
  sys.exit(1 if bad or r["failed"] else 0)
