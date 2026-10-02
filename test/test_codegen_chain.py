# Acceptance test for coral/codegen/chain.py (offline only, never touches the device): whole chains of VALID convolutions, 2x2 max
# pools and a final linear layer as ONE Edge TPU program, against edgetpu_compiler compiling the same TFLite model.
#   1. MNIST (examples/mnist.py's CNN, models/mnist.safetensors, quantized by coral.quantize, exported by tools/export.tflite_chain)
#      at B = 1, 2, 4, 8, 16, 32 images per program (B > 1: one tall image, the linear layer a conv over each image's map);
#   2. random small CNNs (VALID convs k in 1/3/5 with or without ReLU, Cin / Cout 1..128, 2x2 / stride 2 max pools, a final linear
#      layer or not) at B = 1 and stacked into tall images.
#   Per model: the caching and the execution program byte-exact (gen_chain with the default allocation, nothing taken from the
#   compiler), the parameter blob (chain_blob from the model's weights), the host contract (chain_io: DMA sizes from the hints, the
#   output_layout, the blob size). The first mismatch of a failing model is printed: program, instruction index, opcode and the
#   differing decoded fields (ours, compiler). Models the compiler does not turn into one program pair and models chain.py refuses
#   (NotImplementedError) are counted as excluded, with the reason.
#   Compiles run in docker (edgetpu_compiler) and are cached in .compile/; MNIST needs tinygrad's MNIST download (DEV=CPU).
#
#   python test/test_codegen_chain.py [--n N] [--seed S] [--skip-mnist]          or        python -m coral.codegen.chain [...]
from __future__ import annotations
import sys, os, pathlib, argparse, random
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "examples"):
  if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np
from coral.isa import split, opcode, LAYOUTS
from coral.codegen import chain as CH
from coral.codegen.conv2d import conv2d_quant

# ============================== reporting ==============================
def decode(ws:list[int]) -> dict: return LAYOUTS[op].decode(ws) if (op:=opcode(ws[0])) in LAYOUTS else {"word": ws[0]}
def name(op:int) -> str: return LAYOUTS[op].name if op in LAYOUTS else f"op{op:#x}"

def first_diff(ours:bytes, ref:bytes) -> str|None:
  """first differing instruction after the start word (whose length differs whenever anything else does): index, word, opcode and
  the differing decoded fields (ours, compiler)"""
  sa, sb = split(ours), split(ref)
  for k, ((ia, wa), (ib, wb)) in [*list(enumerate(zip(sa, sb)))[1:], *list(enumerate(zip(sa, sb)))[:1]]:
    if wa == wb: continue
    oa, ob = opcode(wa[0]), opcode(wb[0])
    if oa != ob or len(wa) != len(wb): return f"instruction #{k} (word {ia}/{ib}): opcode {name(oa)} (ours) vs {name(ob)} (compiler)"
    da, db = decode(wa), decode(wb)
    return f"instruction #{k} (word {ia}) {name(oa)}: differing fields (ours, compiler) {({n: (da[n], db[n]) for n in da if da[n] != db.get(n)})}"
  return None if len(sa) == len(sb) else f"{len(sa)} vs {len(sb)} instructions"

class Report:
  def __init__(self): self.rows: list[tuple[str, int, int]] = []
  def add(self, title:str, results:list[tuple[object, str|None]], excluded:list|None=None):
    fails = [(k, m) for k, m in results if m is not None]
    self.rows.append((title, len(results) - len(fails), len(results)))
    ex = f"   ({len(excluded)} excluded)" if excluded else ""
    print(f"{title:58s} {len(results) - len(fails):5d}/{len(results)}{ex}")
    for k, m in fails[:5]: print(f"    FAIL {k}: {m}")
    reasons: dict = {}
    for k, why in excluded or []: reasons.setdefault(why, []).append(k)
    for why, ks in reasons.items(): print(f"    excluded {len(ks)}: {why}  (e.g. {ks[0]})")
    sys.stdout.flush()
  @property
  def ok(self) -> bool: return all(p == t for _, p, t in self.rows)

# ============================== TFLite <-> layers ==============================
def tflite_layers(model:bytes) -> tuple[list, list]:
  """(layers, params) of a chain TFLite model (CONV_2D VALID / MAX_POOL_2D / FULLY_CONNECTED, uint8): chain.py's layer
  description and chain_blob's params (conv weights [Cout, kh, kw, Cin] as stored, FC weights [N, K] in the map's (y, x, c) order)"""
  import tflite
  m = tflite.Model.GetRootAsModel(model, 0)
  sg = m.Subgraphs(0)
  def tq(i:int):
    t = sg.Tensors(i)
    q = t.Quantization()
    return (float(q.Scale(0)), int(q.ZeroPoint(0))), [t.Shape(j) for j in range(t.ShapeLength())]
  def data(i:int, dt):
    b = m.Buffers(sg.Tensors(i).Buffer())
    return np.frombuffer(b.DataAsNumpy().tobytes(), dt)
  layers, params = [], []
  for k in range(sg.OperatorsLength()):
    op = sg.Operators(k)
    code = m.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    ins = [op.Inputs(j) for j in range(op.InputsLength())]
    (xq, xs), (yq, _) = tq(ins[0]), tq(op.Outputs(0))
    if code == tflite.BuiltinOperator.CONV_2D:
      (wq, ws) = tq(ins[1])
      o = tflite.Conv2DOptions()
      o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
      assert o.Padding() == tflite.Padding.VALID, "VALID convolutions only"
      layers.append(CH.Conv(xs[1], xs[2], xs[3], ws[0], ws[1], ws[2], o.StrideH(), conv2d_quant(xq, wq, yq, o.FusedActivationFunction())))
      params.append((data(ins[1], np.uint8).reshape(ws), data(ins[2], np.int32) if len(ins) > 2 and ins[2] >= 0 else None))
    elif code == tflite.BuiltinOperator.MAX_POOL_2D:
      o = tflite.Pool2DOptions()
      o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
      layers.append(CH.Pool(o.FilterHeight(), o.StrideH(), yq))
    elif code == tflite.BuiltinOperator.FULLY_CONNECTED:
      (wq, ws) = tq(ins[1])
      o = tflite.FullyConnectedOptions()
      o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
      layers.append(CH.Linear(ws[0], ws[1], conv2d_quant(xq, wq, yq, o.FusedActivationFunction())))
      params.append((data(ins[1], np.uint8).reshape(ws), data(ins[2], np.int32) if len(ins) > 2 and ins[2] >= 0 else None))
    else: raise NotImplementedError(f"operator {code}")
  return layers, params

def random_model(desc:dict, seed:int) -> bytes:
  """a chain TFLite model with random weights and quantization: desc = dict(inp=(H, W, Cin), images=B, layers=[("conv", Cout, kh, kw,
  relu) | ("pool",) | ("linear", N)]); with B > 1 the linear layer is a conv over each image's map (tools/export.py's form)"""
  import tflite
  from tools.tflite_gen import Model, conv_options, fc_options
  from tools.export import _pool
  rng = np.random.default_rng(seed)
  m = Model()
  (H, W, C), B = desc["inp"], desc.get("images", 1)
  q = (float(np.float32(rng.uniform(0.005, 0.05))), int(rng.integers(0, 256)))
  shape = [1, H * B, W, C]
  t = inp = m.tensor("input", shape, np.uint8, *q)
  pitch = H
  for i, L in enumerate(desc["layers"]):
    if L[0] == "pool":
      shape = [1, (shape[1] - 2) // 2 + 1, (shape[2] - 2) // 2 + 1, shape[3]]
      pitch //= 2
      o = m.tensor(f"pool{i}", shape, np.uint8, *q)
      m.op(tflite.BuiltinOperator.MAX_POOL_2D, [t], [o], tflite.BuiltinOptions.Pool2DOptions, _pool(2, 2))
      t = o
      continue
    if L[0] == "conv": _, O, kh, kw, relu = L
    elif B == 1: (_, O), kh, kw, relu = L, None, None, False
    else: (_, O), kh, kw, relu = L, shape[1] - (B - 1) * pitch, shape[2], False
    wq = (float(np.float32(rng.uniform(0.002, 0.02))), int(rng.integers(0, 256)))
    yq = (float(np.float32(rng.uniform(0.02, 0.2))), 0 if relu else int(rng.integers(0, 256)))
    bt = m.tensor(f"b{i}", [O], np.int32, q[0] * wq[0], 0, data=rng.integers(-3000, 3000, O).astype(np.int32))
    if kh is None:                                               # FULLY_CONNECTED on the flattened map
      K = int(np.prod(shape[1:]))
      wt = m.tensor(f"w{i}", [O, K], np.uint8, *wq, data=rng.integers(0, 256, (O, K), dtype=np.uint8))
      shape = [1, O]
      o = m.tensor(f"y{i}", shape, np.uint8, *yq)
      m.op(tflite.BuiltinOperator.FULLY_CONNECTED, [t, wt, bt], [o], tflite.BuiltinOptions.FullyConnectedOptions, fc_options(0))
    else:
      wt = m.tensor(f"w{i}", [O, kh, kw, shape[3]], np.uint8, *wq, data=rng.integers(0, 256, (O, kh, kw, shape[3]), dtype=np.uint8))
      shape = [1, shape[1] - kh + 1, shape[2] - kw + 1, O]
      o = m.tensor(f"y{i}", shape, np.uint8, *yq)
      m.op(tflite.BuiltinOperator.CONV_2D, [t, wt, bt], [o], tflite.BuiltinOptions.Conv2DOptions, conv_options(1, 1, int(relu)))
    t, q = o, yq
  return m.build([inp], [t])

def random_desc(rng:random.Random, images:int=1) -> dict|None:
  """a random chain description (None when the draw does not make a valid chain)"""
  n_pool_max = 2
  H = rng.randint(8, 32) if images == 1 else rng.choice([8, 12, 16, 20, 24, 28])
  W = H if rng.random() < 0.6 else rng.randint(8, 32)
  C = rng.choice([1, 3, 4, 8, 16, 32, rng.randint(1, 64)])
  layers, h, w, pools = [], H, W, 0
  for _ in range(rng.randint(1, 4)):
    ks = [k for k in (1, 3, 5) if k <= h and k <= w]
    if not ks: break
    k = rng.choice(ks)
    layers.append(("conv", rng.choice([rng.randint(1, 128), 8, 16, 32, 64]), k, k, rng.random() < 0.7))
    h, w = h - k + 1, w - k + 1
    if pools < n_pool_max and h >= 4 and w >= 4 and rng.random() < 0.4 and (images == 1 or (H >> pools) % 2 == 0):
      layers.append(("pool",))
      h, w, pools = h // 2, w // 2, pools + 1
  if not layers: return None
  if layers[-1] == ("pool",) or rng.random() < 0.6: layers.append(("linear", rng.randint(1, 64)))
  if images > 1 and layers[-1][0] == "linear" and (h % 2 == 0 or w % 2 == 0): return None   # the conv over an image's map: odd kernels only
  return dict(inp=(H, W, C), images=images, layers=layers)

# ============================== the checks ==============================
def compare(layers:list, params:list, exes:dict, images:int=1) -> str|None:
  """both programs byte-exact, the blob, the host contract"""
  caching, exe = CH.gen_chain(layers)
  msgs = [f"{n}: {first_diff(a, exes[t].bitstreams[0].data)}" for n, t, a in
          (("caching", "PARAMETER_CACHING", caching), ("execution", "EXECUTION_ONLY", exe)) if a != exes[t].bitstreams[0].data]
  eo = exes["EXECUTION_ONLY"]
  if len(eo.bitstreams) != 1: msgs.append(f"{len(eo.bitstreams)} execution bitstreams")
  io = CH.chain_io(layers, images)
  sizes = tuple(sum(h.size for h in eo.hints if h.kind == "dma" and h.desc == d) for d in ("INPUT", "OUTPUT"))
  if sizes != (io["input_bytes"], io["output_bytes"]): msgs.append(f"io sizes {(io['input_bytes'], io['output_bytes'])} != compiler {sizes}")
  lay = eo.outputs[0].output_layout
  if io["output_layout"] != {k: list(v) for k, v in (lay or {}).items()}: msgs.append("output_layout differs")
  blob = exes["PARAMETER_CACHING"].parameters
  if len(blob) != io["param_bytes"]: msgs.append(f"blob size {io['param_bytes']} != compiler {len(blob)}")
  if CH.chain_blob(layers, params) != blob: msgs.append("parameter blob differs")
  return "; ".join(msgs) or None

def check_model(model:bytes, images:int=1) -> tuple[str, str|None]:
  """('ok' | 'fail' | 'excluded', message)"""
  from tools.compiler import compile_tflite
  try: exes = {e.type: e for e in compile_tflite(model)[0]}
  except Exception as e:
    return "excluded", "edgetpu_compiler failed (internal compiler error)" if "Internal compiler error" in str(e) else f"compile error {e}"
  if "EXECUTION_ONLY" not in exes or "PARAMETER_CACHING" not in exes: return "excluded", f"compiler: {sorted(exes)}"
  if len(exes["EXECUTION_ONLY"].bitstreams) != 1: return "excluded", "compiler: several execution bitstreams"
  if any(h.kind == "dma" and h.desc == "PARAMETER" for h in exes["EXECUTION_ONLY"].hints): return "excluded", "compiler: streams parameters"
  layers, params = tflite_layers(model)
  try: m = compare(layers, params, exes, images)
  except NotImplementedError as e: return "excluded", f"chain.py: {e}"
  except Exception as e:
    import traceback
    return "fail", f"exception {type(e).__name__}: {e} {traceback.format_exc().splitlines()[-3].strip()}"
  return ("fail", m) if m else ("ok", None)

def mnist_chain():
  """examples/mnist.py's CNN quantized by coral.quantize (bench/mnist_google.py's build): [(QLayer, ops after it)]"""
  os.environ.setdefault("DEV", "CPU")
  from tinygrad import nn
  from tinygrad.nn.datasets import mnist
  from mnist import Model, WEIGHTS
  from coral.quantize import quantize
  model = Model()
  nn.state.load_state_dict(model, nn.state.safe_load(str(WEIGHTS)))
  return quantize(model, mnist()[0][:256].float() / 255.0)["chain"]

def specs_of(chain:list) -> tuple[list, list]:
  """chain_layers' specs and the NCHW-style weights of a coral.quantize chain (an example adapter)"""
  specs, weights = [], []
  for q, how in chain:
    qq = dict(x_q=(q.xs, q.xz), w_q=(q.ws, q.wz), y_q=(q.ys, q.yz), act=int(how is not None and "relu" in how))
    specs.append({"conv": dict(Cout=q.wq.shape[0], kh=q.kernel[0], kw=q.kernel[1], **qq)} if q.conv else {"linear": dict(N=q.wq.shape[0], **qq)})
    if how is not None and "maxpool2" in how.split(","): specs.append({"pool": dict(k=2, stride=2)})
    weights.append((q.wq.numpy(), q.bq.numpy()))
  return specs, weights

def check_mnist(rep:Report):
  """per B: the TFLite model's layers (tflite_layers) and the same chain built from the quantized layers (chain_layers, chain_params)"""
  from tools.export import tflite_chain
  from tools.compiler import compile_tflite
  chain = mnist_chain()
  specs, weights = specs_of(chain)
  res = []
  for B in (1, 2, 4, 8, 16, 32):
    model = tflite_chain(chain, B)
    status, m = check_model(model, B)
    res.append((f"MNIST B={B}", m if status != "ok" else None))
    layers = CH.chain_layers(28, 28, 1, specs, images=B)
    exes = {e.type: e for e in compile_tflite(model)[0]}
    try: m2 = compare(layers, CH.chain_params(layers, weights), exes, B)
    except Exception as e: m2 = f"exception {type(e).__name__}: {e}"
    res.append((f"MNIST B={B} via chain_layers", m2))
  rep.add("MNIST, B = 1, 2, 4, 8, 16, 32 (programs, blob, host contract)", res)

def check_random(rep:Report, n:int, seed:int, images:int):
  rng, descs = random.Random(f"{seed}/{images}"), []
  while len(descs) < n:
    d = random_desc(rng, images)
    if d is not None: descs.append(d)
  jobs = [(d, random_model(d, seed * 1000 + k)) for k, d in enumerate(descs)]
  with ThreadPoolExecutor(4) as ex: out = list(ex.map(lambda j: (j[0], check_model(j[1], images)), jobs))
  res = [(_short(d), m) for d, (s, m) in out if s != "excluded"]
  excluded = [(_short(d), m) for d, (s, m) in out if s == "excluded"]
  rep.add(f"random chains, B = {images} ({n} drawn)", res, excluded)

def _short(d:dict) -> str:
  L = " ".join(f"c{l[2]}x{l[3]}:{l[1]}{'r' if l[4] else ''}" if l[0] == "conv" else "p" if l[0] == "pool" else f"fc{l[1]}" for l in d["layers"])
  return f"{d['inp'][0]}x{d['inp'][1]}x{d['inp'][2]}{'*' + str(d['images']) if d['images'] > 1 else ''} {L}"

def main(argv:list[str]) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, default=40, help="random chains per batch size")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--skip-mnist", action="store_true")
  args = ap.parse_args(argv)
  rep = Report()
  if not args.skip_mnist: check_mnist(rep)
  check_random(rep, args.n, args.seed, 1)
  for B in (2, 4): check_random(rep, max(args.n // 4, 1), args.seed, B)
  print("PASS" if rep.ok else "FAIL")
  return 0 if rep.ok else 1

if __name__ == "__main__": sys.exit(main(sys.argv[1:]))
