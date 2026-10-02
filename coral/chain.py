# whole layer chains as ONE Edge TPU call. tinygrad's JIT captures a forward pass as a list of kernel calls; on DEV=CORAL a model
# quantized by coral.quantize is a run of selected kernels (convolutions, a final linear layer) joined by "glue" kernels that are
# exactly a uint8 max pool or a copy (coral.select.select_glue). fuse() replaces each such run by one call to a chain program that
# keeps the activations on the chip; the intermediate buffers, which nothing else reads, disappear from the JIT.
# install() hooks it into tinygrad's jit_lower (the lowering of the captured calls, before memory planning and compilation), the
# way coral/ops_coral.py hooks kernel selection into do_to_program. Without TinyJit every kernel still runs on its own.
from __future__ import annotations
import json, functools, collections, dataclasses, zlib, numpy as np
from dataclasses import dataclass, asdict
from tinygrad.helpers import DEBUG, getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo
from tinygrad.renderer import Estimates

CHAIN = getenv("CORAL_CHAIN", 1)       # CORAL_CHAIN=0: no fusion, every selected kernel is its own TPU call
CHAIN_PROGRAM = getenv("CORAL_CHAIN_PROGRAM", 1)   # 0: run fused chains layer by layer instead of as one program

@dataclass(frozen=True)
class ChainSpec:
  """the layers in order, each {"conv2d": ConvSpec fields} / {"fc": FCSpec fields} (with "bias": bool) / {"pool": PoolSpec fields}
  / {"copy": CopySpec fields}. The call's buffers: (out, x, then w[, b] of every conv and fc in order)."""
  layers: tuple
  def dumps(self) -> str: return json.dumps({"chain": list(self.layers)})
  @staticmethod
  def loads(s:str|bytes) -> ChainSpec: return ChainSpec(tuple(json.loads(s)["chain"]))

# *** finding the chains ***

def _node(call:UOp):
  """("layer" | "glue", match) for a CORAL kernel call the TPU can run in a chain, else None"""
  if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK or not str(call.src[1].device).startswith("CORAL"): return None
  import coral.select as sel
  try:
    if (m:=sel.select(call.src[0])) is not None: return ("layer", m)
    if (g:=sel.select_glue(call.src[0])) is not None: return ("glue", g)
  except Exception as e: print(f"CORAL: chain selection failed, not fusing this kernel: {e!r}")
  return None

def _buf(call:UOp, p:UOp) -> UOp: return call.src[1 + p.arg.slot]

def chains(linear:UOp, held_bufs:set, input_uops:list) -> list[list[tuple[int, str, object]]]:
  """the runs of calls to fuse: [(index in linear.src, kind, match)], each starting and ending with a layer, at least two layers,
  every intermediate buffer written by one call and read by the next one only (not held by a live Tensor, not a JIT input)"""
  calls, uses = linear.src, collections.Counter(b for c in linear.src if c.op is Ops.CALL for b in c.src[1:])
  nodes, out, run = [_node(c) for c in calls], [], []
  def flush():
    while run and run[-1][1] != "layer": run.pop()
    if sum(k == "layer" for _, k, _ in run) >= 2: out.append(list(run))
    run.clear()
  for i, (c, n) in enumerate(zip(calls, nodes)):
    if n is None:
      flush()
      continue
    kind, m = n
    if run:
      prev_call, prev_m = calls[run[-1][0]], run[-1][2]
      y = _buf(prev_call, prev_m.params[0])
      if _buf(c, m.params[1]) is y and uses[y] == 2 and y not in held_bufs and y not in input_uops:
        run.append((i, kind, m))
        continue
      flush()
    if kind == "layer": run.append((i, kind, m))
  flush()
  return out

def _spec_dict(kind:str, m) -> dict:
  from coral.select import ConvSpec
  s = asdict(m.spec)
  if kind == "glue": return {type(m.spec).__name__.lower().replace("spec", ""): s}
  return {("conv2d" if isinstance(m.spec, ConvSpec) else "fc"): s, "bias": m.b is not None}

def fused_call(linear:UOp, run:list) -> UOp:
  """one CALL for the run: a PROGRAM whose SOURCE is the ChainSpec, buffers (out, x, w0, b0, w1, ...)"""
  from tinygrad.device import Device
  from tinygrad.runtime.ops_coral import CORAL_SRC
  calls = linear.src
  first, last = calls[run[0][0]], calls[run[-1][0]]
  bufs, params = [_buf(last, run[-1][2].params[0]), _buf(first, run[0][2].params[1])], [run[-1][2].params[0], run[0][2].params[1]]
  for i, kind, m in run:
    if kind == "layer":
      bufs += [_buf(calls[i], p) for p in m.params[2:]]
      params += list(m.params[2:])
  params = [p.replace(arg=dataclasses.replace(p.arg, slot=j)) for j, p in enumerate(params)]
  spec = ChainSpec(tuple(_spec_dict(k, m) for _, k, m in run))
  ops = sum(2 * m.spec.fc().M * m.spec.fc().N * m.spec.fc().K if hasattr(m.spec, "fc") else 2 * m.spec.M * m.spec.N * m.spec.K
            for _, k, m in run if k == "layer")
  nbytes = sum(b.max_numel() * b.dtype.itemsize for b in bufs)
  name = "tpu_chain_" + "_".join(next(iter(d)) for d in spec.layers)
  sink = UOp.sink(*params, arg=KernelInfo(name=name[:60], estimates=Estimates(ops, nbytes, nbytes)))
  info = ProgramInfo(globals=tuple(range(len(params))), outs=(0,), ins=tuple(range(1, len(params))), target=Device["CORAL"].renderer.target)
  prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=CORAL_SRC + spec.dumps())), arg=info)
  return UOp(Ops.CALL, src=(prg, *bufs), arg=first.arg)

def fuse(linear:UOp, held_bufs:set, input_uops:list) -> UOp:
  if not CHAIN or not (runs:=chains(linear, held_bufs, input_uops)): return linear
  drop, put = {i for r in runs for i, _, _ in r}, {r[0][0]: fused_call(linear, r) for r in runs}
  if DEBUG >= 1: print(f"CORAL: fused {len(drop)} kernels into {len(runs)} chain{'s' if len(runs) > 1 else ''}: " +
                       ", ".join("-".join(next(iter(_spec_dict(k, m))) for _, k, m in r) for r in runs))
  return linear.replace(src=tuple(put.get(i, c) for i, c in enumerate(linear.src) if i not in drop or i in put))

def install():
  import tinygrad.engine.jit as jit
  if getattr(jit.jit_lower, "__coral_chain__", False): return
  jit_lower = jit.jit_lower
  @functools.wraps(jit_lower)
  def jit_lower_chain(linear:UOp, held_bufs:set, input_uops:list):
    if any(str(c.src[1].device).startswith("CORAL") for c in linear.src if c.op is Ops.CALL and len(c.src) > 1):
      linear = fuse(linear, held_bufs, input_uops)
    return jit_lower(linear, held_bufs, input_uops)
  setattr(jit_lower_chain, "__coral_chain__", True)
  jit.jit_lower = jit_lower_chain

# *** running a chain ***

def buffers(spec:ChainSpec) -> list[tuple[int, type]]:
  """(elements, numpy dtype) of the call's buffers: out, x, then w[, b] of every conv and fc"""
  def io(d):
    if "conv2d" in d:
      c = d["conv2d"]
      oh, ow = ((c["H"] + 2*c["padding"] - c["kh"]) // c["stride"] + 1, (c["W"] + 2*c["padding"] - c["kw"]) // c["stride"] + 1)
      return c["N"]*c["Cin"]*c["H"]*c["W"], c["N"]*c["Cout"]*oh*ow
    if "fc" in d:
      f = d["fc"]
      return f["M"]*f["K"], f["M"]*f["N"]
    raise ValueError("a chain starts and ends with a conv or fc")
  out = [(io(spec.layers[-1])[1], np.uint8), (io(spec.layers[0])[0], np.uint8)]
  for d in spec.layers:
    if "conv2d" in d:
      c = d["conv2d"]
      out += [(c["Cout"]*c["Cin"]*c["kh"]*c["kw"], np.uint8)] + [(c["Cout"], np.int32)] * d["bias"]
    elif "fc" in d:
      f = d["fc"]
      out += [(f["N"]*f["K"], np.uint8)] + [(f["N"], np.int32)] * d["bias"]
  return out

def _pool(x:np.ndarray, k:int, s:int) -> np.ndarray:
  return np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s].max((4, 5))

# *** one program for the whole chain (coral/codegen/chain.py): G images per call, stacked into one tall image ***

TALL = 32                                     # images per call: the generator's programs are the compiler's up to 32 (B >= 64: row bands)

def codegen_layers(spec:ChainSpec, G:int) -> list|None:
  """the chain as coral/codegen/chain.py layers for G images stacked into one tall image (VALID 2x2 pools and VALID convs never
  mix two images' rows; the final linear layer becomes a conv over each image's map), or None where the generator doesn't apply"""
  from coral.codegen.chain import Conv, Pool, Linear, check_chain
  from coral.programs import Quant
  def q(d): return Quant(d["in_zp"], d["w_zp"], d["out_zp"], d["mult"], d.get("lo", 0), d.get("hi", 255)).fields()
  out, h, w, c, pitch, zp = [], None, None, None, None, 0
  for i, d in enumerate(spec.layers):
    if "conv2d" in d:
      cs = d["conv2d"]
      if cs["stride"] != 1 or cs["padding"] != 0: return None
      if h is None: h, w, c, pitch = cs["H"] * G, cs["W"], cs["Cin"], cs["H"]
      out.append(Conv(h, w, c, cs["Cout"], cs["kh"], cs["kw"], 1, q(cs)))
      h, w, c, zp = h - cs["kh"] + 1, w - cs["kw"] + 1, cs["Cout"], cs["out_zp"]
    elif "copy" in d:
      if (d["copy"].get("lo", 0), d["copy"].get("hi", 255)) != (0, 255): return None   # the identity: nothing to do on the chip
    elif "pool" in d:
      p = d["pool"]
      if pitch % p["stride"]: return None
      out.append(Pool(p["k"], p["stride"], (1.0, zp)))
      h, w, pitch = (h - p["k"]) // p["stride"] + 1, (w - p["k"]) // p["stride"] + 1, pitch // p["stride"]
    elif "fc" in d:
      f = d["fc"]
      if i != len(spec.layers) - 1 or f["K"] != (h - (G - 1) * pitch) * w * c: return None
      out.append(Linear(f["N"], f["K"], q(f)) if G == 1 else Conv(h, w, c, f["N"], h - (G - 1) * pitch, w, 1, q(f)))
  if not out or "fc" not in spec.layers[-1]: return None
  try: check_chain(out)
  except (NotImplementedError, ValueError): return None
  return out

@functools.cache
def chain_program(spec_json:str, G:int):
  """(caching, execution Executables, layers, host contract) of the chain for G images per call, or None"""
  from coral.codegen.chain import gen_chain, chain_io
  from coral.programs import _exe
  from coral.executable import Hint
  if (layers:=codegen_layers(ChainSpec.loads(spec_json), G)) is None: return None
  try: pc, eo = gen_chain(layers)
  except NotImplementedError: return None
  io = chain_io(layers, images=G)
  caching = _exe("PARAMETER_CACHING", pc, [Hint("dma", "INFEED", "PARAMETER", None, 0, io["param_bytes"])])
  exe = _exe("EXECUTION_ONLY", eo,
             [Hint("dma", "INFEED", "INPUT", "x", 0, io["input_bytes"]), Hint("dma", "OUTFEED", "OUTPUT", "y", 0, io["output_bytes"])])
  L, N = io["output_layout"], layers[-1].N if G == 1 else layers[-1].Cout
  rows = io.get("image_rows", [0])
  starts = [L["tile_byte_offset"][L["y_tile"][r] + L["x_tile"][0]] + L["y_local_y_offset"][r] * L["x_local_row_size"][0] + L["x_local_byte_offset"][0]
            for r in rows]
  gather = np.array(starts)[:, None] + np.arange(N)            # image b's N outputs in the output buffer
  return caching, exe, layers, io, gather

def run_chain_tpu(r, spec:ChainSpec, x:np.ndarray, weights:list[np.ndarray]) -> np.ndarray|None:
  """the chain as one program per G images on the device (TPURunner r), or None if the generator doesn't cover it"""
  from coral.codegen.chain import chain_params, chain_blob
  first, last = spec.layers[0]["conv2d"], spec.layers[-1]["fc"] if "fc" in spec.layers[-1] else None
  if last is None: return None
  n = first["N"]
  G = min(n, TALL)
  if (prog:=chain_program(spec.dumps(), G)) is None: return None
  caching, exe, layers, io, gather = prog
  key = (spec.dumps(), G, hash(tuple(zlib.crc32(np.ascontiguousarray(w)) for w in weights)))
  if key not in r.blobs:
    ws, out_ws = iter(weights), []
    for d in spec.layers:
      if "conv2d" in d:
        cs = d["conv2d"]
        out_ws.append((next(ws).reshape(cs["Cout"], cs["Cin"], cs["kh"], cs["kw"]), next(ws) if d["bias"] else np.zeros(cs["Cout"], np.int32)))
      elif "fc" in d:
        f = d["fc"]
        out_ws.append((next(ws).reshape(f["N"], f["K"]), next(ws) if d["bias"] else np.zeros(f["N"], np.int32)))
    r.blobs[key] = chain_blob(layers, chain_params(layers, out_ws))
  H, W, Cin = first["H"], first["W"], first["Cin"]
  m = -(-n // G) * G                                            # whole calls: pad the batch with zero images
  xs = np.zeros((m, Cin, H, W), np.uint8)
  xs[:n] = x.reshape(n, Cin, H, W)
  xs = xs.transpose(0, 2, 3, 1).reshape(m // G, G * H * W * Cin)  # NHWC, G images per call as one tall image
  pad = bytes(io["input_bytes"] - xs.shape[1])
  outs = r.chain(key, caching, exe, [row.tobytes() + pad for row in xs])
  return np.concatenate([np.frombuffer(o, np.uint8)[gather] for o in outs])[:n].reshape(-1)

def run_chain(spec:ChainSpec, x:np.ndarray, weights:list[np.ndarray]) -> np.ndarray:
  """the chain on the uint8 input buffer x (flat) with the weight buffers (flat, in call order) -> the output buffer (flat).
  On the device: one program for the whole chain when coral/codegen/chain.py covers it (CORAL_CHAIN_PROGRAM=0: never). Else
  layer by layer: every conv and fc on the TPU (coral.tpu.run_conv / run_fc; with MOCKCORAL=1 their bit-exact numpy models),
  the pools and copies on the host."""
  from coral.select import ConvSpec
  from coral.tpu import FCSpec, run_fc, run_conv, runner
  if (r:=runner()) is not None and CHAIN_PROGRAM and (y:=run_chain_tpu(r, spec, x, weights)) is not None: return y
  ws, v = iter(weights), x
  for d in spec.layers:
    if "conv2d" in d:
      c = ConvSpec(**d["conv2d"])
      w, b = next(ws), (next(ws) if d["bias"] else None)
      v = run_conv(c, v.reshape(c.N, c.Cin, c.H, c.W), w.reshape(c.Cout, c.Cin, c.kh, c.kw), b)
    elif "fc" in d:
      f = FCSpec(**d["fc"])
      w, b = next(ws), (next(ws) if d["bias"] else np.zeros(f.N, np.int32))
      v = run_fc(f, v.reshape(f.M, f.K), w.reshape(f.N, f.K), b)
    elif "pool" in d: v = _pool(v, d["pool"]["k"], d["pool"]["stride"])     # v is the conv's out[N, C, OH, OW]
    elif "copy" in d:
      c = d["copy"]
      v = np.clip(v, c.get("lo", 0), c.get("hi", 255)).astype(np.uint8)
  return np.ascontiguousarray(v).reshape(-1)
