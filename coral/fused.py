# whole blocks of a model as single Edge TPU programs, so that intermediate tensors never cross the (slow) usb link:
#   "conv":   y = x @ W.T                     a 1x1 conv over the grid of M positions (one row: FULLY_CONNECTED)
#   "ffn":    y = w2(silu(w1 x) * (w3 x))      conv, conv, logistic, mul, mul, conv in one program
#   "argmax": the classifier turned around: the vocabulary is the image (160x200 positions of 288 channels, streamed in), the
#             M token activations are the conv weights, and an 8x8 max pool sends back the max of every block of 64 logits
#             (128 KB instead of 8 MB per step). The host recomputes the best blocks exactly (refine).
# The programs come from our code generator (coral/codegen/fused.py) and its placement plan, which keeps all blocks resident.
from __future__ import annotations
import functools, math, dataclasses, numpy as np
from dataclasses import dataclass, field

def quantize_params(lo:float, hi:float) -> tuple[float, int]:
  """asymmetric uint8 quantization covering [lo, hi] (always including 0) -> (scale, zero point)"""
  lo, hi = min(lo, 0.0), max(hi, 0.0)
  scale = (hi - lo) / 255.0 if hi > lo else 1.0
  return scale, int(np.clip(round(-lo / scale), 0, 255))
def quant(W:np.ndarray, s:float, z:int) -> np.ndarray: return np.clip(np.round(W / s) + z, 0, 255).astype(np.uint8)
def dequant(v:np.ndarray, q:tuple[float, int]) -> np.ndarray: return (v.astype(np.float32) - q[1]) * q[0]
def silu_range(lo:float, hi:float) -> tuple[float, float]:
  def f(v): return v / (1 + math.exp(-v))
  vals = [f(lo), f(hi)] + ([f(-1.278464542761074)] if lo < -1.278464542761074 < hi else [])   # silu's minimum
  return min(vals), max(vals)

@dataclass
class Block:
  name: str
  kind: str
  K: int                                      # x[M,K] uint8 -> y[M,N] uint8 ("argmax": N = vocabulary size)
  N: int
  in_q: tuple[float, int]
  out_q: tuple[float, int]
  quant: dict                                 # (scale, zero point) of every tensor, as coral/codegen/fused.py takes them
  weights: dict                               # the quantized weights (uint8)
  extra: dict = field(default_factory=dict)

REGISTRY: dict[str, Block] = {}
def register(b:Block) -> Block:
  REGISTRY[b.name] = b
  programs.cache_clear()
  return b

def conv_block(name:str, W:np.ndarray, in_range, out_range) -> Block:
  wq_ = quantize_params(float(W.min()), float(W.max()))
  q = dict(x=quantize_params(*in_range), w=wq_, y=quantize_params(*out_range))
  return register(Block(name, "conv", W.shape[1], W.shape[0], q["x"], q["y"], q, dict(w=quant(W, *wq_))))

def ffn_block(name:str, W1:np.ndarray, W3:np.ndarray, W2:np.ndarray, x_range, h1_range, h3_range, m_range, y_range) -> Block:
  q = {k: quantize_params(*r) for k, r in dict(x=x_range, h1=h1_range, h3=h3_range, a=silu_range(*h1_range), m=m_range, y=y_range).items()}
  q["g"] = (1 / 256, 0)                       # uint8 LOGISTIC output, fixed by TFLite's definition
  q.update({k: quantize_params(float(W.min()), float(W.max())) for k, W in dict(w1=W1, w3=W3, w2=W2).items()})
  return register(Block(name, "ffn", W1.shape[1], W1.shape[1], q["x"], q["y"], q, {k: quant(W, *q[k]) for k, W in dict(w1=W1, w3=W3, w2=W2).items()}))

VOCAB_GRID, POOL = (160, 200), 8
def argmax_block(name:str, Wv:np.ndarray, x_range, logit_range) -> Block:
  (V, K), (Hv, Wg) = Wv.shape, VOCAB_GRID
  assert V == Hv * Wg and Hv % POOL == 0 and Wg % POOL == 0
  q = dict(vocab=quantize_params(float(Wv.min()), float(Wv.max())), tokens=quantize_params(*x_range), logits=quantize_params(*logit_range))
  wq = quant(Wv, *q["vocab"])
  # vocabulary ids of block (r, c) of the 20x25 pooled grid: rows 8r..8r+7, columns 8c..8c+7 of the 160x200 image
  r, c, i = np.arange(Hv // POOL), np.arange(Wg // POOL), np.arange(POOL)
  ids = (r[:, None, None, None] * POOL + i[None, None, :, None]) * Wg + c[None, :, None, None] * POOL + i[None, None, None, :]
  return register(Block(name, "argmax", K, V, q["tokens"], q["logits"], q, dict(vocab=wq),
                        extra=dict(Wv=Wv.astype(np.float32), block_ids=ids.reshape(-1, POOL * POOL), vocab_bytes=wq.tobytes())))

def reference(b:Block, x:np.ndarray, h:np.ndarray|None=None) -> np.ndarray:
  """the block in dequantized float math (MOCKCORAL; close to, not bit-exact with, the TPU)"""
  if b.kind == "argmax": return (h.astype(np.float32) @ b.extra["Wv"].T).argmax(1).astype(np.int32)
  W = {k: dequant(w, b.quant[k]) for k, w in b.weights.items()}
  if b.kind == "conv": return quant(dequant(x, b.quant["x"]) @ W["w"].T, *b.quant["y"])
  xf, rq = dequant(x, b.quant["x"]), lambda v, k: dequant(quant(v, *b.quant[k]), b.quant[k])
  h1, h3 = rq(xf @ W["w1"].T, "h1"), rq(xf @ W["w3"].T, "h3")
  return quant(rq(rq(h1 / (1 + np.exp(-h1)), "a") * h3, "m") @ W["w2"].T, *b.quant["y"])

SAMPLE = {"temperature": 0.0, "rng": np.random.default_rng()}   # > 0: sample among the candidates instead of the argmax

def refine(b:Block, blockmax:np.ndarray, h:np.ndarray, top:int=4) -> np.ndarray:
  """blockmax [M, 500] uint8 from the TPU, h [M, K] float -> exact float argmax over the `top` best blocks of every row
  (one GEMM over the union of the candidate blocks: a few thousand of the 32000 logits). With SAMPLE["temperature"] > 0 it
  samples among those candidates (Gumbel max)"""
  M, ids = len(h), b.extra["block_ids"]
  best = np.argpartition(-blockmax.astype(np.int32), top, axis=1)[:, :top]          # [M, top] block indices
  ub = np.unique(best)
  logits = (h.astype(np.float32) @ b.extra["Wv"][ids[ub].reshape(-1)].T).reshape(M, len(ub), -1)
  li = np.searchsorted(ub, best)                                                       # [M, top] -> index into ub
  cand = logits[np.arange(M)[:, None], li].reshape(M, -1)
  if (t:=SAMPLE["temperature"]) > 0: cand = cand / t + SAMPLE["rng"].gumbel(size=cand.shape)
  flat = cand.argmax(1)
  return ids[ub[li[np.arange(M), flat // logits.shape[2]]], flat % logits.shape[2]].astype(np.int32)

@functools.cache
def programs(Mp:int) -> dict[str, dict]:
  """every registered block for Mp rows, generated and placed by coral/codegen/fused.py:
  {name: {"PARAMETER_CACHING": executable carrying the parameter blob, "EXECUTION_ONLY": executable}}"""
  from coral.codegen import fused as cg
  blocks = [b for b in REGISTRY.values() if Mp > 1 or b.kind != "argmax"]
  pl = cg.plan(blocks, Mp)
  assert not (bad:=cg.check_plan(blocks, Mp, pl)), bad
  exes = cg.gen_set(blocks, Mp, pl, {b.name: b.quant for b in blocks})
  for b in blocks:
    if b.kind == "argmax": continue           # its parameters are the token activations of every call
    blob = cg.conv_blob(b.weights["w"], b.quant["w"][1]) if b.kind == "conv" else \
           cg.ffn_params(b.weights["w1"], b.weights["w3"], b.weights["w2"], zps=(b.quant["w1"][1], b.quant["w3"][1], b.quant["w2"][1]))
    exes[b.name]["PARAMETER_CACHING"] = dataclasses.replace(exes[b.name]["PARAMETER_CACHING"], parameters=blob)
  return exes

def relayout(exe, buf:bytes, H:int, W:int, C:int) -> np.ndarray:
  """the executable's output tensor [1, H, W, C] (tiled over the chip) -> [H*W, C]"""
  L, b = exe.outputs[0].output_layout, np.frombuffer(buf, np.uint8)
  if not L or not L.get("y_tile"): return b[:H * W * C].reshape(H * W, C)
  return np.stack([b[s:s+C] for s in (L["tile_byte_offset"][L["y_tile"][y] + L["x_tile"][x]] + L["y_local_y_offset"][y] * L["x_local_row_size"][x] +
                                      L["x_local_byte_offset"][x] for y in range(H) for x in range(W))])
