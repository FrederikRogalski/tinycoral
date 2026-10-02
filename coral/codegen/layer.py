# A whole TinyStories-15M transformer layer (and all six layers) at batch 1 as ONE Edge TPU program (docs/isa/codegen_layer.md):
#   x (the uint8 residual stream) -> L2_NORMALIZATION (the RMSNorm, its weight and sqrt(288) folded into the next FC's weights)
#     -> the attention block (q, k, v FCs with the q / k rows in the order SIGMA, RoPE with swap(q) read as a half swap of every
#        16-byte group, attention over the host's KV rows + this token's k', v, wo) -> ADD (the residual) -> L2_NORMALIZATION
#     -> the FFN (w1, LOGISTIC, MUL, w3, MUL, w2; fused.py's FC form) -> ADD
#   caching, bitstreams, io = gen_model(P, quants, plan(6), local=True)   all six layers, one call per token (P = 1..256)
#   x_out, kv = model_ref(x, cos, sin, Ks, Vs, weights, quants)           the numpy bit model, built from the pieces' bit models
#   qm, qb = calibrate_split(w, stories)                                  the quantization (the BOS position separately)
# `python -m coral.codegen.layer` runs the acceptance test (test/test_codegen_layer.py), offline.
from __future__ import annotations
import struct, pathlib
import numpy as np
from coral.isa import f32, bits_f32, eltwise as EL
from coral.codegen import attention as A, attention_block as B, eltops as E

DM, HD, NH, DH = 288, 768, 6, 48
ROOT = pathlib.Path(__file__).resolve().parents[2]

# ***** the float model (llama2.c's TinyStories-15M in float64 numpy, batch 1, KV cache) *****
def load_checkpoint(path:pathlib.Path=ROOT / "models/stories15M.bin") -> tuple[dict, dict]:
  """karpathy's llama2.c checkpoint -> (config, float32 weights: emb, att_norm, wq, wk, wv, wo, ffn_norm, w1, w2, w3, norm, out)"""
  raw = path.read_bytes()
  dim, hidden, n_layers, n_heads, n_kv, vocab, seq_len = struct.unpack("7i", raw[:28])
  shared, vocab, hs = vocab > 0, abs(vocab), dim // n_heads
  arr, off = np.frombuffer(raw, np.float32, offset=28), 0
  def take(*shape):
    nonlocal off
    n = int(np.prod(shape))
    o = arr[off:off + n].reshape(shape)
    off += n
    return o
  w = dict(emb=take(vocab, dim), att_norm=take(n_layers, dim), wq=take(n_layers, dim, dim), wk=take(n_layers, n_kv * hs, dim),
           wv=take(n_layers, n_kv * hs, dim), wo=take(n_layers, dim, dim), ffn_norm=take(n_layers, dim), w1=take(n_layers, hidden, dim),
           w2=take(n_layers, dim, hidden), w3=take(n_layers, hidden, dim), norm=take(dim))
  take(seq_len, hs // 2)
  take(seq_len, hs // 2)
  w["out"] = w["emb"] if shared else take(vocab, dim)
  return dict(dim=dim, hidden=hidden, n_layers=n_layers, n_heads=n_heads, vocab=vocab, seq_len=seq_len), w

def folded(w:dict, l:int) -> dict[str, np.ndarray]:
  """layer l's float matrices as the chip uses them: RMSNorm(x) * g = (x / ||x||) * sqrt(288) * g, so g * sqrt(288) goes into the
  columns of the next FC (wqkv = wq | wk | wv rows, w1, w3); wo and w2 unchanged"""
  ga, gf = w["att_norm"][l].astype(np.float64) * np.sqrt(DM), w["ffn_norm"][l].astype(np.float64) * np.sqrt(DM)
  return dict(wqkv=np.concatenate([w["wq"][l], w["wk"][l], w["wv"][l]]) * ga[None], wo=w["wo"][l].astype(np.float64),
              w1=w["w1"][l] * gf[None], w3=w["w3"][l] * gf[None], w2=w["w2"][l].astype(np.float64))

def rope_f(x:np.ndarray, pos:int) -> np.ndarray:
  """llama2.c's RoPE of a [288] vector (interleaved pairs, per head of 48)"""
  ang = B.rope_angles(pos)
  o = np.empty_like(x)
  o[0::2] = x[0::2] * np.cos(ang[0::2]) - x[1::2] * np.sin(ang[0::2])
  o[1::2] = x[0::2] * np.sin(ang[0::2]) + x[1::2] * np.cos(ang[0::2])
  return o

class Ranges:
  """min / max of every recorded tensor (calibration)"""
  def __init__(self): self.r: dict[str, tuple[float, float]] = {}
  def __call__(self, key:str, x) -> None:
    x = np.asarray(x)
    lo, hi = self.r.get(key, (np.inf, -np.inf))
    self.r[key] = (min(lo, float(x.min())), max(hi, float(x.max())))

class FloatModel:
  """TinyStories-15M in float64 numpy (the reference): step(token, pos) -> logits, with the layers computed as the chip computes
  them in exact arithmetic (folded norm weights). rec: a Ranges that records every quantization point of every layer"""
  def __init__(self, w:dict, rec:Ranges|None=None):
    self.w, self.rec, self.L = w, rec, w["wq"].shape[0]
    self.F = [folded(w, l) for l in range(self.L)]
    self.K = np.zeros((self.L, 256, DM))
    self.V = np.zeros((self.L, 256, DM))
  def r(self, l:int, key:str, x):
    if self.rec is not None: self.rec(f"{l}.{key}", x)
  def layer(self, l:int, x:np.ndarray, pos:int) -> np.ndarray:
    F, r = self.F[l], lambda k, v: self.r(l, k, v)
    r("res", x)
    xn = x / np.linalg.norm(x)
    r("x", xn)
    h = F["wqkv"] @ xn
    q0, k0, v = h[:DM], h[DM:2 * DM], h[2 * DM:]
    r("qf", q0)
    r("kf", k0)
    r("v", v)
    ang = B.rope_angles(pos)
    cs, sn = np.cos(ang), np.where(np.arange(DM) % 2 == 0, -1.0, 1.0) * np.sin(ang)
    r("qc", q0 * cs)
    r("qc", q0[B.SWAP] * sn)
    r("kc", k0 * cs)
    r("kc", k0[B.SWAP] * sn)
    q, k = rope_f(q0, pos), rope_f(k0, pos)
    r("q", q)
    r("k", k)
    self.K[l, pos], self.V[l, pos] = k, v
    Kh, Vh = self.K[l, :pos + 1].reshape(-1, NH, DH), self.V[l, :pos + 1].reshape(-1, NH, DH)
    qh = q.reshape(NH, DH)
    r("qk", qh[None] * Kh)
    s = np.einsum("hc,phc->hp", qh, Kh)
    r("s", s)
    p = np.exp(s * DH ** -0.5 - (s * DH ** -0.5).max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    r("pv", p.T[:, :, None] * Vh)
    att = np.einsum("hp,phc->hc", p, Vh).reshape(DM)
    r("att", att)
    o = F["wo"] @ att
    r("o", o)
    h = x + o
    r("mid", h)
    hn = h / np.linalg.norm(h)
    r("hn", hn)
    h1, h3 = F["w1"] @ hn, F["w3"] @ hn
    r("h1", h1)
    r("h3", h3)
    a = h1 / (1 + np.exp(-h1))
    r("a", a)
    m = a * h3
    r("m", m)
    y = F["w2"] @ m
    r("y", y)
    out = h + y
    r("out", out)
    return out
  def hidden(self, tok:int, pos:int) -> np.ndarray:
    x = self.w["emb"][tok].astype(np.float64)
    for l in range(self.L): x = self.layer(l, x, pos)
    return x
  def logits(self, x:np.ndarray) -> np.ndarray:
    return self.w["out"] @ (x / np.sqrt((x * x).mean() + 1e-5) * self.w["norm"])
  def step(self, tok:int, pos:int) -> np.ndarray: return self.logits(self.hidden(tok, pos))

# ***** quantization ((scale, zero point) per tensor, per layer) *****
# a layer's dict holds attention_block.block_quant's keys (x = the attention L2 norm's output, wqkv, wo, qf, kf, v, cos, sin, qc, kc,
# q, k, qk, s, p, pv, att, o, beta) and: res (the layer input), mid (after the attention residual), out (the layer output = the next
# layer's res), hn (the FFN L2 norm's output), w1, w3, w2, h1, h3, g (LOGISTIC: (1/256, 0)), a, m, y
L2_OUT = (1 / 128, 128)          # TFLite's L2_NORMALIZATION output quantization (the op's own: any scale works, see l2_quant)
WEIGHTS = ("wqkv", "wo", "w1", "w3", "w2")

def calibrate(w:dict, stories:list[list[int]]) -> Ranges:
  """record every quantization point of the float model over teacher-forced token sequences"""
  rec = Ranges()
  for toks in stories:
    fm = FloatModel(w, rec)
    for pos, t in enumerate(toks[:256]): fm.hidden(t, pos)
  return rec

def quants_from(w:dict, rec:Ranges, l2_out:tuple=L2_OUT, margin:float=0.0) -> list[dict]:
  """per layer the full quantization from recorded ranges (every range widened by `margin` of its span on both sides; the weights
  from their own full range); x / hn: the L2 norms' output quantization l2_out"""
  from coral.fused import quantize_params
  L = w["wq"].shape[0]
  def qp(lo, hi):
    d = (hi - lo) * margin
    return quantize_params(lo - d, hi + d)
  out = []
  for l in range(L):
    F = folded(w, l)
    q = {k: qp(*rec.r[f"{l}.{k}"]) for k in ("res", "qf", "kf", "v", "qc", "kc", "q", "k", "qk", "s", "pv", "att", "o", "mid", "h1",
                                               "h3", "a", "m", "y", "out")}
    q |= {k: quantize_params(float(F[k].min()), float(F[k].max())) for k in WEIGHTS}
    q |= dict(x=l2_out, hn=l2_out, cos=(1 / 127, 128), sin=(1 / 127, 128), p=(1 / 256, 0), g=(1 / 256, 0), beta=DH ** -0.5)
    out.append(q)
  for l in range(1, L): out[l]["res"] = out[l - 1]["out"]     # one quantization of the residual between two layers
  return out

class Recorder(Ranges):
  """Ranges that record only while `on`"""
  def __init__(self):
    super().__init__()
    self.on = True
  def __call__(self, key:str, x) -> None:
    if self.on: super().__call__(key, x)

def float_stories(w:dict, n:int, T:int=200, seed:int=0, temperature:float=0.8) -> list[list[int]]:
  """n stories of the float model, sampled from BOS, each ending before the next BOS (the end of a story) or at T tokens"""
  rng, out = np.random.default_rng(seed), []
  for _ in range(n):
    fm, s = FloatModel(w), [1]
    for pos in range(T - 1):
      lg = fm.step(s[-1], pos) / temperature
      t = int(np.argmax(lg + rng.gumbel(size=lg.shape)))
      if t == 1: break
      s.append(t)
    out.append(s)
  return out

def calibrate_split(w:dict, stories:list[list[int]], l2_out:tuple=L2_OUT, margin:float=0.0) -> tuple[list[dict], list[dict]]:
  """(main, bos): the quantization of P >= 2 calibrated on every position but the BOS token, and of P = 1 (the BOS token at position 0)
  on that position alone. The BOS token drives a few residual channels to +-15 (massive activations in layers 2-5) while every
  other token stays within a few units: one quantization for both wastes most of the residual stream's 256 levels. Shared by both:
  the weights, and the K / V cache rows (k, v cover both, the P = 1 call writes row 0 for the later calls)."""
  rm, rb = Recorder(), Recorder()
  for toks in stories:
    fm = FloatModel(w, rm)
    for pos, t in enumerate(toks[:256]):
      fm.rec = rb if pos == 0 else rm
      fm.hidden(t, pos)
  for l in range(w["wq"].shape[0]):
    for k in ("k", "v"):
      (a0, a1), (b0, b1) = rm.r[f"{l}.{k}"], rb.r[f"{l}.{k}"]
      rm.r[f"{l}.{k}"] = rb.r[f"{l}.{k}"] = (min(a0, b0), max(a1, b1))
  qm, qb = quants_from(w, rm, l2_out, margin), quants_from(w, rb, l2_out, margin)
  for a, b in zip(qm, qb):
    for k in WEIGHTS: b[k] = a[k]
  return qm, qb

def quant_weights(w:dict, quants:list[dict]) -> list[dict[str, np.ndarray]]:
  """the uint8 weight matrices of every layer (folded, quantized per tensor with the layer's quantization)"""
  from coral.fused import quant
  return [{k: quant(folded(w, l)[k], *quants[l][k]) for k in WEIGHTS} for l in range(len(quants))]

# ***** numpy bit models of the pieces this module adds (the attention block's, MUL's and ADD's are attention_block's) *****
def l2_quant(s_in:float, out_q:tuple=L2_OUT) -> dict:
  """L2_NORMALIZATION's second op for any output quantization: mult = f32(f32(s * f32(1/K)) * f32(1/s_out)) (for (1/128, 128)
  exactly eltops.l2norm_quant's), clamps = out_clamps(s_out, zp_out); the first op (sum of squares) and the NLU are unchanged"""
  s = f32(s_in)
  lo, hi = EL.out_clamps(out_q[0], out_q[1])
  return dict(sq=EL.mul32(s, s), mult=EL.mul32(EL.mul32(s, EL.recip32(E.rsqrt_scale(s))), EL.recip32(out_q[0])), clamp_min=lo,
              clamp_max=hi, out_zp=out_q[1])

def l2norm_ref(xq:np.ndarray, in_q:tuple, out_q:tuple=L2_OUT) -> np.ndarray:
  """eltops.l2norm_ref with any output quantization: acc = sum (x - zp)^2; v = max(f32(acc) f32(s^2), f32(s^2)); r = NLU rsqrt (16
  bit); y = clamp(rne(f32((x - zp) r) * mult), lo, hi) + zp_out"""
  F = np.float32
  q = l2_quant(in_q[0], out_q)
  d = np.asarray(xq).astype(np.int64).reshape(-1) - in_q[1]
  v = np.maximum(F(float((d * d).sum())) * F(q["sq"]), F(q["sq"]))
  r = int(E.rsqrt_nlu(np.array([v], F), in_q[0])[0])
  y = np.clip((d * r).astype(F) * F(q["mult"]), F(q["clamp_min"]), F(q["clamp_max"]))
  return (np.rint(y).astype(np.int64) + q["out_zp"]).astype(np.uint8)

LOG_C = np.array(EL.LOGISTIC_SLOTS[:40], np.uint32).view(np.float32).reshape(8, 5)
LOG_B = np.array(EL.LOGISTIC_SLOTS[40:47], np.uint32).view(np.float32)
LOG_CLAMP = np.float32(bits_f32(EL.LOGISTIC_CLAMP))

def logistic_ref(xq:np.ndarray, in_q:tuple, out_zp:int=0) -> np.ndarray:
  """LOGISTIC (output scale 1/256): x = clamp(f32(q - zp) * f32(s), -c, c), c = 10.396732; the NLU's segment s covers
  [b_{s-1}, b_s); p = c0 + x(c1 + x(c2 + x(c3 + x c4))) in float32 (each product and sum rounded); y = clamp(rne(p) + zp_out, 0, 255)"""
  F = np.float32
  x = np.clip((np.asarray(xq).astype(np.int64) - in_q[1]).astype(F) * F(in_q[0]), -LOG_CLAMP, LOG_CLAMP)
  c = LOG_C[np.searchsorted(LOG_B, x, side="right")]
  p = c[..., 4]
  for j in (3, 2, 1, 0): p = (p * x + c[..., j]).astype(F)
  return np.clip(np.rint(p).astype(np.int64) + out_zp, 0, 255).astype(np.uint8)

def ffn_ref(hn:np.ndarray, W:dict, q:dict, parts:bool=False):
  """the FFN's bit model: h1 = FC(w1), g = LOGISTIC(h1), a = MUL(h1, g), h3 = FC(w3), m = MUL(a, h3), y = FC(w2)"""
  h1 = B.fc_ref(hn, W["w1"], q["hn"], q["w1"], q["h1"])
  g = logistic_ref(h1, q["h1"], q["g"][1])
  a = A.mul_ref(h1, g, q["h1"], q["g"], q["a"])
  h3 = B.fc_ref(hn, W["w3"], q["hn"], q["w3"], q["h3"])
  m = A.mul_ref(a, h3, q["a"], q["h3"], q["m"])
  y = B.fc_ref(m, W["w2"], q["m"], q["w2"], q["y"])
  return (y, dict(h1=h1, g=g, a=a, h3=h3, m=m)) if parts else y

# ***** RoPE with permuted rows: swap(q) without a second FC *****
# llama2.c rotates the pairs (2j, 2j+1): q' = q cos + swap(q) sin' (attention_block's form, sin' carrying the sign). The q and k FCs
# compute their outputs in the order SIGMA: per group of 16 outputs the first elements of 8 consecutive pairs, then their second
# elements (16 divides 48 and 64: every group lies inside one head and one tile's chunk). swap(q) is then the half swap of every
# 16-byte group (SWAPS), which MUL step 1 reads with the walk (1, 8), (-8, 2), (16, groups) from base + 8: its only negative
# increment (-15) is of the kind the compiler's rewinds use. The attention is unchanged: q and k carry the same permutation inside
# every head, so every score sums the same products; the KV cache holds k' in this order, cos / sin' come permuted.
SIGMA = np.array([2 * (8 * (p // 16) + p % 8) + int(p % 16 >= 8) for p in range(DM)])
SWAPS = np.arange(DM) ^ 8                # the partner of position p in the permuted order: q[SIGMA][SWAPS] = swap(q)[SIGMA]

def rope_inputs(pos:int, quant:dict|None=None) -> tuple[np.ndarray, np.ndarray]:
  """(cos, sin') [288] uint8 of position pos in the permuted order (attention_block.rope_inputs[SIGMA])"""
  c, s = B.rope_inputs(pos, quant)
  return c[SIGMA], s[SIGMA]

def attention_ref(xn:np.ndarray, cos:np.ndarray, sin:np.ndarray, K:np.ndarray|None, V:np.ndarray|None, W:dict, q:dict):
  """the attention half's bit model (attention_block_ref with q, k in the order SIGMA): -> (o, k', v, parts)"""
  bq = B.block_quant(q)
  Wq, Wk, Wv = W["wqkv"][:DM][SIGMA], W["wqkv"][DM:2 * DM][SIGMA], W["wqkv"][2 * DM:]
  qf, kf = B.fc_ref(xn, Wq, bq["x"], bq["wqkv"], bq["qf"]), B.fc_ref(xn, Wk, bq["x"], bq["wqkv"], bq["kf"])
  v = B.fc_ref(xn, Wv, bq["x"], bq["wqkv"], bq["v"])
  qc, qs = A.mul_ref(qf, cos, bq["qf"], bq["cos"], bq["qc"]), A.mul_ref(qf[SWAPS], sin, bq["qf"], bq["sin"], bq["qs"])
  kc, ks = A.mul_ref(kf, cos, bq["kf"], bq["cos"], bq["kc"]), A.mul_ref(kf[SWAPS], sin, bq["kf"], bq["sin"], bq["ks"])
  q2, k2 = B.add_ref(qc, qs, bq["qc"], bq["qs"], bq["q"]), B.add_ref(kc, ks, bq["kc"], bq["ks"], bq["k"])
  Kp = np.concatenate([np.asarray(K, np.uint8).reshape(-1, DM), k2[None]]) if K is not None and len(K) else k2[None]
  Vp = np.concatenate([np.asarray(V, np.uint8).reshape(-1, DM), v[None]]) if V is not None and len(V) else v[None]
  att, ap = A.attention_ref(q2, Kp, Vp, B.core_quant(bq), parts=True)
  o = B.fc_ref(att, W["wo"], bq["att"], bq["wo"], bq["o"])
  return o, k2, v, dict(qf=qf, kf=kf, qc=qc, qs=qs, q=q2, k=k2, v=v, att=att, o=o, **ap)

def layer_ref(x:np.ndarray, cos:np.ndarray, sin:np.ndarray, K:np.ndarray|None, V:np.ndarray|None, W:dict, q:dict, parts:bool=False):
  """one layer's bit model: x [288] uint8 (q['res']), cos / sin' (rope_inputs: permuted), K / V the layer's cache rows [P-1][288] (k'
  in the order SIGMA, as the program writes them; None at P = 1), W the layer's uint8 weights (quant_weights, natural order) ->
  (x_out [288], k' [288] (order SIGMA), v [288])"""
  x = np.asarray(x, np.uint8).reshape(DM)
  xn = l2norm_ref(x, q["res"], q["x"])
  o, k2, v, ap = attention_ref(xn, cos, sin, K, V, W, q)
  h = B.add_ref(o, x, q["o"], q["res"], q["mid"])                 # operand 1 = o (the op's in_base), operand 2 = x
  hn = l2norm_ref(h, q["mid"], q["hn"])
  y, fp = ffn_ref(hn, W, q, parts=True)
  out = B.add_ref(y, h, q["y"], q["mid"], q["out"])               # operand 1 = y, operand 2 = h
  if parts: return (out, k2, v), dict(x=x, xn=xn, h=h, hn=hn, y=y, out=out, **ap, **fp)
  return out, k2, v

def model_ref(x:np.ndarray, cos:np.ndarray, sin:np.ndarray, Ks:list, Vs:list, Ws:list[dict], quants:list[dict], parts:bool=False):
  """all layers: x = the quantized embedding (quants[0]['res']), Ks / Vs per layer the cache rows [P-1][288] (None at P = 1)
  -> (x_out [288] (quants[-1]['out']), kv [L][576] = k' | v per layer)"""
  kv, ps = [], []
  for l, (W, q) in enumerate(zip(Ws, quants)):
    (x, k2, v), p = layer_ref(x, cos, sin, Ks[l], Vs[l], W, q, parts=True)
    kv.append(np.concatenate([k2, v]))
    ps.append(p)
  return (x, np.stack(kv), ps) if parts else (x, np.stack(kv))

class ChipModel:
  """the whole model as the chip computes it (model_ref, bit-exact with gen_model's program), with the host's part: the quantized
  embedding in, the uint8 KV cache, the output dequantized, the final RMSNorm and the classifier in float. step(token, pos) -> logits"""
  def __init__(self, w:dict, qm:list[dict], qb:list[dict], Ws:list[dict]):
    self.w, self.qm, self.qb, self.Ws, self.L = w, qm, qb, Ws, len(qm)
    self.K = np.zeros((self.L, 256, DM), np.uint8)
    self.V = np.zeros((self.L, 256, DM), np.uint8)
  def quants(self, P:int) -> list[dict]: return self.qb if P == 1 else self.qm
  def inputs(self, tok:int, pos:int) -> tuple:
    q = self.quants(pos + 1)
    x = np.clip(np.rint(self.w["emb"][tok] / q[0]["res"][0]) + q[0]["res"][1], 0, 255).astype(np.uint8)
    cos, sin = rope_inputs(pos, q[0])
    def rows(C): return [C[l, :pos] if pos else None for l in range(self.L)]
    return x, cos, sin, rows(self.K), rows(self.V)
  def finish(self, pos:int, x_out:np.ndarray, kv:np.ndarray) -> np.ndarray:
    """append this token's k', v to the cache, dequantize x_out, final RMSNorm + classifier -> logits"""
    self.K[:, pos], self.V[:, pos] = kv[:, :DM], kv[:, DM:]
    qo = self.quants(pos + 1)[-1]["out"]
    x = (x_out.astype(np.float64) - qo[1]) * qo[0]
    return self.w["out"] @ (x / np.sqrt((x * x).mean() + 1e-5) * self.w["norm"])
  def step(self, tok:int, pos:int) -> np.ndarray:
    x, cos, sin, Ks, Vs = self.inputs(tok, pos)
    return self.finish(pos, *model_ref(x, cos, sin, Ks, Vs, self.Ws, self.quants(pos + 1)))

def evaluate(w:dict, chip:"ChipModel|callable", stories:list[list[int]]) -> dict:
  """teacher-forced quality of the chip's model against the float model on token sequences (stories the float model wrote, kept out of
  the calibration): perplexity of the sequences' own next tokens under each, and how often the chip's argmax is the float model's.
  chip: a ChipModel factory (called once per story)"""
  def nll(lgs, s):
    lg = np.stack(lgs)[:-1]
    lg = lg - lg.max(-1, keepdims=True)
    return list(np.log(np.exp(lg).sum(-1)) - lg[np.arange(len(s) - 1), s[1:]])
  nf, nc, agree = [], [], []
  for s in stories:
    fm, cm = FloatModel(w), chip()
    lf = [fm.step(t, p) for p, t in enumerate(s)]
    lc = [cm.step(t, p) for p, t in enumerate(s)]
    nf += nll(lf, s)
    nc += nll(lc, s)
    agree += [int(np.argmax(a) == np.argmax(b)) for a, b in zip(lc, lf)]
  return dict(float_ppl=float(np.exp(np.mean(nf))), chip_ppl=float(np.exp(np.mean(nc))), agree=float(np.mean(agree)), tokens=len(agree))

# ***** program geometry *****
from coral.isa import cdiv, round_up, op as OP, wide_narrow as WN, ring_mesh as RM, scalar as SC
from coral.codegen import Emitter, caching_program, WIDE_TOP, WIDE_OUT_FIFO
from coral.codegen import fc as FC, conv as CC

G288, G768, G2 = FC.FCGeom(DM, DM), FC.FCGeom(HD, DM), FC.FCGeom(DM, HD)
MATS = ("q", "k", "v", "wo", "w1", "w3", "w2")                 # per layer, in blob order
GEOM = dict(q=G288, k=G288, v=G288, wo=G288, w1=G768, w3=G768, w2=G2)
QKEYS = dict(q=("x", "wqkv", "qf"), k=("x", "wqkv", "kf"), v=("x", "wqkv", "v"), wo=("att", "wo", "o"), w1=("hn", "w1", "h1"),
             w3=("hn", "w3", "h3"), w2=("m", "w2", "y"))      # (input, weights, output) quantization keys of each matmul
IDENT = WIDE_TOP - 4                                            # the 4x4 identity row (MUL step 2, LOGISTIC, copies), the whole program
# wide FIFOs (64-byte units) of verified programs, used at the same places: the L2 norm's (eltops.gen_l2norm) and the FFN FC form's
# broadcasts of its input to tiles 1..11 and of m to tiles 1..4 (fused.FC_WIDE fwd3 / fwd2)
L2FIFO, L2FIFO2, HFIFO, MFIFO = E.ROW_FIFO, E.ROW_FIFO2, 0x1f70 - 4, 0x1f70

def parts_1d(n:int) -> list[tuple[int, dict, int]]:
  """an n-byte vector in edgetpu_compiler's 1-D layout (tile t holds bytes [64t, 64t + 64) at base + 64t): per tile (tile mask,
  eltwise geometry, byte offset), the PARTS of attention_block / fused's FC form"""
  return [(1 << t, EL.tile_geometry([1, n], t), 64 * t) for t in range(cdiv(n, 64))]

def chunk_tiles(n:int) -> int: return (1 << cdiv(n, 64)) - 1

# ***** stages: every one an instruction sequence that ran on the device (per-tile ops in the compiler's 1-D layout) *****
def group_tiles(per:list[dict]) -> list[tuple[int, dict]]:
  """tiles whose op fields are equal but for the tile mask share one instruction (in the order of their lowest tile)"""
  return A._group([{k: v for k, v in f.items() if k != "tile_mask"} for f in per])

def fc_spread(e:Emitter, g:FC.FCGeom, X:int, Y:int, off:int, q:dict, tail:bool=False, local:bool=False):
  """fc.ops_block: the compute fence, the bias loads, one FC op per tile (tile t's outputs at Y + 64t), reset17 (tail: + fence).
  local: every tile writes its outputs at Y and tiles with equal fields share one op, as edgetpu_compiler emits an FC feeding a
  per-tile op (one op for tiles 0..11 of a 768-output FC, writing narrow 0 on each)"""
  if not local: return FC.ops_block(e, g, FC.Place(), X, Y, off, q, tail)
  e.sync(SC.signal_fence)
  for n in sorted({g.gt(t) for t in range(g.T)}, reverse=True):
    e.tile(WN.w2n_bias(sum(1 << t for t in range(g.T) if g.gt(t) == n), off, n, e.seq))
  per = [OP.fc_tile_fields(g.N, g.K, t, 0, X, Y, g.bias_rows(t) + off, q["w_zp"], q["in_zp"], q["mult"], q["out_zp"], q.get("clamp_min"),
                           q.get("clamp_max")) for t in range(g.T)]
  for m, f in group_tiles(per): e.tile(OP.encode_op(**(f | dict(tile_mask=m, seq=e.seq))))
  e.sync(SC.sync_reset17)
  if tail: e.sync(SC.signal_fence)

# streamed FULLY_CONNECTED (copied from coral/codegen/fused.py's _stream_op / _stream_stage / _fills, which are private there):
# every 64-output group streams from its parameter tile (any tile, any offset: one ringProducer per group) into the compute tile's
# 32-row weight FIFO every call
STREAM_SLOTS, STREAM_FIFO, STREAM_BIAS = 32, 8064, 8056
def _fills(kw:int) -> tuple[int, int]: return cdiv(kw, STREAM_SLOTS), kw - STREAM_SLOTS * (cdiv(kw, STREAM_SLOTS) - 1)

def _stream_op(g:FC.FCGeom, t:int, seq:int, X:int, Y:int, q:dict) -> dict:
  """(fused._stream_op) the FC op of tile t reading its weights from the FIFO in b fills; outputs at Y + 64 P t"""
  kw, D, (b, r) = g.kw, STREAM_SLOTS, _fills(g.kw)
  if r not in (D, 8): raise NotImplementedError(f"streamed FC with a partial last FIFO fill of {r} rows (verified: 32 and 8)")
  f = OP.fc_tile_fields(g.N, g.K, t, seq, X, Y + 64 * g.P * t, 0, q["w_zp"], q["in_zp"], q["mult"], q["out_zp"], q.get("clamp_min"),
                        q.get("clamp_max"))
  f |= dict(loop0=D - 1, loop2=b - 1, loop3=0, **FC.ttu("in_", [1, kw, D], [D, 1, b], 8), **FC.ttu("par_", [1, 0, 0], [D, 1, b], 8),
            **FC.ttu("psum_", [0, 1, 0], [D, 1, b], 8), **OP.wide("par", STREAM_FIFO), par_tflags=0x80, par_fifo=D, cfg1=0x14F, sync2=0xB6,
            sync3=0x80, sync4=0x80, psum_hmode=0, psum_tflags=0)
  if r != D:
    f |= dict(rsv174=0x8000 | (r - 1), in_twait=0x2000, rsv536=(D - r) << 22 | 1 << 18 | 3, par_fifo=D | 0x2000,
              rsv1230=(D - r) << 18 | 1 << 14 | 3, rsv1523=(r - 1) << 9 | 1 << 24)
  return f

def fc_streamed(e:Emitter, g:FC.FCGeom, X:int, Y:int, pieces:list[tuple[int, int]], q:dict, bias:int=STREAM_BIAS, tail:bool=False,
                local:bool=False):
  """(fused._stream_stage) compute fence, drain, per compute tile its weight-FIFO ringConsumer1, the bias load, one op per tile, then
  group t (bias row + kw weight rows) streams from pieces[t] = (parameter tile, offset in 64-byte units) to tile t; wait for all rows"""
  D, per, (b, r) = STREAM_SLOTS, 1 + g.kw, _fills(g.kw)
  assert g.P == 1 and len(pieces) == g.G
  e.sync(SC.signal_fence)
  e.sync(SC.sync_drain)
  for t in range(g.T):
    e.tile(RM.encode_ringConsumer(opcode=0x12, tile_mask=1 << t, seq=e.seq, addr=STREAM_FIFO, aux_en0=1, aux_en1=1, s_en_a=1, s_en_b=1,
                                  s_y=1, sdims=1, cbuf=1, mode=3, s_id=(1 << 5) | 1, slots=D, aux_addr0=bias + 2, aux_addr1=bias + 4, s_val=-1,
                                  **FC.ttu("", [1, 0, 0, 0], [D, b, 1, 1], 4), **(dict(grp=r - 1, gstride=4 * (D - r) + 1) if r != D else {})))
  e.tile(CC.bias_load(e.seq, (1 << g.T) - 1, bias, b))
  for t in range(g.T): e.tile(OP.encode_op(**(_stream_op(g, t, e.seq, X, Y, q) | (dict(out_base=Y) if local else {}))))
  for t, (pt, po) in enumerate(pieces): e.tile(CC.rprod_param(e.seq, pt, po, per, per * t, dest=1 << t))
  e.sync(SC.sync_reset17)
  e.sync(SC.broadcast_wait, per * g.G)
  if tail: e.sync(SC.signal_fence)

def fc_stage(e:Emitter, g:FC.FCGeom, X:int, Y:int, pl:tuple, q:dict, tail:bool=False, local:bool=False):
  """pl = ("spread", offset units: the groups resident on their compute tiles) or ("stream", [(tile, offset units)] per group)"""
  if pl[0] == "spread": fc_spread(e, g, X, Y, pl[1], q, tail, local)
  else: fc_streamed(e, g, X, Y, pl[1], q, tail=tail, local=local)

def parts_local(n:int) -> list[tuple[int, dict]]:
  """an n-byte vector in the LOCAL 1-D layout (tile t's bytes [64t, 64t + 64) at the same address on every tile): (tile mask,
  eltwise geometry) per run of tiles with equal geometry (the compiler's geometry: an op differs from the 1-D form in the addresses)"""
  out: list[list] = []
  for m, tg, _ in parts_1d(n):
    if out and out[-1][1] == tg: out[-1][0] |= m
    else: out.append([m, tg])
  return [(m, tg) for m, tg in out]

def swap_walk(w:int) -> dict:
  """MUL step 1's in TTU reading a chunk of w words in SWAPS order (the halves of every 16-byte group exchanged), from base + 8"""
  return OP.ttu_fine("in", [1, -8, 16], [8, 2, 4 * w // 16])

def mul_stage(e:Emitter, n:int, a:int, b:int, tmp:int, out:int, fifo:int, q:dict, swap:bool=False, local:bool=False, b_global:bool=False):
  """out = MUL(a, b) of two n-byte vectors in the 1-D layout: attention_block.mul_1d / fused._mul_stage (FC form), one op per tile and
  step: step 1 (a through the in TTU; swap: a read in SWAPS order, i.e. MUL(a[SWAPS], b)), the FIFO feeds of b, step 2.
  local: a, out (and b unless b_global) in the local layout, tiles of equal geometry share each op and feed"""
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_reset_n2w)
  e.sync(SC.sync_reset_par)
  def geo(tg): return (tg["w"], tg["cols"], tg["rows"])
  if local:
    for m, tg in parts_local(n):
      f = EL.mul_op1_fields(*geo(tg), tg["R"], m, e.seq, a, tmp, fifo, **q)
      if swap: f |= swap_walk(tg["w"]) | dict(in_base=a + 8)
      e.tile(OP.encode_op(**f))
    feeds = [(m, tg, b + o) for m, tg, o in parts_1d(n)] if b_global else [(m, tg, b) for m, tg in parts_local(n)]
    for m, tg, bb in feeds: e.tile(WN.encode_narrow_to_wide(**EL.mul_feed_n2w_fields(e.seq, m, bb, fifo, *geo(tg))))
    e.sync(SC.sync_wn_fence)
    for m, tg in parts_local(n): e.tile(OP.encode_op(**EL.mul_op2_fields(*geo(tg), tg["R"], m, e.seq, tmp, out, IDENT)))
    e.sync(SC.sync_reset_n2w)
    e.sync(SC.sync_reset_par)
    e.sync(SC.sync_reset17)
    return
  for m, tg, o in parts_1d(n):
    f = EL.mul_op1_fields(*geo(tg), tg["R"], m, e.seq, a + o, tmp, fifo, **q)
    if swap: f |= swap_walk(tg["w"]) | dict(in_base=a + o + 8)
    e.tile(OP.encode_op(**f))
  for m, tg, o in parts_1d(n): e.tile(WN.encode_narrow_to_wide(**EL.mul_feed_n2w_fields(e.seq, m, b + o, fifo, *geo(tg))))
  e.sync(SC.sync_wn_fence)
  for m, tg, o in parts_1d(n): e.tile(OP.encode_op(**EL.mul_op2_fields(*geo(tg), tg["R"], m, e.seq, tmp, out + o, IDENT)))
  e.sync(SC.sync_reset_n2w)
  e.sync(SC.sync_reset_par)
  e.sync(SC.sync_reset17)

def logistic_stage(e:Emitter, n:int, src:int, dst:int, in_q:tuple, out_q:tuple, a:dict|None=None, local:bool=False):
  """(fused._logistic_stage, private there, FC form) dst = LOGISTIC(src): reset17, [the identity row: as edgetpu_compiler's program
  for L2_NORMALIZATION -> FULLY_CONNECTED -> LOGISTIC and the FFN FC form, right before the op fence], op fence, the NLU's spline (all
  chunk tiles), one op per tile, reset17"""
  e.sync(SC.sync_reset17)
  if a is not None: e.tile(*EL.ident_prologue(e.seq, 0xffff, a["C"], a["const"]))
  e.sync(SC.sync_op_fence)
  e.tile(EL.encode_nlu(**EL.nlu_fields(chunk_tiles(n), e.seq)))
  for m, tg, o in ([(m, tg, 0) for m, tg in parts_local(n)] if local else parts_1d(n)):
    e.tile(OP.encode_op(**EL.logistic_op_fields(tg["P"], tg["w"], tg["R"][0], tg["R"][0], m, e.seq, src + o, dst + o, IDENT,
                                                **EL.logistic_quant(in_q, out_q[1]), flat=tg["flat"])))
  e.sync(SC.sync_reset17)

def add_stage(e:Emitter, n:int, a:int, b:int, out:int, wn:int, ww:int, q:dict, local:bool=False, b_global:bool=False):
  """out = ADD(a, b) (attention_block.add_1d for n bytes): two reset17, the 16-bit weight matrix (w1, w2) by 10 mesh fills at narrow wn
  and a narrowToWide to wide ww, one op per tile (operand 1 at a + 64t, operand 2 at b + 64t), reset17"""
  e.sync(SC.sync_reset17)
  e.sync(SC.sync_reset17)
  tiles = chunk_tiles(n)
  e.tile(*EL.add_weight_fills(e.seq, tiles, wn, q["w1"], q["w2"]))
  e.tile(WN.encode_narrow_to_wide(**EL.add_weight_n2w_fields(e.seq, tiles, wn, ww)))
  assert b > a and (b - a) % 4 == 0
  if local:     # operand 1 and the output local, operand 2 local (one op per tile group) or global (b_global: one op per tile)
    per = [(m, tg, 0, 64 * t) for t, (m, tg, _) in enumerate(parts_1d(n))] if b_global else [(m, tg, 0, 0) for m, tg in parts_local(n)]
  else: per = [(m, tg, o, 0) for m, tg, o in parts_1d(n)]
  for m, tg, o, bo in per:
    e.tile(OP.encode_op(**EL.add_op_fields(tg["w"], tg["cols"], tg["rows"], tg["R"], (b + bo - a) // 4, tg["R"], m, e.seq, a + o, out + o, ww,
                                           q["offset"], q["mult"], q["out_zp"], q["clamp_min"], q["clamp_max"])))
  e.sync(SC.sync_reset17)

def gather(e:Emitter, n:int, X:int, R:int, local:bool=False):
  """the 1-D-layout n-byte vector (tile t's chunk at X + 64t) onto tile 0 at X..X+n: fc.gather (the FULLY_CONNECTED input gather);
  local: tile t's chunk at X (gather_local)"""
  if local: return gather_local(e, n, X, R)
  FC.gather(e, FC.FCGeom(DM, n), X, R, lambda t: 1 << t)

def gather_local(e:Emitter, n:int, X:int, R:int):
  """the local-layout n-byte vector (tile t's chunk at narrow X) onto tile 0 at X..X+n: fc.gather's mesh phases (west within each row
  of 4 tiles, then north along column 0, relays through R) with every sender reading its chunk at X and row r's first tile collecting
  its row contiguously at X (west receivers write X + 64k for the row's k-th tile, north receivers X + 256r)"""
  g = FC.FCGeom(n, n)
  rows = [list(range(4 * r, min(4 * r + 4, g.T_in))) for r in range(cdiv(g.T_in, 4))]
  def mask(ts): return sum(1 << t for t in ts)
  def words(nbytes): return cdiv(nbytes, 4)
  e.sync(SC.sync_reset17)
  cum, last = 0, 0xffff
  for k in range(1, min(4, g.T_in)):
    rs = [r for r in rows if len(r) > k]
    for r in rs: e.tile(FC._mesh(0x16, mask([r[k]]), e.seq, out=(X, words(g.piece(r[k])))))
    if k >= 2:
      classes: list[tuple[int, list]] = []
      for r in rs:
        if classes and classes[-1][0] == words(g.piece(r[k])): classes[-1][1].append(r)
        else: classes.append((words(g.piece(r[k])), [r]))
      for w, crs in classes: e.tile(FC.relay(0x16, mask([t for r in crs for t in r[1:k]]), e.seq, R, w, cum + 1))
    for r in rs: e.tile(FC._mesh(0x16, mask([r[0]]), e.seq, inn=(X + g.b * k, words(g.piece(r[k])))))
    if k >= 2:
      pk = cdiv(classes[0][0], 4)
      cum += pk
      last = 0xffff & ~mask([t for w, crs in classes if cdiv(w, 4) == pk for r in crs for t in r[1:k]])
    e.sync(SC.sync_mesh, 0x16, last, cum + 1)
  e.sync(SC.sync_drain, last)
  e.sync(SC.sync_drain)
  cum = 0
  for k in range(1, len(rows)):
    nw = words(sum(g.piece(t) for t in rows[k]))
    e.tile(FC._mesh(0x17, mask([4 * k]), e.seq, out=(X, nw), mode=0))
    if k >= 2: e.tile(FC.relay(0x17, mask([4 * j for j in range(1, k)]), e.seq, R, nw, cum))
    e.tile(FC._mesh(0x17, mask([0]), e.seq, inn=(X + g.b * 4 * k, nw), mode=0))
    if k >= 2:
      cum += cdiv(nw, 4)
      last = 0xffff & ~mask([4 * j for j in range(1, k)])
      e.sync(SC.sync_mesh, 0x17, last, cum)
  if len(rows) >= 3: e.sync(SC.sync_drain, last)
  e.sync(SC.sync_reset_mesh)

def output_block(e:Emitter, g:FC.FCGeom, Y:int, local:bool=False):
  """fc.output_block (one output vector from the compute tiles, one DMA); local: every tile's outputs at Y"""
  if not local: return FC.output_block(e, g, FC.Place(), Y)
  e.scalar(SC.output_dma_head())
  for t in range(g.T): e.tile(WN.n2w_output(g.n_out(t), 1 << t, Y // 4, WIDE_OUT_FIFO, e.seq))
  e.scalar(SC.output_dma_tail(g.out_bytes))
  for t in range(g.T):
    e.scalar(SC.outfeed(8 * cdiv(g.n_out(t), 8)))
    e.tile(FC.rprod_output(g.n_out(t), 1 << t, t, e.seq))
  e.scalar(SC.output_wait(e.seq, g.T), seqs=2)

def bcast(e:Emitter, n:int, X:int, dest:int, fifo:int):
  """the n-byte vector at narrow X of tile 0 to the tiles dest (fc.broadcast, with its reset17 / scalar fences around it)"""
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_reset17)
  FC.broadcast(e, FC.FCGeom(64, n), X, 1, dest, fifo)
  e.sync(SC.sync_reset17)

def ident(e:Emitter, a:dict):
  """the 4x4 identity row: 4 mesh fills at narrow C, narrowToWide to wide IDENT (EL.ident_prologue, as the verified block at P = 1)"""
  e.tile(*EL.ident_prologue(e.seq, 0xffff, a["C"], a["const"]))
  e.sync(SC.sync_reset17)

# ***** memory *****
def narrow_sizes(g:A.Geom, L:int) -> dict:
  """narrow buffers (bytes, every tile), in address order. 1-D-layout vectors take the whole vector (tile t uses [64t, 64t + 64);
  tile 0 holds it all once gathered, every tile once broadcast). XG / Xc / Xs: the three inputs; Q: q' | k' | v of every layer (they
  stay until the outputs); XR: the residual stream between layers; an ADD's operand 2 sits above its operand 1 (Yo < XG, XR;
  Y2 < H); QC / KC hold the RoPE products qc | qs 288 bytes apart (add_1d's distance); the 288-byte FC inputs XN, HN, O take 384
  bytes: a streamed FC's op walks its input in fills of 32 words (the last one partial)"""
  core = A.narrow_sizes(g)
  return dict(Stg=core["Stg"], C=16, Yo=320, Y2=320, Xc=320, Xs=320, R=192, Z=16, XN=384, HN=384, Yq=320, Yk=320, Mr=256, QC=2 * DM,
              KC=2 * DM, Wm=64, Q=3 * DM * L, **{k: core[k] for k in ("XK", "XV", "Ms", "Ys", "S", "Pn", "Mv", "Yv")}, O=384, H1=HD,
              G=HD, Af=HD, H3=HD, Mm=HD, H=320, XG=320, XR=320)

def model_alloc(P:int, L:int) -> dict:
  """our memory plan, nothing live at the same time shares memory. Narrow: narrow_sizes in order, 64-byte aligned. Wide (64-byte
  units, from the top): the identity row, the output FIFO, 8 units (the three inputs' ring FIFO [8304, 8320) before the identity row
  is loaded), then the attention core's FIFOs and relay block, the ADD weight rows, the broadcast FIFOs (attention_block.block_alloc's
  order). Fixed: the L2 norms' FIFOs (eltops'), the FFN broadcast FIFOs (the FFN FC form's), the streamed FCs' weight FIFO
  [8064, 8192) and bias rows [8056, 8064); they overlap other stages' buffers in space only (every stage ends with a drain).
  Scalar memory: the attention core's scores and p."""
  g = A.Geom(P)
  a, at = {}, 0
  for name, size in narrow_sizes(g, L).items():
    a[name] = at
    at += round_up(size, 64)
  a["narrow_end"] = at
  wt = WIDE_TOP
  for name, size in [("const", 4), ("outfifo", 4), ("unused", 8), ("in_kv", 8 * A.g4_of(P).c_in), ("mulfifo", 32), ("sumfifo", 16),
                     ("smfifo", 8), ("pfifo", 4), ("bfifo", 4), ("relay", 4 * 2 * cdiv(P, 4)), ("addw", 16), ("qkvfifo", 4), ("ofifo", 4),
                     ("xfifo", 4)]:
    wt -= size
    a[name] = wt
  a |= dict(l2fifo=L2FIFO, l2fifo2=L2FIFO2, hfifo=HFIFO, mfifo=MFIFO)
  a["wide_bottom"] = min(wt, STREAM_BIAS, HFIFO, L2FIFO2)
  a["M"] = P
  a["p_smem_w"] = A.SMEM_TOP // 4 - 6 * P
  a["s_smem_w"] = a["p_smem_w"] - 6 * P
  assert a["const"] == IDENT and a["outfifo"] == WIDE_OUT_FIFO and at <= 192 * 1024 and a["s_smem_w"] >= 0
  assert a["unused"] >= L2FIFO + 16 and L2FIFO2 + 4 <= L2FIFO and MFIFO + 4 <= STREAM_BIAS
  # every FIFO of more than one wide row starts at a multiple of 8 units, as in every program that ran (the first composed program,
  # whose L2 FIFO sat at 8172, hung on the device)
  assert all(a[k] % 8 == 0 for k in ("in_kv", "mulfifo", "sumfifo", "l2fifo")) and STREAM_FIFO % 8 == 0
  return a

def model_limit(P:int=256) -> int:
  """the lowest wide unit the execution program writes on any tile: resident parameters must end at or below it (the floor over every
  P = 1..256 is P = 256's)"""
  return min(model_alloc(p, 1)["wide_bottom"] for p in (1, P))

# ***** placement of the parameters *****
def plan(L:int=6, limit:int|None=None, stream:set|None=None) -> list[dict[str, tuple]]:
  """our placement of every matmul of L layers -> per layer {matmul: ("spread", offset units) | ("stream", [(tile, offset units)] per
  group)}. The FCs always compute on tiles 0..T-1 (q, k, v, wo, w2 on 0..4, w1 / w3 on 0..11). A matmul is spread (its groups resident
  on their compute tiles at one offset) while that fits below the limit, else streamed from the tiles with the most room, every group
  wherever it fits (w1 / w3 first: they need 12 tiles). stream: {(layer, matmul)} streamed even if they would fit (for tests). Every
  region is 256-byte aligned and ends at or below the limit."""
  lim = model_limit() if limit is None else limit
  lim -= lim % 4
  fill, out, stream = [0] * 16, [dict() for _ in range(L)], stream or set()
  def per_tile(m): return GEOM[m].param_bytes // 64 // GEOM[m].T          # units of one group (one compute tile's share)
  order = [(l, m) for m in ("w1", "w3") for l in range(L)] + [(l, m) for l in range(L) for m in ("q", "k", "v", "wo", "w2")]
  streamed = []
  for l, m in order:
    g, u = GEOM[m], per_tile(m)
    o = max(fill[:g.T])
    if o + u <= lim and (l, m) not in stream:
      out[l][m] = ("spread", o)
      for t in range(g.T): fill[t] = o + u
    else: streamed.append((l, m))
  for l, m in streamed:
    g, u, ps = GEOM[m], per_tile(m), []
    for _ in range(g.G):
      t = min(range(16), key=lambda t: (fill[t], -t))
      if fill[t] + u > lim: raise ValueError(f"the parameters of {L} layers do not fit below {lim} units")
      ps.append((t, fill[t]))
      fill[t] += u
    out[l][m] = ("stream", ps)
  return out

def regions(pl:list[dict]) -> list[tuple[int, int, int, str]]:
  """the resident parameter regions [(tile, start unit, end unit, name)] of a plan"""
  out = []
  for l, d in enumerate(pl):
    for m, p in d.items():
      g, u = GEOM[m], GEOM[m].param_bytes // 64 // GEOM[m].T
      if p[0] == "spread": out += [(t, p[1], p[1] + u, f"{l}.{m}") for t in range(g.T)]
      else: out += [(t, o, o + u, f"{l}.{m}") for t, o in p[1]]
  return out

def check_plan(pl:list[dict], limit:int|None=None) -> list[str]:
  lim = model_limit() if limit is None else limit
  regs, bad = regions(pl), []
  for t, a, z, n in regs:
    if z > lim: bad.append(f"{n}: tile {t} [{a}, {z}) above {lim}")
    if a % 4: bad.append(f"{n}: offset {a} not 256-byte aligned")
  for i, (t1, a1, z1, n1) in enumerate(regs):
    for t2, a2, z2, n2 in regs[i + 1:]:
      if t1 == t2 and a1 < z2 and a2 < z1: bad.append(f"{n1} and {n2} overlap on tile {t1}")
  return bad

def model_matrices(W:dict) -> dict[str, np.ndarray]:
  """a layer's uint8 weights (quant_weights) as the seven matmuls of the program: q and k rows in the order SIGMA"""
  return dict(q=W["wqkv"][:DM][SIGMA], k=W["wqkv"][DM:2 * DM][SIGMA], v=W["wqkv"][2 * DM:], wo=W["wo"], w1=W["w1"], w3=W["w3"], w2=W["w2"])

def model_params(Ws:list[dict], quants:list[dict]) -> bytes:
  """the parameter blob: per layer conv_blob of q, k, v, wo, w1, w3, w2 (fused.conv_blob: per 64 outputs a zero bias row, then
  [K/4][64][4] weights), each with its weight zero point"""
  from coral.codegen.fused import conv_blob
  return b"".join(conv_blob(model_matrices(W)[m], q[QKEYS[m][1]][1]) for W, q in zip(Ws, quants) for m in MATS)

def model_caching(pl:list[dict]) -> bytes:
  """PARAMETER_CACHING of model_params: per matmul (in blob order) its ringConsumers, then its infeeds; spread matmuls as fc.py caches
  them, streamed groups as one piece each on their parameter tile"""
  parts, base = [], 0
  for d in pl:
    for m in MATS:
      g, p = GEOM[m], d[m]
      if p[0] == "spread": parts.append(FC.caching_part(g, FC.Place(), p[1], base))
      else:
        rows = g.group_bytes // 256
        parts.append([(dict(tile_mask=1 << t, addr=2 + o, sdims=1, **FC.ttu("", [1], [rows], 4)), base + i * g.group_bytes, g.group_bytes, 4)
                      for i, (t, o) in enumerate(p[1])])
      base += g.param_bytes
  return caching_program(base, parts)

# ***** the program *****
def fc_quants(q:dict) -> dict[str, dict]:
  return {m: B.fc_quant(q[i], q[w], q[o]) for m, (i, w, o) in QKEYS.items()}

STOPS = ("x", "xn", "q", "k", "qc", "qs", "qkv", "att", "h", "hn", "h1", "g", "a", "h3", "m1", "m")   # prefix programs that end after a stage of a
FFN_STOPS = ("h1", "g", "a", "h3", "m1")    # layer (bisection on the device); these output their 768-byte vector from tiles 0..11

class Stop(Exception): pass

def stop_outputs(name:str, a:dict, l:int) -> list[tuple[int, int]]:
  """(narrow address, bytes) of the output DMAs of a prefix program that stops at `name`: from tile 0 (every DMA 288, 384 or 576 bytes),
  or for FFN_STOPS one DMA of the 1-D-layout vector from tiles 0..11 (fc.output_block, as gen_fc(768, 288) outputs)"""
  Q = a["Q"] + 3 * DM * l
  return dict(x=[(a["XG"] if l == 0 else a["XR"], DM)], xn=[(a["XN"], DM)], q=[(a["Yq"], DM)], k=[(a["Yk"], DM)], qc=[(a["QC"], DM)],
              qs=[(a["QC"] + DM, DM)], qkv=[(Q, DM), (Q + DM, 2 * DM)],
              att=[(a["O"], DM)], h=[(a["H"], DM)], hn=[(a["HN"], DM)], h1=[(a["H1"], HD)], g=[(a["G"], HD)], a=[(a["Af"], HD)],
              h3=[(a["H3"], HD)], m1=[(a["Mm"], HD)], m=[(a["Mm"], HD // 2), (a["Mm"] + HD // 2, HD // 2)])[name]

def layer_stages(e:Emitter, l:int, g:A.Geom, a:dict, q:dict, pl:dict, stop:tuple|None=None, reload_ident:bool=True, st:dict|None=None,
                 local:bool=False):
  """layer l: the residual x (1-D layout at XG for layer 0, XR after) -> x_out (1-D layout at XR). stop = (layer, name): after that
  stage, put the stage's vector on tile 0 and raise Stop.
  The identity row: an L2 norm leaves it unusable on tile 0 and a streamed FC on its compute tiles (measured: the next MUL computes
  garbage there; edgetpu_compiler loads the identity after an L2 norm). st['dirty'] tracks that across stages and layers; every stage
  that reads the identity (the K / V input copies, MUL, LOGISTIC, the slot copies, the attention core) loads it again first
  (reload_ident=False: never, for the device diagnosis). local: the tile-local layout (fc_spread / mul_stage / ... local=True; the
  inputs x of layer 0, cos and sin' stay in the 1-D layout of their input DMA)"""
  P, mq, first = g.P, fc_quants(q), l == 0
  X0, Q = (a["XG"] if first else a["XR"]), a["Q"] + 3 * DM * l
  st = {"dirty": False} if st is None else st
  def at(name:str, gathered:tuple|None=None):
    if stop is None or stop != (l, name): return
    if gathered: gather(e, *gathered, a["R"], local)
    raise Stop
  def use_ident():
    if st["dirty"] and reload_ident:
      e.sync(SC.sync_reset17)
      ident(e, a)
      st["dirty"] = False
  def fc(geom, X, Y, m):
    fc_stage(e, geom, X, Y, pl[m], mq[m], local=local)
    if pl[m][0] == "stream": st["dirty"] = True
  def l2(X, Y, in_q, out_q):
    E.l2norm_stage(e, DM, X, Y, a["Z"], in_q, out_q, a["l2fifo"], a["l2fifo2"])
    st["dirty"] = True
  # attention RMSNorm: x onto tile 0, L2_NORMALIZATION (its output at XN), broadcast to tiles 1..4
  gather(e, DM, X0, a["R"], local and not first)
  at("x")
  l2(X0, a["XN"], q["res"], q["x"])
  at("xn")
  use_ident()
  bcast(e, DM, a["XN"], 0x1e, a["xfifo"])
  # this layer's K and V cache rows (their input path copies through the identity row)
  if P > 1:
    use_ident()
    A.input4d(e, P, a["Stg"], a["XK"], a["C"], a["const"], a["in_kv"], False)
    e.sync(SC.sync_reset17)
    A.input4d(e, P, a["Stg"], a["XV"], a["C"], a["const"], a["in_kv"], False)
    e.sync(SC.sync_reset17)
  # q, k, v (1-D layout on tiles 0..4, q and k in the order SIGMA; v straight into its slot of Q), RoPE (swap(q) read as q[SWAPS])
  fc(G288, a["XN"], a["Yq"], "q")
  at("q", (DM, a["Yq"]))
  fc(G288, a["XN"], a["Yk"], "k")
  at("k", (DM, a["Yk"]))
  fc(G288, a["XN"], Q + 2 * DM, "v")
  bq = B.block_quant(q)
  use_ident()
  mul_stage(e, DM, a["Yq"], a["Xc"], a["Mr"], a["QC"], a["mulfifo"], EL.mul_quant(bq["qf"], bq["cos"], bq["qc"]), local=local, b_global=True)
  at("qc", (DM, a["QC"]))
  mul_stage(e, DM, a["Yq"], a["Xs"], a["Mr"], a["QC"] + DM, a["mulfifo"], EL.mul_quant(bq["qf"], bq["sin"], bq["qs"]), swap=True, local=local,
            b_global=True)
  at("qs", (DM, a["QC"] + DM))
  add_stage(e, DM, a["QC"], a["QC"] + DM, Q, a["Wm"], a["addw"], EL.add_quant(bq["qc"], bq["qs"], bq["q"]), local=local)
  mul_stage(e, DM, a["Yk"], a["Xc"], a["Mr"], a["KC"], a["mulfifo"], EL.mul_quant(bq["kf"], bq["cos"], bq["kc"]), local=local, b_global=True)
  mul_stage(e, DM, a["Yk"], a["Xs"], a["Mr"], a["KC"] + DM, a["mulfifo"], EL.mul_quant(bq["kf"], bq["sin"], bq["ks"]), swap=True, local=local,
            b_global=True)
  add_stage(e, DM, a["KC"], a["KC"] + DM, Q + DM, a["Wm"], a["addw"], EL.add_quant(bq["kc"], bq["ks"], bq["k"]), local=local)
  # q' | k' | v onto tile 0 (they stay there for the output), to every tile that holds positions; k' and v into row P-1 of K / V
  for k in range(3): gather(e, DM, Q + DM * k, a["R"], local)
  at("qkv")
  e.scalar(SC.scsync_nop())
  e.sync(SC.sync_reset17)
  if dest := sum(1 << t for t in range(1, 16) if g.cols[t % 4]): FC.broadcast(e, FC.FCGeom(64, 3 * DM), Q, 1, dest, a["qkvfifo"])
  e.sync(SC.sync_reset17)
  use_ident()
  B.slot_copies(e, g, Q + DM, a["XK"])
  B.slot_copies(e, g, Q + 2 * DM, a["XV"])
  e.sync(SC.sync_reset17)
  # the attention core (attention_block's, from the scores on)
  cq = B.core_quant(bq)
  def q_bcast(t): return (Q + DH * A.HEAD0[t // 4], [1, 0, DH, DH * A.ROWS[t // 4]], [DH, g.cols[t % 4], A.ROWS[t // 4]])
  A.mul_stage(e, g, q_bcast, a["XK"], a["Ms"], a["Ys"], a["mulfifo"], a["const"], EL.mul_quant(cq["q"], cq["k"], cq["qk"]),
              lambda t: A.HW * g.cols[t % 4])
  A.chansum_stage(e, g, a["Ys"], a["S"], a["sumfifo"], A.sum_quant(cq["qk"], cq["s"]))
  A.tiles_to_smem(e, g, a["S"], a["s_smem_w"], a["M"], a["smfifo"])
  e.scalar(A.softmax_words(NH, P, a["s_smem_w"], a["M"], a["p_smem_w"], cq["s"][1], f32(cq["s"][0]), cq["beta"], cq["p"][0], cq["p"][1]))
  A.smem_to_tiles(e, 1, a["Pn"], a["p_smem_w"], P, NH, a["pfifo"], collapse=False)
  if dest:
    e.scalar(SC.scsync_nop())
    e.sync(SC.sync_reset17)
    FC.broadcast(e, FC.FCGeom(64, 24 * P), a["Pn"], 1, dest, a["bfifo"])
    e.sync(SC.sync_reset17)
  def p_bcast(t):
    r, c = divmod(t, 4)
    return a["Pn"] + 4 * (P * A.HEAD0[r] + g.p0(c)), [0, 4, 4 * P, 4 * P * A.ROWS[r]], [DH, g.cols[c], A.ROWS[r]]
  A.mul_stage(e, g, p_bcast, a["XV"], a["Mv"], a["Yv"], a["mulfifo"], a["const"], EL.mul_quant(cq["p"], cq["v"], cq["pv"]),
              lambda t: A.HW * (P - g.p0(t % 4)))
  A.possum_any(e, g, a["Yv"], a["O"], a["relay"], a["const"], A.sum_quant(cq["pv"], cq["o"]))
  e.sync(SC.sync_reset17)
  B.gather_north(e, a["O"], a["R"])
  at("att")
  bcast(e, DM, a["O"], 0x1e, a["ofifo"])
  # o = wo(att) (1-D layout), h = o + x
  fc(G288, a["O"], a["Yo"], "wo")
  add_stage(e, DM, a["Yo"], X0, a["H"], a["Wm"], a["addw"], EL.add_quant(q["o"], q["res"], q["mid"]), local=local, b_global=local and first)
  # FFN RMSNorm: h onto tile 0, L2_NORMALIZATION (at HN), broadcast to tiles 1..11
  gather(e, DM, a["H"], a["R"], local)
  at("h")
  l2(a["H"], a["HN"], q["mid"], q["hn"])
  at("hn")
  use_ident()
  bcast(e, DM, a["HN"], 0xffe, a["hfifo"])
  # the FFN (fused's FC form, 1-D layout on tiles 0..11): h1, g = LOGISTIC(h1), a = h1 g, h3, m = a h3; m onto tile 0, to tiles
  # 1..4; y = w2 m. The LOGISTIC stage loads the identity row itself (edgetpu_compiler's sequence)
  fc(G768, a["HN"], a["H1"], "w1")
  at("h1")
  logistic_stage(e, HD, a["H1"], a["G"], q["h1"], q["g"], a if reload_ident else None, local=local)
  if reload_ident: st["dirty"] = False
  at("g")
  use_ident()
  mul_stage(e, HD, a["H1"], a["G"], a["Mr"], a["Af"], a["mulfifo"], EL.mul_quant(q["h1"], q["g"], q["a"]), local=local)
  at("a")
  fc(G768, a["HN"], a["H3"], "w3")
  at("h3")
  use_ident()
  mul_stage(e, HD, a["Af"], a["H3"], a["Mr"], a["Mm"], a["mulfifo"], EL.mul_quant(q["a"], q["h3"], q["m"]), local=local)
  at("m1")
  gather(e, HD, a["Mm"], a["R"], local)
  at("m")
  bcast(e, HD, a["Mm"], 0x1e, a["mfifo"])
  fc(G2, a["Mm"], a["Y2"], "w2")
  # x_out = y + h (1-D layout, at XR)
  add_stage(e, DM, a["Y2"], a["H"], a["XR"], a["Wm"], a["addw"], EL.add_quant(q["y"], q["mid"], q["out"]), local=local)

def gen_model(P:int, quants:list[dict], pl:list[dict]|None=None, stop:tuple|None=None, reload_ident:bool=True,
              local:bool=False) -> tuple[bytes, list[bytes], dict]:
  """(PARAMETER_CACHING, [EXECUTION_ONLY bitstreams], host contract model_io(P, L)) of L = len(quants) layers at P = 1..256 positions
  (the new token at P-1). quants: per layer the quantization (keys above; quants[l]['out'] == quants[l+1]['res']); pl: the parameter
  placement (plan(L)). stop = (layer, name in STOPS): a prefix program that ends after that stage and returns its vector instead
  (stop_outputs), for bisection on the device. reload_ident: load the identity row again after every L2 norm. local: the tile-local
  layout of the 1-D vectors (one op for every tile group instead of one per tile: about a third fewer instruction words)"""
  if not 1 <= P <= 256: raise NotImplementedError("P must be 1..256 (the scalar memory holds 2 x 6 x 256 words)")
  L = len(quants)
  pl = plan(L) if pl is None else pl
  assert len(pl) == L and not check_plan(pl, model_limit(P)), check_plan(pl, model_limit(P))
  for l in range(1, L): assert tuple(quants[l]["res"]) == tuple(quants[l - 1]["out"]), "the residual quantization must chain"
  assert stop is None or (0 <= stop[0] < L and stop[1] in STOPS)
  g, a = A.Geom(P), model_alloc(P, L)
  e = Emitter()
  e.scalar(SC.exe_prologue(), seqs=1)
  # inputs: x (the residual), cos and sin' (1-D layout on tiles 0..4); then the identity row (as the verified attention block at P = 1:
  # its narrow staging at C, the 4x4 identity at wide IDENT, over the inputs' ring FIFO)
  B.scatter_input(e, a["XG"])
  B.scatter_input(e, a["Xc"])
  B.scatter_input(e, a["Xs"])
  ident(e, a)
  st = {"dirty": False}                      # the identity row needs loading again (layer_stages)
  try:
    for l in range(L): layer_stages(e, l, g, a, quants[l], pl[l], stop, reload_ident, st, local)
    # outputs: x_out (gathered onto tile 0), then every layer's k' | v (576 bytes from its Q): DMAs of sizes that ran before (288, 576)
    gather(e, DM, a["XR"], a["R"], local)
    outs = [(a["XR"], DM)] + [(a["Q"] + 3 * DM * l + DM, 2 * DM) for l in range(L)]
  except Stop: outs = stop_outputs(stop[1], a, stop[0])
  e.sync(SC.sync_reset17)
  if stop is not None and stop[1] in FFN_STOPS:
    e.sync(SC.signal_fence)
    output_block(e, G768, outs[0][0], local)
  else:
    for addr, n in outs: B.output_tile0(e, addr, n)
  e.sync(SC.epilogue)
  from coral.codegen.fused import split_bitstreams
  return model_caching(pl), split_bitstreams(e.words), model_io(P, L, stop)

# ***** host contract *****
def model_io(P:int, L:int, stop:tuple|None=None) -> dict:
  """inputs in DMA order: x [288] (the quantized embedding, quants[0]['res']), cos, sin' [288] (rope_inputs: the order SIGMA), then per
  layer K, V [6][P][48] (P >= 2; rows 0..P-2 the cache, row P-1 a placeholder); outputs in DMA order: x_out [288] (quants[-1]['out']),
  then per layer kv = k' | v [576] (k' in the order SIGMA). A stop program returns stop_outputs instead."""
  kv_layers = L if stop is None else stop[0] + (stop[1] not in ("x", "xn"))   # a prefix program ends before the later layers' K / V inputs
  ins = [("x", DM), ("cos", DM), ("sin", DM)] + ([(f"{kv}{l}", DM * P) for l in range(kv_layers) for kv in "kv"] if P > 1 else [])
  if stop is None: outs = [("out", DM)] + [(f"kv{l}", 2 * DM) for l in range(L)]
  else: outs = [(f"{stop[1]}{i}", n) for i, (_, n) in enumerate(stop_outputs(stop[1], model_alloc(P, L), stop[0]))]
  return dict(inputs=ins, outputs=outs, param_bytes=L * sum(GEOM[m].param_bytes for m in MATS))

def model_executables(caching:bytes, bitstreams:list[bytes], io:dict) -> tuple:
  """coral.executable.Executables (PARAMETER_CACHING, EXECUTION_ONLY) for coral.runtime.run_executable / tools.hw.run: the inputs in one
  host buffer (model_inputs), the outputs in one buffer (model_outputs); every DMA's hint has exactly the size of its descriptor"""
  from coral.executable import Executable, Bitstream, Hint
  from coral.codegen.fused import program_hints
  dmas = [(t, n) for bs in bitstreams for t, n in SC.dma_seqs(bs) if t in (SC.TAG_INPUT, SC.TAG_OUTPUT)]
  assert dmas == [(SC.TAG_INPUT, n) for _, n in io["inputs"]] + [(SC.TAG_OUTPUT, n) for _, n in io["outputs"]], dmas
  def mk(kind, bss, hs): return Executable(None, kind, 1, 0, [Bitstream(b, []) for b in bss], b"", hs, True, [], [], "beagle", 0, 0, 0, 0)
  pc = mk("PARAMETER_CACHING", [caching], [Hint("instruction", "INFEED", chunk=0), Hint("dma", "INFEED", "PARAMETER", "", 0, io["param_bytes"]),
                                           Hint("interrupt", "OUTFEED", interrupt=0)])
  ex = mk("EXECUTION_ONLY", bitstreams, program_hints(bitstreams, "x", "out"))
  outs = [h.size for h in ex.hints if h.kind == "dma" and h.direction == "OUTFEED"]
  assert outs == [n for _, n in io["outputs"]], (outs, io["outputs"])
  return pc, ex

def model_inputs(x:np.ndarray, cos:np.ndarray, sin:np.ndarray, Ks:list|None=None, Vs:list|None=None) -> bytes:
  """the host buffer: x, cos, sin' [288] uint8; Ks / Vs per layer the cache rows 0..P-2 as [P-1][288] (None / empty at P = 1); row P-1
  goes as zeros (the program writes k', v there)"""
  buf = b"".join(np.asarray(t, np.uint8).reshape(DM).tobytes() for t in (x, cos, sin))
  for K, V in zip(Ks or [], Vs or []):
    if K is None or not len(K): continue
    for C in (K, V): buf += A.heads(np.concatenate([np.asarray(C, np.uint8).reshape(-1, DM), np.zeros((1, DM), np.uint8)])).tobytes()
  return buf

def model_outputs(buf:bytes, L:int, stop:tuple|None=None):
  """the execution's output bytes -> (x_out [288], kv [L][576]); a stop program's -> its vector (the stage's whole vector)"""
  b = np.frombuffer(buf, np.uint8)
  if stop is not None: return b.copy()
  return b[:DM].copy(), b[DM:DM + 2 * DM * L].reshape(L, 2 * DM).copy()

def stop_ref(parts:dict, name:str) -> np.ndarray:
  """the bit model's vector of a stop program (layer_ref's parts)"""
  return dict(x=parts["x"], xn=parts["xn"], q=parts["qf"], k=parts["kf"], qc=parts["qc"], qs=parts["qs"],
              qkv=np.concatenate([parts["q"], parts["k"], parts["v"]]),
              att=parts["att"], h=parts["h"], hn=parts["hn"], h1=parts["h1"], g=parts["g"], a=parts["a"], h3=parts["h3"], m1=parts["m"],
              m=parts["m"])[name]

if __name__ == "__main__":
  from coral.codegen import run_test
  run_test("test_codegen_layer")
