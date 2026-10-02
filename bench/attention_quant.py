# would TinyStories-15M survive attention (and more) in uint8 on the Edge TPU? A numpy simulation at batch 1, before any code
# generation: each variant quantizes more of the model (uint8 per tensor, ranges calibrated on float runs), and is compared with
# the float model by teacher-forced top-1 agreement on stories the float model wrote.
#   F  float
#   M  every matmul uint8 (inputs, weights, outputs), as the TPU runs them today; norms, RoPE, attention, residuals float
#   A  M + attention in uint8: q, k, v, the scores (q.K), softmax probabilities (1/256, as TFLite quantizes them), p.V
#   N  A + the RMSNorms on uint8 inputs and outputs (an L2 normalization on the chip)
#   R  N + the residual stream itself in uint8 between the sublayers
#   Mp, Ap, Np, Rp: the same with weights quantized per output channel (the chip supports a float32 multiplier per output)
#   E  A as coral/codegen/attention.py runs it today: elementwise MUL, so every q*k and p*v product is rounded to uint8 before the SUM
#   .venv/bin/python bench/attention_quant.py [stories] [length] [variants, e.g. M,Rp]
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import stories
from coral.fused import quantize_params

def fq(x, r):
  """quantize to uint8 over range r, then back (fake quantization, round half to even as the TPU does)"""
  s, z = quantize_params(*r)
  return (np.clip(np.round(x / s) + z, 0, 255) - z) * s

class Sim:
  def __init__(self, cfg, w, mode:str, ranges:dict|None):
    self.c, self.mode, self.ranges, self.rec = cfg, mode, ranges, {}
    L, D = cfg["n_layers"], cfg["dim"]
    self.W = [{k: w[f"layers.{l}.{k}"] for k in ("attention_norm.weight", "ffn_norm.weight", "attention.wq.weight", "attention.wk.weight",
                                                 "attention.wv.weight", "attention.wo.weight", "feed_forward.w1.weight", "feed_forward.w2.weight",
                                                 "feed_forward.w3.weight")} for l in range(L)]
    for d in self.W:
      d["wqkv"] = np.concatenate([d["attention.wq.weight"], d["attention.wk.weight"], d["attention.wv.weight"]])
      d["w13"] = np.concatenate([d["feed_forward.w1.weight"], d["feed_forward.w3.weight"]])
      if mode != "F":                                    # uint8 weights, per tensor (or per output channel: variants ending in p)
        for k in ("wqkv", "attention.wo.weight", "w13", "feed_forward.w2.weight"):
          d[k] = np.stack([fq(r, (r.min(), r.max())) for r in d[k]]) if mode.endswith("p") else fq(d[k], (d[k].min(), d[k].max()))
    self.emb, self.norm, self.out = w["tok_embeddings.weight"], w["norm.weight"], w["output.weight"]
    hd, T = D // cfg["n_heads"], cfg["max_context"]
    f = 1.0 / (10000.0 ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.arange(T)[:, None] * f[None]
    self.cos, self.sin = np.cos(ang), np.sin(ang)
  def q(self, name:str, x, level:str):
    """record the range (calibration), or fake-quantize when this variant quantizes at `level`"""
    if self.ranges is None:
      lo, hi = self.rec.get(name, (np.inf, -np.inf))
      self.rec[name] = (min(lo, float(x.min())), max(hi, float(x.max())))
      return x
    if self.mode[0] == "E": return fq(x, self.ranges[name]) if level in "MAE" else x      # E: A with every product rounded
    return fq(x, self.ranges[name]) if level in "MANR"[:"FMANR".index(self.mode[0])] else x
  def mm(self, name, x, wt): return self.q(name + ".out", self.q(name + ".in", x, "M") @ wt.T, "M")
  def rms(self, name, x, wt):
    x = self.q(name + ".in", x, "N")
    return self.q(name + ".out", x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * wt, "N")
  def rope(self, x, pos):                              # x [T, H, hd], interleaved pairs like llama2.c
    x0, x1 = x[..., 0::2], x[..., 1::2]
    c, s = self.cos[pos][:, None], self.sin[pos][:, None]
    o = np.empty_like(x)
    o[..., 0::2] = x0 * c - x1 * s
    o[..., 1::2] = x0 * s + x1 * c
    return o
  def run(self, toks) -> np.ndarray:
    """teacher-forced logits [T, vocab] for the token sequence (batch 1, causal)"""
    c, T = self.c, len(toks)
    H, D = c["n_heads"], c["dim"]
    hd = D // H
    x = self.emb[toks].astype(np.float64)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for l, d in enumerate(self.W):
      x = self.q(f"{l}.res1", x, "R")
      h = self.mm(f"{l}.qkv", self.rms(f"{l}.n1", x, d["attention_norm.weight"]), d["wqkv"])
      qh, kh, vh = (h[:, i*D:(i+1)*D].reshape(T, H, hd).transpose(1, 0, 2) for i in range(3))
      qh, kh = self.rope(qh.transpose(1, 0, 2), np.arange(T)).transpose(1, 0, 2), self.rope(kh.transpose(1, 0, 2), np.arange(T)).transpose(1, 0, 2)
      qh, kh, vh = self.q(f"{l}.q", qh, "A"), self.q(f"{l}.k", kh, "A"), self.q(f"{l}.v", vh, "A")
      if self.mode[0] == "E" or self.ranges is None:  # the elementwise form the chip runs today: every product rounded to uint8
        prod = self.q(f"{l}.qk_prod", qh[:, :, None, :] * kh[:, None, :, :] / np.sqrt(hd), "E")   # [H, Tq, Tk, hd]
        s_e = prod.sum(-1)
      s = self.q(f"{l}.scores", s_e if self.mode[0] == "E" else qh @ kh.transpose(0, 2, 1) / np.sqrt(hd), "A")
      p = np.exp(s + mask - (s + mask).max(-1, keepdims=True))
      p /= p.sum(-1, keepdims=True)
      if self.ranges is not None and self.mode[0] in "ANRE": p = np.clip(np.round(p * 256), 0, 255) / 256      # softmax output (1/256, 0)
      if self.mode[0] == "E" or self.ranges is None:
        pv = self.q(f"{l}.pv_prod", p[:, :, :, None] * vh[:, None, :, :], "E").sum(2)                   # [H, Tq, hd]
      o = self.q(f"{l}.att", (pv if self.mode[0] == "E" else p @ vh).transpose(1, 0, 2).reshape(T, D), "A")
      x = x + self.mm(f"{l}.wo", o, d["attention.wo.weight"])
      x = self.q(f"{l}.res2", x, "R")
      g = self.mm(f"{l}.w13", self.rms(f"{l}.n2", x, d["ffn_norm.weight"]), d["w13"])
      a, b = g[:, :c["hidden_dim"]], g[:, c["hidden_dim"]:]
      x = x + self.mm(f"{l}.w2", (a / (1 + np.exp(-a))) * b, d["feed_forward.w2.weight"])
    x = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * self.norm
    return x @ self.out.T

if __name__ == "__main__":
  cfg, w = stories.load_checkpoint(stories.ROOT / "models/stories15M.bin")
  tok = stories.Tokenizer(stories.ROOT / "models/tokenizer.bin", cfg["vocab_size"])
  n, T = (int(sys.argv[1]) if len(sys.argv) > 1 else 8), (int(sys.argv[2]) if len(sys.argv) > 2 else 200)
  rng, F = np.random.default_rng(0), Sim(cfg, w, "F", None)
  seqs = []
  for i in range(2 * n):                               # stories by the float model, sampled (temperature 0.8) for variety
    s = [1]
    for _ in range(T - 1):
      lg = F.run(np.array(s))[-1] / 0.8
      s.append(int(np.argmax(lg + rng.gumbel(size=lg.shape))))
    seqs.append(np.array(s))
  calib, test = seqs[:n], seqs[n:]                     # ranges from half of them, agreement on the other half
  for s in calib: F.run(s)                             # (F's recorder holds the ranges)
  def nll(lg, s):                                      # mean negative log-likelihood of the story's actual next tokens
    lg = lg[:-1] - lg[:-1].max(-1, keepdims=True)
    return float(np.mean(np.log(np.exp(lg).sum(-1)) - lg[np.arange(len(s) - 1), s[1:]]))
  fl = [F.run(s) for s in test]
  ref = [lg.argmax(-1) for lg in fl]
  print(f"F: perplexity {np.exp(np.mean([nll(lg, s) for lg, s in zip(fl, test)])):6.3f}")
  print("example:", tok.decode(1, int(test[0][1])) + "".join(tok.decode(int(a), int(b)) for a, b in zip(test[0][1:60], test[0][2:61])))
  for mode in sys.argv[3].split(",") if len(sys.argv) > 3 else ["M", "A", "N", "R", "Mp", "Ap", "Np", "Rp"]:
    sim = Sim(cfg, w, mode, F.rec)
    lgs = [sim.run(s) for s in test]
    agree = np.concatenate([lg.argmax(-1) == r for lg, r in zip(lgs, ref)])
    print(f"{mode}: perplexity {np.exp(np.mean([nll(lg, s) for lg, s in zip(lgs, test)])):6.3f}, "
          f"top-1 agreement with float {agree.mean() * 100:6.2f}%  ({agree.size} positions)", flush=True)
