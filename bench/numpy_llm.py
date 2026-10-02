# a strong CPU baseline: TinyStories-15M batched greedy decoding in plain numpy (float32), all matmuls through numpy's
# BLAS (Apple Accelerate on macOS: AMX units, all cores). Same model, same KV-cache decoding as examples/stories.py.
#   .venv/bin/python bench/numpy_llm.py 1 8 32 128 256
import sys, os, time, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import stories

def rmsnorm(x, w, eps=1e-5): return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w

class NumpyLlama:
  def __init__(self, cfg, w, B):
    self.c, self.B = cfg, B
    self.L = [{k: w[f"layers.{l}.{k}"] for k in ("attention_norm.weight", "ffn_norm.weight", "attention.wq.weight", "attention.wk.weight",
                                                 "attention.wv.weight", "attention.wo.weight", "feed_forward.w1.weight", "feed_forward.w2.weight",
                                                 "feed_forward.w3.weight")} for l in range(cfg["n_layers"])]
    for d in self.L:                                     # fused, pre-transposed weights: one GEMM for qkv and one for w1|w3
      d["wqkv"] = np.ascontiguousarray(np.concatenate([d["attention.wq.weight"], d["attention.wk.weight"], d["attention.wv.weight"]]).T)
      d["wo"], d["w2"] = np.ascontiguousarray(d["attention.wo.weight"].T), np.ascontiguousarray(d["feed_forward.w2.weight"].T)
      d["w13"] = np.ascontiguousarray(np.concatenate([d["feed_forward.w1.weight"], d["feed_forward.w3.weight"]]).T)
    self.emb, self.norm, self.out = w["tok_embeddings.weight"], w["norm.weight"], np.ascontiguousarray(w["output.weight"].T)
    hd, T = cfg["dim"] // cfg["n_heads"], cfg["max_context"]
    f = 1.0 / (10000.0 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    ang = np.arange(T, dtype=np.float32)[:, None] * f[None]
    self.cos, self.sin = np.cos(ang), np.sin(ang)
    self.kc = np.zeros((cfg["n_layers"], B, cfg["n_heads"], T, hd), np.float32)
    self.vc = np.zeros_like(self.kc)
  def rope(self, x, pos):                                # x [B, H, hd], interleaved pairs like llama2.c / tinygrad
    x0, x1 = x[..., 0::2], x[..., 1::2]
    c, s = self.cos[pos], self.sin[pos]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out
  def step(self, toks, pos):
    c, B = self.c, self.B
    H, D = c["n_heads"], c["dim"]
    hd = D // H
    x = self.emb[toks]
    for l, d in enumerate(self.L):
      h = rmsnorm(x, d["attention_norm.weight"]) @ d["wqkv"]
      q, k, v = h[:, :D].reshape(B, H, hd), h[:, D:2*D].reshape(B, H, hd), h[:, 2*D:].reshape(B, H, hd)
      q, k = self.rope(q, pos), self.rope(k, pos)
      self.kc[l, :, :, pos], self.vc[l, :, :, pos] = k, v
      K, V = self.kc[l, :, :, :pos+1], self.vc[l, :, :, :pos+1]
      att = np.einsum("bhd,bhtd->bht", q, K) / np.sqrt(hd)
      att = np.exp(att - att.max(-1, keepdims=True))
      att /= att.sum(-1, keepdims=True)
      x = x + np.einsum("bht,bhtd->bhd", att, V).reshape(B, D) @ d["wo"]
      g = rmsnorm(x, d["ffn_norm.weight"]) @ d["w13"]
      a, b = g[:, :c["hidden_dim"]], g[:, c["hidden_dim"]:]
      x = x + ((a / (1 + np.exp(-a))) * b) @ d["w2"]
    return (rmsnorm(x, self.norm) @ self.out).argmax(-1)

if __name__ == "__main__":
  cfg, w = stories.load_checkpoint(stories.ROOT / "models/stories15M.bin")
  res = {}
  for B in [int(a) for a in sys.argv[1:]] or [1, 256]:
    m, toks, ts = NumpyLlama(cfg, w, B), np.full(B, 9038), []
    for pos in range(24):
      st = time.perf_counter()
      toks = m.step(toks, pos)
      ts.append(time.perf_counter() - st)
    step = float(np.median(ts[4:]))
    res[B] = {"step_ms": step * 1e3, "tok_s": B / step}
    print(f"numpy B={B:4d}: {step*1e3:7.2f} ms/step  {B/step:8.1f} tok/s", flush=True)
  print(json.dumps(res))
