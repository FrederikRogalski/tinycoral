# TinyStories 15M (karpathy/llama2.c) in tinygrad, with every transformer matmul running on the Google Coral Edge TPU
#   python examples/stories.py --prompt "Once upon a time"
# float reference on the host:   --float      fake TPU (numpy):   MOCKCORAL=1
import sys, os, struct, pathlib, argparse, time, numpy as np
os.environ.setdefault("WQKV", "1")                                     # tinygrad's fused qkv path: one matmul per layer
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tinygrad-src"))
from tinygrad import Tensor, Device, nn
from tinygrad.helpers import Timing
from extra.models.llama import Transformer
from coral.nn import QLinear

def load_checkpoint(path:pathlib.Path):
  raw = path.read_bytes()
  dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq_len = struct.unpack("7i", raw[:28])
  shared, vocab = vocab > 0, abs(vocab)
  hs = dim // n_heads
  arr, off = np.frombuffer(raw, np.float32, offset=28), 0
  def take(*shape):
    nonlocal off
    n = int(np.prod(shape)); out = arr[off:off+n].reshape(shape); off += n
    return out
  w = {"tok_embeddings.weight": take(vocab, dim)}
  att_norm, wq, wk, wv, wo = take(n_layers, dim), take(n_layers, dim, dim), take(n_layers, n_kv_heads*hs, dim), take(n_layers, n_kv_heads*hs, dim), take(n_layers, dim, dim)
  ffn_norm, w1, w2, w3 = take(n_layers, dim), take(n_layers, hidden, dim), take(n_layers, dim, hidden), take(n_layers, hidden, dim)
  w["norm.weight"] = take(dim)
  take(seq_len, hs // 2); take(seq_len, hs // 2)    # legacy freq_cis_real / freq_cis_imag
  w["output.weight"] = w["tok_embeddings.weight"] if shared else take(vocab, dim)
  for l in range(n_layers):
    p = f"layers.{l}."
    w[p+"attention_norm.weight"], w[p+"ffn_norm.weight"] = att_norm[l], ffn_norm[l]
    w[p+"attention.wq.weight"], w[p+"attention.wk.weight"], w[p+"attention.wv.weight"], w[p+"attention.wo.weight"] = wq[l], wk[l], wv[l], wo[l]
    w[p+"feed_forward.w1.weight"], w[p+"feed_forward.w2.weight"], w[p+"feed_forward.w3.weight"] = w1[l], w2[l], w3[l]
  return dict(dim=dim, hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, vocab_size=vocab, max_context=seq_len), w

class Tokenizer:
  def __init__(self, path:pathlib.Path, vocab_size:int):
    raw, off = path.read_bytes(), 4
    self.vocab, self.scores = [], []
    for _ in range(vocab_size):
      score, n = struct.unpack_from("fi", raw, off); off += 8
      self.vocab.append(raw[off:off+n]); self.scores.append(score); off += n
    self.lookup = {t: i for i, t in enumerate(self.vocab)}
  def encode(self, text:str) -> list[int]:
    toks = [self.lookup[b" "]] if text else []                       # llama2.c prepends a dummy space
    for ch in text.encode():
      toks.append(self.lookup[bytes([ch])] if bytes([ch]) in self.lookup else self.lookup[f"<0x{ch:02X}>".encode()])
    while True:  # greedy bpe merges by score
      best = max(((self.scores[j], i, j) for i in range(len(toks) - 1) if (j:=self.lookup.get(self.vocab[toks[i]] + self.vocab[toks[i+1]])) is not None), default=None)
      if best is None: break
      _, i, j = best; toks[i:i+2] = [j]
    return [1] + toks                                                 # BOS
  def decode(self, prev:int, tok:int) -> str:
    piece = self.vocab[tok]
    if prev == 1 and piece.startswith(b" "): piece = piece[1:]
    if piece.startswith(b"<0x") and piece.endswith(b">") and len(piece) == 6: piece = bytes([int(piece[3:5], 16)])
    return piece.decode(errors="replace")

class CalibLinear:
  """float linear that records the range of its inputs and outputs (for post-training quantization)"""
  def __init__(self, in_features, out_features, bias=False):
    self.weight = Tensor.zeros(out_features, in_features)
    self.lo_in = self.lo_out = np.inf; self.hi_in = self.hi_out = -np.inf
  def __call__(self, x:Tensor) -> Tensor:
    y = x.linear(self.weight.T)
    xn, yn = x.numpy(), y.numpy()
    self.lo_in, self.hi_in = min(self.lo_in, float(xn.min())), max(self.hi_in, float(xn.max()))
    self.lo_out, self.hi_out = min(self.lo_out, float(yn.min())), max(self.hi_out, float(yn.max()))
    return y

class CoralFFN:
  def __init__(self, w13:QLinear, w2:QLinear, hidden:int): self.w13, self.w2, self.hidden = w13, w2, hidden
  def __call__(self, x:Tensor) -> Tensor:
    h = self.w13(x)
    return self.w2(h[..., :self.hidden].silu() * h[..., self.hidden:])

def fuse_qkv(weights, cfg):
  # rows interleaved per head [q_h, k_h, v_h], as extra/models/llama.py expects with WQKV (n_rep=1)
  hd, nh, out = cfg["dim"] // cfg["n_heads"], cfg["n_heads"], dict(weights)
  for l in range(cfg["n_layers"]):
    p = f"layers.{l}.attention."
    out[p+"wqkv.weight"] = np.stack([weights[p+f"w{x}.weight"].reshape(nh, hd, -1) for x in "qkv"], axis=1).reshape(3*nh*hd, -1)
  return out

def build(cfg, weights, linear, jit:bool):
  model = Transformer(cfg["dim"], cfg["hidden_dim"], cfg["n_heads"], cfg["n_layers"], 1e-5, cfg["vocab_size"], linear=linear,
                      n_kv_heads=cfg["n_kv_heads"], max_context=cfg["max_context"], jit=jit)
  nn.state.load_state_dict(model, {k: Tensor(v.copy()) for k, v in fuse_qkv(weights, cfg).items()}, strict=False, verbose=False)
  # the RoPE table as data: computed from constants inside the attention kernel, it was ~2/3 of a batch-256 decoding step on the CPU
  model.freqs_cis = Tensor(model.freqs_cis.numpy()).realize()
  return model

def generate(model, tok:Tokenizer, prompt:str, steps:int, temperature:float=0.0) -> tuple[str, list[int], float]:
  toks = tok.encode(prompt)
  out, st, n, plen = "", None, 0, len(toks)
  for pos in range(min(steps, 255)):
    if pos == plen - 1: st = time.perf_counter()                    # time only the generated tokens
    nxt = int(model(Tensor([[toks[pos]]]), pos, temperature).item())
    if pos >= plen - 1:
      if nxt == 1: break
      out += tok.decode(toks[-1], nxt); toks.append(nxt); n += 1
      print(tok.decode(toks[-2], nxt), end="", flush=True)
  print()
  return out, toks, n / (time.perf_counter() - st) if st else 0.0

CALIB = ("Once upon a time there was a little girl named Lily. She loved to play outside in the park.|"
         "Tom had a red ball. He liked to throw it high in the sky. One day the ball got stuck in a tree.")

def calibrate(cfg, weights, calib_text:str, tok:Tokenizer) -> dict[str, tuple[float, float, float, float]]:
  """(lo_in, hi_in, lo_out, hi_out) of every matmul, from running the float model over calib_text (cached in models/)"""
  import json, hashlib
  cache = ROOT / "models" / f"calib_{hashlib.sha1(calib_text.encode()).hexdigest()[:12]}.json"
  if cache.exists(): return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
  calib = build(cfg, weights, CalibLinear, jit=False)
  cout = CalibLinear(cfg["dim"], cfg["vocab_size"]); cout.weight = calib.output.weight; calib.output = cout
  with Timing("calibration: "):
    for text in calib_text.split("|"):
      for l in calib.layers:
        if hasattr(l.attention, "cache_kv"): l.attention.cache_kv = Tensor.zeros_like(l.attention.cache_kv).contiguous().realize()
      for pos, t in enumerate(tok.encode(text)[:96]): calib(Tensor([[t]]), pos, float("nan"))
  r = {"output": cout}
  for l, cl in enumerate(calib.layers):
    r.update({f"{l}.wqkv": cl.attention.wqkv, f"{l}.wo": cl.attention.wo, f"{l}.w1": cl.feed_forward.w1, f"{l}.w3": cl.feed_forward.w3, f"{l}.w2": cl.feed_forward.w2})
  ranges = {k: (c.lo_in, c.hi_in, c.lo_out, c.hi_out) for k, c in r.items()}
  cache.write_text(json.dumps(ranges))
  return ranges

def coralize(model, cfg, weights, calib_text:str=CALIB, tok:Tokenizer|None=None, classifier:bool=False):
  """calibrate activation ranges with the float model, then replace every transformer matmul (and optionally the classifier)
  by a uint8 QLinear on the Edge TPU"""
  tok = tok or Tokenizer(ROOT / "models/tokenizer.bin", cfg["vocab_size"])
  R = calibrate(cfg, weights, calib_text, tok)
  pad = lambda lo, hi: (lo - 0.05*(hi-lo), hi + 0.05*(hi-lo))
  ispan = lambda *ks: pad(min(R[k][0] for k in ks), max(R[k][1] for k in ks))
  span = lambda *ks: pad(min(R[k][2] for k in ks), max(R[k][3] for k in ks))
  fused = fuse_qkv(weights, cfg)
  for l in range(cfg["n_layers"]):
    p, q = f"layers.{l}.", f"{l}."
    model.layers[l].attention.wqkv = QLinear(fused[p+"attention.wqkv.weight"], None, ispan(q+"wqkv"), span(q+"wqkv"), device=Device.DEFAULT)
    model.layers[l].attention.wo = QLinear(weights[p+"attention.wo.weight"], None, ispan(q+"wo"), span(q+"wo"), device=Device.DEFAULT)
    W13 = np.concatenate([weights[p+"feed_forward.w1.weight"], weights[p+"feed_forward.w3.weight"]])
    model.layers[l].feed_forward = CoralFFN(QLinear(W13, None, ispan(q+"w1", q+"w3"), span(q+"w1", q+"w3"), device=Device.DEFAULT),
                                            QLinear(weights[p+"feed_forward.w2.weight"], None, ispan(q+"w2"), span(q+"w2"), device=Device.DEFAULT),
                                            cfg["hidden_dim"])
  if classifier: model.output = QLinear(weights["output.weight"], None, ispan("output"), span("output"), device=Device.DEFAULT)
  return model

def coralize_fused(model, cfg, weights, calib_text:str=CALIB, tok:Tokenizer|None=None, classifier:bool=True):
  """every transformer layer as 3 Edge TPU programs (fused qkv, wo, the whole FFN: w1, w3, silu, mul, w2) and, with
  classifier=True, the classifier as an on-chip block-argmax (coral/fused.py): the forward pass then returns token ids
  instead of logits. It streams the 9.2 MB vocabulary every step, so it only pays off for big batches."""
  from coral import fused
  from coral.nn import FusedBlock, FusedArgmax
  tok = tok or Tokenizer(ROOT / "models/tokenizer.bin", cfg["vocab_size"])
  R = calibrate(cfg, weights, calib_text, tok)
  pad = lambda lo, hi: (lo - 0.05*(hi-lo), hi + 0.05*(hi-lo))
  ispan = lambda *ks: pad(min(R[k][0] for k in ks), max(R[k][1] for k in ks))
  span = lambda *ks: pad(min(R[k][2] for k in ks), max(R[k][3] for k in ks))
  fq = fuse_qkv(weights, cfg)
  for l in range(cfg["n_layers"]):
    p, q = f"layers.{l}.", f"{l}."
    fused.conv_block(f"L{l}.qkv", fq[p+"attention.wqkv.weight"], ispan(q+"wqkv"), span(q+"wqkv"))
    fused.conv_block(f"L{l}.wo", weights[p+"attention.wo.weight"], ispan(q+"wo"), span(q+"wo"))
    fused.ffn_block(f"L{l}.ffn", weights[p+"feed_forward.w1.weight"], weights[p+"feed_forward.w3.weight"], weights[p+"feed_forward.w2.weight"],
                    ispan(q+"w1", q+"w3"), span(q+"w1"), span(q+"w3"), ispan(q+"w2"), span(q+"w2"))
    model.layers[l].attention.wqkv, model.layers[l].attention.wo = FusedBlock(f"L{l}.qkv"), FusedBlock(f"L{l}.wo")
    model.layers[l].feed_forward = FusedBlock(f"L{l}.ffn")
  if classifier:
    fused.argmax_block("cls", weights["output.weight"], ispan("output"), span("output"))
    model.output = FusedArgmax("cls")
  return model

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--prompt", default="Once upon a time")
  ap.add_argument("--steps", type=int, default=128)
  ap.add_argument("--float", action="store_true", help="run the float model on the host instead")
  ap.add_argument("--calib", default=CALIB)
  args = ap.parse_args()
  cfg, weights = load_checkpoint(ROOT / "models/stories15M.bin")
  tok = Tokenizer(ROOT / "models/tokenizer.bin", cfg["vocab_size"])
  model = build(cfg, weights, nn.Linear, jit=True)
  if not args.float: coralize(model, cfg, weights, args.calib, tok)
  where = "in numpy (MOCKCORAL)" if os.getenv("MOCKCORAL") else "on the Edge TPU"
  print(f"device {Device.DEFAULT}, {'float' if args.float else f'uint8 matmuls {where}'}")
  text, toks, tps = generate(model, tok, args.prompt, args.steps)
  print(f"\n{len(toks)} tokens, {tps:.1f} tok/s")
  if not args.float:
    from coral.tpu import runner, MOCK_STATS
    print("tpu stats:", r.stats if (r:=runner()) is not None else MOCK_STATS)
