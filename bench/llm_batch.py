# throughput of batched decoding (B sequences in parallel) for TinyStories-15M in tinygrad
#   DEV=CPU .venv/bin/python bench/llm_batch.py float 1 8 32       (Mac CPU only, float32)
#   DEV=CORAL .venv/bin/python bench/llm_batch.py matmuls 1 8 32   (every matmul one TPU program: plain tinygrad + kernel selection)
#   DEV=CORAL .venv/bin/python bench/llm_batch.py fused 1 8 32     (3 fused TPU programs per layer, on-chip argmax classifier from batch 64)
import sys, os, time, json, numpy as np
os.environ.setdefault("WQKV", "1")
sys.argv, mode, Bs = sys.argv, sys.argv[1], [int(x) for x in sys.argv[2:]]
sys.path.insert(0, "examples")
import stories
from tinygrad import Tensor, nn, TinyJit
cfg, weights = stories.load_checkpoint(stories.ROOT / "models/stories15M.bin")
res = {}
for B in Bs:
  model = stories.build(cfg, weights, nn.Linear, jit=False)
  # the classifier (9.3 MB of weights) streams over usb every step: worth it once the batch is big enough
  if mode == "matmuls":
    stories.coralize(model, cfg, weights, "Once upon a time there was a girl.", classifier=B >= int(os.getenv("CLS_B", "1000000")))
  ids = mode == "fused" and B >= int(os.getenv("CLS_B", "64"))       # the on-chip block-argmax classifier returns token ids
  if mode == "fused": stories.coralize_fused(model, cfg, weights, "Once upon a time there was a girl.", classifier=ids)
  fwd = TinyJit(lambda toks, pos: model.forward(toks, pos, float("nan"), 0, 0.8, 0.0, 0.0).realize())
  from tinygrad import Variable
  toks = Tensor(np.full((B, 1), 9038, np.int32)).realize()      # "Once"
  steps, ts = 24, []
  for pos in range(1, steps):
    st = time.perf_counter()
    logits = fwd(toks, Variable("start_pos", 1, cfg["max_context"]-1).bind(pos))
    nxt = logits.numpy()[:, -1] if ids else logits[:, -1, :].argmax(axis=-1).numpy()
    ts.append(time.perf_counter() - st)
    toks = Tensor(nxt.reshape(B, 1).astype(np.int32)).realize()
  step = float(np.median(ts[4:]))
  res[B] = {"step_ms": step * 1e3, "tok_s": B / step}
  print(f"{mode} B={B:4d}: {step*1e3:7.2f} ms/step  {B/step:8.1f} tok/s", flush=True)
print(json.dumps(res))
