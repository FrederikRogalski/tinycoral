# B TinyStories generated in parallel: every matmul of the batch runs on the Edge TPU (1x1-conv programs over 16 tiles, all weights resident)
#   DEV=CORAL python examples/stories_batch.py --batch 256 --steps 100
#   DEV=CPU   python examples/stories_batch.py --batch 256 --steps 100 --float      (the same on the Mac's CPU)
import os, sys, time, argparse, itertools, numpy as np
os.environ.setdefault("WQKV", "1")
sys.path.insert(0, os.path.dirname(__file__))
import stories
from tinygrad import Tensor, Device, nn, TinyJit, Variable

NAMES = ["Lily", "Tom", "Ben", "Sue", "Max", "Mia", "Sam", "Anna", "Tim", "Lucy", "Jack", "Emma", "Leo", "Zoe", "Bob", "Kate"]
WHO = ["a little girl", "a little boy", "a small dog", "a big cat", "a happy bird", "a shy bunny", "a brave frog", "a kind bear"]
def prompts(B:int) -> list[str]:
  return [f"Once upon a time, there was {w} named {n}." for w, n in itertools.islice(itertools.cycle(itertools.product(WHO, NAMES)), B)]

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--batch", type=int, default=128)
  ap.add_argument("--steps", type=int, default=100)
  ap.add_argument("--float", action="store_true", help="float model on the host instead of the TPU")
  ap.add_argument("--show", type=int, default=3)
  ap.add_argument("--matmuls", action="store_true", help="the matmuls one by one (QLinear) instead of the fused blocks")
  args = ap.parse_args()
  cfg, weights = stories.load_checkpoint(stories.ROOT / "models/stories15M.bin")
  tok = stories.Tokenizer(stories.ROOT / "models/tokenizer.bin", cfg["vocab_size"])
  model = stories.build(cfg, weights, nn.Linear, jit=False)
  fused = not args.float and not args.matmuls and args.batch >= 64   # on-chip block-argmax classifier (token ids out)
  if not args.float and not args.matmuls: stories.coralize_fused(model, cfg, weights, tok=tok, classifier=fused)
  elif not args.float: stories.coralize(model, cfg, weights, tok=tok, classifier=args.batch >= 32)
  B, ps = args.batch, [[1] + tok.encode(p)[1:] if tok.encode(p)[:1] == [1] else tok.encode(p) for p in prompts(args.batch)]
  fwd = TinyJit(lambda t, pos: model.forward(t, pos, float("nan"), 0, 0.8, 0.0, 0.0).realize())
  seqs, done = [list(p[:1]) for p in ps], [False] * B
  ts = []
  for pos in range(args.steps):
    st = time.perf_counter()
    cur = Tensor(np.array([[s[-1]] for s in seqs], np.int32))
    out = fwd(cur, Variable("start_pos", 0, cfg["max_context"] - 1).bind(pos))
    nxt = out.numpy()[:, -1] if fused else out[:, -1, :].argmax(axis=-1).numpy()
    for i, s in enumerate(seqs):   # the prompt is fed token by token (rows have different lengths), then greedy decoding
      s.append(ps[i][pos + 1] if pos + 1 < len(ps[i]) else int(nxt[i]))
    ts.append(time.perf_counter() - st)
  step = float(np.median(ts[3:]))
  print(f"device {Device.DEFAULT}, {'float on the host' if args.float else 'fused blocks on the Edge TPU' if fused else 'uint8 matmuls on the Edge TPU'}: batch {B}, "
        f"{step*1e3:.1f} ms per step, {B/step:.0f} tok/s ({sum(ts):.1f} s for {B*len(ts)} tokens)")
  for i in np.linspace(0, B - 1, min(args.show, B)).astype(int):
    text = "".join(tok.decode(a, b) for a, b in zip(seqs[i][:-1], seqs[i][1:]) if b != 1).split("\n")[0]
    print(f"--- story {i}: {text}")
  if not args.float:
    from coral.tpu import runner, MOCK_STATS
    print("tpu stats:", r.stats if (r:=runner()) is not None else MOCK_STATS)
