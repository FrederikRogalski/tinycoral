# the fused LLM blocks (coral/fused.py) on hardware, one at a time and with a log line before every call,
# so that a hang points at its program:  .venv/bin/python test/test_hw_fused.py 16 64 128
import sys, os, time, numpy as np
os.environ.setdefault("WQKV", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import stories
from tinygrad import nn
from coral import fused
from coral.tpu import runner

if __name__ == "__main__":
  cfg, weights = stories.load_checkpoint(stories.ROOT / "models/stories15M.bin")
  stories.coralize_fused(stories.build(cfg, weights, nn.Linear, jit=False), cfg, weights)
  r, rng = runner(), np.random.default_rng(0)
  for Mp in [int(a) for a in sys.argv[1:]] or [16]:
    for name in ["L0.qkv", "L0.wo", "L0.ffn"] + (["cls"] if Mp > 1 else []):
      blk = fused.REGISTRY[name]
      # realistic activations: within the calibrated range (random uint8 would saturate the classifier's logits)
      x = np.clip(blk.in_q[1] + rng.normal(0, 12, (Mp, blk.K)), 0, 255).astype(np.uint8)
      h = (x.astype(np.float32) - blk.in_q[1]) * blk.in_q[0]
      print(f"Mp={Mp:3d} {name:6s} ...", end=" ", flush=True)
      r.run_block(name, x, h)                    # first call: compile lookup + upload of the weights
      st = time.perf_counter()
      y = r.run_block(name, x, h)
      t = time.perf_counter() - st
      if blk.kind == "argmax":
        exact = (h @ blk.extra["Wv"].T).argmax(1)
        print(f"{t*1e3:7.2f} ms, argmax agrees with float on {np.mean(y == exact)*100:.1f}% of rows", flush=True)
      else:
        d = np.abs(y.astype(int) - fused.reference(blk, x).astype(int))
        print(f"{t*1e3:7.2f} ms, vs float reference: max diff {d.max()}, mean {d.mean():.3f}", flush=True)
  print("stats", r.stats)
