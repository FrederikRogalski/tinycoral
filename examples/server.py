# a little web app for TinyStories-15M on the Coral: write a story together with the model, or watch a batch of stories
# grow at once; live tokens/s, usb traffic, and where every layer's weights sit in the 16 tiles
#   DEV=CORAL python examples/server.py [--port 8642]        then open http://localhost:8642
#   DEV=CORAL CHIP=1 python examples/server.py                writing runs all 6 layers on the chip, one program per token
import os, sys, json, time, threading, pathlib, argparse, numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
os.environ.setdefault("WQKV", "1")
sys.path.insert(0, os.path.dirname(__file__))
import stories, stories_batch
from tinygrad import Tensor, Device, nn, TinyJit, Variable

class Engine:
  """the models (one per batch size: each keeps its own KV cache and JIT) and the generation loop; one request at a time.
  CHIP=1: writing (batch 1) runs the whole transformer on the Edge TPU, one program per token (examples/stories_chip.py)"""
  def __init__(self):
    self.cfg, self.weights = stories.load_checkpoint(stories.ROOT / "models/stories15M.bin")
    self.tok = stories.Tokenizer(stories.ROOT / "models/tokenizer.bin", self.cfg["vocab_size"])
    self.models, self.lock, self.chip = {}, threading.Lock(), None
    if os.getenv("CHIP") and Device.DEFAULT == "CORAL":
      import stories_chip
      from coral.codegen import layer as LY
      from coral.tpu import runner
      _, w = LY.load_checkpoint()
      self.chip = stories_chip.Chip(w, *stories_chip.calibration(w), 256, local=True, tpu=runner().tpu)
  def chip_generate(self, prompt:str, steps:int, temperature:float):
    """batch 1 on the chip: one TPU call per token runs all 6 layers; the host embeds, classifies and samples"""
    from coral.runtime import TRAFFIC
    from coral.tpu import runner
    r = runner()
    if r.owner != "chip": self.chip.upload(); r._own("chip")     # its weights fill the tiles: coral.tpu re-uploads its own later
    rng, toks = np.random.default_rng(), self.tok.encode(prompt)
    toks = toks if toks[:1] == [1] else [1] + toks
    for pos in range(min(len(toks) + steps, self.cfg["max_context"]) - 1):
      t0, out0, in0 = time.perf_counter(), TRAFFIC["to_device"], TRAFFIC["from_device"]
      lg = self.chip.step(toks[pos], pos).astype(np.float64)
      dt = time.perf_counter() - t0
      if pos + 1 < len(toks): yield dict(pos=pos, deltas=[""], ms=dt * 1e3, tok_s=1 / dt, tpu_calls=1, to_device=TRAFFIC["to_device"] - out0,
                                         from_device=TRAFFIC["from_device"] - in0, done=False); continue
      nxt = int((lg / temperature + rng.gumbel(size=lg.shape)).argmax() if temperature > 0 else lg.argmax())
      done = nxt == 1
      if not done: toks.append(nxt)
      yield dict(pos=pos, deltas=["" if done else self.tok.decode(toks[-2], toks[-1])], ms=dt * 1e3, tok_s=1 / dt, tpu_calls=1,
                 to_device=TRAFFIC["to_device"] - out0, from_device=TRAFFIC["from_device"] - in0, done=done)
      if done: break
  def model(self, B:int):
    if B not in self.models:
      m = stories.build(self.cfg, self.weights, nn.Linear, jit=False)
      on_chip = B >= 64                          # the classifier as an on-chip block-argmax from batch 64 (token ids out)
      if Device.DEFAULT == "CORAL": stories.coralize_fused(m, self.cfg, self.weights, tok=self.tok, classifier=on_chip)
      fwd = TinyJit(lambda t, pos: m.forward(t, pos, float("nan"), 0, 0.8, 0.0, 0.0).realize())
      self.models[B] = (fwd, on_chip and Device.DEFAULT == "CORAL")
    return self.models[B]
  def generate(self, prompts:list[str], steps:int, temperature:float):
    """yields one event per decoding step: the new text of every row and the step's stats"""
    if len(prompts) == 1 and self.chip is not None: yield from self.chip_generate(prompts[0], steps, temperature); return
    from coral import fused
    from coral.runtime import TRAFFIC
    from coral.tpu import runner
    B, rng = len(prompts), np.random.default_rng()
    fwd, ids_out = self.model(B)
    fused.SAMPLE.update(temperature=temperature, rng=rng)
    ps = [[1] + self.tok.encode(p)[1:] if self.tok.encode(p)[:1] == [1] else self.tok.encode(p) for p in prompts]
    seqs, done = [list(p[:1]) for p in ps], [False] * B
    r = runner() if Device.DEFAULT == "CORAL" else None
    for pos in range(min(steps + max(map(len, ps)), self.cfg["max_context"] - 1)):
      calls0, t0, out0, in0 = r.stats["calls"] if r else 0, time.perf_counter(), TRAFFIC["to_device"], TRAFFIC["from_device"]
      out = fwd(Tensor(np.array([[s[-1]] for s in seqs], np.int32)), Variable("start_pos", 0, self.cfg["max_context"] - 1).bind(pos)).numpy()
      if ids_out: nxt = out[:, -1]
      else:
        lg = out[:, -1, :].astype(np.float64)
        nxt = (lg / temperature + rng.gumbel(size=lg.shape)).argmax(1) if temperature > 0 else lg.argmax(1)
      dt, deltas = time.perf_counter() - t0, []
      for i, s in enumerate(seqs):
        if pos + 1 < len(ps[i]): s.append(ps[i][pos + 1]); deltas.append(""); continue   # still feeding the prompt
        if done[i] or int(nxt[i]) == 1: done[i] = True; deltas.append(""); continue      # end of story
        s.append(int(nxt[i])); deltas.append(self.tok.decode(s[-2], s[-1]))
      yield dict(pos=pos, deltas=deltas, ms=dt * 1e3, tok_s=B / dt, tpu_calls=(r.stats["calls"] - calls0) if r else 0,
                 to_device=TRAFFIC["to_device"] - out0, from_device=TRAFFIC["from_device"] - in0, done=all(done))
      if all(done): break

  def tiles(self, B:int) -> list[list[dict]]:
    """where the weights sit: per tile, the regions [start, end) bytes of every block (our placement plan)"""
    from coral import fused
    from coral.codegen import fused as cg
    from coral.tpu import bucket
    if Device.DEFAULT != "CORAL": return dict(tiles=[[] for _ in range(16)], limit=524288)
    self.model(B)
    Mp = 1 if B == 1 else bucket(max(B, 16))
    blocks = [b for b in fused.REGISTRY.values() if Mp > 1 or b.kind != "argmax"]
    pl, out = cg.plan(blocks, Mp), [[] for _ in range(16)]
    for s in cg.block_specs(blocks):
      for t, a, z in cg.regions(s, Mp, pl[s.name]): out[t].append(dict(name=s.name, start=a * 64, end=z * 64))
    return dict(tiles=out, limit=cg.set_limit(cg.block_specs(blocks), Mp) * 64)   # above the limit: the programs' own buffers

ENGINE, PAGE = None, pathlib.Path(__file__).with_name("coral.html")

class Handler(BaseHTTPRequestHandler):
  def log_message(self, *a): pass
  def send(self, code:int, body:bytes, ctype:str):
    self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers()
    self.wfile.write(body)
  def do_GET(self):
    if self.path in ("/", "/index.html"): return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
    if self.path.startswith("/api/info"):
      info = dict(device=Device.DEFAULT, model="TinyStories-15M (llama2.c), uint8 on the Edge TPU", prompts=stories_batch.prompts(256),
                  chip=ENGINE.chip is not None)
      return self.send(200, json.dumps(info).encode(), "application/json")
    if self.path.startswith("/api/tiles"):
      B = int(self.path.split("B=")[1]) if "B=" in self.path else 1
      with ENGINE.lock: body = json.dumps(ENGINE.tiles(B)).encode()
      return self.send(200, body, "application/json")
    self.send(404, b"not found", "text/plain")
  def do_POST(self):
    if self.path != "/api/generate": return self.send(404, b"not found", "text/plain")
    req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
    prompts = req.get("prompts") or [req.get("prompt", "Once upon a time")]
    self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
    with ENGINE.lock:                             # one generation at a time: the device and the KV caches are shared
      for ev in ENGINE.generate(prompts, int(req.get("steps", 200)), float(req.get("temperature", 0.8))):
        try: self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): break

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--port", type=int, default=8642)
  args = ap.parse_args()
  ENGINE = Engine()
  for B in ((64,) if ENGINE.chip else (1, 64)): ENGINE.model(B)   # build (and calibrate) the models before the first request
  print(f"device {Device.DEFAULT}: http://localhost:{args.port}", flush=True)
  ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
