# coral/loop.py, "instructions once": one instruction upload runs a compiled program N times.
#   offline (default, never touches the device): every execution program of the corpus (tools/data/corpus.pkl.xz) is looped and
#     checked: it re-splits into instructions, `start` carries the new length, the branch lands on the body and the body runs N times.
#   --hw (the device): Google's MobileNet v1 (ref/edgetpu/test_data, as test_mobilenet.py) on the images of test_data, cycled over
#     N frames (default 100): every frame through the normal path (one instruction upload per frame) and through the loop (one upload
#     for all N), byte for byte; the top-5 labels per image, and the CPU TFLite reference when mobilenet_v1_1.0_224_quant.tflite and
#     ai-edge-litert are there; afterwards the normal path must still work. More single-chunk models can be given (random inputs).
#   .venv/bin/python test/test_loop.py [--hw] [--frames N] [MODEL_edgetpu.tflite ...]
from __future__ import annotations
import sys, time, pathlib, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
from coral.loop import looped, check, run_loop, frame_size
from coral.executable import load_edgetpu_tflite

TD = ROOT / "ref/edgetpu/test_data"
MOBILENET = TD / "mobilenet_v1_1.0_224_quant_edgetpu.tflite"

def offline(frames:int) -> bool:
  from tools.corpus import load_corpus
  progs = [p for p in load_corpus() if "CACHING" not in str(p["exe"])]
  bad = []
  for i, p in enumerate(progs):
    try:
      prog, info = looped(p["bitstream"], frames)
      res = check(prog, frames, info)
      assert info["total_words"] == len(p["bitstream"]) // 16 + 4 and res["counter_final"] == 0 and not res["p0_final"]
    except AssertionError as e: bad.append((i, p["kind"], str(e)[:60]))
  print(f"corpus execution programs looped x{frames} and checked: {len(progs) - len(bad)}/{len(progs)}"
        + (f"   first failures: {bad[:3]}" if bad else ""))
  return not bad

def images() -> list[tuple[str, bytes]]:
  """the test_data bitmaps as MobileNet inputs: 224x224 RGB uint8"""
  from PIL import Image
  return [(p.name, np.asarray(Image.open(p).convert("RGB").resize((224, 224)), dtype=np.uint8).tobytes()) for p in sorted(TD.glob("*.bmp"))]

def cpu_reference(x:bytes) -> np.ndarray | None:
  try: from ai_edge_litert.interpreter import Interpreter
  except ImportError: return None
  if not (TD / "mobilenet_v1_1.0_224_quant.tflite").exists(): return None
  it = Interpreter(str(TD / "mobilenet_v1_1.0_224_quant.tflite"))
  it.allocate_tensors()
  it.set_tensor(it.get_input_details()[0]["index"], np.frombuffer(x, np.uint8).reshape(1, 224, 224, 3))
  it.invoke()
  return it.get_tensor(it.get_output_details()[0]["index"])[0]

def top5(labels:list[str], v:np.ndarray) -> list[tuple[str, int]]: return [(labels[i], int(v[i])) for i in np.argsort(-v.astype(int))[:5]]

def compare(tpu, exe, name:str, xs:list[bytes]) -> tuple[bool, list[bytes]]:
  """every frame through the normal path and through one looped upload -> (all bit-exact and the normal path still works, looped outputs)"""
  from coral.runtime import run_executable
  frames = len(xs)
  t0 = time.perf_counter()
  ref = [run_executable(tpu, exe, x) for x in xs]
  t1 = time.perf_counter()
  prog, info = looped(exe.bitstreams[0].data, frames)
  check(prog, frames, info)
  t2 = time.perf_counter()
  got = run_loop(tpu, exe, xs, prog)
  t3 = time.perf_counter()
  exact = sum(a == b for a, b in zip(ref, got))
  after = run_executable(tpu, exe, xs[0]) == ref[0]
  ok = exact == frames and after
  print(f"{name}: {len(exe.bitstreams[0].data) // 16} -> {info['total_words']} words; normal {1e3 * (t1 - t0) / frames:.2f} ms/frame, "
        f"looped {1e3 * (t3 - t2) / frames:.2f} ms/frame; {frames} frames bit-exact {exact}/{frames}; normal path afterwards: {after}"
        f"  {'OK' if ok else 'FAIL'}", flush=True)
  return ok, got

def hw(paths:list[pathlib.Path], frames:int) -> bool:
  from coral.device import EdgeTPU
  from coral.runtime import run_executable
  tpu = EdgeTPU()
  rng = np.random.default_rng(0)
  allok = True
  for path in paths:
    exes = load_edgetpu_tflite(path)
    for pc in [e for e in exes if e.type == "PARAMETER_CACHING"]: run_executable(tpu, pc)
    exe = [e for e in exes if e.type != "PARAMETER_CACHING"][0]
    if len(exe.bitstreams) != 1:
      print(f"{path.name}: {len(exe.bitstreams)} chunks, not loopable")
      allok = False
      continue
    imgs = images() if path == MOBILENET else []
    if imgs:
      xs = [imgs[i % len(imgs)][1] for i in range(frames)]                      # the images in turn, frames times
      ok, got = compare(tpu, exe, f"MobileNet v1 on {', '.join(n for n, _ in imgs)} in turn", xs)
      labels = (TD / "imagenet_labels.txt").read_text().splitlines()
      for i, (n, x) in enumerate(imgs):
        out = np.frombuffer(got[i], np.uint8)[:1001]
        print(f"   {n:22s} looped top5: {top5(labels, out)}")
        ref = cpu_reference(x)
        if ref is not None:
          print(f"   {'':22s} cpu    top5: {top5(labels, ref)}   max abs diff vs cpu: {np.abs(out.astype(int) - ref.astype(int)).max()}")
    else:
      xs = [rng.integers(0, 256, frame_size(exe), dtype=np.uint8).tobytes() for _ in range(frames)]
      ok, _ = compare(tpu, exe, f"{path.name} (random inputs)", xs)
    allok &= ok
  return allok

if __name__ == "__main__":
  ap = argparse.ArgumentParser(description="coral/loop.py: one instruction upload, N frames")
  ap.add_argument("--hw", action="store_true", help="run on the device: MobileNet v1 from ref/edgetpu/test_data on its images, plus MODEL ...")
  ap.add_argument("--frames", type=int, default=100)
  ap.add_argument("models", nargs="*", type=pathlib.Path)
  a = ap.parse_args()
  ok = offline(a.frames) if not a.hw else hw(([MOBILENET] if MOBILENET.exists() else []) + a.models, a.frames)
  sys.exit(0 if ok else 1)
