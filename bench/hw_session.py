# hardware tests and benchmarks in sequence, each in its own process with a timeout; stops when the device stops answering.
# The other test/test_hw_*.py take stage arguments and run on their own (see their headers).
#   .venv/bin/python bench/hw_session.py [step ...]        results -> bench/hw_session/<step>.txt
import sys, os, subprocess, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT, PY = ROOT / "bench" / "hw_session", str(ROOT / ".venv" / "bin" / "python")
ALIVE = [PY, "-c", "from coral.device import EdgeTPU; t = EdgeTPU(); print('alive', hex(t.read32('scu_ctrl_0')))"]
LLM = ["1", "8", "32", "128", "256"]
STEPS = {
  "mobilenet":   ([PY, "test/test_mobilenet.py"], {}, 120),
  "programs":    ([PY, "test/test_hw_programs.py"], {}, 300),
  "fc":          ([PY, "test/test_hw_fc.py"], {}, 300),
  "select":      ([PY, "test/test_hw_select.py"], {"DEV": "CORAL"}, 300),
  "fused":       ([PY, "test/test_hw_fused.py", "1", "16", "128", "256"], {}, 600),
  "fc_bench":    ([PY, "bench/fc.py"], {}, 900),
  "llm_fused":   ([PY, "bench/llm_batch.py", "fused", *LLM], {"DEV": "CORAL"}, 1800),
  "llm_matmuls": ([PY, "bench/llm_batch.py", "matmuls", *LLM], {"DEV": "CORAL"}, 1800),
  "stories":     ([PY, "examples/stories_batch.py", "--batch", "256"], {"DEV": "CORAL"}, 900),
  "llm_cpu":     ([PY, "bench/llm_batch.py", "float", *LLM], {"DEV": "CPU"}, 1800),
  "llm_numpy":   ([PY, "bench/numpy_llm.py", *LLM], {}, 900),
  "llm_metal":   ([PY, "bench/llm_batch.py", "float", *LLM], {"DEV": "METAL"}, 1800),
}
HOST_ONLY = {"llm_cpu", "llm_numpy", "llm_metal"}

def run(name, cmd, env, timeout) -> bool:
  st = time.perf_counter()
  try: p = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, capture_output=True, text=True, timeout=timeout)
  except subprocess.TimeoutExpired as e:
    (OUT / f"{name}.txt").write_text(f"TIMEOUT after {timeout}s\n{e.stdout or ''}\n{e.stderr or ''}")
    print(f"{name}: TIMEOUT", flush=True)
    return False
  (OUT / f"{name}.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr[-20000:])
  tail = [l for l in p.stdout.strip().splitlines() if l.strip()][-3:]
  print(f"{name}: exit {p.returncode} in {time.perf_counter()-st:.0f}s", *tail, sep="\n  ", flush=True)
  return p.returncode == 0

if __name__ == "__main__":
  OUT.mkdir(exist_ok=True)
  for name in sys.argv[1:] or list(STEPS):
    if name not in HOST_ONLY and not run("alive", ALIVE, {}, 60):
      print("device not responding: stopping (replug needed)")
      break
    run(name, *STEPS[name])
