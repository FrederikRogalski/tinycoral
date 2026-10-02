# run Google's edgetpu_compiler (x86-64 linux) inside a persistent docker container, cached by content hash
import hashlib, pathlib, subprocess, os
from coral.executable import load_edgetpu_tflite, Executable

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORK = pathlib.Path(os.environ.get("CORAL_COMPILE_DIR", ROOT / ".compile"))
CONTAINER = "etpc"

def _ensure_container():
  r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER], capture_output=True, text=True)
  if r.stdout.strip() == "true" and _mounted_ok(): return
  subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
  WORK.mkdir(parents=True, exist_ok=True)
  subprocess.run(["docker", "run", "-d", "--name", CONTAINER, "--platform", "linux/amd64",
                  "-v", f"{ROOT}/ref/edgetpu/compiler/x86_64:/compiler:ro", "-v", f"{WORK}:/work",
                  "debian:bookworm-slim", "sleep", "infinity"], check=True, capture_output=True)

def _mounted_ok() -> bool:
  r = subprocess.run(["docker", "inspect", "-f", "{{range .Mounts}}{{.Source}} {{end}}", CONTAINER], capture_output=True, text=True)
  return str(WORK) in r.stdout

def compile_tflite(model:bytes, flags:tuple[str, ...]=()) -> tuple[list[Executable], str]:
  WORK.mkdir(parents=True, exist_ok=True)
  h = hashlib.sha256(model + repr(flags).encode()).hexdigest()[:16]
  src, out, log = WORK / f"{h}.tflite", WORK / f"{h}_edgetpu.tflite", WORK / f"{h}.stdout"
  if not out.exists():
    src.write_bytes(model)
    _ensure_container()
    r = subprocess.run(["docker", "exec", "-w", "/work", CONTAINER, "/compiler/edgetpu_compiler", "-s", *flags, f"{h}.tflite"],
                       capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr)
    if not out.exists(): raise RuntimeError(f"edgetpu_compiler failed:\n{r.stdout}\n{r.stderr}")
  return load_edgetpu_tflite(out), log.read_text()

def compiled_path(model:bytes, flags:tuple[str, ...]=()) -> pathlib.Path:
  compile_tflite(model, flags)
  return WORK / f"{hashlib.sha256(model + repr(flags).encode()).hexdigest()[:16]}_edgetpu.tflite"
