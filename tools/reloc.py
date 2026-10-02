# parameter memory relocation: find the fields that hold a program's parameter base address (per tile, in some unit)
# by co-compiling it behind spacer models of two different sizes and diffing.
from __future__ import annotations
import subprocess, hashlib, numpy as np
from tools.tflite_gen import fc_model
from tools.compiler import _ensure_container, WORK, CONTAINER
from coral.executable import load_edgetpu_tflite

Q = dict(in_q=(1/32, 128), w_q=(1/64, 128), out_q=(1/4, 128))

def cocompile(models:list[bytes]) -> list[dict]:
  _ensure_container()
  h = hashlib.sha256(b"".join(models)).hexdigest()[:16]
  d = WORK / f"co_{h}"
  if not (d / f"m{len(models)-1}_edgetpu.tflite").exists():
    d.mkdir(exist_ok=True)
    for i, m in enumerate(models): (d / f"m{i}.tflite").write_bytes(m)
    r = subprocess.run(["docker", "exec", "-w", f"/work/{d.name}", CONTAINER, "/compiler/edgetpu_compiler",
                        *[f"m{i}.tflite" for i in range(len(models))]],
                       capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stdout + r.stderr)
  return [{e.type: e for e in load_edgetpu_tflite(d / f"m{i}_edgetpu.tflite")} for i in range(len(models))]

def spacer(n:int, kp:int, seed:int) -> bytes:
  # same tile count as the target, per-tile parameter bytes = 256 + 64*kp
  return fc_model(np.random.default_rng(seed).integers(0, 256, (n, kp), dtype=np.uint8), np.zeros(n, np.int32), **Q)

def _fields(a:bytes, b:bytes, delta:int) -> list[tuple[int, int, int]]:
  x, y = int.from_bytes(a, "little"), int.from_bytes(b, "little")
  var = [i for i in range(len(a) * 8) if ((x ^ y) >> i) & 1]
  fields, i = [], 0
  while i < len(var):
    lo, best = var[i], None
    for unit in (64, 256, 128, 32, 16, 512, 1024):                   # parameter memory units seen so far: 64B (dma), 256B (op)
      if delta % unit: continue
      for start in range(lo, max(lo - 16, -1), -1):
        for width in (13,):                                            # 13 bits x 64B = the 512 KiB wide memory of a tile
          if not all(start <= v < start + width for v in var[i:i+1]): continue
          fx, fy = (x >> start) & ((1 << width) - 1), (y >> start) & ((1 << width) - 1)
          if (fy - fx) % (1 << width) == delta // unit:
            best = (start, width, unit)
            break
        if best: break
      if best: break
    assert best is not None, f"no consistent field for bit {lo}"
    fields.append(best)
    while i < len(var) and var[i] < best[0] + best[1]: i += 1
  return fields

def reloc_fields(target:bytes, n:int, k1:int=64, k2:int=128) -> dict|None:
  """-> {"pc": [(bit, width, unit)], "eo": [...]}: (field - base) * unit == byte offset of the parameters in each tile"""
  delta = max(1, n // 1024) * 64 * (k2 - k1)        # per tile: (N/1024 groups of 64 outputs) x 64 x K bytes for N > 1024
  for order in (0, 1):
    def mk(kp, seed): return cocompile([spacer(n, kp, seed), target] if order == 0 else [target, spacer(n, kp, seed)])[1 - order]
    a, b = mk(k1, 1), mk(k2, 2)
    out = {name: _fields(a[t].bitstreams[0].data, b[t].bitstreams[0].data, delta)
           for name, t in [("pc", "PARAMETER_CACHING"), ("eo", "EXECUTION_ONLY")]}
    if out["pc"] or out["eo"]: return out
  return None
