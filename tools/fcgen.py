# the FC table: edgetpu_compiler's FULLY_CONNECTED programs for every canonical shape (N = 64*t, t <= 16, or 1024*n; K in KS) with
# their parameter relocation fields (tools/reloc.py), the oracle of test/test_codegen.py.  python -m tools.fcgen  rebuilds it.
from __future__ import annotations
import functools, pathlib, pickle, lzma
import numpy as np

TABLE = pathlib.Path(__file__).resolve().parent / "data" / "fc_table.pkl.xz"

KS = list(range(64, 1025, 64)) + [2048, 3072, 4096]
def _compile(Np:int, Kp:int):
  from tools.tflite_gen import fc_model
  from tools.compiler import compile_tflite
  exes = compile_tflite(fc_model(np.full((Np, Kp), 128, np.uint8), np.zeros(Np, np.int32), in_q=(1/32, 128), w_q=(1/64, 128), out_q=(1/4, 128)))[0]
  d = {e.type: e for e in exes}
  if set(d) != {"PARAMETER_CACHING", "EXECUTION_ONLY"}: return None
  pc, eo = d["PARAMETER_CACHING"], d["EXECUTION_ONLY"]
  io = tuple(sum(h.size for h in eo.hints if h.kind == "dma" and h.desc == x) for x in ("INPUT", "OUTPUT"))
  return pc.bitstreams[0].data, eo.bitstreams[0].data, len(pc.parameters), io

def _reloc(Np:int, Kp:int):
  try: return _reloc_(Np, Kp)
  except Exception as e:
    print(f"no relocation fields for N={Np} K={Kp}: {e}")
    return None

def _reloc_(Np:int, Kp:int):
  from tools.reloc import reloc_fields, Q
  from tools.tflite_gen import fc_model
  f = reloc_fields(fc_model(np.random.default_rng(Np * 7919 + Kp).integers(0, 256, (Np, Kp), dtype=np.uint8), np.zeros(Np, np.int32), **Q), Np)
  return None if f is None else (tuple(f["pc"]), tuple(f["eo"]))

def build_table(workers:int=8) -> dict:
  from concurrent.futures import ThreadPoolExecutor
  jobs = [(64*t, k) for t in range(1, 17) for k in KS] + [(1024*n, k) for n in (2, 3, 4) for k in KS]
  with ThreadPoolExecutor(workers) as ex: res = list(ex.map(lambda j: _compile(*j), jobs))
  ok = [j for j, r in zip(jobs, res) if r is not None]
  with ThreadPoolExecutor(workers) as ex: rel = dict(zip(ok, ex.map(lambda j: _reloc(*j), ok)))
  return {j: (*r, rel[j]) for j, r in zip(jobs, res) if r is not None}

def save_table(tab:dict):
  TABLE.parent.mkdir(parents=True, exist_ok=True)
  TABLE.write_bytes(lzma.compress(pickle.dumps(tab), preset=9 | lzma.PRESET_EXTREME))

@functools.cache
def load_table() -> dict: return pickle.loads(lzma.decompress(TABLE.read_bytes()))

def relocate(bs:bytes, fields, delta:int) -> bytes:
  """move a table program's per-tile parameter region by delta bytes (fields: (bit, width, unit) from tools/reloc.py)"""
  v = int.from_bytes(bs, "little")
  for b, w, u in fields:
    assert delta % u == 0, f"offset {delta} not aligned to {u}"
    m = (1 << w) - 1
    v = (v & ~(m << b)) | ((((v >> b) & m) + delta // u) & m) << b
  return v.to_bytes(len(bs), "little")

if __name__ == "__main__":
  import time
  st = time.perf_counter()
  tab = build_table()
  save_table(tab)
  print(f"{len(tab)} programs in {time.perf_counter()-st:.0f}s, table {TABLE.stat().st_size/1e6:.2f} MB at {TABLE}")
