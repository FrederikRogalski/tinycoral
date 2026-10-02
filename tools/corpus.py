# a corpus of edgetpu_compiler programs with metadata, the oracle of test/test_isa.py
#   python -m tools.corpus            builds tools/data/corpus.pkl.xz
from __future__ import annotations
import pathlib, pickle, lzma, functools
from concurrent.futures import ThreadPoolExecutor
import numpy as np

CORPUS = pathlib.Path(__file__).resolve().parent / "data" / "corpus.pkl.xz"
Q = dict(in_q=(1/32, 128), w_q=(1/64, 128), out_q=(1/4, 128))

def _fc(N, K):
  from tools.tflite_gen import fc_model
  from tools.compiler import compile_tflite
  exes = compile_tflite(fc_model(np.full((N, K), 128, np.uint8), np.zeros(N, np.int32), **Q))[0]
  return {e.type: e for e in exes}

def _relu(n):
  import tflite
  from tools.tflite_gen import Model
  from tools.compiler import compile_tflite
  m = Model()
  x = m.tensor("x", [1, n], np.uint8, 1/16, 128)
  y = m.tensor("y", [1, n], np.uint8, 1/16, 128)
  m.op(tflite.BuiltinOperator.RELU, [x], [y])
  return {e.type: e for e in compile_tflite(m.build([x], [y]))[0]}

def _conv1x1(M, N, K):
  from tools.tflite_gen import conv_model
  from tools.compiler import compile_tflite
  return {e.type: e for e in compile_tflite(conv_model(np.full((N, 1, 1, K), 128, np.uint8), np.zeros(N, np.int32), H=1, W=M, **Q))[0]}

def build() -> list[dict]:
  jobs = [("fc", dict(N=N, K=K)) for N in [16, 64, 128, 192, 256, 320, 448, 512, 640, 768, 1024]
          for K in [16, 32, 64, 96, 128, 192, 256, 288, 320, 512, 768, 1024, 2048]]
  jobs += [("fc", dict(N=N, K=K)) for N, K in [(10, 30), (100, 300), (288, 288), (864, 288), (1536, 288), (2048, 1024), (4096, 1024)]]
  jobs += [("relu", dict(n=n)) for n in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 65536]]
  jobs += [("conv1x1", dict(M=M, N=N, K=K)) for M in [2, 8, 32, 128] for N, K in [(64, 64), (288, 288), (1024, 320)]]
  fns = {"fc": _fc, "relu": _relu, "conv1x1": _conv1x1}
  def run(j):
    kind, meta = j
    try: exes = fns[kind](**meta)
    except Exception: return None
    out = []
    for t, e in exes.items():
      for ci, b in enumerate(e.bitstreams):
        out.append(dict(kind=kind, exe=t, chunk=ci, bitstream=b.data, params=len(e.parameters), hints=[repr(h) for h in e.hints], **meta))
    return out
  with ThreadPoolExecutor(8) as ex: res = list(ex.map(run, jobs))
  return [p for r in res if r for p in r]

@functools.cache
def load_corpus() -> list[dict]: return pickle.loads(lzma.decompress(CORPUS.read_bytes()))

if __name__ == "__main__":
  c = build()
  CORPUS.parent.mkdir(parents=True, exist_ok=True)
  CORPUS.write_bytes(lzma.compress(pickle.dumps(c)))
  print(f"{len(c)} programs, {sum(len(p['bitstream']) for p in c)//16} words -> {CORPUS} ({CORPUS.stat().st_size/1e3:.0f} KB)")
