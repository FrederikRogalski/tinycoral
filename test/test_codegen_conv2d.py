# Acceptance test for coral/codegen/conv2d.py (offline only, never touches the device):
#   1. the 4 convolutions of a small MNIST CNN (28x28x1 -5x5-> 24x24x32, 24x24x32 -5x5-> 20x20x32, 10x10x32 -3x3-> 8x8x64,
#      8x8x64 -3x3-> 6x6x64), compiled alone: caching + execution program byte-exact, the parameter blob (conv2d_blob, random weights
#      and int32 biases) byte-exact, the host contract (input / output DMA sizes, output_layout, blob size);
#   2. shape classes: random shapes per (kernel 1/3/5, stride 1/2, VALID/SAME), H, W in 8..64, Cin in 1..128, Cout in 8..256, the
#      same checks. Layers the compiler splits over several tiles are re-checked with param_tile=compiler_tiles(); shapes the
#      compiler streams (STAND_ALONE) and shapes gen_conv2d refuses (NotImplementedError) are counted as excluded;
#   3. other quantizations (zero points, scales, fused activations) through conv2d_quant;
#   4. 20 convs co-compiled in one compiler call: every program regenerated from the compiler's (tile, offset) placement.
#   The first mismatch of a failing shape is printed: instruction index, opcode and the differing decoded fields (ours, compiler).
#   Compiles run in docker (edgetpu_compiler) and are cached in .compile/ (a cold run compiles ~400 models, a few minutes).
#
#   python test/test_codegen_conv2d.py [--n N] [--seed S] [--classes k3s1v,...]        or        python -m coral.codegen.conv2d [...]
from __future__ import annotations
import sys, pathlib, argparse, random
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
from coral.isa import split, opcode, LAYOUTS, ring_mesh as RM
from coral.codegen import conv2d as C2

PAD = {"VALID": 1, "SAME": 0}          # tools.tflite_gen.conv_model's padding argument
MNIST = [(28, 28, 1, 32, 5, 5, 1, "VALID"), (24, 24, 32, 32, 5, 5, 1, "VALID"), (10, 10, 32, 64, 3, 3, 1, "VALID"),
         (8, 8, 64, 64, 3, 3, 1, "VALID")]

# ============================== reporting ==============================
def decode(ws:list[int]) -> dict: return LAYOUTS[op].decode(ws) if (op:=opcode(ws[0])) in LAYOUTS else {"word": ws[0]}
def name(op:int) -> str: return LAYOUTS[op].name if op in LAYOUTS else f"op{op:#x}"

def first_diff(ours:bytes, ref:bytes) -> str|None:
  """first differing instruction after the start word (whose length differs whenever anything else does): index, word, opcode and
  the differing decoded fields (ours, compiler)"""
  sa, sb = split(ours), split(ref)
  for k, ((ia, wa), (ib, wb)) in [*list(enumerate(zip(sa, sb)))[1:], *list(enumerate(zip(sa, sb)))[:1]]:
    if wa == wb: continue
    oa, ob = opcode(wa[0]), opcode(wb[0])
    if oa != ob or len(wa) != len(wb): return f"instruction #{k} (word {ia}/{ib}): opcode {name(oa)} (ours) vs {name(ob)} (compiler)"
    da, db = decode(wa), decode(wb)
    return f"instruction #{k} (word {ia}) {name(oa)}: differing fields (ours, compiler) {({n: (da[n], db[n]) for n in da if da[n] != db.get(n)})}"
  return None if len(sa) == len(sb) else f"{len(sa)} vs {len(sb)} instructions"

class Report:
  def __init__(self): self.rows: list[tuple[str, int, int]] = []
  def add(self, name:str, results:list[tuple[object, str|None]], excluded:list|None=None):
    fails = [(k, m) for k, m in results if m is not None]
    self.rows.append((name, len(results) - len(fails), len(results)))
    ex = f"   ({len(excluded)} excluded: {_short(excluded)})" if excluded else ""
    print(f"{name:58s} {len(results) - len(fails):5d}/{len(results)}{ex}")
    for k, m in fails[:3]: print(f"    FAIL {k}: {m}")
    sys.stdout.flush()
  @property
  def ok(self) -> bool: return all(p == t for _, p, t in self.rows)

def _short(xs:list, n:int=4) -> str: return ", ".join(map(str, xs[:n])) + (", ..." if len(xs) > n else "")

# ============================== the oracle ==============================
def model(shape:tuple, seed:int=0, quant:tuple|None=None, act:int=0, bias:bool=False):
  """(TFLite model bytes, weights [Cout,kh,kw,Cin], bias or None) of tools.tflite_gen.conv_model for shape; quant = ((in scale, zp),
  (weight scale, zp), (out scale, zp)) or None for conv_model's defaults"""
  from tools.tflite_gen import conv_model
  H, W, Cin, Cout, kh, kw, s, pad = shape
  rng = np.random.default_rng(seed)
  w = rng.integers(0, 256, (Cout, kh, kw, Cin), dtype=np.uint8)
  b = rng.integers(-5000, 5000, Cout).astype(np.int32) if bias else None
  q = dict(zip(("in_q", "w_q", "out_q"), quant)) if quant else {}
  return conv_model(w, b, H=H, W=W, stride=s, padding=PAD[pad], act=act, **q), w, b

def compile_shape(shape:tuple, seed:int=0, quant:tuple|None=None, act:int=0, bias:bool=False):
  from tools.compiler import compile_tflite
  m, w, b = model(shape, seed, quant, act, bias)
  return {e.type: e for e in compile_tflite(m)[0]}, w, b

def _compile_all(jobs:list, workers:int=4) -> dict:
  def run(j):
    try: return j, compile_shape(*j) if isinstance(j[0], tuple) else compile_shape(j)
    except Exception as e: return j, e
  with ThreadPoolExecutor(workers) as ex: return dict(ex.map(run, jobs))

def compare(shape:tuple, exes:dict, w, b, quant:dict|None=None, param_tile=0) -> str|None:
  """both programs byte-exact, the blob, and the host contract (DMA sizes from the hints, output_layout)"""
  H, W, Cin, Cout, kh, kw, s, pad = shape
  ours = C2.gen_conv2d(H, W, Cin, Cout, kh, kw, s, pad, param_tile=param_tile, quant=quant)
  msgs = [f"{n}: {first_diff(a, exes[t].bitstreams[0].data)}" for n, t, a in
          (("caching", "PARAMETER_CACHING", ours[0]), ("execution", "EXECUTION_ONLY", ours[1])) if a != exes[t].bitstreams[0].data]
  if len(exes["EXECUTION_ONLY"].bitstreams) != 1: msgs.append(f"{len(exes['EXECUTION_ONLY'].bitstreams)} execution bitstreams")
  io, eo = C2.conv2d_io(H, W, Cin, Cout, kh, kw, s, pad), exes["EXECUTION_ONLY"]
  sizes = tuple(sum(h.size for h in eo.hints if h.kind == "dma" and h.desc == d) for d in ("INPUT", "OUTPUT"))
  if sizes != (io["input_bytes"], io["output_bytes"]): msgs.append(f"io sizes {(io['input_bytes'], io['output_bytes'])} != compiler {sizes}")
  if {k: list(v) for k, v in eo.outputs[0].output_layout.items()} != io["output_layout"]: msgs.append("output_layout differs")
  blob = exes["PARAMETER_CACHING"].parameters
  if len(blob) != io["param_bytes"]: msgs.append(f"blob size {io['param_bytes']} != compiler {len(blob)}")
  zp = (quant or {}).get("w_zp", 128)
  if C2.conv2d_blob(w, zp, b, H, W, s, pad) != blob: msgs.append("parameter blob differs")
  return "; ".join(msgs) or None

def caching_tiles(exes:dict) -> tuple[int, ...]:
  """the tiles holding the parameters in the compiler's caching program"""
  return tuple(RM.decode_ringConsumer(ws)["tile_mask"].bit_length() - 1 for _, ws in split(exes["PARAMETER_CACHING"].bitstreams[0].data)
               if opcode(ws[0]) == 0x11)

def check(rep:Report, title:str, jobs:list, quants:dict|None=None):
  """jobs: shapes, or (shape, seed, tflite quant, act, bias) tuples; quants: job -> gen_conv2d quant overrides. Layers the compiler
  splits over several tiles must raise NotImplementedError with param_tile=0 and are re-checked with compiler_tiles()"""
  res, excluded, multi = [], [], []
  for j, r in _compile_all(jobs).items():
    shape = j[0] if isinstance(j[0], tuple) else j
    if isinstance(r, Exception):
      res.append((shape, f"compile error {r}"))
      continue
    exes, w, b = r
    if "STAND_ALONE" in exes or "EXECUTION_ONLY" not in exes:
      excluded.append((*shape, "streamed"))
      continue
    q = (quants or {}).get(j)
    try: res.append((shape, compare(shape, exes, w, b, q)))
    except NotImplementedError as e:
      if len(caching_tiles(exes)) > 1:
        try: multi.append((shape, compare(shape, exes, w, b, q, C2.compiler_tiles(C2.Conv2DGeom(*shape)))))
        except Exception as e2: multi.append((shape, f"exception {type(e2).__name__}: {e2}"))
      else: excluded.append((*shape, str(e)[:60]))
    except Exception as e: res.append((shape, f"exception {type(e).__name__}: {e}"))
  rep.add(title, res, excluded)
  if multi: rep.add("  + layers split over tiles (param_tile=compiler_tiles())", multi)

# ============================== shape classes ==============================
def random_shapes(n:int, seed:int, k:int, s:int, pad:str) -> list[tuple]:
  rng, out = random.Random(f"{seed}/{k}/{s}/{pad}"), set()
  while len(out) < n:
    H = rng.randint(8, 64)
    W = H if rng.random() < 0.5 else rng.randint(8, 64)
    Cin = rng.choice([rng.randint(1, 128), rng.choice([1, 3, 4, 8, 16, 32, 64, 128])])
    Cout = rng.choice([rng.randint(8, 256), rng.choice([8, 16, 32, 64, 128, 256])])
    if pad == "VALID" and (H < k or W < k): continue
    out.add((H, W, Cin, Cout, k, k, s, pad))
  return sorted(out)

CLASSES = {f"k{k}s{s}{p[0].lower()}": (k, s, p) for k in (1, 3, 5) for s in (1, 2) for p in ("VALID", "SAME")}

def check_coset(rep:Report, n:int=20, seed:int=4):
  """n random convs co-compiled in one edgetpu_compiler call: the compiler gives each model its own tile, and once the tiles are
  used up, nonzero offsets. Every program is regenerated from that placement (param_tile / param_offset); splits get the set's
  lowest parameter limit."""
  from tools.reloc import cocompile
  rng, shapes = random.Random(seed), []
  while len(shapes) < n:
    k, s, pad = rng.choice([1, 3, 5]), rng.choice([1, 2]), rng.choice(["VALID", "SAME"])
    H, W = rng.randint(8, 32), rng.randint(8, 32)
    if pad == "SAME" or (H >= k and W >= k):
      shapes.append((H, W, rng.choice([1, 3, 8, 16, 32, 64]), rng.choice([16, 32, 64, 128]), k, k, s, pad))
  models = [model(sh, i, None, 0, True) for i, sh in enumerate(shapes)]
  limit = min(C2.param_limit(C2.Conv2DGeom(*sh)) for sh in shapes)
  res = []
  for sh, exes in zip(shapes, cocompile([m for m, _, _ in models])):
    if "PARAMETER_CACHING" not in exes:
      res.append((sh, "streamed"))
      continue
    tiles, H, W, Cin, Cout, kh, kw, s, pad = caching_tiles(exes), *sh
    rc0 = next(RM.decode_ringConsumer(ws) for _, ws in split(exes["PARAMETER_CACHING"].bitstreams[0].data) if opcode(ws[0]) == 0x11)
    off = 64 * (rc0["addr"] - 2)
    ours = C2.gen_conv2d(H, W, Cin, Cout, kh, kw, s, pad, param_tile=tiles if len(tiles) > 1 else tiles[0], param_offset=off,
                         param_limit_units=limit if len(tiles) > 1 else None)
    msgs = [f"{n_}: {first_diff(a, exes[t].bitstreams[0].data)}" for n_, t, a in
            (("caching", "PARAMETER_CACHING", ours[0]), ("execution", "EXECUTION_ONLY", ours[1])) if a != exes[t].bitstreams[0].data]
    res.append(((*sh, tiles, off), "; ".join(msgs) or None))
  rep.add(f"co-compiled set of {n} convs (param_tile / param_offset)", res)

def quant_jobs() -> tuple[list, dict]:
  """the same shapes compiled with other quantizations; gen_conv2d gets the op fields from conv2d_quant"""
  jobs, quants = [], {}
  for i, (shape, xq, wq, yq, act) in enumerate([(MNIST[0], (0.02, 3), (0.01, 140), (0.1, 250), 0), (MNIST[1], (0.05, 100), (1/64, 77), (0.3, 0), 1),
                                                (MNIST[2], (1/32, 128), (1/256, 0), (1/8, 128), 3),
                                                (MNIST[3], (0.003, 200), (0.02, 90), (0.07, 7), 1),
                                                ((16, 16, 8, 24, 3, 3, 2, "SAME"), (0.1, 10), (0.03, 128), (0.2, 50), 0)]):
    j = (shape, i, (xq, wq, yq), act, True)
    jobs.append(j)
    quants[j] = C2.conv2d_quant(xq, wq, yq, act)
  return jobs, quants

def main(argv:list[str]|None=None) -> int:
  ap = argparse.ArgumentParser(description="gen_conv2d vs edgetpu_compiler (offline)")
  ap.add_argument("--n", type=int, default=30, help="random shapes per class (default 30)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--classes", default=",".join(CLASSES), help="comma-separated subset of " + ",".join(CLASSES))
  args = ap.parse_args(argv)
  rep = Report()
  check(rep, "MNIST CNN convolutions (random weights and biases)", [(s, 0, None, 0, True) for s in MNIST])
  for c in args.classes.split(","):
    k, s, p = CLASSES[c]
    check(rep, f"{c}: {k}x{k} stride {s} {p}, seed {args.seed}", random_shapes(args.n, args.seed, k, s, p))
  jobs, quants = quant_jobs()
  check(rep, "other quantizations (zero points, scales, RELU/RELU6)", jobs, quants)
  check_coset(rep)
  print("PASS" if rep.ok else "FAIL")
  return 0 if rep.ok else 1

def test_codegen_conv2d(): assert main(["--n", "5"]) == 0        # pytest entry point

if __name__ == "__main__": sys.exit(main())
