# Acceptance test for coral/codegen/conv.py (offline only, never touches the device):
#   1. the TinyStories shapes: Mp in GRIDS x (N, K) in {(864,288), (288,288), (1536,288), (288,768)}, compiled alone with
#      tools.oracle.compile_conv (exes PARAMETER_CACHING + EXECUTION_ONLY): byte-exact;
#   2. fresh compiles of random shapes, N in 64..2048, K in 64..1024, Mp in {16, 64, 256}: byte-exact when the weights fit
#      one tile. Shapes the compiler splits over several tiles are excluded (gen_conv1x1 must raise NotImplementedError for
#      exactly those); they are re-checked with the compiler's own tile list (compiler_tiles) and reported separately;
#   3. the co-compiled sets tools.oracle.coset(Mp): the 24 transformer slots per Mp, generated from their param_tile /
#      param_offset (split slots: tile tuple + the set's lowest parameter limit). The streamed classifier is excluded.
#   Every comparison also checks the host-side contract: input/output DMA sizes, output_layout and parameter blob size.
#   --extra adds ~120 edge-case compiles (K % 4 classes, prime K/4, position-outer ops, scalar-memory outputs, layout and
#   single-tile boundaries). Compiles run in docker (edgetpu_compiler) and are cached in .compile/.
#
#   python test/test_codegen_conv.py [--random N] [--seed S] [--extra]        or        python -m coral.codegen.conv [...]
from __future__ import annotations
import sys, pathlib, argparse, random
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from coral.isa import split, opcode, LAYOUTS
from coral.codegen import conv as cc

LLM_SHAPES = [(864, 288), (288, 288), (1536, 288), (288, 768)]

def decode(ws:list[int]) -> dict: return LAYOUTS[op].decode(ws) if (op:=opcode(ws[0])) in LAYOUTS else {"word": ws[0]}
def name(op:int) -> str: return LAYOUTS[op].name if op in LAYOUTS else f"op{op:#x}"

def first_diff(ours:bytes, ref:bytes) -> str|None:
  """first differing instruction: index, word, opcode and the differing decoded fields (ours, compiler)"""
  sa, sb = split(ours), split(ref)
  for k, ((ia, wa), (ib, wb)) in enumerate(zip(sa, sb)):
    if wa == wb: continue
    oa, ob = opcode(wa[0]), opcode(wb[0])
    if oa != ob or len(wa) != len(wb): return f"instruction #{k} (word {ia}/{ib}): opcode {name(oa)} (ours) vs {name(ob)} (compiler)"
    da, db = decode(wa), decode(wb)
    return f"instruction #{k} (word {ia}) {name(oa)}: differing fields (ours, compiler) {({n: (da[n], db[n]) for n in da if da[n] != db.get(n)})}"
  return None if len(sa) == len(sb) else f"{len(sa)} vs {len(sb)} instructions"

def compare(ours:tuple[bytes, bytes], exes:dict, shape:tuple[int, int, int]|None=None) -> str|None:
  """byte-exact programs; with shape also the host-side contract: DMA sizes (hints) and the output layout"""
  msgs = [f"{name}: {first_diff(a, exes[t].bitstreams[0].data)}" for name, t, a in
          (("caching", "PARAMETER_CACHING", ours[0]), ("execution", "EXECUTION_ONLY", ours[1])) if a != exes[t].bitstreams[0].data]
  if shape is not None:
    eo, (Mp, N, K) = exes["EXECUTION_ONLY"], shape
    sizes = tuple(sum(h.size for h in eo.hints if h.kind == "dma" and h.desc == d) for d in ("INPUT", "OUTPUT"))
    if sizes != cc.io_sizes(Mp, N, K): msgs.append(f"io sizes {cc.io_sizes(Mp, N, K)} != compiler {sizes}")
    lay = {k: list(v) for k, v in eo.outputs[0].output_layout.items()}
    if lay != cc.output_layout(Mp, N): msgs.append("output_layout differs")
    if len(exes["PARAMETER_CACHING"].parameters) != cc.ConvGeom(Mp, N, K).param_bytes: msgs.append("parameter blob size differs")
  return "; ".join(msgs) or None

def caching_tiles(exes:dict) -> tuple[tuple[int, ...], int]:
  """(tiles holding the parameters, byte offset) as the compiler's caching program places them"""
  rcs = [cc.RM.decode_ringConsumer(ws) for _, ws in split(exes["PARAMETER_CACHING"].bitstreams[0].data) if opcode(ws[0]) == 0x11]
  return tuple(d["tile_mask"].bit_length() - 1 for d in rcs), 64 * (rcs[0]["addr"] - 2)

class Report:
  def __init__(self): self.rows: list[tuple[str, int, int]] = []
  def add(self, name:str, results:list[tuple[object, str|None]], excluded:list|None=None):
    fails = [(k, m) for k, m in results if m is not None]
    self.rows.append((name, len(results) - len(fails), len(results)))
    ex = f"   ({len(excluded)} excluded: {_short(excluded)})" if excluded else ""
    print(f"{name:62s} {len(results) - len(fails):5d}/{len(results)}{ex}")
    for k, m in fails[:3]: print(f"    FAIL {k}: {m}")
    sys.stdout.flush()
  @property
  def ok(self) -> bool: return all(p == t for _, p, t in self.rows)

def _short(xs:list, n:int=6) -> str: return ", ".join(map(str, xs[:n])) + (", ..." if len(xs) > n else "")

def _compile_all(jobs:list[tuple[int, int, int]], workers:int=4) -> dict:
  from tools.oracle import compile_conv as _compile
  def run(j):
    try: return j, _compile(*j)
    except Exception as e: return j, e
  with ThreadPoolExecutor(workers) as ex: return dict(ex.map(run, jobs))

def check_shapes(rep:Report, name:str, jobs:list[tuple[int, int, int]], multi_name:str|None=None):
  """byte-exact on single-tile shapes; multi-tile shapes must raise and are re-checked with the compiler's tile list"""
  res, excluded, multi, other = [], [], [], []
  for j, exes in _compile_all(jobs).items():
    if isinstance(exes, Exception):
      other.append((j, f"compile error {exes}"))
      continue
    if "STAND_ALONE" in exes:
      excluded.append((*j, "streamed"))
      continue
    tiles, off = caching_tiles(exes)
    try: ours = cc.gen_conv1x1(*j)
    except NotImplementedError as e:
      if len(tiles) == 1: res.append((j, f"raised but the compiler keeps the weights on one tile: {e}"))
      else:
        excluded.append(j)
        try: multi.append((j, compare(cc.gen_conv1x1(*j, param_tile=cc.compiler_tiles(*j)), exes)))
        except Exception as e2: multi.append((j, f"exception {type(e2).__name__}: {e2}"))
      continue
    except Exception as e:
      res.append((j, f"exception {type(e).__name__}: {e}"))
      continue
    res.append((j, compare(ours, exes, j) if len(tiles) == 1 else f"compiler splits over tiles {tiles}, gen_conv1x1 did not raise"))
  rep.add(name, res + other, excluded)
  if multi_name and multi: rep.add(multi_name, multi)

def check_coset(rep:Report):
  from tools.oracle import coset
  for Mp in cc.GRIDS:
    entries = [(i, N, K, exes) for i, ((N, K), exes) in enumerate(coset(Mp))]
    cached = [(i, N, K, exes) for i, N, K, exes in entries if "PARAMETER_CACHING" in exes]
    limit = min(cc.param_limit(cc.ConvGeom(Mp, N, K)) for _, N, K, _ in cached)   # the set shares the high wide memory
    res = []
    for i, N, K, exes in cached:
      tiles, off = caching_tiles(exes)
      try: ours = cc.gen_conv1x1(Mp, N, K, param_tile=tiles if len(tiles) > 1 else tiles[0], param_offset=off, param_limit_units=limit)
      except Exception as e:
        res.append(((Mp, i, N, K), f"exception {type(e).__name__}: {e}"))
        continue
      res.append(((Mp, i, N, K, tiles, off), compare(ours, exes, (Mp, N, K))))
    streamed = [(i, N, K) for i, N, K, exes in entries if "PARAMETER_CACHING" not in exes]
    rep.add(f"coset Mp={Mp}: {len(cached)} slots (param_tile/param_offset)", res, [(*s, "streamed") for s in streamed])

def random_shapes(n:int, seed:int, Mps=(16, 64, 256)) -> list[tuple[int, int, int]]:
  rng, out = random.Random(seed), set()
  for Mp in Mps:
    k = 0
    while k < n:
      s = (Mp, rng.randint(64, 2048), rng.randint(64, 1024))
      if s not in out:
        out.add(s)
        k += 1
  return sorted(out)

def extra_shapes() -> list[tuple[int, int, int]]:
  from coral.codegen.conv import GRIDS
  out = [(Mp, 64, K) for Mp in GRIDS for K in (65, 66, 67, 68, 71, 72, 101, 133, 263, 997)]       # K % 4 classes, prime K/4
  out += [(Mp, 288, K) for Mp in GRIDS for K in (64, 69, 100, 136, 300, 500)]
  out += [(256, N, K) for N in (64, 130, 288) for K in (64, 65, 67, 68, 72)]                     # position-outer (K/4 <= 17)
  out += [(16, N, 64) for N in (100, 300, 1004, 1028, 1204, 1252, 1796, 2044)]                     # scalar-memory outputs
  out += [(16, N, 77) for N in (65, 1012, 1500)]
  out += [(128, n, 100) for n in (200, 201, 205)] + [(32, n, 64) for n in (260, 261)] + [(16, n, 64) for n in (512, 513, 529)]
  out += [(16, 896, 581), (16, 896, 584), (16, 1408, 372), (128, 960, 540), (128, 1216, 428), (256, 960, 529), (256, 1024, 500)]
  out += [(64, 2048, 1024), (16, 2048, 1024), (128, 2048, 640), (256, 1536, 1000)]               # 2..5 tiles
  return sorted(set(out))

def main(argv:list[str]|None=None) -> int:
  ap = argparse.ArgumentParser(description="gen_conv1x1 vs edgetpu_compiler (offline)")
  ap.add_argument("--random", type=int, default=100, help="random shapes per Mp in {16, 64, 256} (default 100)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--extra", action="store_true", help="also compare ~120 edge-case shapes")
  args = ap.parse_args(argv)
  rep = Report()
  check_shapes(rep, "TinyStories shapes, Mp in GRIDS (tools.oracle.compile_conv)", [(Mp, N, K) for Mp in cc.GRIDS for N, K in LLM_SHAPES])
  if args.random:
    check_shapes(rep, f"random N 64..2048, K 64..1024, Mp in (16,64,256), seed {args.seed}", random_shapes(args.random, args.seed),
                 "  excluded multi-tile shapes, with compiler_tiles (not counted)")
  check_coset(rep)
  if args.extra: check_shapes(rep, "extra edge cases", extra_shapes(), "  extra multi-tile shapes, with compiler_tiles")
  print("PASS" if rep.ok else "FAIL")
  return 0 if rep.ok else 1

def test_codegen_conv(): assert main(["--random", "20"]) == 0        # pytest entry point

if __name__ == "__main__": sys.exit(main())
