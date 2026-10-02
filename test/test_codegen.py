# Acceptance test for coral/codegen/fc.py (offline only, never touches the device):
#   1. byte-exact equality with edgetpu_compiler for every entry of the FC table (tools/data/fc_table.pkl.xz);
#   2. byte-exact equality for non-canonical shapes compiled with tools.fcgen._compile (docker compiler, cached in .compile/);
#   3. param_offset == tools.fcgen.relocate(table program, measured relocation fields, offset);
#   4. tile_shift == translate_tiles (below) where legal, and the same legality verdict;
#
#   python test/test_codegen.py [--extra] [--random K] [--quick]        or        python -m coral.codegen.fc [...]
from __future__ import annotations
import sys, pathlib, argparse, random
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from coral.isa import split, opcode, LAYOUTS
from coral.codegen import fc as cg

TILE_OPS = (0x01, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18)
NONCANONICAL = [(N, K) for N in (10, 100, 288, 864) for K in (30, 288, 300)]
# extra shapes that exercise partial pieces/packets, uneven N>1024 tiles, both narrow layouts, full-pass parameter DMAs
EXTRA = [(64, 196), (64, 200), (64, 228), (64, 600), (64, 1000), (64, 1100), (64, 1500), (64, 2000), (64, 3000), (64, 1028),
         (100, 200), (100, 600), (100, 1000), (100, 1500), (64, 900), (64, 948), (64, 772), (64, 516), (752, 128), (752, 256),
         (1000, 320), (1020, 512), (2000, 1024), (1264, 768), (3, 8), (3, 54), (78, 446), (130, 523), (1681, 913), (4011, 154),
         (3908, 1276), (4058, 1050), (64, 1020), (1, 45), (1, 300), (4, 2604), (958, 4), (767, 115), (2045, 1330), (41, 3896),
         (48, 2728), (733, 256), (1021, 512)]

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

def compare(N:int, K:int, ref_pc:bytes, ref_eo:bytes) -> str|None:
  """our programs vs the compiler's for a zero-bias model (the compiler drops an all-zero bias only for N=1)"""
  try: pc, eo = cg.gen_fc(N, K, bias=N > 1)
  except Exception as e: return f"exception {type(e).__name__}: {e}"
  msgs = [f"{name}: {first_diff(a, b)}" for name, a, b in (("caching", pc, ref_pc), ("execution", eo, ref_eo)) if a != b]
  return "; ".join(msgs) or None

class Report:
  def __init__(self): self.rows: list[tuple[str, int, int, list]] = []
  def add(self, name:str, results:list[tuple[object, str|None]]):
    fails = [(k, m) for k, m in results if m is not None]
    self.rows.append((name, len(results) - len(fails), len(results), fails))
    print(f"{name:58s} {len(results) - len(fails):5d}/{len(results)}")
    for k, m in fails[:3]: print(f"    FAIL {k}: {m}")
    sys.stdout.flush()
  @property
  def ok(self) -> bool: return all(p == t for _, p, t, _ in self.rows)

# *** 1 + 2: byte-exact equality with the compiler ***
def check_table(rep:Report):
  from tools.fcgen import load_table
  tab = load_table()
  big = {2048, 3072, 4096}
  parts = {"table: N<=1024, K<=1024": [], "table: K in {2048,3072,4096}, N<=1024": [], "table: N in {2048,3072,4096}": []}
  for (N, K), e in sorted(tab.items()):
    key = "table: N in {2048,3072,4096}" if N in big else ("table: K in {2048,3072,4096}, N<=1024" if K in big else "table: N<=1024, K<=1024")
    parts[key].append(((N, K), compare(N, K, e[0], e[1])))
  for name, res in parts.items(): rep.add(name, res)

def check_compiled(rep:Report, name:str, shapes:list[tuple[int, int]]):
  from tools.fcgen import _compile
  res = []
  for N, K in shapes:
    r = _compile(N, K)
    res.append(((N, K), "compiler produced no caching+execution pair" if r is None else compare(N, K, r[0], r[1])))
  rep.add(name, res)

def random_shapes(n:int, seed:int=0) -> list[tuple[int, int]]:
  rng, out = random.Random(seed), set()
  while len(out) < n:
    r = rng.random()
    if r < 0.4: s = (rng.randint(1, 1024), rng.randint(5, 1024))
    elif r < 0.7: s = (rng.randint(1, 1024), rng.randint(1025, 4096))
    elif r < 0.9: s = (rng.randint(1025, 4096), rng.randint(5, 1500))
    else: s = (rng.randint(1, 64), rng.randint(5, 64))
    out.add(s)
  return sorted(out)

# *** 3: parameter relocation ***
def check_param_offset(rep:Report, offsets=(256, 4096, 256 * 250)):   # the hardware needs 256-byte aligned regions
  from tools.fcgen import load_table
  from tools.fcgen import relocate
  res = []
  for (N, K), e in sorted(load_table().items()):
    for off in offsets:
      pc, eo = cg.gen_fc(N, K, param_offset=off)
      ok = pc == relocate(e[0], e[4][0], off) and eo == relocate(e[1], e[4][1], off)
      res.append(((N, K, off), None if ok else "differs from relocate(table program, reloc fields)"))
  rep.add(f"param_offset {offsets} vs measured relocation fields", res)

# *** 4: tile placement ***
MESH_DIR = {0x15: "south", 0x16: "west", 0x17: "north", 0x18: "east"}
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
def neighbor(t:int, direction:str) -> int|None:
  """mesh neighbour of tile t (row-major 4x4), None at the edge"""
  r, c = divmod(t, 4)
  dr, dc = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}[direction]
  return 4 * (r + dr) + c + dc if 0 <= r + dr < 4 and 0 <= c + dc < 4 else None

def translate_tiles(bitstream:bytes, s:int) -> bytes:
  """the reference for tile_shift (ring_mesh.md section 6): move every physical tile reference t -> t+s, i.e. the tile masks of all
  tile instructions (opcode < 0x20, incl. 0x1a sync; 0xffff stays, other masks are rotated so 'all but X' masks keep their meaning),
  ringProducer destination tiles and infeed destination bitmaps. Raises ValueError when the move is not a pure translation: a used
  tile would wrap past 15, or a mesh hop would no longer connect neighbours. Narrow/wide addresses, output ordinals and wideToNarrow
  lane selections are logical and are left alone."""
  assert 0 <= s < 16, "s must be in [0, 16)"
  def rot(m): return ((m << s) | (m >> (16 - s))) & 0xffff if s else m
  out, used = bytearray(bitstream), 0
  for i, ws in split(bitstream):
    op, v = opcode(ws[0]), sum(w << (128 * j) for j, w in enumerate(ws))
    if op < 0x20 and (m:=(v >> 12) & 0xffff) not in (0, 0xffff):
      if op != 0x1a: used |= m
      v = v & ~(0xffff << 12) | rot(m) << 12
    for lo in (467,) * (op == 0x10) + (409,) * (op == 0x26):     # ringProducer destination, infeed destination bitmaps
      used |= (d:=(v >> lo) & 0xffff)
      v = v & ~(0xffff << lo) | rot(d) << lo
    if op in MESH_DIR:
      dirn, f = MESH_DIR[op], cg.RM.decode_mesh(ws)
      for t in [t for t in range(16) if f["tile_mask"] >> t & 1]:
        for half, dd in (("o_", dirn), ("i_", OPPOSITE[dirn])):
          if not (f[half + "addr"] or f[half + "cnt0"] or f[half + "inc0"]) or (half == "i_" and f["fill_en"]): continue
          a, b = neighbor(t, dd), neighbor((t + s) % 16, dd)
          if a is None or b != (a + s) % 16 or (a + s) > 15: raise ValueError(f"mesh {dirn} hop at tile {t} breaks when moved by {s}")
    out[16 * i:16 * (i + len(ws))] = v.to_bytes(16 * len(ws), "little")
  if used and used.bit_length() - 1 + s > 15:
    raise ValueError(f"tiles {[t for t in range(16) if used >> t & 1]} moved by {s} would wrap past tile 15")
  return bytes(out)

def _tile_set(bs:bytes) -> dict:
  """per role: union of physical tile bits"""
  roles = {"compute": 0, "input": 0, "infeed": 0, "fwd_dest": 0}
  for _, ws in split(bs):
    op = opcode(ws[0])
    d = decode(ws)
    if op == 0x01: roles["compute"] |= d["tile_mask"]
    if op == 0x14 and d["mode"] == 1: roles["input"] |= d["tile_mask"]
    if op == 0x26: roles["infeed"] |= d["tiles"]
    if op == 0x10 and d["dest"] != 1 << cg.RM.SCALAR_CORE: roles["fwd_dest"] |= d["dest"]
  return roles

def check_tile_shift(rep:Report, quick:bool=False):
  from tools.fcgen import load_table
  keys = sorted(load_table())[::7] if quick else sorted(load_table())
  same, agree, structural = [], [], []
  for N, K in keys:
    pc0, eo0 = cg.gen_fc(N, K)
    g = cg.FCGeom(N, K)
    for s in range(1, 16):
      try:
        ref = (translate_tiles(pc0, s), translate_tiles(eo0, s))
        legal_tr = True
      except ValueError: legal_tr = False
      legal = cg.shift_ok(N, K, s)
      agree.append(((N, K, s), None if legal == legal_tr else f"shift_ok={legal}, translate_tiles legal={legal_tr}"))
      if legal: same.append(((N, K, s), None if cg.gen_fc(N, K, tile_shift=s) == ref else "differs from translate_tiles"))
      # compute-only shift: gather stays on [0, T_in), x is broadcast to [s, s+T)
      if cg.shift_ok(N, K, s, move_gather=False):
        pc, eo = cg.gen_fc(N, K, tile_shift=s, move_gather=False)
        r, want_c = _tile_set(eo), ((1 << g.T) - 1) << s
        msg = None
        if r["compute"] != want_c: msg = f"compute tiles {r['compute']:#x} != {want_c:#x}"
        elif r["input"] != (1 << g.T_in) - 1 or r["infeed"] != (1 << g.T_in) - 1: msg = "input tiles moved"
        elif r["fwd_dest"] != want_c & ~1: msg = f"broadcast dest {r['fwd_dest']:#x} != {want_c & ~1:#x}"
        elif _tile_set(pc)["compute"] or not _seq_ok(eo) or not _seq_ok(pc): msg = "bad sequence numbers / caching tiles"
        structural.append(((N, K, s), msg))
  rep.add("tile_shift legality == translate_tiles legality", agree)
  rep.add("tile_shift programs == translate_tiles(gen_fc(N, K), s)", same)
  rep.add("tile_shift move_gather=False: placement + seq self-consistency", structural)

def _seq_ok(bs:bytes) -> bool:
  n = 0
  for _, ws in split(bs):
    op = opcode(ws[0])
    if op == 0x1a:
      if cg.SC.SYNC.decode(ws)["seq"] != n: return False
      n += 1
    elif op in TILE_OPS:
      if (ws[0] >> 46) & 0x3fff != n: return False
      n += 1
  return True

def main(argv:list[str]|None=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--extra", action="store_true", help="also compare the extra edge-case shapes (compiler, cached)")
  ap.add_argument("--random", type=int, default=0, help="also compare this many random shapes (compiler; ~0.5 s each if not cached)")
  ap.add_argument("--quick", action="store_true", help="subsample the tile_shift checks")
  args = ap.parse_args(argv)
  rep = Report()
  check_table(rep)
  check_compiled(rep, "non-canonical N in {10,100,288,864} x K in {30,288,300}", NONCANONICAL)
  if args.extra: check_compiled(rep, "extra edge cases", EXTRA)
  if args.random: check_compiled(rep, f"random shapes ({args.random})", random_shapes(args.random))
  check_param_offset(rep)
  check_tile_shift(rep, args.quick)
  print("PASS" if rep.ok else "FAIL")
  return 0 if rep.ok else 1

def test_codegen(): assert main(["--quick"]) == 0        # pytest entry point

if __name__ == "__main__": sys.exit(main())
