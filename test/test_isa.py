# Corpus test for coral/isa (offline only, never touches the device): every instruction layout round-trips every instruction of the
# corpus (tools/data/corpus.pkl.xz, edgetpu_compiler programs with metadata), and the builders rebuild the corpus instructions
# bit-exactly from the program metadata (shapes, hint sizes) plus the allocation read from the instruction (seq, bases).
#   python test/test_isa.py [--extra] [extra *_edgetpu.tflite ...]
#   --extra: + 162 programs compiled for this study (edgetpu_compiler in docker, cached in .compile/)
from __future__ import annotations
import sys, pathlib, re, math, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from coral.isa import LAYOUTS, cdiv, split, opcode, bits_f32, op as OP, wide_narrow as WN, ring_mesh as RM, scalar as SC, eltwise as EL
from coral.codegen import caching_program
from tools.corpus import load_corpus

TILE_OPS = (0x01, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18)

class Report:
  def __init__(self): self.rows: list[tuple[str, int, int]] = []
  def add(self, name:str, results:list[tuple[object, bool]]):
    bad = [k for k, ok in results if not ok]
    self.rows.append((name, len(results) - len(bad), len(results)))
    print(f"{name:72s} {len(results) - len(bad):6d}/{len(results)}" + (f"   first failures: {bad[:3]}" if bad else ""))
    sys.stdout.flush()
  @property
  def ok(self) -> bool: return all(p == t for _, p, t in self.rows)

# ============================== layouts ==============================
def check_layouts(rep:Report, progs:list[dict], tag:str="", rsv:bool=True):
  """encode(**decode(words)) == words for every instruction with a layout; rsv: and no reserved bit is set (the extra programs use
  reserved fields: MUL / ADD write rsv174, rsv1230, rsv589, ...)"""
  res: dict[str, list] = {}
  for pi, p in enumerate(progs):
    for i, ws in split(p["bitstream"]):
      if (op:=opcode(ws[0])) not in LAYOUTS: continue
      d = LAYOUTS[op].decode(ws)
      ok = LAYOUTS[op].encode(**d) == ws and not (rsv and any(v for k, v in d.items() if k.startswith("rsv")))
      res.setdefault(f"{LAYOUTS[op].name} {op:#04x}", []).append(((pi, i), ok))
  for name, r in sorted(res.items()): rep.add(f"{tag}round trip{', reserved bits 0' if rsv else ''}: {name}", r)

# ============================== op (0x01) ==============================
def check_op(rep:Report, progs:list[dict]):
  """FULLY_CONNECTED ops from (N, K, tile) and RELU requantize ops from (n, tile), + the instruction's allocation and quantization"""
  fc, relu = [], []
  for pi, p in enumerate(progs):
    for i, ws in split(p["bitstream"]):
      if opcode(ws[0]) != 0x01 or p["kind"] not in ("fc", "relu"): continue
      d, t = OP.decode_op(ws), (ws[0] >> 12 & 0xffff).bit_length() - 1
      q = [bits_f32(d[k]) for k in ("mult_bits", "clamp_min_bits", "clamp_max_bits")]
      if p["kind"] == "fc":
        f = OP.fc_tile_fields(p["N"], p["K"], t, d["seq"], d["in_base"], d["out_base"], d["par_base"], d["w_zp"], d["in_zp"], q[0], d["out_zp"],
                              *q[1:])
        fc.append(((pi, i), OP.encode_op(**f) == ws))
      else:   # per tile 64*ceil(ceil(n/64)/16) bytes, each a 1-D requantize op over its words
        n = p["n"]
        nt = min(64 * cdiv(cdiv(n, 64), 16), n - 64 * cdiv(cdiv(n, 64), 16) * t)
        f = EL.requant_op_fields(1, cdiv(nt, 4), cdiv(n, 4), cdiv(n, 4), 1 << t, d["seq"], d["in_base"], d["out_base"],
                                 d["par_base"] | d["par_sel"] << 13,
                                 d["in_zp"], d["out_zp"], q[0], q[1], q[2])
        relu.append(((pi, i), OP.encode_op(**f) == ws))
  rep.add("op: fc_tile_fields reproduces the FC ops", fc)
  rep.add("op: requant_op_fields reproduces the RELU ops", relu)

# ============================== wideToNarrow / narrowToWide ==============================
def wn_strides(d:dict, p:str, levels:int) -> tuple[list[int], list[int]]:
  """(logical strides, iteration counts) of a TTU, the inverse of wide_narrow.ttu"""
  strides, counts, acc = [], [], 0
  for k in range(levels):
    strides.append(d[f"{p}_inc{k}"] + acc)
    counts.append(d[f"{p}_lim{k}"] + 1)
    acc += strides[-1] * d[f"{p}_lim{k}"]
  return strides, counts

def check_wide_narrow(rep:Report, progs:list[dict]):
  ttu_ok, lvl = [], []
  for pi, p in enumerate(progs):
    for i, ws in split(p["bitstream"]):
      if opcode(ws[0]) not in (0x13, 0x14): continue
      d = LAYOUTS[opcode(ws[0])].decode(ws)
      ttu_ok.append(((pi, i), all(WN.ttu(t, *wn_strides(d, t, n), n) == {k: v for k, v in d.items() if k.startswith(t + "_")}
                                  for t, n in (("n", 6), ("w", 4)))))
      lvl.append(((pi, i), d["narrow_lvl_mask"] == sum(1 << k for k, s in enumerate(wn_strides(d, "n", 6)[0]) if s != 0)))
  rep.add("wide_narrow: ttu(strides, counts) == the TTU fields", ttu_ok)
  rep.add("wide_narrow: narrow_lvl_mask = the levels with stride != 0", lvl)
  res = []    # the FC (N <= 1024, execution) and RELU recipes, rebuilt from the shape + seq + the narrow bases
  for pi, p in enumerate(progs):
    if not ((p["kind"] == "fc" and p["exe"] == "EXECUTION_ONLY" and p["N"] <= 1024) or p["kind"] == "relu"): continue
    S = p["K"] if p["kind"] == "fc" else p["n"]
    ins = [ws for _, ws in split(p["bitstream"]) if opcode(ws[0]) in (0x13, 0x14)]
    d14 = [(WN.decode_wide_to_narrow(ws), ws) for ws in ins if opcode(ws[0]) == 0x14]
    d13 = [(WN.decode_narrow_to_wide(ws), ws) for ws in ins if opcode(ws[0]) == 0x13]
    X = next(d["narrow_addr"] for d, _ in d14 if d["mode"] == 1 and d["tile_mask"] == 1)        # allocator choices
    Y = next((d["narrow_addr"] for d, _ in d13 if d["wide_addr"] == 0x2078 and d["tile_mask"] == 1), 0)
    for d, ws in d14:
      if d["mode"] == 1: got = WN.w2n_input(S, d["tile_mask"].bit_length() - 1, X, d["seq"])
      elif d["mode"] == 0: got = WN.w2n_ring_recv(S, d["tile_mask"], X, d["seq"])
      else: got = WN.w2n_bias(d["tile_mask"], d["wide_addr"], 1, d["seq"])
      res.append(((pi, "w2n", d["seq"]), got == ws))
    for d, ws in d13:
      if d["sync_id"] == 5: continue                                    # the identity row (eltwise.ident_prologue)
      t, P = d["tile_mask"].bit_length() - 1, 64 * cdiv(S, 1024)
      if d["wide_addr"] == 0x1f70: got = WN.n2w_output(S, d["tile_mask"], X, 0x1f70, d["seq"])
      elif p["kind"] == "fc": got = WN.n2w_output(min(64, p["N"] - 64 * t), d["tile_mask"], Y + 16 * t, 0x2078, d["seq"])
      else: got = WN.n2w_output(min(P, S - t * P), d["tile_mask"], t * P // 4 if S > 64 else 0xc0, 0x2078, d["seq"])
      res.append(((pi, "n2w", d["seq"]), got == ws))
  rep.add("wide_narrow: FC (N <= 1024) and RELU recipes bit-exact", res)

# ============================== ring / mesh ==============================
def check_ring_mesh(rep:Report, progs:list[dict]):
  """the routing formulas of the FC programs (ring_mesh.md)"""
  chk: dict[str, list] = {}
  def ok(name, key, cond): return chk.setdefault(name, []).append((key, bool(cond)))
  for pi, p in enumerate(progs):
    if p["kind"] != "fc": continue
    N, K = p["N"], p["K"]
    n = max(1, cdiv(N, 1024))
    T = cdiv(N, 64 * n)
    Kq = cdiv(K, 4)
    ds = [LAYOUTS[opcode(ws[0])].decode(ws) for _, ws in split(p["bitstream"]) if opcode(ws[0]) in (0x10, 0x11, 0x12, 0x15, 0x16, 0x17, 0x18)]
    if p["exe"] == "PARAMETER_CACHING":
      cons = [d for d in ds if d["opcode"] == 0x11]
      ok("caching: consumer t on tile t", pi, [d["tile_mask"] for d in cons] == [1 << t for t in range(T)])
      ok("caching: addr = 2+4n, loops (1, Kq-1)", pi,
         all(d["addr"] == 2 + 4 * n and (d["inc0"], d["cnt0"], d["cnt1"]) == (1, Kq - 1, n - 1) for d in cons))
      ok("caching: aux = (0, 4(n-1))", pi, all((d["aux_en0"], d["aux_addr0"], d["aux_addr1"], d["aux_en1"]) == (1, 0, 4 * (n - 1), 1) for d in cons))
      continue
    c = cdiv(int(re.search(r"INPUT:\S* off=0x[0-9a-f]+ size=(0x[0-9a-f]+)", " ".join(p["hints"])).group(1), 16), 256)
    outs = [d for d in ds if d["opcode"] == 0x10 and d["dest"] == 1 << RM.SCALAR_CORE]
    ok("exec: output producer k on tile k, r0_val = k", pi, [(d["tile_mask"], d["r0_val"]) for d in outs] == [(1 << t, t) for t in range(T)])
    fwd = [d for d in ds if d["opcode"] == 0x10 and d["dest"] != 1 << RM.SCALAR_CORE]
    ok("exec: forward producer tile 0 -> tiles 1..T-1", pi, [(d["tile_mask"], d["dest"]) for d in fwd] == ([(1, (1 << T) - 2)] if T > 1 else []))
    grp = [d for d in ds if d["opcode"] == 0x11 and d["addr"] != 8048]
    ok("exec: group consumer addr = 8320-8c, slots = c, s_val = -64c", pi,
       all((d["addr"], d["slots"], d["s_val"], d["s_cnt"]) == (8320 - 8 * c, c, -64 * c, c) for d in grp))
    ok("exec: group consumers on tiles 4g..4g+3", pi,
       all(d["tile_mask"] >> (4 * g) in (1, 3, 7, 15) and d["tile_mask"] % (1 << (4 * g)) == 0 for g, d in enumerate(grp)))
  for name, r in chk.items(): rep.add(f"ring_mesh FC formulas: {name}", r)

# ============================== scalar side ==============================
def _hint_sizes(p:dict) -> dict[str, list[int]]:
  out = {"INPUT": [], "PARAMETER": [], "OUTPUT": []}
  for h in p["hints"]:
    m = re.match(r"<dma (?:in |out) (\w+):\S* off=0x[0-9a-f]+ size=(0x[0-9a-f]+)>", h)
    if m and m.group(1) in out: out[m.group(1)].append(int(m.group(2), 16))
  return out

def sync_variant(ws:list[int], caching:bool=False) -> str|None:
  """the builder that reproduces this 0x1a (sync) or 0x24 (scsync) instruction, or None"""
  if opcode(ws[0]) == 0x1a:
    d = SC.SYNC.decode(ws)
    s, t, c = d["seq"], d["tiles"], d["count"]
    cands = dict(init=SC.sync_init(caching) if s == 0 else None, signal=SC.sync_signal(s), final=SC.sync_final(s, caching), drain=SC.sync_drain(s, t),
                 wn_fence=SC.sync_wn_fence(s), reset17=SC.sync_reset17(s), reset_mesh=SC.sync_reset_mesh(s), rpa=SC.sync_rpa(s), rpb=SC.sync_rpb(s),
                 w2n=SC.sync_w2n(s), mesh_west=SC.sync_mesh(s, 0x16, t, c), mesh_north=SC.sync_mesh(s, 0x17, t, c))
  elif opcode(ws[0]) == 0x24:
    d = SC.SCSYNC.decode(ws)
    cands = dict(init=SC.scsync_init(), fence=SC.scsync_fence(), set_av=SC.scsync_set_av(), av_credit=SC.scsync_av_credit(),
                 set_avpop=SC.scsync_set_avpop(),
                 nop=SC.scsync_nop(), wait_pa=SC.scsync_wait_pa(d["thr5"]), wait_pb=SC.scsync_wait_pb(d["thr6"]), smem_pre=SC.scsync_smem_pre(),
                 smem_post=SC.scsync_smem_post())
  else: return None
  return next((k for k, v in cands.items() if v == ws), None)

def check_scalar(rep:Report, progs:list[dict], tag:str=""):
  """rebuild the scalar-side instructions from the program metadata (hint sizes), check the DMA descriptors (replayed ALU) against
  the hints, find the host DMA sequences and the skeleton blocks verbatim, rebuild every caching program"""
  res: dict[str, list] = {}
  for pi, p in enumerate(progs):
    def chk(name, ok, key=pi): res.setdefault(name, []).append((key, bool(ok)))
    bs, exe, hs = p["bitstream"], p.get("exe"), _hint_sizes(p)
    ops = [(opcode(ws[0]), ws) for _, ws in split(bs)]
    chk("start", ops[0][1] == SC.start(len(bs) // 16))
    chk("end", ops[-1][1] == SC.end())
    n, seq_ok = 0, True     # sync.seq (and tile ops at bit 46) = number of earlier tile-dispatched instructions
    for o, ws in ops:
      if o == 0x1a:
        seq_ok &= SC.SYNC.decode(ws)["seq"] == n
        n += 1
      elif o in TILE_OPS:
        seq_ok &= (ws[0] >> 46) & 0xfff == n
        n += 1
    chk("sequence numbers", seq_ok)
    for i, (o, ws) in enumerate(ops):
      if o in (0x1a, 0x24):
        chk("sync variants" if o == 0x1a else "scsync variants", sync_variant(ws, exe == "PARAMETER_CACHING") is not None, (pi, i))
    chk("halt", [ws for o, ws in ops if o == 0x21] == [SC.halt()])
    pops = [SC.POP.decode(ws) | {"w": ws} for o, ws in ops if o == 0x25]   # activation pops follow the INPUT hints in order
    av = [x for x in pops if x["stream"]]
    chk("av_pop", len(av) == len(hs["INPUT"]) and all(x["w"] == SC.av_pop(s) for x, s in zip(av, hs["INPUT"])))
    if hs["PARAMETER"]:     # a STAND_ALONE program streams its weights: b183 = rows_neg = 0
      want = SC.param_pop(hs["PARAMETER"][0])
      if exe != "PARAMETER_CACHING": want = SC.POP.encode(**(SC.POP.decode(want) | dict(b183=0, rows_neg=0)))
      chk("param_pop", [x["w"] for x in pops if not x["stream"]] == [want])
    feeds = [SC.INFEED.decode(ws) | {"w": ws} for o, ws in ops if o == 0x26]
    lanes = min(4, cdiv(p["N"], 16)) if "N" in p else 4
    for stream, unit, sizes in ((1, 8, hs["INPUT"]), (0, 64, hs["PARAMETER"] if exe == "PARAMETER_CACHING" else [])):
      if not sizes: continue
      groups, cur = [], []
      for f in [f for f in feeds if f["stream"] == stream]:   # each infeed rebuilt from (offset, size, tiles)
        off = (f["pop_wait"] - 1) * unit
        n = (f["count_m1"] + 1 + (0 if stream else f["d1_limit"] * SC.PAR_BUF)) * unit
        if off == 0 and cur:
          groups.append(cur)
          cur = []
        cur.append((off, n))
        chk("av_infeed" if stream else "param_infeed",
            f["w"] == (SC.av_infeed(off, n, f["tiles"]) if stream else SC.param_infeed(off, n, f["tiles"], lanes)), (pi, off))
      groups += [cur] if cur else []
      # per input the chunks are consecutive and cover [0, size); they overlap by one 8-byte unit when a tile boundary is not 8-aligned
      chk("infeeds tile each input",
          len(groups) == len(sizes) and all(g[-1][0] + g[-1][1] == s and all(g[k][0] <= g[k-1][0] + g[k-1][1] for k in range(1, len(g)))
                                                                         for g, s in zip(groups, sizes)))
    outs = [SC.OUTFEED.decode(ws) | {"w": ws} for o, ws in ops if o == 0x27]
    host = [x for x in outs if not x["d3_stride"]]   # host outfeeds: rebuilt from the size; together the OUTPUT hints (padded per tile)
    for x in host: chk("outfeed", x["w"] == SC.outfeed(8 * (x["d0_limit"] + 1)), (pi, x["d0_limit"]))
    if host and len(host) == len(outs): chk("outfeeds total the OUTPUT hints", sum(8 * (x["d0_limit"] + 1) for x in host) == sum(hs["OUTPUT"]))
    want = sorted([(SC.TAG_INPUT, s) for s in hs["INPUT"]] + [(SC.TAG_PARAMETERS, s) for s in hs["PARAMETER"]] +
                  [(SC.TAG_OUTPUT, s) for s in hs["OUTPUT"]] + [(SC.TAG_INT0, 0)])
    chk("DMA descriptors (dma_seqs) == the hints", sorted(SC.dma_seqs(bs)) == want)
    alu = [ws[0] for o, ws in ops if o == 0x20]
    def contains(seq): return any(alu[k:k + len(seq)] == seq for k in range(len(alu) - len(seq) + 1))
    if exe == "PARAMETER_CACHING":
      P = hs["PARAMETER"][0]
      chk("host_dma sequences", contains(SC.host_dma(SC.TAG_PARAMETERS, P, regs="par", load_base=(0, 0))) and contains(SC.interrupt(True)))
      rcs = [ws for o, ws in ops if o == 0x11]   # the blob split equally over the ringConsumer tiles (FC: tile t, conv: one piece)
      part = [({k: v for k, v in RM.decode_ringConsumer(ws).items() if k != "seq"}, t * (P // len(rcs)), P // len(rcs),
               lanes) for t, ws in enumerate(rcs)]
      chk("caching_program", caching_program(P, [part]) == bs)
      continue
    chk("host_dma sequences", all(contains(SC.host_dma(tag, s, load_base=(0, 0))) for tag in (SC.TAG_INPUT, SC.TAG_OUTPUT)
                                  for s in hs["INPUT" if tag == SC.TAG_INPUT else "OUTPUT"]) and contains(SC.interrupt(False)))
    # the skeleton blocks appear verbatim in the scalar-side stream (tile instructions removed but counted for seq; the compiler
    # sometimes interleaves independent tile instructions inside a block)
    flat, seqs_at, n = [], [], 0
    for o, ws in ops:
      if o in TILE_OPS:
        n += 1
        continue
      seqs_at += [n] * len(ws)
      flat += ws
      if o == 0x1a: n += 1
    def has(block_fn):   # block_fn(seq) occurs at a position whose running seq == seq
      cache = {}
      for k in range(len(flat)):
        if (q:=seqs_at[k]) not in cache: cache[q] = block_fn(q)
        b = cache[q]
        if flat[k] == b[0] and flat[k:k + len(b)] == b: return True
      return False
    ok = flat[1:1 + len(SC.exe_prologue())] == SC.exe_prologue() and has(SC.input_head) and has(SC.epilogue)
    ok &= all(has(lambda q, s=s: SC.sync_wn_fence(q) + SC.input_dma(s)) for s in set(hs["INPUT"]))
    if host: ok &= any(has(lambda q, c=c: SC.output_wait(q, c)) for c in range(1, 17))
    chk("skeleton blocks", ok)
  for name, r in res.items(): rep.add(f"{tag}scalar: {name}", r)

def extra_programs(workers:int=4) -> list[dict]:
  """recompile the 162 extra programs of this study (edgetpu_compiler in docker, cached in .compile/)"""
  import numpy as np, tflite
  from concurrent.futures import ThreadPoolExecutor
  from tools.tflite_gen import Model, fc_model, conv_model
  from tools.compiler import compile_tflite
  Q = dict(in_q=(1/32, 128), w_q=(1/64, 128), out_q=(1/4, 128))
  def relu(shape, k=1):
    m = Model()
    xs, ys = [], []
    for i in range(k):
      x = m.tensor(f"x{i}" if k > 1 else "x", shape, np.uint8, 1/16, 128)
      y = m.tensor(f"y{i}" if k > 1 else "y", shape, np.uint8, 1/16, 128)
      m.op(tflite.BuiltinOperator.RELU, [x], [y])
      xs.append(x)
      ys.append(y)
    return m.build(xs, ys)
  def add(shape):
    m = Model()
    a = m.tensor("a", shape, np.uint8, 1/16, 128)
    b = m.tensor("b", shape, np.uint8, 1/16, 128)
    y = m.tensor("y", shape, np.uint8, 1/8, 128)
    def opt(bb):
      tflite.AddOptionsStart(bb)
      return tflite.AddOptionsEnd(bb)
    m.op(tflite.BuiltinOperator.ADD, [a, b], [y], tflite.BuiltinOptions.AddOptions, opt)
    return m.build([a, b], [y])
  jobs = [("relu", dict(n=n), lambda n=n: relu([1, n])) for n in list(range(1, 65)) + [80, 96, 112, 144, 160, 176, 192, 208, 224, 240, 272, 288,
          320, 384, 448, 576, 640, 768, 896, 1152, 1280, 1536, 1792, 2304, 3072, 6144, 12288, 16384, 24576, 32768, 49152]]
  for sh in [(1,2,2,4),(1,4,4,4),(1,4,4,8),(1,4,4,16),(1,8,8,8),(1,8,8,16),(1,8,8,32),(1,16,16,16),(1,16,16,3),(1,7,7,5),(1,32,32,8),(1,2,3,64),
             (1,3,5,7),(1,64,64,4),(1,1,1,100),(1,1,100,1),(1,10,10,1),(1,4,64),(1,16,64),(1,3,100),(1,64,64,64),(1,128,128,8),(1,32,32,128),
             (1,80,64,32),(1,96,64,32),(1,129,128,8),(1,127,128,8),(1,64,64,40),(1,48,64,64),(1,100,100,8)]:
    jobs.append(("relu", dict(shape=sh), lambda sh=sh: relu(list(sh))))
  jobs += [("add", dict(shape=sh), lambda sh=sh: add(list(sh))) for sh in [(1,16),(1,64),(1,256),(1,1024),(1,4096),(1,8,8,16),(1,4,4,4)]]
  jobs += [("relu_multi", dict(n=n, k=k), lambda n=n, k=k: relu([1, n], k)) for n, k in [(16,2),(64,2),(256,2),(64,3),(1024,2)]]
  jobs += [("fc", dict(N=N, K=K), lambda N=N, K=K: fc_model(np.full((N, K), 128, np.uint8), np.zeros(N, np.int32), **Q))
           for N, K in [(32,16),(48,16),(32,64),(48,64),(80,64),(16,4096),(64,4096),(256,4096),(1024,4096),(2048,4096),(4096,4096),(4096,2048)]]
  jobs += [("conv1x1", dict(M=M, N=N, K=K),
            lambda M=M, N=N, K=K: conv_model(np.full((N, 1, 1, K), 128, np.uint8), np.zeros(N, np.int32), H=1, W=M, **Q))
           for M, N, K in [(256,64,320),(200,64,320)]]
  def run(job):
    kind, meta, fn = job
    try: exes = compile_tflite(fn())[0]
    except Exception: return []   # e.g. internal compiler errors for 1-D n > 65536
    return [dict(kind=kind, exe=e.type, bitstream=b.data, hints=[repr(h) for h in e.hints], **meta) for e in exes for b in e.bitstreams]
  with ThreadPoolExecutor(workers) as ex: return [p for r in ex.map(run, jobs) for p in r]

# ============================== NLU (0x19) ==============================
def check_nlu(rep:Report):
  """the LOGISTIC spline: codec round trip, and the decoded piecewise quartic (8 x 5 coefficients, 7 breakpoints) vs 256*sigmoid"""
  w = EL.encode_nlu(**EL.nlu_fields(0xFFFF, 32))
  fl = [bits_f32(s) for s in EL.LOGISTIC_SLOTS]
  def spline(x): return sum(c * x ** j for j, c in enumerate(fl[5 * sum(x >= b for b in fl[40:47]):][:5]))
  err = max(abs(spline(x / 200) - 256 / (1 + math.exp(-x / 200))) for x in range(-2079, 2080))
  rep.add("nlu: 0x19 round trip", [("logistic", EL.encode_nlu(**EL.decode_nlu(w)) == w)])
  rep.add(f"nlu: LOGISTIC spline max |error| vs 256*sigmoid on [-10.4, 10.4] = {err:.4f} < 0.01", [("logistic", err < 0.01)])
  rep.add(f"nlu: clamp {bits_f32(EL.LOGISTIC_CLAMP):.6f}, ADD weights of scales (1/16, 0.0625*pi/10) = (226, 71)",
          [("clamp", abs(bits_f32(EL.LOGISTIC_CLAMP) - 10.396732) < 1e-6), ("add", EL.add_weights(1/16, 0.0625 * math.pi / 10) == (226, 71))])

def main(argv:list[str]|None=None) -> int:
  ap = argparse.ArgumentParser(description="coral/isa layouts and builders vs the edgetpu_compiler corpus (offline)")
  ap.add_argument("--extra", action="store_true", help="also check 162 extra compiled programs (compiler in docker, cached)")
  ap.add_argument("tflite", nargs="*", help="extra *_edgetpu.tflite files: round trip every instruction")
  args = ap.parse_args(argv)
  rep, progs = Report(), load_corpus()
  print(f"{len(progs)} corpus programs")
  check_layouts(rep, progs)
  check_op(rep, progs)
  check_wide_narrow(rep, progs)
  check_ring_mesh(rep, progs)
  check_scalar(rep, progs)
  check_nlu(rep)
  if args.extra:
    extra = extra_programs()
    print(f"{len(extra)} extra programs")
    check_layouts(rep, extra, "extra: ", rsv=False)
    check_scalar(rep, extra, "extra: ")
  if args.tflite:
    from coral.executable import load_edgetpu_tflite
    check_layouts(rep, [dict(bitstream=b.data) for path in args.tflite for e in load_edgetpu_tflite(path) for b in e.bitstreams], "tflite: ",
                  rsv=False)
  print("PASS" if rep.ok else "FAIL")
  return 0 if rep.ok else 1

def test_isa(): assert main([]) == 0        # pytest entry point

if __name__ == "__main__": sys.exit(main())
