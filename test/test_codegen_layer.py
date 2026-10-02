# Acceptance test for coral/codegen/layer.py, offline (never touches the USB device):
#   1. the L2 norm stage with another output quantization: the bit model's arithmetic, (1/128, 128) = eltops.l2norm_ref
#   2. the parameter blob and the caching program of a plan (every region where the plan says; spread and streamed matmuls); plan(6)
#      fits below the program buffers of every P
#   3. gen_model(P) for 1 layer at P = 1..256 and 6 layers at sample P: every instruction decodes and re-encodes; sequence numbers
#      across bitstreams; DMA descriptors = host contract; every narrow range inside one buffer of model_alloc; FC ops and bias loads
#      inside their matmul's region, every other wide address at or above the parameter limit; data flow (on every tile every byte an
#      instruction reads was written earlier in program order); the swap-read MULs read q / k in pair-swapped order; the slot copies are
#      the last writers of exactly row P-1 of K and V
#   4. the bit models on a calibrated model: the layer near float math, the whole model's logits near float
#
#   python test/test_codegen_layer.py [--quick]        or        python -m coral.codegen.layer [--quick]
from __future__ import annotations
import sys, pathlib, argparse, functools
ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "test"):
  if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import numpy as np
from coral.isa import opcode, LAYOUTS, scalar as SC, scalar_core as S, ring_mesh as RM, op as OP
from coral.codegen import attention as A, eltops as E, layer as LY
from test_codegen_attention import Report, narrow_accesses, _decode, _op_levels
from test_codegen_attention_block import dataflow

def one_program(bss:list[bytes]) -> bytes:
  """the bitstreams of a split program as one (start / end and the cut points' halt + nops dropped): for the program-order checks"""
  words = []
  for i, bs in enumerate(bss):
    ins = [ws for _, ws in S.split(bs)][1:-1]
    if i < len(bss) - 1: ins = ins[:-5]                       # halt(1) + 4 nops
    words += [w for ws in ins for w in ws]
  return SC.program(words)

def streamed_exact(prog:bytes) -> bytes:
  """the program with every streamed FC op's input walk cut to the K/4 words it uses: its in TTU nominally walks whole FIFO fills
  of 32 words, the last fill's records (rsv174: r - 1 with bit 15) stop it after r words"""
  out = []
  for _, ws in S.split(prog):
    d = _decode(ws) if opcode(ws[0]) == 1 else None
    if d is not None and d["dp_mode"] == 3 and d["par_tflags"] == 0x80:
      kw = 32 * d["loop2"] + ((d["rsv174"] & 0x7fff) + 1 if d["rsv174"] & 0x8000 else 32)
      ws = OP.encode_op(**(d | OP.ttu_fine("in", [4], [kw])))
    out += ws
  return b"".join(w.to_bytes(16, "little") for w in out)

def accesses(prog:bytes) -> list[tuple[int, str, int, tuple[int, int]]]:
  """test_codegen_attention.narrow_accesses, with an ADD's (dp_mode 5) two operands as two ranges (they sit in different buffers)"""
  out, ins = [], S.split(prog)
  for k, what, tiles, rng in narrow_accesses(prog):
    d = _decode(ins[k][1]) if what == "op.in" else None
    if d is not None and d["dp_mode"] == 5:
      lv = _op_levels(d, "in")
      w = (lv[5][1] - 1) * lv[5][0] + 4                       # the operand's words (level 5: 4 bytes x w)
      for b in (d["in_base"], d["in_base"] + lv[1][0]): out.append((k, "add.in", tiles, (b, b + w)))
    else: out.append((k, what, tiles, rng))
  return out

# ============================== 1. L2 norm with another output scale ==============================
def check_l2(rep:Report):
  rng, res = np.random.default_rng(1), []
  for in_q in ((1/16, 128), (0.0417, 117), (0.0031, 140)):
    x = rng.integers(0, 256, (20, 288)).astype(np.uint8)
    a = np.stack([LY.l2norm_ref(r, in_q) for r in x])
    b = E.l2norm_ref(x, in_q)
    res.append((("(1/128, 128) = eltops", in_q), None if np.array_equal(a, b) else f"{np.abs(a.astype(int) - b).max()} LSB"))
    for out_q in ((1/256, 128), (1/200, 120)):
      y = np.stack([LY.l2norm_ref(r, in_q, out_q) for r in x])
      d = (x.astype(float) - in_q[1])
      f = d / np.linalg.norm(d, axis=1, keepdims=True)
      exact = np.clip(np.rint(f / out_q[0]) + out_q[1], 0, 255)
      err = np.abs(y - exact).max()
      res.append(((in_q, out_q), None if err <= 1 else f"{err} LSB from float"))
  rep.add("L2 norm bit model: TFLite's output = eltops.l2norm_ref, other output scales within 1 LSB of float", res)

# ============================== 2. parameters and placement ==============================
def check_params(rep:Report):
  from coral.codegen.fused import conv_blob
  rng, res = np.random.default_rng(2), []
  L = 6
  pl = LY.plan(L)
  res.append(("plan(6) fits every P (no overlap, aligned, below the limit)", LY.check_plan(pl, LY.model_limit()) or None))
  Ws = [{k: rng.integers(0, 256, s, dtype=np.uint8) for k, s in (("wqkv", (864, 288)), ("wo", (288, 288)), ("w1", (768, 288)), ("w3", (768, 288)),
                                                                   ("w2", (288, 768)))} for _ in range(L)]
  qs = [{k: (0.01, int(rng.integers(100, 156))) for k in LY.WEIGHTS} for _ in range(L)]
  blob = LY.model_params(Ws, qs)
  want = b"".join(conv_blob(src, q[k][1]) for W, q in zip(Ws, qs) for src, k in ((W["wqkv"][:288][LY.SIGMA], "wqkv"),
                                                                                 (W["wqkv"][288:576][LY.SIGMA], "wqkv"),
                                                                                 (W["wqkv"][576:], "wqkv"), (W["wo"], "wo"), (W["w1"], "w1"),
                                                                                 (W["w3"], "w3"), (W["w2"], "w2")))
  res.append(("blob = conv_blob of q, k (rows in the order SIGMA), v, wo, w1, w3, w2 per layer",
              None if blob == want and len(blob) == LY.model_io(2, L)["param_bytes"] else "differs"))
  # the caching program writes every group where the plan puts it: (tile, first unit) of every ringConsumer, in blob order
  pc = LY.model_caching(pl)
  rcs = [RM.decode_ringConsumer(ws) for _, ws in S.split(pc) if opcode(ws[0]) == 0x11]
  got = [(d["tile_mask"], d["addr"] - 2 - (4 if d["aux_en0"] else 0)) for d in rcs]
  want = []
  for d in pl:
    for m in LY.MATS:
      p, g = d[m], LY.GEOM[m]
      want += [(1 << t, p[1]) for t in range(g.T)] if p[0] == "spread" else [(1 << t, o) for t, o in p[1]]
  infeeds = [SC.INFEED.decode(ws) for _, ws in S.split(pc) if opcode(ws[0]) == 0x26]
  res.append(("caching: one ringConsumer + infeed per group, at the plan's (tile, unit)", None if got == want and len(infeeds) == len(want) else
              f"{len(got)} / {len(want)} consumers, first diff {next(((a, b) for a, b in zip(got, want) if a != b), None)}"))
  rep.add("parameter blob, placement, caching program", res)

# ============================== 3. the programs ==============================
def calib_small(L:int=6):
  """a quick calibration (one short story of the float model) for the structural checks"""
  cfg, w = LY.load_checkpoint()
  rec = LY.Ranges()
  fm = LY.FloatModel(w, rec)
  for p, t in enumerate([1, 403, 2501, 727, 931, 278, 263, 931, 2041, 4086, 278, 1034, 29889]): fm.hidden(t, p)
  return LY.quants_from(w, rec)[:L]

def check_program(P:int, L:int, quants:list[dict], stop:tuple|None=None, pl:list|None=None, local:bool=False) -> str|None:
  pl = LY.plan(L) if pl is None else pl
  pc, bss, io = LY.gen_model(P, quants, pl, stop, local=local)
  if any(len(b) // 16 > 16384 for b in bss): return f"a bitstream of more than 16384 words: {[len(b) // 16 for b in bss]}"
  prog = one_program(bss)
  g, a = A.Geom(P), LY.model_alloc(P, L)
  ins = S.split(prog)
  seq = 0
  for k, (_, ws) in enumerate(ins):
    oc = opcode(ws[0])
    if oc in (0x20, 0x22, 0x23): continue
    d = _decode(ws)
    if d is None: return f"#{k}: opcode {oc:#x} does not decode"
    if (S.OP3 if oc == 3 else LAYOUTS[oc]).encode(**d) != ws: return f"#{k}: re-encoding differs"
    if oc == 0x1a or 0x01 <= oc <= 0x19:
      if d["seq"] != seq: return f"#{k}: seq {d['seq']} != {seq}"
      seq += 1
  if seq >= 1 << 14: return f"{seq} tile instructions: the 14-bit seq fields of the DMA instructions overflow"
  try: LY.model_executables(pc, bss, io)
  except AssertionError as ex: return f"host contract: {ex}"
  ffn_stop = stop is not None and stop[1] in LY.FFN_STOPS        # those output a 768-byte vector from tiles 0..11 (fc.output_block)
  if any(n not in ((768,) if ffn_stop else (288, 384, 576)) for _, n in io["outputs"]):
    return f"an output DMA of a size that never ran: {io['outputs']}"
  # narrow: inside one buffer
  bufs = sorted((a[n], a[n] + sz) for n, sz in LY.narrow_sizes(g, L).items())
  for k, what, tiles, (lo, hi) in accesses(streamed_exact(prog)):
    if not any(b0 <= lo and hi <= b1 for b0, b1 in bufs): return f"#{k} {what} tiles {tiles:#06x}: [{lo}, {hi}) outside every buffer"
  # wide: FC ops / bias loads / parameter streams inside their matmul's regions, everything else at or above the limit
  lim = LY.model_limit(P)
  regs = [(t, lo, hi) for t, lo, hi, _ in LY.regions(pl)]
  def inreg(tiles, lo, hi): return all(any(rt == t and r0 <= lo and hi <= r1 for rt, r0, r1 in regs) for t in range(16) if tiles >> t & 1)
  for k, (_, ws) in enumerate(ins):
    oc = opcode(ws[0])
    if oc not in (0x01, 0x02, 0x10, 0x11, 0x12, 0x13, 0x14): continue
    d = _decode(ws)
    if oc == 0x01 and d["dp_mode"] == 3 and d["par_base"] + 8192 * d["par_sel"] != LY.STREAM_FIFO:   # a spread FC op: its weights
      pb = d["par_base"] + 8192 * d["par_sel"]
      if not inreg(d["tile_mask"], pb - 4, pb + 4 * (d["loop0"] + 1)): return f"#{k}: FC op reads parameters at {pb} outside its region"
      continue
    if oc == 0x14 and d["mode"] == 2:                          # bias loads: spread from the region, streamed from the stream bias rows
      if d["wide_addr"] != LY.STREAM_BIAS and not inreg(d["tile_mask"], d["wide_addr"], d["wide_addr"] + 4):
        return f"#{k}: bias load from {d['wide_addr']}"
      continue
    if oc == 0x10 and d.get("to_c1"):                           # parameter streams: from inside a region of the parameter tile
      if not inreg(d["tile_mask"], d["addr"], d["addr"] + 4 * (d["cnt0"] + 1)): return f"#{k}: parameter stream from {d['addr']} outside the regions"
      continue
    addr = d.get("wide_addr", d.get("addr")) if oc != 0x01 else d["par_base"] + 8192 * d["par_sel"]
    if oc in (0x01, 0x02) and d["par_base"] == 0 and d["par_sel"] == 0: continue
    if not lim <= addr < A.WIDE_TOP: return f"#{k} opcode {oc:#x}: wide address {addr} below the parameter limit {lim}"
  # the identity row is loaded before any instruction reads it (wide IDENT through an op's par TTU)
  ident_at = next(k for k, (_, ws) in enumerate(ins) if opcode(ws[0]) == 0x13 and _decode(ws)["wide_addr"] == LY.IDENT)
  for k, (_, ws) in enumerate(ins[:ident_at]):
    if opcode(ws[0]) in (1, 2, 3) and _decode(ws)["par_base"] + 8192 * _decode(ws)["par_sel"] == LY.IDENT:
      return f"#{k} reads the identity row before it is loaded"
  # data flow (attention_block's exceptions: the position SUM's transposing relay over-reads the last quad; the K / V input path's copy
  # ops are issued before the input DMA that fills their staging)
  stg = (a["Stg"], a["Stg"] + LY.narrow_sizes(g, L)["Stg"])
  prob, st = dataflow(streamed_exact(prog), allow_overread=lambda k, d: (opcode(ins[k][1][0]) == 0x13 and d.get("tail_f799") == 1) or
                      (opcode(ins[k][1][0]) == 1 and stg[0] <= d["in_base"] < stg[1]))
  if prob: return prob
  # the swap-read MULs: step 1 reading q (k) from base + 8 in SWAPS order
  starts = {a[b] + 64 * t + 8: t for b in ("Yq", "Yk") for t in range(5)}
  swaps = [k for k, (_, ws) in enumerate(ins) if opcode(ws[0]) == 1 and _decode(ws)["dp_mode"] == 4 and _decode(ws)["in_base"] in starts]
  if stop is None and len(swaps) != 2 * (2 if local else 5) * L: return f"{len(swaps)} swap-read MUL ops, want {(4 if local else 10) * L}"
  for k in swaps:
    d = _decode(ins[k][1])
    lv = _op_levels(d, "in")
    addrs, idx = [], [0] * 8
    n = int(np.prod([c for _, c in lv]))
    for i in range(n):
      addrs.append(sum(s * j for (s, _), j in zip(lv, idx)) + 8)
      for lev in range(8):
        idx[lev] += 1
        if idx[lev] < lv[lev][1]: break
        idx[lev] = 0
    t = starts[d["in_base"]]
    if not np.array_equal(np.array(addrs) + 64 * t, LY.SWAPS[64 * t:64 * t + n]): return f"#{k}: the swap read walks {addrs[:10]}..."
  if stop is not None: return None
  # the slot copies of the last layer are the last writers of exactly row P-1 of K and V
  cl = max(c for c in range(4) if g.cols[c])
  j, s = P - 1 - g.p0(cl), g.cols[cl]
  copies = [k for k, (_, ws) in enumerate(ins) if opcode(ws[0]) == 1 and _decode(ws)["in_tflags"] == 0x15 and
            any(X <= _decode(ws)["out_base"] < X + 96 * g.smax for X in (a["XK"], a["XV"]))]
  for X in (a["XK"], a["XV"]):
    for t in range(16):
      r, c = divmod(t, 4)
      if not g.cols[c]: continue
      blk = np.arange(X, X + 48 * A.ROWS[r] * g.cols[c])
      by_copy = np.isin(st["last"][t, blk], copies)
      want = np.zeros(len(blk), bool)
      if c == cl:
        for h in range(A.ROWS[r]): want[(h * s + j) * 48:(h * s + j + 1) * 48] = True
      if not np.array_equal(by_copy, want): return f"tile {t}: the slot copies wrote {by_copy.sum()} bytes of the block at {X}, want {want.sum()}"
  return None

# ============================== 4. bit models vs float ==============================
@functools.cache
def calibrated(n_stories:int=4, T:int=64):
  """quantization calibrated on float stories (the BOS position separately: P = 1 runs with its own quantization), weights, stories"""
  cfg, w = LY.load_checkpoint()
  stories = LY.float_stories(w, n_stories, T, seed=7)
  qm, qb = LY.calibrate_split(w, stories)
  return w, qm, qb, LY.quant_weights(w, qm), stories

def check_models(rep:Report):
  w, qm, qb, Ws, stories = calibrated()
  test = LY.float_stories(w, 2, 64, seed=8)
  r = LY.evaluate(w, lambda: LY.ChipModel(w, qm, qb, Ws), test)
  print(f"    {r['tokens']} tokens of 2 float stories (calibrated on 4 others): perplexity float {r['float_ppl']:.3f}, chip {r['chip_ppl']:.3f}; "
        f"argmax = float {100 * r['agree']:.1f}%")
  ok = r["chip_ppl"] < 1.5 * r["float_ppl"] and r["agree"] > 0.7
  rep.add("bit models: the whole model on the chip's arithmetic near the float model (teacher forced)", [("quality", None if ok else str(r))])

# ============================== main ==============================
def main(argv:list[str]|None=None) -> int:
  ap = argparse.ArgumentParser(description="coral/codegen/layer.py (offline)")
  ap.add_argument("--quick", action="store_true", help="a sample of every check")
  args = ap.parse_args(argv)
  rep = Report()
  check_l2(rep)
  check_params(rep)
  q6 = calib_small()
  sample = [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33, 64, 65, 127, 128, 200, 255, 256]
  Ps = sample if args.quick else list(range(1, 257))
  rep.add("gen_model(P), 1 layer: decodes, seq, host contract, buffers, wide, data flow, swap reads, slot copies",
          [(P, check_program(P, 1, q6[:1])) for P in Ps])
  rep.add("gen_model(P), 6 layers (bitstreams <= 16384 words), same checks",
          [(P, check_program(P, 6, q6)) for P in ([1, 2, 7, 64, 256] if args.quick else sample)])
  rep.add("prefix programs (stop after a stage of layer 0 / 5) for the bisection, same checks",
          [((P, L, st), check_program(P, L, q6[:L], (L - 1, st)))
                                                                                            for P in (1, 2, 64) for L in (1, 6) for st in LY.STOPS])
  rep.add("local layout (one op per tile group): 1 and 6 layers, prefix programs, streamed, same checks",
          [((P, L), check_program(P, L, q6[:L], local=True)) for P in (1, 2, 7, 64, 256) for L in (1, 6)] +
          [((P, st), check_program(P, 1, q6[:1], (0, st), local=True)) for P in (1, 64) for st in LY.STOPS] +
          [((P, "stream"), check_program(P, 1, q6[:1], None, LY.plan(1, stream={(0, "q"), (0, "w1"), (0, "w2")}), local=True)) for P in (1, 64)])
  forced = {(0, "q"), (0, "w2"), (0, "w1")}
  rep.add("streamed matmuls (1 layer, q / w1 / w2 streamed from other tiles), same checks",
          [(P, check_program(P, 1, q6[:1], None, LY.plan(1, stream=forced)))
                                                                                             for P in (1, 2, 64, 256)])
  check_models(rep)
  print("PASS" if rep.ok else "FAIL")
  return 0 if rep.ok else 1

def test_codegen_layer(): assert main(["--quick"]) == 0        # pytest entry point

if __name__ == "__main__": sys.exit(main())
