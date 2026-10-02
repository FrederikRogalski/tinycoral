# scalar-core-side instructions and the program skeleton (docs/isa/scalar.md): 0x3e start, 0x3f end, 0x21 halt, 0x1a sync (tile fence),
# 0x24 scalar sync, 0x25 avDataPop / parameterPop, 0x26 infeed, 0x27 outfeed, and the scalar-ALU (0x20) words that build host DMA
# descriptors. Fields named after a bit number (k157, ...) are constant within a stream class; their meaning is unknown.
from __future__ import annotations
from coral.isa import Layout, WORD, cdiv, split, opcode

START = Layout("start", (0x3e,), 1, [("length_bits", 19, 32)])   # 128 * (words between start and end, exclusive)
END = Layout("end", (0x3f,), 1, [("length_bits", 19, 32)])       # always 128
HALT = Layout("halt", (0x21,), 1, [("code", 14, 5)])            # 4; 1 ends a bitstream that continues in the next one
SYNC = Layout("sync", (0x1a,), 1, [       # tile fence: dispatched to the tiles, takes a slot in the tile-instruction sequence
  ("tiles", 12, 16),       # tiles that execute the fence
  ("seq", 28, 15),         # tile-instruction sequence number (tile instructions have it at bit 46)
  ("b43", 43, 1),          # set in the first and last sync of PARAMETER_CACHING programs
  ("counters", 44, 19),    # tile sync-counter mask, bit k = TILE_COUNTERS[k]
  ("count", 63, 16),       # counter value: 0 for resets, mesh fences: packets so far (scalar.md splits it into b63 + count)
  ("units", 79, 10),       # tile unit mask, 0x3ff = all; bit k = meshBus opcode 0x0f+k for k = 6..9; bit 5: wideToNarrow or ringBusProducer
  ("signal", 107, 1),      # report completion to the scalar core (paired with a following scsync tokwait)
  ("f109", 109, 2)])       # 3 only in the last sync before the completion interrupt
SCSYNC = Layout("scsync", (0x24,), 2, [   # scalar fence / scalar sync-counter op (not dispatched to the tiles: no sequence number)
  ("tiles", 12, 16),       # 0xffff with tokwait / waits, else 0
  ("smask", 32, 9),        # scalar sync-counter mask, bit k = SCALAR_COUNTERS[k]
  ("value", 43, 13),       # value written to the selected counters
  ("tokwait", 57, 1),      # wait for the completion tokens of the tiles in tmask (syncs with signal=1)
  ("tmask", 73, 16), *[(f"thr{k}", 93 + 16 * k, 16) for k in range(9)],   # per-counter wait threshold
  ("wait", 237, 9),        # per-counter "wait until counter >= thr" enable, bits as smask
  ("sunits", 246, 5)])     # scalar unit mask: 0b01100 normal, 0b11111 in the first scsync, 0b10000 / 0b01000 around scalar-memory outfeeds
POP = Layout("pop", (0x25,), 2, [         # avDataPop (stream=1) / parameterPop (stream=0): host stream -> circular staging buffer
  ("stream", 12, 1),       # 1 activations (DMA tag 1, 8-byte units), 0 parameters (tag 2, 64-byte units)
  ("d0_stride", 28, 16), ("d0_limit", 44, 15),     # buffer walk: stride 1, limit = buffer units - 1
  ("d1_stride", 59, 16), ("d1_limit", 75, 15),     # rewind; limit = buffer passes - 1
  ("d2_stride", 90, 16), ("d2_limit", 106, 15),    # rewind
  ("k121", 121, 1), ("k124", 124, 1), ("par136", 136, 1), ("av140", 140, 1), ("count_m1", 142, 14), ("k157", 157, 1),
  ("count_neg", 159, 14),  # -(units in the last pass) mod 2^14 (activations) / 2^10 (parameters); with count_m1 only for a partial pass
  ("k174", 174, 1), ("par177", 177, 1), ("k178", 178, 1),
  ("b183", 183, 1),        # parameters: 1 iff the unit count is even or >= 2048 (0 for STAND_ALONE streaming)
  ("rows_neg", 184, 15),   # parameters: -min(units // 2, 1024) mod 2^15, 128-byte rows capped at 128 KiB (0 for streaming)
  ("k199", 199, 1), ("k215", 215, 1)])
INFEED = Layout("infeed", (0x26,), 4, [   # staging buffer -> ring bus -> the tiles in `tiles` (received by a ringConsumer)
  ("stream", 12, 1), ("buf_off", 14, 20),          # start offset in the staging buffer (units)
  ("d0_stride", 34, 21), ("d0_limit", 55, 20), ("d1_stride", 75, 21), ("d1_limit", 96, 20), ("d2_stride", 116, 21), ("d2_limit", 137, 20),
  ("k157", 157, 1), ("k160", 160, 1), ("par172", 172, 1), ("av176", 176, 1), ("count_m1", 183, 14), ("k203", 203, 1), ("count_neg", 205, 14),
  ("k311", 311, 1), ("par314", 314, 1),
  ("pop_wait", 320, 16),   # start offset in units + 1: waits until the pop has produced it (hypothesis)
  ("k336", 336, 1), ("k352", 352, 1), ("k392", 392, 1),
  ("tiles", 409, 16),      # ring destination bitmap (= the matching ringConsumer's tile mask)
  ("k434", 434, 4), ("f438", 438, 2)])               # f438: 64-byte lanes per weight row - 1
OUTFEED = Layout("outfeed", (0x27,), 2, [ # ring (from a ringProducer) -> host stream (DMA tag 3), or -> scalar memory
  ("base", 12, 15),        # scalar-memory variant: destination offset (elements); 0 for the host
  ("d0_stride", 27, 16), ("d0_limit", 43, 15), ("d1_stride", 58, 16), ("d1_limit", 74, 15), ("d2_stride", 89, 16), ("d2_limit", 105, 15),
  ("d3_stride", 120, 16),  # 1 in the scalar-memory variant
  ("k173", 173, 2), ("f222", 222, 7)])

TILE_COUNTERS = ["AVDATA", "PARAMETERS", "PARTIAL_SUMS", "MESH_NORTH_IN", "MESH_EAST_IN", "MESH_SOUTH_IN", "MESH_WEST_IN",
                 "MESH_NORTH_OUT", "MESH_EAST_OUT", "MESH_SOUTH_OUT", "MESH_WEST_OUT", "WIDE_TO_NARROW", "WIDE_TO_SCALING",
                 "NARROW_TO_WIDE", "RING_READ_A", "RING_READ_B", "RING_WRITE", "RING_PRODUCER_A", "RING_PRODUCER_B"]
SCALAR_COUNTERS = ["AVDATA_POP", "PARAMETER_POP", "AVDATA_INFEED", "PARAMETER_INFEED", "SCALAR_INFEED", "PRODUCER_A", "PRODUCER_B",
                   "RING_OUTFEED", "SCALAR_PIPELINE"]
def tc(*names) -> int: return sum(1 << TILE_COUNTERS.index(n) for n in names)
def scm(*names) -> int: return sum(1 << SCALAR_COUNTERS.index(n) for n in names)
# the counters of a mesh move (meshBus opcode): the receiving tile's IN side, the sending tile's OUT side
MESH_COUNTERS = {op: (f"MESH_{a}_IN", f"MESH_{b}_OUT") for op, (a, b) in
                 {0x15: ("NORTH", "SOUTH"), 0x16: ("EAST", "WEST"), 0x17: ("SOUTH", "NORTH"), 0x18: ("WEST", "EAST")}.items()}
AV_BUF, PAR_BUF = 1 << 14, 1 << 10       # staging buffers in units: 16384 x 8 B activations, 1024 x 64 B parameters

# *** fences ***
def sync(seq:int, tiles:int=0xffff, counters:int=0, count:int=0, units:int=0x3ff, signal:int=0, b43:int=0, f109:int=0) -> list[int]:
  return SYNC.encode(seq=seq, tiles=tiles, counters=counters, count=count, units=units, signal=signal, b43=b43, f109=f109)
def sync_init(caching:bool=False) -> list[int]: return sync(0, counters=(1 << 19) - 1, signal=1, b43=int(caching))   # 2nd word of every program
def sync_signal(seq:int) -> list[int]: return sync(seq, signal=1)          # drain all units, then a token -> scsync_fence
def sync_final(seq:int, caching:bool=False) -> list[int]: return sync(seq, signal=1, f109=3, b43=int(caching))
def sync_drain(seq:int, tiles:int=0xffff) -> list[int]: return sync(seq, tiles=tiles)   # drain all units, no token
def sync_wn_fence(seq:int) -> list[int]: return sync(seq, units=1 << 5)    # starts each input's DMA
def sync_reset17(seq:int) -> list[int]: return sync(seq, counters=(1 << 17) - 1)       # all counters but RING_PRODUCER_A/B
def sync_reset_mesh(seq:int) -> list[int]: return sync(seq, counters=tc(*TILE_COUNTERS[3:11]))   # the 8 mesh counters
def sync_rpa(seq:int) -> list[int]: return sync(seq, counters=tc("RING_PRODUCER_A"), signal=1)   # after the outfeeds
def sync_rpb(seq:int) -> list[int]: return sync(seq, counters=tc("RING_PRODUCER_B"), signal=1)   # after the PRODUCER_B wait
def sync_w2n(seq:int) -> list[int]: return sync(seq, counters=tc("WIDE_TO_NARROW"), units=0)     # counter only, no fence
def sync_op_fence(seq:int) -> list[int]: return sync(seq, counters=tc("AVDATA", "PARAMETERS", "PARTIAL_SUMS"), units=1)  # before 0x19
def sync_reset_n2w(seq:int) -> list[int]: return sync(seq, counters=tc("NARROW_TO_WIDE"))       # around MUL
def sync_reset_par(seq:int) -> list[int]: return sync(seq, counters=tc("PARAMETERS"))           # around MUL
def sync_mesh(seq:int, op:int, tiles:int, count:int) -> list[int]:
  """fence between the groups of a mesh phase (meshBus opcode op), count = the phase's running packet count"""
  return sync(seq, tiles=tiles, counters=tc(*MESH_COUNTERS[op]), count=count, units=1 << (op - 0x0f))

def scsync(smask:int=0, value:int=0, tokwait:bool=False, thresholds:dict[int, int]|None=None, sunits:int=0b01100) -> list[int]:
  """thresholds: {counter index: value} sets thr<k> and the wait bit k (the counter must also be in smask)"""
  th = thresholds or {}
  wait = sum(1 << k for k in th)
  return SCSYNC.encode(smask=smask, value=value, sunits=sunits, wait=wait, tiles=0xffff if tokwait or wait else 0,
                       **(dict(tokwait=1, tmask=0xffff) if tokwait else {}), **{f"thr{k}": v for k, v in th.items()})
def scsync_fence() -> list[int]: return scsync(tokwait=True)                                   # follows a sync(signal=1)
def scsync_init() -> list[int]: return scsync(smask=0x1ff, tokwait=True, sunits=0b11111)      # 2nd instruction of every program
def scsync_set_av() -> list[int]: return scsync(smask=scm("AVDATA_POP", "AVDATA_INFEED"))      # before each input
def scsync_av_credit() -> list[int]: return scsync(smask=scm("AVDATA_INFEED"), value=0x1fff)   # right before avDataPop
def scsync_set_avpop() -> list[int]: return scsync(smask=scm("AVDATA_POP"))                    # right before avDataPop
def scsync_nop() -> list[int]: return scsync()                                                 # scalar-unit fence only
def scsync_wait_pa(n:int) -> list[int]: return scsync(smask=scm("PRODUCER_A"), thresholds={5: n})   # n = outfeeds of the output
def scsync_wait_pb(n:int) -> list[int]: return scsync(smask=scm("PRODUCER_B"), thresholds={6: n})   # n = parameter blocks broadcast
def scsync_smem_pre() -> list[int]: return scsync(sunits=0b10000)    # before each scalar-memory outfeed
def scsync_smem_post() -> list[int]: return scsync(sunits=0b01000)   # after the last scalar-memory outfeed

# *** host streams: pops fill the staging buffer, infeeds send buffer ranges over the ring, outfeeds drain the ring to the host ***
def _neg(n:int, bits:int) -> int: return (-n) % (1 << bits)
def _passes(u:int, buf:int) -> tuple[int, int]: return cdiv(u, buf), u - (cdiv(u, buf) - 1) * buf   # buffer passes, units in the last

def av_pop(nbytes:int) -> list[int]:
  """avDataPop of a contiguous activation input of nbytes (a multiple of 8)"""
  assert nbytes % 8 == 0 and nbytes > 0
  f, (passes, last) = dict(stream=1, d0_stride=1, k121=1, k124=1, av140=1, k174=1, k178=1, k199=1, k215=1), _passes(nbytes // 8, AV_BUF)
  if nbytes == 8: return POP.encode(**f)
  f.update(d0_limit=AV_BUF - 1, d1_stride=_neg(AV_BUF - 1, 16), d1_limit=passes - 1, d2_stride=_neg(AV_BUF - 1, 16))
  if last != AV_BUF: f.update(count_m1=last - 1, k157=1, count_neg=_neg(last, 14))
  return POP.encode(**f)

def param_pop(nbytes:int) -> list[int]:
  """parameterPop of a PARAMETER_CACHING program (nbytes a multiple of 64)"""
  u = nbytes // 64
  passes, last = _passes(u, PAR_BUF)
  f = dict(stream=0, d0_stride=1, d0_limit=PAR_BUF - 1, d1_stride=_neg(PAR_BUF - 1, 16), d1_limit=passes - 1, d2_stride=_neg(PAR_BUF - 1, 16),
           k121=1, k124=1, par136=1, k174=1, par177=1, k178=1, b183=int(u % 2 == 0 or u >= 2048), rows_neg=_neg(min(u // 2, 1024), 15),
           k199=1, k215=1)
  if last != PAR_BUF: f.update(count_m1=last - 1, k157=1, count_neg=_neg(last, 10))
  return POP.encode(**f)

def av_infeed(offset:int, nbytes:int, tiles:int) -> list[int]:
  """infeed of nbytes activations from byte `offset` of the input, multicast on the ring to `tiles`. A transfer longer than the
  staging buffer takes several passes; its offsets wrap (buf_off at 2^15 units, pop_wait at 2^16, as in the argmax program)"""
  assert offset % 8 == 0 and nbytes % 8 == 0 and nbytes > 0
  off, u = offset // 8, nbytes // 8
  (passes, last), wrap = _passes(u, AV_BUF), u > AV_BUF
  f = dict(stream=1, buf_off=off % (1 << 15) if wrap else off, d0_stride=1, k157=1, k160=1, av176=1, count_m1=last - 1, k311=1,
           pop_wait=(off + 1) % (1 << 16) if wrap else off + 1, k336=1, k352=1, k392=1, tiles=tiles, k434=0xf, f438=3)
  if u > 1: f.update(d0_limit=AV_BUF - 1, d1_stride=_neg(AV_BUF - 1, 21), d1_limit=passes - 1, d2_stride=_neg(AV_BUF - 1, 21), k203=1,
                     count_neg=_neg(last, 14))
  return INFEED.encode(**f)

def param_infeed(offset:int, nbytes:int, tiles:int, lanes:int) -> list[int]:
  """infeed of nbytes of the parameter blob from byte `offset` to `tiles` (PARAMETER_CACHING); lanes = 64-byte lanes per weight row"""
  off, u = offset // 64, nbytes // 64
  passes, last = _passes(u, PAR_BUF)
  f = dict(stream=0, buf_off=off % 2048, d0_stride=1, d0_limit=PAR_BUF - 1, d1_stride=_neg(PAR_BUF - 1, 21), d1_limit=passes - 1,
           d2_stride=_neg(PAR_BUF - 1, 21), k157=1, k160=1, par172=1, k311=1, par314=1, pop_wait=(off + 1) % (1 << 16), k336=1, k352=1,
           k392=1, tiles=tiles, k434=0xf, f438=lanes - 1)
  if last != PAR_BUF: f.update(count_m1=last - 1, k203=1, count_neg=_neg(last, 10))
  return INFEED.encode(**f)

def outfeed(nbytes:int) -> list[int]:
  """host outfeed of nbytes (a multiple of 8) from the ring (one per ringProducer)"""
  assert nbytes % 8 == 0 and nbytes > 0
  return OUTFEED.encode(d0_stride=int(nbytes == 8), d0_limit=nbytes // 8 - 1, k173=3, f222=0x7f)

# *** host DMA descriptors, built by the scalar ALU (0x20): v_op=0xa pushes s[vs_reg] into descriptor word v_offset
# (address lo, address hi, length, tag), v_op=0xc issues it ***
TAG_INPUT, TAG_PARAMETERS, TAG_OUTPUT, TAG_INT0 = 1, 2, 3, 4

def alu(s_op:int=0, s_x:int=0, s_y:int=0, imm:int=0, pred:int|None=None, v_op:int=0, v_offset:int=0, vs_reg:int=0) -> int:
  """s_x = s_y <s_op> imm (s_op + 0x20: immediate form; 1 ADD 3 AND 0xa NEQ 0xc LT (signed) 0xf MOV), run iff p<pred> if given"""
  p = dict(gate=1, pred_reg=pred, yes_pred=1) if pred is not None else {}
  return WORD.encode(enable_scalar=1, s_op=s_op, s_x=s_x, s_y=s_y, imm_scalar=imm & 0xffffffff, v_op=v_op, v_offset=v_offset, vs_reg=vs_reg,
                     **p)[0]
def movi(rd:int, imm:int) -> int: return alu(0x2f, rd, 0, imm)

def add64(hi:int, lo:int, off:int, out_hi:int, out_lo:int, tmp:int) -> list[int]:
  """(out_hi:out_lo) = (hi:lo) + off for 0 <= off < 2^31, the carry through predicates p0/p1 (8 words, as edgetpu_compiler emits them)"""
  assert 0 <= off < (1 << 31)
  return [alu(0x2c, 0, lo, 0),               # p0 = lo < 0 (bit 31 of lo)
          alu(0x0a, 1, lo, lo),              # p1 = false
          alu(0x23, tmp, lo, 0x7fffffff),
          alu(0x21, tmp, tmp, off, pred=0),
          alu(0x2c, 1, tmp, 0, pred=0),      # p0: p1 = carry out of bit 31
          alu(0x21, out_lo, lo, off),
          alu(0x21, out_hi, hi, 0),
          alu(0x21, out_hi, out_hi, 1, pred=1)]

def push_issue(lo:int, hi:int, len_reg:int, tag_reg:int, length:int, tag:int) -> list[int]:
  return [alu(v_op=0xa, v_offset=0, vs_reg=lo), alu(0x2f, len_reg, 0, length, v_op=0xa, v_offset=1, vs_reg=hi),
          alu(0x2f, tag_reg, 0, tag, v_op=0xa, v_offset=2, vs_reg=len_reg), alu(v_op=0xa, v_offset=3, vs_reg=tag_reg), alu(v_op=0xc)]

def _descriptor(r:int, off2:int, length:int, tag:int) -> list[int]:
  """(s[r+3]:s[r+2]) = (s[r]:s[r+1]) + off2, then push and issue the descriptor from s[r+2..r+5]"""
  return add64(r, r + 1, off2, r + 3, r + 2, r + 6) + push_issue(r + 2, r + 3, r + 4, r + 5, length, tag)

def host_dma(tag:int, length:int, off:int=0, off2:int=0, regs:str="act", load_base:tuple[int, int]|None=None) -> list[int]:
  """push and issue a host DMA descriptor of the relocated buffer address + off + off2. regs 'act': an input / output, its address in
  s11:s12, scratch s4..s10; 'par': the parameters of a caching program, address s0:s1, scratch s2..s8. load_base = (hi, lo) also
  emits the two MOVIs that the runtime relocates (a caching program loads lo first)."""
  hi, lo, r = {"act": (11, 12, 4), "par": (0, 1, 2)}[regs]
  movis = [movi(hi, load_base[0]), movi(lo, load_base[1])] if load_base else []
  return (movis if regs == "act" else movis[::-1]) + add64(hi, lo, off, r, r + 1, r + 2) + _descriptor(r, off2, length, tag)

def interrupt(caching:bool=False) -> list[int]:
  """the completion interrupt sc_int_0: a descriptor with tag 4, address 0, length 0"""
  r = 2 if caching else 4
  return [movi(r, 0), movi(r + 1, 0)] + _descriptor(r, 0, 0, TAG_INT0)

def dma_seqs(bitstream:bytes) -> list[tuple[int, int]]:
  """(tag, length) of every host DMA descriptor the program issues, by replaying the scalar ALU (relocated bases = 0)"""
  S, P, desc, out = [0] * 32, [False] * 8, [0] * 4, []
  def sg(v): return v - (1 << 32) if v >> 31 else v
  for _, ws in split(bitstream):
    if opcode(ws[0]) != 0x20: continue
    d = WORD.decode(ws)
    if d["gate"] and P[d["pred_reg"]] != bool(d["yes_pred"]): continue
    if d["v_op"] == 0xa: desc[d["v_offset"]] = S[d["vs_reg"]]
    elif d["v_op"] == 0xc: out.append((desc[3], desc[2]))
    if not (sop := d["s_op"]): continue
    x, a, b, o = d["s_x"], S[d["s_y"]], d["imm_scalar"] if sop & 0x20 else S[d["imm_scalar"] & 0x1f], sop & 0xf
    if o == 0xf: S[x] = b
    elif 9 <= o <= 14: P[x & 7] = {9: a == b, 10: a != b, 11: sg(a) > sg(b), 12: sg(a) < sg(b), 13: a >= b, 14: sg(a) >= sg(b)}[o]
    else: S[x] = {1: a + b, 2: a - b, 3: a & b, 4: a | b, 5: a ^ b, 6: a << (b & 31), 7: a >> (b & 31), 8: sg(a) >> (b & 31)}.get(o, 0) \
                 & 0xffffffff
  return out

# *** the program skeleton; seq = tile instructions (tile ops + syncs) before the block's first sync ***
def start(nwords:int) -> list[int]: return START.encode(length_bits=128 * (nwords - 2))   # nwords: the whole program, start and end included
def end() -> list[int]: return END.encode(length_bits=128)
def halt(code:int=4) -> list[int]: return HALT.encode(code=code)
def nop() -> list[int]: return [WORD.encode(enable_scalar=1)[0]]   # 4 of these follow halt
def program(body:list[int]) -> bytes: return b"".join(w.to_bytes(16, "little") for w in start(len(body) + 2) + body + end())

def exe_prologue() -> list[int]:
  """after start: init fence, then s1/s0 = PARAMETER lo/hi, s3/s2 = SCRATCH lo/hi (relocated MOVIs)"""
  return sync_init() + scsync_init() + [movi(1, 0), movi(0, 0), movi(3, 0), movi(2, 0)]
def signal_fence(seq:int) -> list[int]: return sync_signal(seq) + scsync_fence()   # drain all tile units, wait for their tokens; 1 seq
def input_head(seq:int) -> list[int]: return scsync_set_av() + signal_fence(seq)     # starts every input; 1 seq
def input_dma(nbytes:int, off:int=0) -> list[int]:
  """s11/s12 = INPUT hi/lo (relocated), credit, avDataPop, host DMA (tag 1) of nbytes at `off` into the input"""
  return [movi(11, 0), movi(12, 0)] + scsync_av_credit() + scsync_set_avpop() + av_pop(nbytes) + host_dma(TAG_INPUT, nbytes, off=off)
def output_dma_head(off:int=0) -> list[int]: return [movi(11, 0), movi(12, 0)] + add64(11, 12, off, 4, 5, 6)   # s11/s12 = OUTPUT hi/lo
def output_dma_tail(nbytes:int) -> list[int]: return add64(4, 5, 0, 7, 6, 10) + push_issue(6, 7, 8, 9, nbytes, TAG_OUTPUT)
def output_wait(seq:int, n_outfeeds:int) -> list[int]:
  """after the (outfeed, ringProducer) pairs: wait PRODUCER_A >= n_outfeeds, reset RING_PRODUCER_A (2 seqs)"""
  return sync_reset17(seq) + scsync_wait_pa(n_outfeeds) + sync_rpa(seq + 1) + scsync_fence()
def broadcast_wait(seq:int, rows:int) -> list[int]:
  """after a parameter broadcast: wait PRODUCER_B >= the rows sent, reset RING_PRODUCER_B; 1 seq"""
  return scsync_wait_pb(rows) + sync_rpb(seq) + scsync_fence()
def epilogue(seq:int, caching:bool=False) -> list[int]:
  """final fence, completion interrupt, halt, 4 nops (end follows); 1 seq"""
  return sync_final(seq, caching) + scsync_fence() + interrupt(caching) + halt() + nop() * 4
