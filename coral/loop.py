# Instructions once: wrap a compiled program's body in a scalar-core loop, so one instruction upload serves N frames.
#
# Every call of an edgetpu_compiler program re-sends its whole instruction stream (80 KB .. 1.3 MB) over USB, and the chip cannot
# replay a finished program: after `halt` the scalar core stops, a new input only queues up. But a program that does not halt keeps
# running: `looped()` keeps everything after `start` and before `sync_final` as the loop body and adds four words
#
#   start ; movi s_r, N ; [ sync_init ; scsync_init ; alu x4 ; input phase ; compute ; outputs ] ; s_r -= 1 ; p0 = s_r != 0 ;
#   branch back while p0 ; sync_final ; scsync ; interrupt ; halt ; nop x4 ; end
#
# The host sends the looped program once, then per frame only the input DMA(s) and reads the output DMA(s) (`run_loop`); the
# completion interrupt comes once, after frame N. The body must include the paired sync_init (every tile resets its 19 sync
# counters and signals) and scsync_init (the scalar core waits the 16 tokens and resets its 9 counters): programs whose outputs go
# through scalar memory leave SyncCounter_SCALAR_INFEED / SyncCounter_RING_OUTFEED nonzero at the end of the body, and without the
# resets only the first frame is right. Looping with only one of the pair deadlocks the chip (replug).
#
# Limits: one chunk (the instruction memory holds 16 384 words, 256 KB: the loop body, prologue and epilogue must fit; the 15-bit
# branch offset reaches the whole memory), no streamed parameters and no scratch DMAs in the program, no control flow of its own.
# Measured on the stick (2026-10-19): bit-exact for Google's MobileNet v1 (14 114 -> 14 118 words, 3.94 -> 3.43 ms/frame), 1-D relu
# x1000, 3x3 convs, SiLU backbones up to 15 808 body words, and 640x480 two-input YUV420 YOLOv5n detectors (5-8k words, 20 frames).
from __future__ import annotations
from coral.isa import opcode, WORD
from coral.isa import scalar as SC
from coral.isa import scalar_core as CORE
from coral.executable import Executable

def classify(words:list[int]) -> str:
  """one tag per instruction: start/end/halt/alu/branch/hwloop, sync/sync_final, scsync/scsync_set_av, opXX for the tile side"""
  op = opcode(words[0])
  if op == 0x24:
    d = SC.SCSYNC.decode(words)
    return "scsync_set_av" if d["smask"] == SC.scm("AVDATA_POP", "AVDATA_INFEED") and not d["tokwait"] else "scsync"
  if op == 0x1a: return "sync_final" if SC.SYNC.decode(words)["f109"] == 3 else "sync"
  return {0x3e: "start", 0x3f: "end", 0x21: "halt", 0x20: "alu", 0x22: "branch", 0x23: "hwloop"}.get(op, f"op{op:02x}")

def regs_written(words:list[int]) -> set[int]:
  """scalar registers written by the 0x20 words: ALU destinations (not the compares/predicate ops), vector-slot loads, sfu results"""
  out = set()
  for w in words:
    if opcode(w) != 0x20: continue
    d = WORD.decode([w])
    if d["s_op"] and (d["s_op"] & 0x1f) not in (0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e): out.add(d["s_x"])
    if d["enable_vector"] in (2, 3): out.add(d["vs_reg_v1"])
    if d["v_op_2"] in (2, 3): out.add(d["unk_3"] & 0x1f)
  return out

def _flat(part:list[tuple[int, list[int]]]) -> list[int]: return [w for _, ws in part for w in ws]

def looped(bitstream:bytes, frames:int, counter_reg:int|None=None, body_start:int=1) -> tuple[bytes, dict]:
  """the program of one chunk, with instructions body_start .. the one before sync_final repeated `frames` times
  -> (looped bitstream, info: body_words, total_words, counter_reg, branch_off, body_descriptors, epilogue_descriptors)"""
  ins = CORE.split(bitstream)
  kinds = [classify(ws) for _, ws in ins]
  assert kinds[0] == "start" and kinds[-1] == "end" and kinds.count("halt") == 1, "not a single complete program"
  assert "branch" not in kinds and "hwloop" not in kinds, "the program has control flow of its own; not handled"
  b1 = kinds.index("sync_final")                             # the epilogue starts here
  assert 1 <= body_start < b1, body_start
  prologue, body, epilogue = _flat(ins[1:body_start]), _flat(ins[body_start:b1]), _flat(ins[b1:-1])
  used = regs_written(prologue + body + epilogue)
  r = counter_reg if counter_reg is not None else next(i for i in range(31, 12, -1) if i not in used)
  assert r not in used, f"s{r} is written by the program"
  tail = [CORE.alu(CORE.SUB, r, r, imm=1), CORE.alu(CORE.NE, 0, r, imm=0)]   # s_r -= 1 ; p0 = (s_r != 0)
  branch_off = -(len(body) + len(tail))                      # from the branch word back to body[0]
  assert -(1 << 14) <= branch_off < 0, f"a body of {len(body)} words is too long for the 15-bit branch"
  new = prologue + [CORE.movi(r, frames)] + body + tail + [CORE.branch(branch_off)] + epilogue
  info = dict(body_words=len(body), total_words=len(new) + 2, counter_reg=r, branch_off=branch_off,
              body_descriptors=SC.dma_seqs(CORE.words([w for w in body if opcode(w) == 0x20])),
              epilogue_descriptors=SC.dma_seqs(CORE.words([w for w in epilogue if opcode(w) == 0x20])))
  return SC.program(new), info

def check(prog:bytes, frames:int, info:dict) -> dict:
  """offline: the looped program re-splits into instructions, `start` carries the new length, the branch lands on the body's first
  word and the body runs exactly `frames` times. A minimal control-flow simulation at instruction granularity: only the counter
  (MOV / SUB / NE immediates on s_r) and the 0x22 branch are modelled; payload words of multi-word instructions are never decoded."""
  ins = CORE.split(prog)
  kinds = [classify(ws) for _, ws in ins]
  assert kinds[0] == "start" and kinds[-1] == "end" and kinds.count("halt") == 1 and kinds.count("branch") == 1
  assert SC.START.decode(ins[0][1])["length_bits"] == 128 * (len(prog) // 16 - 2)
  r = info["counter_reg"]
  at = {widx: i for i, (widx, _) in enumerate(ins)}         # word index -> instruction index
  nwords = len(prog) // 16
  pc, s, p0, body_runs, steps, b0 = 0, 0, False, 0, 0, None
  while pc < nwords:
    assert pc in at, f"pc {pc} is not an instruction boundary"
    _, ws = ins[at[pc]]
    w = ws[0]
    op = opcode(w)
    steps += 1
    assert steps < 10**7
    if op == 0x20 and len(ws) == 1:
      d = WORD.decode([w])
      if d["s_op"] == (CORE.MOV | 0x20) and d["s_x"] == r:
        s = d["imm_scalar"]
        b0 = pc + 1
      elif d["s_op"] == (CORE.SUB | 0x20) and d["s_x"] == r and d["s_y"] == r: s = (s - d["imm_scalar"]) & 0xffffffff
      elif d["s_op"] == (CORE.NE | 0x20) and d["s_y"] == r: p0 = s != d["imm_scalar"]
    if op == 0x22:
      off = (w >> 17) & 0x7fff
      if off & 0x4000: off -= 0x8000
      if p0:
        pc += off
        assert pc == b0, f"the branch lands at {pc}, the body starts at {b0}"
        continue
    if pc == b0: body_runs += 1
    pc += len(ws)
  assert body_runs == frames, f"the body ran {body_runs} times, expected {frames}"
  return dict(counter_final=s, p0_final=p0, body_runs=body_runs, kinds={k: kinds.count(k) for k in set(kinds)})

def frame_size(exe:Executable) -> int:
  """bytes of one frame for run_loop: all INPUT DMAs of the executable, concatenated in its layout (offset / size per hint)"""
  return max(h.offset + h.size for h in exe.hints if h.kind == "dma" and h.direction == "INFEED" and h.desc == "INPUT")

def run_loop(tpu, exe:Executable, inputs:list[bytes], prog:bytes|None=None) -> list[bytes]:
  """device: the looped program once, then per frame OUT its input DMA(s) and IN its output DMA(s), in hint order (the single-endpoint
  rule: no OUT while an IN is pending), and the completion interrupt once. Parameters must be resident (run the PARAMETER_CACHING
  executable first); programs with streamed parameters or scratch DMAs are refused."""
  from coral.device import TAG_INSTRUCTIONS, TAG_INPUT, TAG_INT0
  assert len(exe.bitstreams) == 1, "one chunk only"
  prog = prog or looped(exe.bitstreams[0].data, len(inputs))[0]
  in_hints = [h for h in exe.hints if h.kind == "dma" and h.direction == "INFEED"]
  out_hints = [h for h in exe.hints if h.kind == "dma" and h.direction == "OUTFEED"]
  assert in_hints and all(h.desc == "INPUT" for h in in_hints), "streamed parameters: not loopable"
  assert all(h.desc == "OUTPUT" for h in out_hints), "scratch DMAs: not loopable"
  need = frame_size(exe)
  tpu.send(TAG_INSTRUCTIONS, prog)
  outs = []
  for x in inputs:
    assert len(x) == need, f"a frame has {len(x)} bytes, the executable's inputs need {need}"
    for h in in_hints: tpu.send(TAG_INPUT, x[h.offset:h.offset + h.size])
    outs.append(b"".join(tpu.read_output(h.size) for h in out_hints))
  tag, _, _ = tpu.read_event()
  assert tag == TAG_INT0, f"expected the completion interrupt, got tag {tag}"
  return outs
