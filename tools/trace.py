# single-step a whole program on hardware while a feeder thread streams the dma hints: ground truth instruction boundaries
from __future__ import annotations
import threading
from coral.device import EdgeTPU, TAG_INSTRUCTIONS, TAG_INPUT, TAG_PARAMETERS
from coral.executable import Executable

class Tracer:
  def __init__(self, tpu:EdgeTPU): self.tpu = tpu
  def step(self) -> int:
    self.tpu.write64("scalarCoreRunControl", 3)
    return self.tpu.read64("currentPc")

  def _feed(self, exe:Executable, inputs:bytes, out:bytearray, events:list):
    t = self.tpu
    try:
      for h in exe.hints[1:]:
        if h.kind == "dma" and h.direction == "INFEED":
          t.send(TAG_INPUT if h.desc == "INPUT" else TAG_PARAMETERS, (inputs if h.desc == "INPUT" else exe.parameters)[h.offset:h.offset+h.size],
                 timeout=0)
        elif h.kind == "dma" and h.direction == "OUTFEED": out += t.read_output(h.size, timeout=0)
        elif h.kind == "interrupt": events.append(("interrupt", t.read_event(timeout=0)))
        elif h.kind == "instruction": t.send(TAG_INSTRUCTIONS, exe.bitstreams[h.chunk].data, timeout=0)
        events.append(("hint done", repr(h)))
    except Exception as e: events.append(("feeder error", repr(e)))

  def trace(self, exe:Executable, inputs:bytes=b"", max_steps:int=500000, idle_steps:int=200, max_infeed:int=16384) -> tuple[list[int], bytes, list]:
    t = self.tpu
    # while the scalar core is halted the usb firmware blocks inside a bulk-out it can't drain and stops answering control
    # transfers (needs a replug). only trace programs whose infeed dmas fit in the on-chip queues.
    big = [h for h in exe.hints if h.kind == "dma" and h.direction == "INFEED" and h.size > max_infeed]
    if big: raise ValueError(f"refusing to trace: infeed dma of {big[0].size} bytes would wedge the device")
    assert exe.hints[0].kind == "instruction", exe.hints[0]
    t.write64("scalarCoreBreakPoint", (0 << 1) | 1)       # break on the start instruction (always 1 word), then single-step
    t.send(TAG_INSTRUCTIONS, exe.bitstreams[exe.hints[0].chunk].data)
    t.poll("scalarCoreRunStatus", 4)
    t.write64("scalarCoreBreakPoint", 0)
    out, events = bytearray(), []
    feeder = threading.Thread(target=self._feed, args=(exe, inputs, out, events), daemon=True)
    feeder.start()
    pcs, same = [t.read64("currentPc")], 0
    for _ in range(max_steps):
      pc = self.step()
      if pc != pcs[-1]:
        pcs.append(pc)
        same = 0
      else:
        same += 1
        if same > idle_steps and not feeder.is_alive(): break
    t.write64("scalarCoreRunControl", 1)
    feeder.join(timeout=5)
    return pcs, bytes(out), events

def boundaries(pcs:list[int]) -> tuple[dict[int, int], list[tuple[int, int]]]:
  """instruction start -> length, and backwards jumps (loops). pc reads back the last word of the retired instruction"""
  lens, jumps = {}, []
  for a, b in zip(pcs, pcs[1:]):
    if b > a: lens[a + 1] = b - a
    else: jumps.append((a, b))
  return lens, jumps
