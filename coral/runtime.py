# run DarwiNN executables on the Edge TPU by following the compiler's DMA hints
from __future__ import annotations
import os, struct, time
from tinygrad.helpers import DEBUG
from coral.device import EdgeTPU, TAG_INSTRUCTIONS, TAG_INPUT, TAG_PARAMETERS, TAG_INT0, EP_OUT, EP_DATA_IN, EP_EVENT_IN
from coral.executable import Executable

ASYNC = int(os.getenv("CORAL_ASYNC", "1"))     # 0: one blocking transfer at a time, the path of programs with other hints (fences)
TRAFFIC = {"to_device": 0, "from_device": 0}   # bytes over usb, for the stats

def run_executable(tpu:EdgeTPU, exe:Executable, inputs:bytes=b"", parameters:bytes|None=None, bitstreams:list[bytes]|None=None) -> bytes:
  params = exe.parameters if parameters is None else parameters
  if ASYNC and all(h.kind in ("instruction", "dma", "interrupt") for h in exe.hints):
    # the hints in segments: OUTs (instructions, inputs, parameters), then the INs (outputs, events) due before the next OUT.
    # within a segment every bulk IN is queued before the OUTs (fully overlapped usb). A segment's OUTs may only start once all
    # earlier INs completed: in single-endpoint mode a bulk-out can block a pending bulk-in until the device deadlocks
    # (libedgetpu usb_driver.cc enforces the same rule; programs with interleaved input/output DMAs hang without it)
    bss = [b.data for b in exe.bitstreams] if bitstreams is None else bitstreams
    segs, out = [([], [])], []
    for h in exe.hints:
      if h.kind == "instruction" or (h.kind == "dma" and h.direction == "INFEED"):
        if segs[-1][1]: segs.append(([], []))
        if h.kind == "instruction": segs[-1][0].append((EP_OUT, struct.pack("<II", len(bss[h.chunk]), TAG_INSTRUCTIONS) + bss[h.chunk]))
        else:
          data = (inputs if h.desc == "INPUT" else params)[h.offset:h.offset+h.size]
          assert len(data) == h.size, f"{h.desc} has {len(data)} bytes, the program expects {h.size}"
          segs[-1][0].append((EP_OUT, struct.pack("<II", len(data), TAG_INPUT if h.desc == "INPUT" else TAG_PARAMETERS) + bytes(data)))
      else: segs[-1][1].append((EP_DATA_IN, h.size) if h.kind == "dma" else (EP_EVENT_IN, 16))
    for outs, ins in segs:
      try: res = tpu.batch.run(ins + outs)
      except RuntimeError as e:
        from coral.device import mark_hung
        mark_hung(f"{exe.type} {exe.name!r}: {e}")
        raise
      out += [r for (ep, _), r in zip(ins, res) if ep == EP_DATA_IN]
      TRAFFIC["to_device"] += sum(len(d) for _, d in outs)
      TRAFFIC["from_device"] += sum(n for ep, n in ins if ep == EP_DATA_IN)
    return b"".join(out)
  bss = [b.data for b in exe.bitstreams] if bitstreams is None else bitstreams
  out, st = bytearray(), time.perf_counter()
  for h in exe.hints:
    if h.kind == "instruction": tpu.send(TAG_INSTRUCTIONS, bss[h.chunk])
    elif h.kind == "dma" and h.direction == "INFEED":
      if h.desc == "INPUT": tpu.send(TAG_INPUT, inputs[h.offset:h.offset+h.size])
      elif h.desc == "PARAMETER": tpu.send(TAG_PARAMETERS, params[h.offset:h.offset+h.size])
      else: raise NotImplementedError(f"infeed {h}")
    elif h.kind == "dma" and h.direction == "OUTFEED": out += tpu.read_output(h.size)
    elif h.kind == "interrupt":
      tag, _, _ = tpu.read_event()
      assert tag == TAG_INT0 + h.interrupt, f"expected interrupt {h.interrupt}, got tag {tag}"
    elif h.kind == "fence": pass
    else: raise NotImplementedError(f"hint {h}")
  if DEBUG >= 2: print(f"ran {exe.type} {exe.name!r} in {(time.perf_counter()-st)*1e3:.2f} ms")
  return bytes(out)

class Model:
  """an *_edgetpu.tflite with one edgetpu custom op: caches parameters once, then runs inferences"""
  def __init__(self, tpu:EdgeTPU, exes:list[Executable]):
    self.tpu = tpu
    self.caching = [e for e in exes if e.type == "PARAMETER_CACHING"]
    self.infer = [e for e in exes if e.type != "PARAMETER_CACHING"]
    assert len(self.infer) == 1, f"expected one inference executable, got {[e.type for e in exes]}"
    self.cached_token = None

  def __call__(self, inputs:bytes) -> bytes:
    exe = self.infer[0]
    if self.caching and self.cached_token != exe.caching_token:
      for pc in self.caching: run_executable(self.tpu, pc)
      self.cached_token = exe.caching_token
    return run_executable(self.tpu, exe, inputs)
