# runs generated program pairs on the device for the hardware tests. By default only programs byte-identical to edgetpu_compiler's
# for the same model run (oracle=...): in our runtime those run like the compiler's own. new=True runs programs without such an
# oracle. coral.device does the rest: one process at a time, and after a failed transfer the next open re-initializes the chip and
# needs a canary program to come back bit-exact (else it refuses until the stick is replugged). Never reset the USB port; after a
# failure, stop and find the cause offline before running again.
from __future__ import annotations
import dataclasses, time
from coral.executable import Executable

def flatten_inputs(exe:Executable, x:bytes|dict[str, bytes]) -> tuple[Executable, bytes]:
  """x: the input bytes, or {input name: bytes} for an executable with several inputs (edgetpu_compiler gives each input hint offset
  0 into its own tensor) -> the executable with every input hint pointing into one buffer, and that buffer. The program is unchanged:
  over USB the input DMAs are just sent in hint order."""
  if isinstance(x, (bytes, bytearray)): return exe, bytes(x)
  hints, buf = [], b""
  for h in exe.hints:
    if h.kind == "dma" and h.direction == "INFEED" and h.desc == "INPUT":
      data = bytes(x[h.name][h.offset:h.offset + h.size])
      assert len(data) == h.size, f"input {h.name!r}: {len(data)} bytes, the program expects {h.size}"
      hints.append(dataclasses.replace(h, offset=len(buf)))
      buf += data
    else: hints.append(h)
  return dataclasses.replace(exe, hints=hints), buf

def _check(caching:Executable|None, execution:Executable, params:bytes|None, oracle:dict[str, Executable]|None, new:bool):
  if new: return
  if oracle is None: raise PermissionError("pass the compiler's executables for the same model (oracle=...) or new=True")
  pairs = (("PARAMETER_CACHING", caching), ("EXECUTION_ONLY", execution)) if caching is not None else (("STAND_ALONE", execution),)
  for kind, e in pairs:
    if kind not in oracle or [b.data for b in e.bitstreams] != [b.data for b in oracle[kind].bitstreams]:
      raise PermissionError(f"{kind} differs from edgetpu_compiler's program: not run (new=True runs it anyway)")
  if caching is not None and (params if params is not None else caching.parameters) != oracle["PARAMETER_CACHING"].parameters:
    raise PermissionError("the parameter blob differs from edgetpu_compiler's: not run")

def _execute(tpu, caching, execution, inputs, params, times) -> list[bytes]:
  from coral.runtime import run_executable
  if caching is not None: run_executable(tpu, caching, parameters=params)
  out = []
  for x in inputs:
    exe, buf = flatten_inputs(execution, x)
    t0 = time.perf_counter()
    out.append(run_executable(tpu, exe, buf))
    if times is not None: times.append(time.perf_counter() - t0)
  return out

def run(caching:Executable|None, execution:Executable, inputs:list[bytes|dict[str, bytes]], params:bytes|None=None,
        oracle:dict[str, Executable]|None=None, new:bool=False, times:list|None=None) -> list[bytes]:
  """caching / execution: the program pair (Executables with the oracle's hints, e.g. dataclasses.replace(oracle_exe,
  bitstreams=[Bitstream(our_bytes, [])])); caching=None for a STAND_ALONE program without parameters (eltops, attention), checked
  against oracle["STAND_ALONE"]; params: the parameter blob (default: caching.parameters); oracle: the compiler's executables for
  the same model; inputs: per call the input bytes or {input name: bytes}; times: if a list, the wall time of every call (seconds,
  the USB transfers included) is appended. Returns the execution's output bytes for every input."""
  _check(caching, execution, params, oracle, new)
  from coral.device import EdgeTPU
  return _execute(EdgeTPU(), caching, execution, inputs, params, times)

def run_jobs(jobs:list[dict], new:bool=False, on_done=None) -> list[list[bytes]]:
  """several program pairs in ONE device session (one EdgeTPU open), every job checked as run() checks it before anything runs:
  jobs = [dict(caching=, execution=, inputs=[...], params=None, oracle=None, times=None)]. The jobs run in order; a failed transfer
  raises (and coral.device marks the device hung) without running the rest. on_done(i, outputs) is called after job i (report it
  before the next job starts: a hang in a later job then still shows how far the session got)."""
  for j in jobs: _check(j.get("caching"), j["execution"], j.get("params"), j.get("oracle"), new)
  from coral.device import EdgeTPU
  tpu, out = EdgeTPU(), []
  for i, j in enumerate(jobs):
    out.append(_execute(tpu, j.get("caching"), j["execution"], j["inputs"], j.get("params"), j.get("times")))
    if on_done is not None: on_done(i, out[-1])
  return out
