# after a program stalled on the chip (an output that never came, a transfer that timed out), the usb firmware usually still
# answers. Then re-initializing the chip (EdgeTPU.recover: drain both IN pipes, run the open sequence again) brings it back without
# a replug. A canary decides: a known program with known output must come back bit-exact before the device counts as healthy. If
# the firmware doesn't answer at all (e.g. the single-endpoint deadlock), only a replug helps. Never a USB port reset: on a wedged
# stick that makes it vanish until a replug.
from __future__ import annotations
import numpy as np

def canary(tpu) -> bool:
  """MNIST's first conv from our generator (byte-identical to edgetpu_compiler's), fixed weights and input, vs its numpy model"""
  from coral.select import ConvSpec, conv_reference
  from coral.programs import conv2d_program
  from coral.runtime import run_executable
  rng = np.random.default_rng(0)
  x, w = rng.integers(0, 256, (1, 1, 28, 28), dtype=np.uint8), rng.integers(0, 256, (32, 1, 5, 5), dtype=np.uint8)
  b = rng.integers(-2000, 2000, 32).astype(np.int32)
  spec = ConvSpec(1, 1, 28, 28, 32, 5, 5, 1, 0, 120, 130, 100, 0.004)
  prog = conv2d_program(28, 28, 1, 32, 5, 5, 1, "VALID")
  caching, exe = prog.programs(0, 0, spec.quant())
  run_executable(tpu, caching, parameters=prog.params(w.transpose(0, 2, 3, 1), b, 130))
  y = prog.output_array(run_executable(tpu, exe, prog.input_bytes(x[0].transpose(1, 2, 0))))
  return bool(np.array_equal(y, conv_reference(spec, x, w, b)[0].transpose(1, 2, 0)))

def recover(tpu) -> bool:
  """re-initialize a chip whose last program stalled; True if the canary then runs bit-exact"""
  try:
    tpu.recover()
    return canary(tpu)
  except RuntimeError: return False
