# runs Edge TPU kernels for tinygrad: quantized matmuls (coral/programs.py) and fused model blocks (coral/fused.py) on the
# device, keeping their weights resident in the 16 tiles' wide memory; MOCKCORAL=1 swaps the device for numpy references
from __future__ import annotations
import json, functools, hashlib, os, numpy as np
from dataclasses import dataclass, asdict, replace
from coral.programs import Quant, WIDE_BYTES, bucket, fc_program, conv_program, conv2d_program

@dataclass(frozen=True)
class FCSpec:
  M: int                                                   # y[M,N] = requant(x[M,K] @ w[N,K].T + b[N])
  N: int
  K: int
  in_zp: int
  w_zp: int
  out_zp: int
  mult: float
  lo: int = 0                                              # clamp in the quantized output domain (fused relu etc.)
  hi: int = 255
  def quant(self) -> Quant: return Quant(self.in_zp, self.w_zp, self.out_zp, self.mult, self.lo, self.hi)
  def dumps(self) -> str: return json.dumps(asdict(self))

def fc_reference(spec:FCSpec, x:np.ndarray, w:np.ndarray, b:np.ndarray) -> np.ndarray:
  """what the Edge TPU computes: int32 accumulate, float32 rescale, clamp, round half to even (measured with exact .5 ties
  on the device), add the zero point. float64 holds every partial sum exactly (|acc| < 255*255*K << 2**53) and uses BLAS."""
  acc = ((x.astype(np.float64) - spec.in_zp) @ (w.astype(np.float64) - spec.w_zp).T).astype(np.int64) + b.astype(np.int64)
  y = np.clip(acc.astype(np.float32) * np.float32(spec.mult), np.float32(spec.lo - spec.out_zp), np.float32(spec.hi - spec.out_zp))
  return (np.round(y) + spec.out_zp).astype(np.uint8)

def rows_max(K:int) -> int: return max([m for m in (16, 32, 64, 128, 256) if m * K <= 256 * 3840] or [16])

class TPURunner:
  """owns the usb device and the on-chip parameter memory.
  matmuls: every weight matrix gets ntiles consecutive tiles at one offset (first fit at the lowest offset); when a new one
  doesn't fit, all known ones are re-packed (largest first) and re-uploaded on their next use.
  fused blocks: the whole set is placed by coral/codegen/fused.py's plan."""
  def __init__(self):
    from coral.device import EdgeTPU
    self.tpu = EdgeTPU()
    self.stats = {"calls": 0, "param_uploads": 0, "param_bytes": 0, "repacks": 0}
    self.items: dict = {}           # key -> (ntiles, bytes per tile) of every weight matrix seen
    self.where: dict = {}           # key -> (first tile, byte offset) of the placed ones
    self.cached: set = set()        # keys (or fused placements) whose parameters are on chip right now
    self.blobs: dict = {}           # key -> parameter blob
    self.owner = None               # "matmul" or the fused set (Mp) that laid out the wide memory last
    self.floor = WIDE_BYTES         # the lowest wide buffer of any program that ran: nothing is placed above it

  def _own(self, owner):
    if self.owner != owner: self.owner, self.cached, self.where, self.items, self.floor = owner, set(), {}, {}, WIDE_BYTES

  def _lower_floor(self, limit:int):
    """a program whose own wide buffers start at `limit` overwrites everything above it on every tile, every call: from now on
    nothing is placed above it, and what was is placed again on its next use"""
    if limit >= self.floor: return
    self.floor = limit
    for k in [k for k, (t0, off) in self.where.items() if off + self.items[k][1] > limit]:
      del self.where[k]
      self.cached.discard(k)

  # *** matmuls ***
  def fc(self, spec:FCSpec, x:np.ndarray, w:np.ndarray, b:np.ndarray, wkey) -> np.ndarray:
    # at most rows_max(K) rows per program: above ~960 KB of input (256 positions x 4096) edgetpu_compiler switches the 1x1 conv to
    # a mode coral/codegen/conv.py doesn't model (its programs differ from the compiler's there, and compute wrong results)
    if x.shape[0] > (R:=rows_max(spec.K)): return np.concatenate([self.fc(spec, x[m:m+R], w, b, wkey) for m in range(0, x.shape[0], R)])
    prog = fc_program(spec.N, spec.K) if x.shape[0] == 1 else conv_program(bucket(x.shape[0]), spec.N, spec.K)
    if not prog.fits():             # more than one tile's worth per tile: independent chunks of outputs, each its own spec
      n = max(64, 64 * (spec.N // 64 // 2))
      return np.concatenate([self.fc(replace(spec, N=len(w[i:i+n])), x, w[i:i+n], b[i:i+n], (wkey, i)) for i in range(0, spec.N, n)], axis=1)
    key = (wkey, x.shape[0] == 1)   # an FC and a conv layout of the same weights are different regions
    self._own("matmul")
    self._lower_floor(prog.buffers)
    if key not in self.where: self._place(key, prog.ntiles, prog.tile_bytes, prog.limit)
    caching, exe = prog.programs(*self.where[key], spec.quant())
    if key not in self.cached:
      blob = self.blobs.get(key) or self.blobs.setdefault(key, prog.params(w, b, spec.w_zp))
      self._upload(caching, blob)
      self.cached.add(key)
    self.stats["calls"] += 1
    from coral.runtime import run_executable
    return prog.output_array(run_executable(self.tpu, exe, prog.input_bytes(x, spec.in_zp)))[:x.shape[0]]

  def _place(self, key, ntiles:int, size:int, limit:int):
    """limit: where the region must end (the program's own wide buffers, or a rule like the small-group one), and never above
    the lowest buffer of any program that ran (self.floor)"""
    limit = min(limit, self.floor)
    self.items[key] = (ntiles, size, limit)
    if (pl:=self._fit([0] * 16 if not self.where else self._fill(), ntiles, size, limit)) is not None:
      self.where[key] = pl
      return
    # nothing free: re-pack everything seen so far, largest first (re-uploaded lazily); if even that fails, keep only this one
    self.stats["repacks"] += 1
    fill, plan = [0] * 16, {}
    for k, (n, s, lim) in sorted(self.items.items(), key=lambda kv: -kv[1][0] * kv[1][1]):
      if (pl:=self._fit(fill, n, s, min(lim, self.floor))) is None:
        plan = None
        break
      plan[k] = pl
      for t in range(pl[0], pl[0] + n): fill[t] = pl[1] + s
    self.where, self.cached = plan or {key: (0, 0)}, set()
    if plan is None: self.items = {key: (ntiles, size, limit)}
  def _fill(self) -> list[int]:
    fill = [0] * 16
    for k, (t0, off) in self.where.items():
      for t in range(t0, t0 + self.items[k][0]): fill[t] = max(fill[t], off + self.items[k][1])
    return fill
  @staticmethod
  def _fit(fill:list[int], n:int, size:int, limit:int=WIDE_BYTES) -> tuple[int, int]|None:
    fits = [(max(fill[s:s+n]), s) for s in range(17 - n) if max(fill[s:s+n]) + size <= limit]
    return (min(fits)[1], min(fits)[0]) if fits else None

  def conv2d(self, spec, x:np.ndarray, w:np.ndarray, b:np.ndarray, wkey) -> np.ndarray:
    """a selected CONV_2D (coral.select.ConvSpec) on our conv2d programs: x[N,Cin,H,W] -> y[N,Cout,OH,OW]. G images per call as
    one tall image: each image padded with in_zp (its own padding, then rows up to a multiple of the stride), stacked, and the
    conv run VALID; image g's output rows start at g * (padded height) / stride"""
    from coral.runtime import run_executable
    p, st = spec.padding, spec.stride
    Hp, Wp = -(-(spec.H + 2 * p) // st) * st, spec.W + 2 * p
    G, prog = self._conv2d_group(spec, Hp, Wp)
    key = (wkey, "conv2d", prog.args)
    self._own("matmul")
    self._lower_floor(prog.buffers)
    lim = min(prog.limit, self.floor)
    if (n:=prog.ntiles(lim)) > 16: raise NotImplementedError("the weights don't fit the chip")
    # weights bigger than a tile: pieces on n consecutive tiles, each filled from offset 0 up to the limit (whole tiles)
    if key not in self.where: self._place(key, n, prog.tile_bytes if n == 1 else lim, prog.limit)
    t0, off = self.where[key]
    caching, exe = prog.programs(t0, off, spec.quant()) if n == 1 else prog.programs(tuple(range(t0, t0 + n)), off, spec.quant(), lim)
    if key not in self.cached:
      blob = self.blobs.get(key) or self.blobs.setdefault(key, prog.params(w.transpose(0, 2, 3, 1), b, spec.w_zp))
      self._upload(caching, blob)
      self.cached.add(key)
    n = -(-spec.N // G) * G                       # whole calls: pad the batch with images of zero points
    xp = np.full((n, spec.Cin, Hp, Wp), spec.in_zp, np.uint8)
    xp[:spec.N, :, p:p + spec.H, p:p + spec.W] = x
    xt = xp.transpose(0, 2, 3, 1).reshape(n // G, G * Hp, Wp, spec.Cin)
    rows = np.arange(G)[:, None] * (Hp // st) + np.arange(spec.OH)
    out = np.empty((n, spec.OH, spec.OW, spec.Cout), np.uint8)
    for i in range(n // G):
      self.stats["calls"] += 1
      out[i * G:(i + 1) * G] = prog.output_array(run_executable(self.tpu, exe, prog.input_bytes(xt[i])))[rows]
    return out[:spec.N].transpose(0, 3, 1, 2)

  @functools.cache
  def _conv2d_group(self, spec, Hp:int, Wp:int):
    """the most images per call (up to 32) whose tall image our generator covers (weights on one tile, or split over several)"""
    for G in sorted({min(spec.N, g) for g in (32, 16, 8, 4, 2, 1)}, reverse=True):
      prog = conv2d_program(G * Hp, Wp, spec.Cin, spec.Cout, spec.kh, spec.kw, spec.stride, "VALID")
      # outputs through scalar memory arrive as several DMAs (coral/usbdev.py reads on until the hint's size); CORAL_SMEM=0 skips them
      if prog.smem and not SMEM: continue
      try:
        if (n:=prog.ntiles(prog.limit)) > 16: continue
        prog.programs(0, 0, spec.quant()) if n == 1 else prog.programs(tuple(range(n)), 0, spec.quant(), prog.limit)
      except NotImplementedError: continue
      return G, prog
    raise NotImplementedError("no conv2d program for this shape")

  # *** whole chains (coral/chain.py): the program owns the wide memory, placed as edgetpu_compiler would ***
  def chain(self, key, caching, exe, inputs:list[bytes]) -> list[bytes]:
    from coral.runtime import run_executable
    self._own(("chain", key))
    if key not in self.cached:
      self._upload(caching, self.blobs[key])
      self.cached.add(key)
    self.stats["calls"] += len(inputs)
    return [run_executable(self.tpu, exe, x) for x in inputs]

  def _upload(self, caching, blob:bytes):
    from coral.runtime import run_executable
    run_executable(self.tpu, caching, parameters=blob)
    self.stats["param_uploads"] += 1
    self.stats["param_bytes"] += len(blob)

  # *** fused blocks ***
  def run_block(self, name:str, x:np.ndarray, h:np.ndarray|None=None) -> np.ndarray:
    """a fused block (coral/fused.py) on M rows of x; "argmax" blocks return token ids (the classifier's block maxima, refined
    on the host with the float activations h)"""
    from coral import fused
    from coral.codegen.fused import argmax_params
    from coral.runtime import run_executable
    b, M = fused.REGISTRY[name], x.shape[0]
    if M > 256: return np.concatenate([self.run_block(name, x[m:m+256], None if h is None else h[m:m+256]) for m in range(0, M, 256)])
    Mp = 1 if M == 1 else bucket(max(M, 16))       # one row: the FULLY_CONNECTED forms
    ex = fused.programs(Mp)[name]
    self._own(("blocks", Mp))
    caching, infer = ex["PARAMETER_CACHING"], ex["EXECUTION_ONLY"]
    self.stats["calls"] += 1
    if b.kind == "argmax":                         # the token activations are this program's weights: uploaded every call
      self._upload(caching, argmax_params(x, Mp, b.in_q[1]))
      out = run_executable(self.tpu, infer, b.extra["vocab_bytes"])
      Hv, Wg = fused.VOCAB_GRID
      return fused.refine(b, fused.relayout(infer, out, Hv // fused.POOL, Wg // fused.POOL, Mp)[:, :M].T, h)
    if (pl:=hashlib.sha1(caching.bitstreams[0].data).digest()) not in self.cached:
      self._upload(caching, caching.parameters)
      self.cached.add(pl)
    xp = np.full((Mp, b.K), b.in_q[1], np.uint8)
    xp[:M] = x
    in_size = sum(hh.size for hh in infer.hints if hh.desc == "INPUT")
    out = run_executable(self.tpu, infer, xp.tobytes() + bytes([b.in_q[1]]) * (in_size - xp.size))
    if Mp == 1: return np.frombuffer(out, np.uint8)[:b.N].reshape(1, b.N)
    from coral.codegen.conv import GRIDS
    return fused.relayout(infer, out, *GRIDS[Mp], b.N)[:M]

@functools.cache
def runner() -> TPURunner|None: return None if os.getenv("MOCKCORAL") else TPURunner()

MOCK_STATS = {"calls": 0}
def run_fc(spec:FCSpec, x:np.ndarray, w:np.ndarray, b:np.ndarray, wkey=None) -> np.ndarray:
  if (r:=runner()) is None:
    MOCK_STATS["calls"] += 1
    return fc_reference(spec, x, w, b)
  return r.fc(spec, x, w, b, wkey if wkey is not None else hash((w.tobytes(), b.tobytes())))

NATIVE_CONV = int(os.getenv("CORAL_CONV", "1"))   # CORAL_CONV=0: convolutions as im2col + matmul programs
SMEM = int(os.getenv("CORAL_SMEM", "1"))          # conv programs whose output goes through scalar memory
def run_conv(spec, x:np.ndarray, w:np.ndarray, b:np.ndarray|None, wkey=None) -> np.ndarray:
  """a selected CONV_2D (coral.select.ConvSpec) on the buffers x, w, b (any shape, b None without bias) -> out[N,Cout,OH,OW]: our
  conv2d program when it covers the shape, else im2col on the host and our matmul programs"""
  from coral.select import conv_reference, run_conv as run_im2col
  x, w = x.reshape(spec.N, spec.Cin, spec.H, spec.W), w.reshape(spec.Cout, spec.Cin, spec.kh, spec.kw)
  b = np.zeros(spec.Cout, np.int32) if b is None else np.asarray(b, np.int32).reshape(spec.Cout)
  if (r:=runner()) is None:
    MOCK_STATS["calls"] += 1
    return conv_reference(spec, x, w, b)
  wkey = wkey if wkey is not None else hash((w.tobytes(), b.tobytes()))
  if NATIVE_CONV:
    try: return r.conv2d(spec, x, w, b, wkey)
    except NotImplementedError: pass
  return run_im2col(spec, x, w, b, wkey)

def run_block(name:str, x:np.ndarray, h:np.ndarray|None=None) -> np.ndarray:
  if (r:=runner()) is not None: return r.run_block(name, x, h)
  from coral import fused                          # MOCKCORAL: dequantized float math (close to, not bit-exact with, the TPU)
  MOCK_STATS["calls"] += 1
  return fused.reference(fused.REGISTRY[name], x, h)
