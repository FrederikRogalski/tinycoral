# Google Coral Edge TPU (USB). Buffers live in host memory. Generic kernels are compiled with clang and run on the host,
# kernels that carry an Edge TPU program are executed on the TPU through our own USB driver (coral.device). Edge TPU programs
# come from kernel selection (a scheduled kernel that is a quantized FC or CONV_2D, see coral/select.py and coral/qops.py) or
# from custom kernels (coral.nn).
from __future__ import annotations
import ctypes, functools, platform, zlib
import numpy as np
from tinygrad.device import Compiled, HostAllocator, Program, TinyELF, Compiler
from tinygrad.helpers import DEBUG, getenv
from tinygrad.renderer import Target, Estimates
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.ops_cpu import CPUProgram
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo

CORAL_SRC, CORAL_LIB = "//CORAL-TPU:", b"CORALTPU"
SELECT = getenv("CORAL_SELECT", 1)   # CORAL_SELECT=0: no kernel selection, every scheduled kernel is a clang kernel

class CoralCompiler(Compiler):
  def __init__(self, cpu:Compiler):
    super().__init__("compile_coral_" + str(cpu.cachekey))
    self.cpu = cpu
  def compile(self, src:str) -> bytes:
    if src.startswith(CORAL_SRC): return CORAL_LIB + src[len(CORAL_SRC):].encode()
    return self.cpu.compile(src)

class CoralRenderer(ClangRenderer):
  def __init__(self, target:Target):
    super().__init__(target)
    self.compiler = CoralCompiler(self.compiler)

  def select_program(self, ast:UOp) -> UOp|None:
    """Kernel selection. A scheduled kernel (SINK) that computes a quantized FULLY_CONNECTED or CONV_2D becomes a PROGRAM whose
    SOURCE is the Edge TPU spec: CoralCompiler passes it through and CoralProgram runs it on the TPU. None: the kernel goes to clang."""
    if not SELECT or ast.op is not Ops.SINK: return None
    try:
      from coral.select import select, why_not, FCMatch
      if (m:=select(ast)) is None:
        if DEBUG >= 4 and any(u.op is Ops.REDUCE for u in ast.toposort()): print(f"CORAL: clang kernel, not an Edge TPU op: {why_not(ast)}")
        return None
    except Exception as e:  # a bug in the matcher must not break compilation: clang computes the kernel correctly
      print(f"CORAL: kernel selection failed, using clang: {e!r}")
      return None
    s, slots = m.spec, tuple(p.arg.slot for p in m.params)
    if DEBUG >= 3: print(f"CORAL: Edge TPU kernel {s}, buffers (out, x, w, b) = slots {slots}")
    if isinstance(m, FCMatch): name, (M, N, K), nx, ny = f"tpu_fc_{s.M}x{s.K}x{s.N}", (s.M, s.N, s.K), s.M*s.K, s.M*s.N
    else:
      f = s.fc()
      name, (M, N, K) = f"tpu_conv_{s.N}x{s.Cin}x{s.H}x{s.W}_{s.Cout}x{s.kh}x{s.kw}_s{s.stride}p{s.padding}", (f.M, f.N, f.K)
      nx, ny = s.N*s.Cin*s.H*s.W, s.N*s.Cout*s.OH*s.OW
    nbytes = nx + N*K + 4*N*(m.b is not None) + ny
    sink = UOp.sink(*m.params, arg=KernelInfo(name=name, estimates=Estimates(2*M*N*K, nbytes, nbytes)))
    # the runtime launches a program's buffers in globals order: (out, x, w[, b]) is the order CoralProgram reads them in
    info = ProgramInfo(globals=slots, outs=slots[:1], ins=tuple(sorted(set(slots[1:]))), target=self.target)
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=CORAL_SRC + s.dumps())), arg=info)

def _install_select_hook():
  # tinygrad has no renderer or compiler hook that sees a kernel before codegen: render() gets the optimized, lowered program
  # (upcasted, unrolled, devectorized loads/stores), where a matmul is no longer recognizable. The scheduler's kernel AST only
  # passes through codegen.do_to_program, which already accepts a precompiled PROGRAM. So the hook wraps do_to_program: a
  # renderer with select_program() may turn the kernel AST into a PROGRAM first; every other kernel and device is untouched.
  # It is installed at import, so that compile workers (which import this module to unpickle the renderer) have it too.
  import tinygrad.codegen as codegen
  if getattr(codegen.do_to_program, "__coral_select__", False): return
  do_to_program = codegen.do_to_program
  @functools.wraps(do_to_program)
  def do_to_program_select(ast:UOp, renderer) -> UOp:
    if ast.op is Ops.SINK and (sel:=getattr(renderer, "select_program", None)) is not None and (prg:=sel(ast)) is not None: ast = prg
    return do_to_program(ast, renderer)
  setattr(do_to_program_select, "__coral_select__", True)
  setattr(codegen, "do_to_program", do_to_program_select)
_install_select_hook()
from coral.chain import install as _install_chain_hook   # TinyJit: runs of TPU kernels become one chain program (coral/chain.py)
_install_chain_hook()

def host_array(addr:int, n:int, dtype) -> np.ndarray:
  return np.ctypeslib.as_array((ctypes.c_uint8 * (n * np.dtype(dtype).itemsize)).from_address(addr)).view(dtype)

class CoralProgram(Program['CORALDevice']):
  def __init__(self, dev:CORALDevice, obj:TinyELF):
    self.dev, self.name = dev, obj.name
    self.tpu = obj.lib.startswith(CORAL_LIB)
    if self.tpu:
      import json
      from coral.tpu import FCSpec
      from coral.select import ConvSpec
      from coral.chain import ChainSpec
      d = json.loads(obj.lib[len(CORAL_LIB):].decode())
      self.block = d if "block" in d else None    # a fused block of coral.fused, a chain, a CONV_2D, or else one quantized FC
      self.chain = ChainSpec(tuple(d["chain"])) if "chain" in d else None
      self.conv = ConvSpec(**d["conv2d"]) if "conv2d" in d else None
      self.spec = None if self.block or self.chain or self.conv else FCSpec(**d)
      self.no_bias = None if self.spec is None else np.zeros(self.spec.N, np.int32)   # a selected FC without bias has 3 buffers
    else: self.cpu = CPUProgram(dev, obj)  # type: ignore[arg-type]

  def __call__(self, *bufs:int, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, timeout=None):
    if not self.tpu: return self.cpu(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait, timeout=timeout)
    if (b:=self.block) is not None:
      from coral.tpu import run_block
      x = host_array(bufs[1], b["M"] * b["K"], np.uint8).reshape(b["M"], b["K"])
      if b.get("argmax"):
        h = host_array(bufs[2], b["M"] * b["K"], np.float32).reshape(b["M"], b["K"])
        host_array(bufs[0], b["M"], np.int32)[:] = run_block(b["block"], x, h)
      else: host_array(bufs[0], b["M"] * b["N"], np.uint8)[:] = run_block(b["block"], x).reshape(-1)
      return None
    if self.chain is not None:
      from coral.chain import buffers, run_chain
      arrs = [host_array(b, n, dt) for b, (n, dt) in zip(bufs, buffers(self.chain))]
      arrs[0][:] = run_chain(self.chain, arrs[1], arrs[2:])
      return None
    if (c:=self.conv) is not None:
      from coral.tpu import run_conv
      x, w = host_array(bufs[1], c.N*c.Cin*c.H*c.W, np.uint8), host_array(bufs[2], c.Cout*c.Cin*c.kh*c.kw, np.uint8)
      b = host_array(bufs[3], c.Cout, np.int32) if len(bufs) > 3 else None
      wkey = (bufs[2], zlib.crc32(w), zlib.crc32(b) if b is not None else 0)
      host_array(bufs[0], c.N*c.Cout*c.OH*c.OW, np.uint8)[:] = run_conv(c, x, w, b, wkey).reshape(-1)
      return None
    from coral.tpu import run_fc
    s = self.spec
    out, x = host_array(bufs[0], s.M * s.N, np.uint8), host_array(bufs[1], s.M * s.K, np.uint8)
    w = host_array(bufs[2], s.N * s.K, np.uint8).reshape(s.N, s.K)
    b = host_array(bufs[3], s.N, np.int32) if len(bufs) > 3 else self.no_bias
    wkey = (bufs[2], zlib.crc32(w), zlib.crc32(b))
    out[:] = run_fc(s, x.reshape(s.M, s.K), w, b, wkey).reshape(-1)
    return None

class CORALDevice(Compiled):
  def __init__(self, device:str=""):
    self.remote = None
    super().__init__(device, HostAllocator(self), [CoralRenderer], CoralProgram,
                     arch={'amd64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")
