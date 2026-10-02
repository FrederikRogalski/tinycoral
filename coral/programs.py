# one matmul y[M,N] = requant(x[M,K] @ W[N,K].T + b[N]) as an Edge TPU program from our code generator:
#   M = 1: a FULLY_CONNECTED program (coral/codegen/fc.py) on n consecutive tiles from tile s at one parameter offset
#   M > 1: a 1x1 conv over a grid of positions (coral/codegen/conv.py), the parameters on one tile
# and one CONV_2D (coral/codegen/conv2d.py): an image x[H,W,Cin] (NHWC) per call, the parameters on one tile
# Programs are generated per (placement, quantization) and cached; the host side lays out parameters, inputs and outputs.
from __future__ import annotations
import functools, numpy as np
from dataclasses import dataclass
from coral.executable import Executable, Bitstream, Hint
from coral.codegen.fc import gen_fc, FCGeom
from coral.codegen.conv import gen_conv1x1, io_sizes, output_layout, GRIDS, ConvGeom, param_limit as conv_param_limit
from coral.codegen.fused import conv_blob
from coral.codegen.conv2d import gen_conv2d, conv2d_io, conv2d_blob, Conv2DGeom, param_limit, smem_output

WIDE_BYTES = 496640   # parameter bytes per tile that stay below every program's own wide-memory buffers

@dataclass(frozen=True)
class Quant:
  in_zp: int                                               # mult = in_scale * w_scale / out_scale
  w_zp: int
  out_zp: int
  mult: float
  lo: int = 0                                              # clamp in the quantized output domain (fused relu etc.)
  hi: int = 255
  def fields(self) -> dict:
    return dict(w_zp=self.w_zp, in_zp=self.in_zp, out_zp=self.out_zp, mult=float(np.float32(self.mult)),
                clamp_min=float(self.lo - self.out_zp), clamp_max=float(self.hi - self.out_zp))

def bucket(M:int) -> int: return next(m for m in GRIDS if M <= m)
def _exe(kind:str, bitstream:bytes, hints:list[Hint]) -> Executable:
  return Executable(None, kind, 1, 0, [Bitstream(bitstream, [])], b"", [Hint("instruction", "INFEED"), *hints, Hint("interrupt", "OUTFEED")],
                    True, [], [], "beagle", 0, 0, 0, 0)

class Matmul:
  """the host side shared by both forms: n >= 64 outputs (N < 64 runs the 64-output program with padded rows: the compiler's
  smaller groups read wrong weights at >= 256 KiB), parameters as edgetpu_compiler lays them out"""
  ntiles: int
  tile_bytes: int
  in_size: int
  out_size: int
  def __init__(self, N:int, K:int): self.N, self.K, self.n = N, K, max(N, 64)
  def fits(self) -> bool: return self.tile_bytes <= self.limit
  def params(self, w:np.ndarray, b:np.ndarray, w_zp:int) -> bytes: return conv_blob(w, w_zp, b)
  def programs(self, shift:int, offset:int, q:Quant) -> tuple[Executable, Executable]:
    pc, eo = self._gen(shift, offset, q)
    return (_exe("PARAMETER_CACHING", pc, [Hint("dma", "INFEED", "PARAMETER", None, 0, self.param_bytes)]),
            _exe("EXECUTION_ONLY", eo,
                 [Hint("dma", "INFEED", "INPUT", "x", 0, self.in_size), Hint("dma", "OUTFEED", "OUTPUT", "y", 0, self.out_size)]))

class FCProgram(Matmul):
  limit = buffers = WIDE_BYTES
  def __init__(self, N:int, K:int):
    super().__init__(N, K)
    g = FCGeom(self.n, K)
    self.ntiles, self.tile_bytes = g.T, 256 * -(-(g.P * g.group_bytes) // 256)   # parameter regions are 256-byte aligned
    self.in_size, self.out_size, self.param_bytes = g.S, g.out_bytes, g.param_bytes
  @functools.cache
  def _gen(self, shift:int, offset:int, q:Quant):
    # the compute tiles move to [shift, shift+T); the input gather stays on tiles 0.. (a structure edgetpu_compiler never emits)
    return gen_fc(self.n, self.K, param_offset=offset, tile_shift=shift, move_gather=False, quant=q.fields())
  def input_bytes(self, x:np.ndarray, in_zp:int) -> bytes:
    return np.ascontiguousarray(x, np.uint8).reshape(-1).tobytes() + bytes([in_zp]) * (self.in_size - self.K)
  def output_array(self, out:bytes) -> np.ndarray: return np.frombuffer(out, np.uint8)[:self.N].reshape(1, self.N)

class ConvProgram(Matmul):
  def __init__(self, Mp:int, N:int, K:int):
    super().__init__(N, K)
    (self.H, self.W), self.Mp = GRIDS[Mp], Mp
    self.k = max(K, -(-256 // self.W))                     # an image row must fill a 256-byte ring packet: pad K (with zero points)
    self.ntiles, self.param_bytes = 1, len(conv_blob(np.zeros((self.n, self.k), np.uint8), 0))
    self.tile_bytes = 256 * -(-self.param_bytes // 256)
    self.in_size, self.out_size = io_sizes(Mp, self.n, self.k)
    self.layout = output_layout(Mp, self.n)
    # its own wide buffers start lower for big inputs (e.g. 458 496 B at 256 x 2304), and every call writes them on all 16 tiles
    self.buffers = conv_param_limit(ConvGeom(Mp, self.n, self.k)) * 64
    self.limit = min(WIDE_BYTES, self.buffers)
  def params(self, w:np.ndarray, b:np.ndarray, w_zp:int) -> bytes:
    return conv_blob(np.pad(w, ((0, 0), (0, self.k - self.K)), constant_values=w_zp), w_zp, b)
  @functools.cache
  def _gen(self, tile:int, offset:int, q:Quant): return gen_conv1x1(self.Mp, self.n, self.k, tile, offset, q.fields())
  def input_bytes(self, x:np.ndarray, in_zp:int) -> bytes:
    xp = np.full((self.Mp, self.k), in_zp, np.uint8)                             # K bytes per position, no row padding
    xp[:len(x), :self.K] = x
    return xp.tobytes() + bytes([in_zp]) * (self.in_size - xp.size)
  def output_array(self, buf:bytes) -> np.ndarray:
    L, b, out = self.layout, np.frombuffer(buf, np.uint8), np.empty((self.Mp, self.N), np.uint8)
    for y in range(self.H):
      for x in range(self.W):
        s = L["tile_byte_offset"][L["y_tile"][y] + L["x_tile"][x]] + L["y_local_y_offset"][y] * L["x_local_row_size"][x] + L["x_local_byte_offset"][x]
        out[y * self.W + x] = b[s:s+self.N]
    return out

@functools.cache
def fc_program(N:int, K:int) -> FCProgram: return FCProgram(N, K)
@functools.cache
def conv_program(Mp:int, N:int, K:int) -> ConvProgram: return ConvProgram(Mp, N, K)

class Conv2dProgram:
  """CONV_2D y[OH,OW,Cout] = requant(conv(x[H,W,Cin], w[Cout,kh,kw,Cin]) + b), one image per call; weights on one tile"""
  def __init__(self, H:int, W:int, Cin:int, Cout:int, kh:int, kw:int, stride:int, padding:str):
    self.args, (self.Cout, self.Cin) = (H, W, Cin, Cout, kh, kw, stride, padding), (Cout, Cin)
    io, g = conv2d_io(*self.args), Conv2DGeom(*self.args)
    self.in_size, self.out_size, self.param_bytes, self.layout = io["input_bytes"], io["output_bytes"], io["param_bytes"], io["output_layout"]
    self.OH, self.OW = len(self.layout["y_tile"]), len(self.layout["x_tile"])
    self.tile_bytes, self.smem, self.blocks = 256 * -(-self.param_bytes // 256), smem_output(g), g.blocks
    # buffers: where this program's own wide buffers start (it writes them on every tile, every call). limit: where its weights
    # must end; groups of fewer than 64 outputs also stay below 256 KiB (FC programs with such groups read wrong weights above,
    # see NOTES.md; not tested for convs, so the same rule)
    self.buffers = param_limit(g) * 64
    self.limit = min(WIDE_BYTES, self.buffers, *([262144] if g.cg < 64 else []))
  def fits(self) -> bool: return self.tile_bytes <= self.limit
  def params(self, w:np.ndarray, b:np.ndarray, w_zp:int) -> bytes:
    H, W, _, _, _, _, stride, padding = self.args
    return conv2d_blob(w, w_zp, b, H, W, stride, padding)  # w [Cout, kh, kw, Cin]
  def ntiles(self, limit:int) -> int:
    """tiles the weights take from offset 0 when every tile holds rows up to `limit` bytes (edgetpu_compiler's split of big layers)"""
    return 1 if self.tile_bytes <= limit else -(-self.blocks // (limit // 256))
  @functools.cache
  def programs(self, tile:int|tuple[int, ...], offset:int, q:Quant, limit:int|None=None) -> tuple[Executable, Executable]:
    """tile: the weights' tile, or a tuple of tiles that hold them in pieces up to `limit` bytes each"""
    pc, eo = gen_conv2d(*self.args, param_tile=tile, param_offset=offset, quant=q.fields(), param_limit_units=None if limit is None else limit // 64)
    return (_exe("PARAMETER_CACHING", pc, [Hint("dma", "INFEED", "PARAMETER", None, 0, self.param_bytes)]),
            _exe("EXECUTION_ONLY", eo,
                 [Hint("dma", "INFEED", "INPUT", "x", 0, self.in_size), Hint("dma", "OUTFEED", "OUTPUT", "y", 0, self.out_size)]))
  def input_bytes(self, x_hwc:np.ndarray) -> bytes:
    b = np.ascontiguousarray(x_hwc, np.uint8).tobytes()
    return b + bytes(self.in_size - len(b))
  @functools.cached_property
  def gather(self) -> np.ndarray:
    """the byte index of every y[OH, OW, Cout] in the output buffer"""
    L = {k: np.array(v) for k, v in self.layout.items()}
    tile = L["y_tile"][:, None] + L["x_tile"][None, :]
    starts = L["tile_byte_offset"][tile] + L["y_local_y_offset"][:, None] * L["x_local_row_size"][None, :] + L["x_local_byte_offset"][None, :]
    return starts[..., None] + np.arange(self.Cout)
  def output_array(self, buf:bytes) -> np.ndarray:
    """-> y[OH, OW, Cout]"""
    return np.frombuffer(buf, np.uint8)[self.gather]

@functools.cache
def conv2d_program(H:int, W:int, Cin:int, Cout:int, kh:int, kw:int, stride:int, padding:str) -> Conv2dProgram:
  return Conv2dProgram(H, W, Cin, Cout, kh, kw, stride, padding)
