# shared by the code generators: the instruction emitter, the PARAMETER_CACHING program, the execution programs' fixed wide buffers
from __future__ import annotations
from coral.isa import f32, ttu, scalar as SC, ring_mesh as RM

WIDE_TOP = 8320          # 64-byte units: the execution programs' own wide buffers sit right below this, above every parameter region
WIDE_OUT_FIFO = 8312     # narrowToWide output -> ringProducer -> outfeed
REF_QUANT = dict(w_zp=128, in_zp=128, out_zp=128, mult=f32((1/32) * (1/64) / (1/4)))   # the quantization of the compiler references

class Emitter:
  """collects 128-bit words; seq counts the tile-dispatched instructions (0x01-0x19) and the tile fences (0x1a)"""
  def __init__(self): self.words, self.seq = [], 0
  def tile(self, *insns:list[int]):
    for w in insns:
      self.words += w
      self.seq += 1
  def sync(self, fn, *args, **kw): self.tile(fn(self.seq, *args, **kw))     # fn(seq) holds exactly one tile fence
  def scalar(self, words:list[int], seqs:int=0):                                         # seqs: tile fences inside a skeleton block
    self.words += words
    self.seq += seqs
  def program(self) -> bytes: return SC.program(self.words)

Piece = tuple[int, int, int]     # (tile, wide offset in 64-byte units, parameter rows): one contiguous part of a blob on one tile

def caching_program(nbytes:int, parts:list[list[tuple[dict, int, int, int]]]) -> bytes:
  """PARAMETER_CACHING of an nbytes blob: per part its ringConsumers, then its parameter infeeds. Part entries are (ringConsumer
  fields without seq, blob byte offset, bytes, 64-byte lanes per weight row); the infeed goes to the ringConsumer's tiles."""
  e = Emitter()
  e.scalar(SC.sync_init(True) + SC.scsync_init() + SC.host_dma(SC.TAG_PARAMETERS, nbytes, regs="par", load_base=(0, 0)) + SC.param_pop(nbytes),
           seqs=1)
  for part in parts:
    for rc, *_ in part: e.tile(RM.encode_ringConsumer(seq=e.seq, **rc))
    for rc, off, n, lanes in part: e.scalar(SC.param_infeed(off, n, rc["tile_mask"], lanes))
  e.sync(SC.epilogue, caching=True)
  return e.program()

def piece_parts(pieces:list[list[Piece]], row_bytes:int=256) -> list[list[tuple[dict, int, int, int]]]:
  """caching parts of consecutive blobs cut into pieces (one list per blob): the ringConsumer writes a piece's rows from offset + 2"""
  out, first = [], 0
  for ps in pieces:
    out.append([])
    for t, o, n in ps:
      rc = dict(tile_mask=1 << t, addr=2 + o, sdims=int(n > 1), **ttu("", [1], [n], 4))
      out[-1].append((rc, row_bytes * first, row_bytes * n, row_bytes // 64))
      first += n
  return out

def run_test(name:str):
  """`python -m coral.codegen.<x>` runs its acceptance test test/<name>.py (loaded by path: `test` is also a stdlib package)"""
  import sys, pathlib, importlib.util
  spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).resolve().parents[2] / "test" / f"{name}.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  sys.exit(mod.main(sys.argv[1:]))
