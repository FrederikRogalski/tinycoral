# parse *_edgetpu.tflite -> DarwiNN executables (schema: libedgetpu/executable/executable.fbs)
from __future__ import annotations
import struct, pathlib
from dataclasses import dataclass
from flatbuffers import flexbuffers
from coral.fb import Table

DESC = {0: "OUTPUT", 1: "INPUT", 2: "PARAMETER", 3: "SCRATCH"}
EXE_TYPE = {0: "STAND_ALONE", 1: "PARAMETER_CACHING", 2: "EXECUTION_ONLY"}
DTYPE = {0: "u8", 1: "u16", 2: "i32", 3: "bf16", 4: "f16", 5: "f32", 8: "i8", 9: "i16"}
HINT = {1: "dma", 2: "instruction", 3: "interrupt", 4: "fence"}

@dataclass
class FieldOffset:
  desc: str
  batch: int
  name: str|None
  upper: bool
  offset_bit: int

@dataclass
class Bitstream:
  data: bytes
  relocs: list[FieldOffset]

@dataclass
class Hint:
  kind: str
  direction: str
  desc: str|None = None
  name: str|None = None
  offset: int = 0
  size: int = 0
  chunk: int = 0
  interrupt: int = 0
  def __repr__(self):
    d = "in " if self.direction == "INFEED" else "out"
    if self.kind == "dma": return f"<dma {d} {self.desc}:{self.name} off={self.offset:#x} size={self.size:#x}>"
    if self.kind == "instruction": return f"<instr {d} chunk={self.chunk}>"
    if self.kind == "interrupt": return f"<interrupt {d} sc_int_{self.interrupt}>"
    return f"<{self.kind} {d}>"

@dataclass
class Layer:
  name: str
  size_bytes: int
  y: int
  x: int
  z: int
  zero_point: int
  scale: float
  dtype: str
  is_output: bool
  shape: list[tuple[int, int]]
  execution_count: int
  output_layout: dict|None = None

@dataclass
class Executable:
  name: str|None
  type: str
  batch_size: int
  scratch_size: int
  bitstreams: list[Bitstream]
  parameters: bytes
  hints: list[Hint]
  fully_deterministic: bool
  inputs: list[Layer]
  outputs: list[Layer]
  chip: str|None
  estimated_cycles: int
  narrow_bytes_per_tile: int
  caching_token: int
  version: int

def _meta(t:Table|None):
  if t is None: return None, 0, None, False
  return DESC[t.scalar(0, "<h")], t.scalar(1), t.string(2), bool(t.scalar(3, "<h"))

def _layer(t:Table, is_output:bool) -> Layer:
  num = t.table(5)
  shape_t = t.table(11)
  shape = []
  if shape_t is not None:
    s, n = shape_t._vec(0)
    shape = [struct.unpack_from("<ii", t.buf, s + 8*i) for i in range(n)]
  layout = None
  if is_output and t.scalar(7, "<B") == 1 and (ol:=t.table(8)) is not None and (lt:=ol.table(0)) is not None:
    layout = {k: lt.scalars(i) for i, k in enumerate(["y_tile", "x_tile", "tile_byte_offset", "x_local_byte_offset", "y_local_y_offset",
                                                      "x_local_row_size"])}
  return Layer(t.string(0), t.scalar(1), t.scalar(2), t.scalar(3), t.scalar(4),
               num.scalar(0) if num else 0, num.scalar(1, "<f") if num else 0.0, DTYPE.get(t.scalar(6, "<h"), "?"), is_output,
               shape, t.scalar(9, default=1), layout)

def parse_executable(buf:bytes) -> Executable:
  e = Table.root(buf)
  bitstreams = []
  for b in e.tables(5):
    relocs = []
    for fo in b.tables(1):
      desc, batch, name, upper = _meta(fo.table(0))
      relocs.append(FieldOffset(desc, batch, name, upper, fo.scalar(1)))
    bitstreams.append(Bitstream(b.blob(0), relocs))
  hints, det = [], False
  if (dh:=e.table(7)) is not None:
    det = bool(dh.scalar(1, "<B"))
    for h in dh.tables(0):
      kind, ht, direction = HINT.get(h.scalar(0, "<B"), "?"), h.table(1), "OUTFEED" if h.scalar(2, "<h") else "INFEED"
      if kind == "dma":
        desc, _, name, _ = _meta(ht.table(0))
        hints.append(Hint(kind, direction, desc, name, ht.scalar(1), ht.scalar(2)))
      elif kind == "instruction": hints.append(Hint(kind, direction, chunk=ht.scalar(0)))
      elif kind == "interrupt": hints.append(Hint(kind, direction, interrupt=ht.scalar(0, "<h")))
      else: hints.append(Hint(kind, direction))
  return Executable(e.string(1), EXE_TYPE[e.scalar(13, "<h")], e.scalar(3), e.scalar(4), bitstreams, e.blob(6), hints, det,
                    [_layer(t, False) for t in e.tables(8)], [_layer(t, True) for t in e.tables(9)], e.string(10),
                    e.scalar(16, "<q") or e.scalar(11), e.scalar(12), e.scalar(14, "<Q"), e.scalar(0))

def parse_package(buf:bytes) -> list[Executable]:
  pkg = Table.root(buf)
  assert buf[4:8] == b"DWN1", f"bad package identifier {buf[4:8]!r}"
  multi = Table.root(pkg.blob(1))
  return [parse_executable(s) for s in multi.strings(0)]

def tflite_custom_ops(buf:bytes) -> list[tuple[str, bytes]]:
  model = Table.root(buf)
  codes = [(oc.string(1), max(oc.scalar(0, "<b"), oc.scalar(3))) for oc in model.tables(1)]
  out = []
  for sg in model.tables(2):
    for op in sg.tables(3):
      custom, _ = codes[op.scalar(0, "<I")]
      if custom is not None: out.append((custom, op.blob(5)))
  return out

def load_edgetpu_tflite(path:str|pathlib.Path) -> list[Executable]:
  exes = []
  for custom, opts in tflite_custom_ops(pathlib.Path(path).read_bytes()):
    if custom != "edgetpu-custom-op": continue
    m = flexbuffers.GetRoot(opts).AsMap
    exes += parse_package(bytes(m["4"].AsStringBytes))
    try: rest = m["7"].AsVector
    except KeyError: rest = []
    for r in rest: exes += parse_package(bytes(r.AsStringBytes))
  return exes

def summarize(exe:Executable) -> str:
  lines = [f"Executable {exe.name!r} type={exe.type} chip={exe.chip} batch={exe.batch_size} scratch={exe.scratch_size} "
           f"params={len(exe.parameters)} cycles={exe.estimated_cycles} narrow/tile={exe.narrow_bytes_per_tile} token={exe.caching_token:#x}"]
  for i, b in enumerate(exe.bitstreams):
    lines.append(f"  bitstream[{i}]: {len(b.data)} bytes, {len(b.relocs)} relocs")
    for r in b.relocs:
      lines.append(f"    reloc {r.desc}:{r.name} batch={r.batch} {'hi' if r.upper else 'lo'} @bit {r.offset_bit} (byte {r.offset_bit/8:.2f})")
  lines.append(f"  hints (deterministic={exe.fully_deterministic}): {exe.hints}")
  for l in exe.inputs + exe.outputs:
    lines.append(f"  {'out' if l.is_output else 'in '} {l.name!r} {l.dtype} yxz=({l.y},{l.x},{l.z}) size={l.size_bytes} zp={l.zero_point} "
                 f"scale={l.scale:.6g} shape={l.shape} exec_count={l.execution_count}")
    if l.output_layout: lines.append(f"      layout: { {k: (v[:8], len(v)) for k, v in l.output_layout.items()} }")
  return "\n".join(lines)

if __name__ == "__main__":
  import sys
  for e in load_edgetpu_tflite(sys.argv[1]): print(summarize(e))
