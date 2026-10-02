# minimal schema-less flatbuffer reader: fields are addressed by their index in the .fbs table (unions take two slots)
import struct

class Table:
  def __init__(self, buf:bytes, pos:int): self.buf, self.pos = buf, pos
  @classmethod
  def root(cls, buf:bytes, off:int=0): return cls(buf, off + struct.unpack_from("<I", buf, off)[0])

  def _off(self, idx:int) -> int|None:
    vt = self.pos - struct.unpack_from("<i", self.buf, self.pos)[0]
    vtlen = struct.unpack_from("<H", self.buf, vt)[0]
    if 4 + 2*idx >= vtlen: return None
    fo = struct.unpack_from("<H", self.buf, vt + 4 + 2*idx)[0]
    return self.pos + fo if fo else None

  def scalar(self, idx:int, fmt:str="<i", default=0):
    return default if (o:=self._off(idx)) is None else struct.unpack_from(fmt, self.buf, o)[0]
  def _ind(self, o:int) -> int: return o + struct.unpack_from("<I", self.buf, o)[0]
  def table(self, idx:int) -> "Table|None": return None if (o:=self._off(idx)) is None else Table(self.buf, self._ind(o))
  def _vec(self, idx:int) -> tuple[int, int]:
    if (o:=self._off(idx)) is None: return 0, 0
    v = self._ind(o)
    return v + 4, struct.unpack_from("<I", self.buf, v)[0]
  def blob(self, idx:int) -> bytes:
    s, n = self._vec(idx)
    return self.buf[s:s+n]
  def string(self, idx:int) -> str|None: return None if self._off(idx) is None else self.blob(idx).decode()
  def scalars(self, idx:int, fmt:str="<i") -> list:
    s, n = self._vec(idx)
    return list(struct.unpack_from(f"<{n}{fmt.lstrip('<')}", self.buf, s))
  def tables(self, idx:int) -> list["Table"]:
    s, n = self._vec(idx)
    return [Table(self.buf, self._ind(s + 4*i)) for i in range(n)]
  def strings(self, idx:int) -> list[bytes]:
    s, n = self._vec(idx)
    out = []
    for i in range(n):
      v = self._ind(s + 4*i)
      out.append(self.buf[v+4:v+4+struct.unpack_from("<I", self.buf, v)[0]])
    return out
