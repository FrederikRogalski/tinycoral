# Edge TPU ("beagle" DarwiNN) instructions: one or more little endian 128-bit words; bit 0 is the lsb of byte 0 of word 0, word j
# holds bits [128j, 128j+128). Bits 6-11 of the first word are the opcode, which fixes the length (LEN).
from __future__ import annotations
import struct

def cdiv(a:int, b:int) -> int: return -(-a // b)
def round_up(a:int, b:int) -> int: return cdiv(a, b) * b
def f32(x:float) -> float: return struct.unpack("<f", struct.pack("<f", x))[0]
def f32_bits(x:float) -> int: return struct.unpack("<I", struct.pack("<f", x))[0]
def bits_f32(b:int) -> float: return struct.unpack("<f", struct.pack("<I", b))[0]
def word(b:bytes, i:int=0) -> int: return int.from_bytes(b[i*16:(i+1)*16], "little")
def opcode(w:int) -> int: return (w >> 6) & 0x3f

# instruction lengths in 128-bit words: verified by single-stepping on hardware for 3e 1a 24 20 25 14 11 26 17 13 01 27 10 21 3f,
# for 02 15 18 19 23 by the decoded instruction layouts; the rest learned statistically
LEN = {0x00: 14, 0x01: 17, 0x02: 17, 0x03: 8, 0x04: 1, 0x05: 4, 0x06: 3, 0x07: 7, 0x08: 1, 0x0b: 2, 0x0c: 7, 0x0e: 1, 0x10: 4, 0x11: 4,
       0x12: 4, 0x13: 7, 0x14: 7, 0x15: 6, 0x16: 6, 0x17: 6, 0x18: 6, 0x19: 13, 0x1a: 1, 0x1c: 5, 0x1d: 5, 0x1e: 4, 0x1f: 1, 0x20: 1,
       0x21: 1, 0x22: 29, 0x23: 1, 0x24: 2, 0x25: 2, 0x26: 4, 0x27: 2, 0x28: 1, 0x29: 3, 0x2a: 15, 0x2b: 1, 0x2c: 1, 0x2d: 1, 0x2f: 1,
       0x30: 11, 0x32: 15, 0x33: 1, 0x34: 2, 0x38: 1, 0x39: 35, 0x3a: 1, 0x3c: 12, 0x3e: 1, 0x3f: 1}

def split(prog:bytes) -> list[tuple[int, list[int]]]:
  """-> [(word index, [words])] of the program's instructions"""
  out, i, n = [], 0, len(prog) // 16
  while i < n:
    ln = LEN.get(opcode(word(prog, i)), 1)
    out.append((i, [word(prog, j) for j in range(i, min(i + ln, n))]))
    i += ln
  return out

def ttu(p:str, strides:list[int], counts:list[int], levels:int, cnt:str="cnt") -> dict[str, int]:
  """fields {p}inc{d}, {p}{cnt}{d} (= count - 1) of an address generator (TTU) loop nest given innermost first, padded to `levels`
  with stride 0 / count 1. The hardware adds inc_d when level d steps and the inner levels wrap, so
  inc_d = stride_d - sum_{e<d} (count_e - 1) * stride_e (the padding levels rewind to the base, as edgetpu_compiler encodes it)"""
  assert max(len(strides), len(counts)) <= levels and all(c >= 1 for c in counts)
  out, acc = {}, 0
  for d, (s, c) in enumerate(zip([*strides, *[0] * (levels - len(strides))], [*counts, *[1] * (levels - len(counts))])):
    out[f"{p}inc{d}"], out[f"{p}{cnt}{d}"] = s - acc, c - 1
    acc += s * (c - 1)
  return out

LAYOUTS: dict[int, Layout] = {}     # opcode -> the Layout of every decoded instruction (registered by the instruction modules)
HEADER = [("gate", 0, 1), ("pred_reg", 1, 3), ("pred_pol", 4, 1), ("opcode", 6, 6)]   # if gate: run iff p[pred_reg] == pred_pol

class Layout:
  """a fixed-length instruction: HEADER + fields (name, lo bit, width[, signed]). Bits no field covers become reserved fields
  rsv<lo>, so encode(**decode(words)) == words for every bit pattern. encode() starts from the defaults (opcode = ops[0])."""
  def __init__(self, name:str, ops:tuple[int, ...], nwords:int, fields:list[tuple], **defaults):
    self.name, self.ops, self.nwords, self.defaults = name, ops, nwords, ({"opcode": ops[0]} if ops else {}) | defaults
    self.fields, pos = [], 0
    for n, lo, w, *s in sorted((HEADER if ops else []) + fields, key=lambda f: f[1]) + [("", 128 * nwords, 0)]:
      assert lo >= pos, f"{name}: field {n} overlaps"
      if lo > pos: self.fields.append((f"rsv{pos}", pos, lo - pos, False))
      if w:
        self.fields.append((n, lo, w, bool(s and s[0])))
        pos = lo + w
    self.index = {n: (lo, w, s) for n, lo, w, s in self.fields}
    assert len(self.index) == len(self.fields), f"{name}: duplicate field names"
    for op in ops: LAYOUTS[op] = self
  def decode(self, words:list[int]) -> dict[str, int]:
    assert len(words) == self.nwords, f"{self.name}: expected {self.nwords} words, got {len(words)}"
    v = sum(w << (128 * j) for j, w in enumerate(words))
    assert not self.ops or opcode(v) in self.ops, f"{self.name}: opcode {opcode(v):#x}"
    d = {n: (v >> lo) & ((1 << w) - 1) for n, lo, w, _ in self.fields}
    for n, _, w, s in self.fields:
      if s and d[n] >> (w - 1): d[n] -= 1 << w
    return d
  def encode(self, **kw) -> list[int]:
    v = 0
    for n, x in (self.defaults | kw).items():
      assert n in self.index, f"{self.name}: unknown field {n}"
      lo, w, s = self.index[n]
      assert (-(1 << (w - 1)) <= x < (1 << (w - 1))) if s else 0 <= x < (1 << w), f"{self.name}.{n}={x} doesn't fit {w} bits"
      v |= (x & ((1 << w) - 1)) << lo
    assert not self.ops or opcode(v) in self.ops, f"{self.name}: opcode {opcode(v):#x}"
    return [(v >> (128 * j)) & ((1 << 128) - 1) for j in range(self.nwords)]

# geohot's view of a scalar-core word (edgetpuxray decompiler.py); bits 6-10 "branch" and bit 11 "enable_scalar" are the opcode
WORD = Layout("word", (), 1, [
  ("gate", 0, 1), ("pred_reg", 1, 3), ("yes_pred", 4, 1), ("unk_0", 5, 1), ("branch", 6, 5), ("enable_scalar", 11, 1),
  ("enable_vector", 12, 2), ("vs_reg_v1", 14, 5), ("imm_size", 19, 12), ("v_op", 31, 5), ("v_offset", 36, 8), ("v_cmd", 44, 5),
  ("vs_reg", 49, 5), ("s_op", 54, 6), ("s_x", 60, 5), ("s_y", 65, 5), ("imm_scalar", 70, 32), ("v_op_2", 102, 3),
  ("vs_reg_w", 105, 5), ("unk_3", 110, 18)])
