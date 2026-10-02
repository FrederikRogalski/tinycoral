# the scalar core's compute instructions (docs/isa/codegen_eltops.md 3): what edgetpu_compiler emits when it runs SOFTMAX on the scalar
# core. Every word is a 0x20 bundle in geohot's field view (coral.isa.WORD) with these slots, read from the softmax programs:
#   scalar ALU    s_op (54-59), s_x (60-64), s_y (65-69), imm (70-101). s_op bit 5 = immediate form, s_op & 0x1f = the operation
#                 (0x01-0x0f integer: add sub and or xor shl shr asr eq ne gt lt ge ges mov; 0x10-0x16 float32: i2f f2i fadd fsub
#                 fmul fmax fmin); i2f / f2i take the source register in the immediate field.
#   memory        enable_vector (12-13) = 2: load s[vs_reg_v1] = smem32[s[imm_size]]; 3: the same with a post-increment of the address
#                 register (imm_size bit 5 set). v_op (31-35) = 4: store smem32[s[v_offset]] = s[vs_reg]; 8: push s[vs_reg] into the
#                 host DMA stream (the descriptor issued before, v_op 0xa / 0xc).
#   special unit  v_op_2 (102-104) = 2 / 3 with source vs_reg_w (105-109) and destination unk_3 (110-114): sfu 2 = reciprocal,
#                 sfu 3 = exp2 (names from the softmax data flow; precision unknown).
#   0x23 loop     repeat the next `body` words `count` times: count at bits 12-27, body at bit 28 (1 word)
#   0x22 branch   1 word: relative word offset (15 bits signed) at bit 17, taken when p0 is set (the only form seen; bits 12-16 = 0x11, bit 33)
# coral.isa.LEN has 8 / 29 words for opcodes 0x03 / 0x22 (statistical guesses); split() here uses the decoded lengths (0x03 = the 17-word op
# with opcode 3, the reduction op of SUM / MEAN).
from __future__ import annotations
from coral.isa import LEN as _LEN, LAYOUTS, WORD, word, opcode, f32_bits, Layout
from coral.isa import op as OP

LEN = dict(_LEN) | {0x03: 17, 0x22: 1, 0x23: 1}

def split(prog:bytes) -> list[tuple[int, list[int]]]:
  """-> [(word index, [words])] with the corrected lengths of 0x03 / 0x22 / 0x23"""
  out, i, n = [], 0, len(prog) // 16
  while i < n:
    ln = LEN.get(opcode(word(prog, i)), 1)
    out.append((i, [word(prog, j) for j in range(i, min(i + ln, n))]))
    i += ln
  return out

# opcode 3: the 17-word op layout of 0x01 / 0x02 (the SUM / MEAN reduction: the operand is read through the par TTU only)
OP3 = Layout("op3", (0x03,), 17, [f[:3] + ((True,) if f[3] else ()) for f in OP.OP.fields
                                 if not f[0].startswith("rsv") and f[0] not in ("gate", "pred_reg", "pred_pol", "opcode")])
del LAYOUTS[0x03]   # not in the global registry: coral.isa.split still cuts opcode 3 into 8 words, its decoders would misread it
encode_op3, decode_op3 = OP3.encode, OP3.decode

# *** scalar ALU ***
I2F, F2I, FADD, FSUB, FMUL, FMAX, FMIN = range(0x10, 0x17)
ADD, SUB, AND, OR, XOR, SHL, SHR = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
NE, LT, MOV = 0x0a, 0x0c, 0x0f

def _w(**f) -> int: return WORD.encode(enable_scalar=1, **f)[0]
def alu(op:int, x:int, y:int=0, imm:int|None=None, reg:int|None=None, **kw) -> int:
  """s_x = s_y <op> imm (immediate form, s_op | 0x20) or s_y <op> s_reg"""
  if reg is not None: return _w(s_op=op, s_x=x, s_y=y, imm_scalar=reg, **kw)
  return _w(s_op=op | 0x20, s_x=x, s_y=y, imm_scalar=imm & 0xffffffff, **kw)
def mov(x:int, reg:int) -> int: return alu(MOV, x, reg=reg)
def movi(x:int, imm:int) -> int: return alu(MOV, x, imm=imm)
def fimm(op:int, x:int, y:int, v:float) -> int: return alu(op, x, y, imm=f32_bits(v))   # float op with an f32 immediate
def i2f(x:int, src:int) -> int: return alu(I2F, x, reg=src)
def f2i(x:int, src:int) -> int: return alu(F2I, x, reg=src)
def ld(x:int, addr:int, inc:bool=False, **kw) -> int:
  """s_x = smem32[s_addr] (inc: s_addr += 1 afterwards)"""
  return _w(enable_vector=3 if inc else 2, vs_reg_v1=x, imm_size=addr | (0x20 if inc else 0), **kw)
def st(addr:int, val:int) -> int: return _w(v_op=4, v_offset=addr, vs_reg=val)   # smem32[s_addr] = s_val
def push(reg:int, **kw) -> int: return WORD.encode(enable_scalar=1, **({"v_op": 8, "vs_reg": reg} | kw))[0]
def sfu(op:int, x:int, src:int) -> int: return _w(v_op_2=op, vs_reg_w=src, unk_3=x)  # s_x = f(s_src): 2 reciprocal, 3 exp2
RECIP, EXP2 = 2, 3
NOP = _w()
def loop(count:int, body:int) -> int:
  assert 0 < count < 1 << 16 and 0 < body < 1 << 10
  return 0x8c0 | count << 12 | body << 28
def branch(off:int) -> int:
  """jump by off words (relative to this instruction) while p0"""
  return 0x880 | 0x11 << 12 | (off & 0x7fff) << 17 | 1 << 33

def words(ws:list[int]) -> bytes: return b"".join(w.to_bytes(16, "little") for w in ws)

# *** an interpreter of these words (offline check that a generated scalar program computes what its numpy model says) ***
import struct as _st
def _f(b:int) -> float: return _st.unpack("<f", _st.pack("<I", b & 0xffffffff))[0]
def _b(x:float) -> int:
  try: return _st.unpack("<I", _st.pack("<f", x))[0]
  except OverflowError: return 0x7f800000 if x > 0 else 0xff800000
def _s32(v:int) -> int: return v - (1 << 32) if v >> 31 else v

def run(ws:list[int], smem:bytearray, regs:list[int]|None=None, exp2=None, recip=None, f2i=None, max_steps:int=10**8,
        load_latency:int=0) -> dict:
  """execute scalar-core words (0x20 bundles, 0x23 loops, 0x22 branches; anything else is skipped) on `smem` (bytes, little endian
  32-bit words). The float slots compute in float32 (numpy) and round to nearest; exp2 / recip / f2i are the unknown units (defaults:
  correctly rounded 2^x and 1/x, round half to even). load_latency L > 0: a load issued in bundle t writes its register at the end of
  bundle t + L - 1 (over anything written there in between), so a program that reads it earlier sees the old value; the device's
  latency is not known (edgetpu_compiler first uses a loaded register 5 bundles later, or 2 in its pipelined push loop).
  -> {'regs', 'pred', 'pushed'}"""
  import numpy as np
  F32 = np.float32
  exp2 = exp2 or (lambda x: _b(float(F32(2.0 ** x)) if x > -160 else 0.0))
  recip = recip or (lambda x: _b(float(F32(1.0) / F32(x))))
  f2i = f2i or (lambda x: int(np.rint(x)))
  R, P, pushed = (regs or [0] * 32)[:], [False] * 8, []
  pending = []                                     # loads in flight: [bundles left, register, value]
  def retire():
    for q in pending: q[0] -= 1
    for q in [q for q in pending if q[0] <= 0]: R[q[1]] = q[2]
    pending[:] = [q for q in pending if q[0] > 0]
  def fop(o, a, b):
    x, y = F32(_f(a)), F32(_f(b))
    with np.errstate(all="ignore"):
      r = {0x12: x + y, 0x13: x - y, 0x14: x * y, 0x15: max(x, y), 0x16: min(x, y)}[o]
    return _b(float(r))
  def bundle(w):
    if load_latency:
      _bundle(w)
      retire()
    else: _bundle(w)
  def _bundle(w):
    d = WORD.decode([w])
    if d["gate"] and P[d["pred_reg"]] != bool(d["yes_pred"]): return
    srcs = dict(R=R[:])
    if d["enable_vector"] in (2, 3):              # load
      a = srcs["R"][d["imm_size"] & 0x1f]
      v = int.from_bytes(smem[4 * a: 4 * a + 4], "little")
      if load_latency: pending.append([load_latency, d["vs_reg_v1"], v])
      else: R[d["vs_reg_v1"]] = v
      if d["enable_vector"] == 3: R[d["imm_size"] & 0x1f] = (a + 1) & 0xffffffff
    if d["v_op"] == 4: smem[4 * srcs["R"][d["v_offset"]]: 4 * srcs["R"][d["v_offset"]] + 4] = srcs["R"][d["vs_reg"]].to_bytes(4, "little")
    elif d["v_op"] == 8: pushed.append(srcs["R"][d["vs_reg"]])
    if d["v_op_2"] in (RECIP, EXP2):
      x = _f(srcs["R"][d["vs_reg_w"]])
      R[d["unk_3"] & 0x1f] = (exp2 if d["v_op_2"] == EXP2 else recip)(x)
    so = d["s_op"]
    if not so: return
    o, x, a = so & 0x1f, d["s_x"], srcs["R"][d["s_y"]]
    b = d["imm_scalar"] if so & 0x20 else srcs["R"][d["imm_scalar"] & 0x1f]
    if o == MOV: R[x] = b
    elif o == I2F: R[x] = _b(float(F32(_s32(b))))
    elif o == F2I: R[x] = f2i(_f(b)) & 0xffffffff
    elif o in (0x12, 0x13, 0x14, 0x15, 0x16): R[x] = fop(o, a, b)
    elif o in (0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e):
      P[x & 7] = {9: a == b, 10: a != b, 11: _s32(a) > _s32(b), 12: _s32(a) < _s32(b), 13: a >= b, 14: _s32(a) >= _s32(b)}[o]
    else: R[x] = {ADD: a + b, SUB: a - b, AND: a & b, OR: a | b, XOR: a ^ b, SHL: a << (b & 31), SHR: a >> (b & 31)}[o] & 0xffffffff
  pc, steps = 0, 0
  while pc < len(ws):
    w = ws[pc]
    op = (w >> 6) & 0x3f
    steps += 1
    assert steps < max_steps
    if op == 0x23:
      count, body = (w >> 12) & 0xffff, w >> 28
      for _ in range(count):
        for bw in ws[pc + 1: pc + 1 + body]: bundle(bw)
      pc += 1 + body
      continue
    if op == 0x22:
      off = (w >> 17) & 0x7fff
      if load_latency: retire()
      pc += (off - 0x8000 if off & 0x4000 else off) if P[0] else 1
      continue
    if op == 0x20: bundle(w)
    pc += 1
  while pending: retire()
  return dict(regs=R, pred=P, pushed=pushed)

def dma_seqs(prog:bytes) -> list[tuple[int, int]]:
  """coral.isa.scalar.dma_seqs on the corrected instruction split: (tag, length) of every host DMA descriptor, in issue order"""
  from coral.isa import scalar as SC
  return SC.dma_seqs(b"".join(b"".join(w.to_bytes(16, "little") for w in ws) for _, ws in split(prog) if opcode(ws[0]) == 0x20))
