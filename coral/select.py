# Kernel selection for DEV=CORAL: find a quantized FULLY_CONNECTED in a tinygrad kernel AST and turn it into an Edge TPU
# program spec (coral.tpu.FCSpec). The AST is the scheduler's kernel, before codegen optimizes it:
#
#   out[m,n] = uint8(R(clip(float32(sum_k (x[m,k] - in_zp) * (w[n,k] - w_zp) + b[n]) * mult, lo - out_zp, hi - out_zp)) + out_zp)
#
# with R = round half to even (measured on the device). Every node of the kernel is checked, and only rewrites that are
# bit-exact in int32/float32 are accepted; anything else raises NoMatch and the kernel stays a clang kernel. Accepted:
#   - layout: x[M,K] and w[N,K] row-major (w read through the transposed index n*K+k), b[N] int32 or no bias, out[M,N];
#     the rows may be any number of output ranges that flatten row-major (x[B,T,K] @ w.T); M = 1 and N = 1 drop a range
#   - operands in any order (commuted MUL/ADD/MAX/CMPLT, bias first or last, x and w swapped in the product)
#   - zero points folded: x.cast(int32) + (-zp), x.cast(int32) - zp, or the bare cast when zp = 0; widening cast chains
#   - mult as a float32 constant (a missing MUL is mult = 1.0)
#   - clamps as MAX, as the CMPLT/WHERE forms of clip/clamp/maximum/minimum/relu, or tinygrad's float minimum -max(-y, -c),
#     before the rounding (any float bound b acts as R(b), since R is monotone) and after it (integer bounds, also after
#     adding out_zp, also in an int domain after a cast); several clamps compose into one [lo, hi]
#   - R as tinygrad lowers Tensor.round(): ((y > 0) == (trunc(y) even)) ? ceil(y - 0.5) : floor(y + 0.5), with the condition
#     in any boolean form and the branches in either order (its truth table is checked exhaustively, see round_half_even).
#     Round half away from zero, half up (floor(y + 0.5)), half down, half to odd and trunc are not R and stay clang kernels.
# The forms are UPats (a src list matches both operand orders, a repeated name the same node); the values of the constants,
# the index arithmetic and the clamp bounds are plain code.
#
# The same kernel over a convolution (coral.qops.qconv2d) is a quantized CONV_2D (ConvSpec, see match_conv below). Entry points:
# select(ast) gives the FCMatch or ConvMatch of a kernel (None: clang), why_not(ast) why it gives None; select_fc and
# select_conv look for one kind. select_glue(ast) recognizes the uint8 copies and max pools between two layers (CopyMatch,
# PoolMatch) for coral/chain.py's fusion; alone such a kernel stays clang.
from __future__ import annotations
import itertools, json, math, operator, numpy as np
from dataclasses import dataclass, asdict
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops, AxisType
from tinygrad.dtype import dtypes, DType, AddrSpace
import coral.tpu as tpu
from coral.tpu import FCSpec, fc_reference

class NoMatch(Exception): pass
def need(cond, why:str):
  if not cond: raise NoMatch(why)

F32, I32, U8 = dtypes.float32, dtypes.int32, dtypes.uint8

@dataclass(frozen=True)
class FCMatch:
  spec: FCSpec
  out: UOp                                 # the kernel's PARAMs in Edge TPU argument order (bias None: no bias)
  x: UOp
  w: UOp
  b: UOp|None
  @property
  def params(self) -> tuple[UOp, ...]: return (self.out, self.x, self.w) + ((self.b,) if self.b is not None else ())

# ***** constants, buffers and index arithmetic *****

def cval(u:UOp, dt:DType):
  """the value of u as a constant operand of an op computing in dt (a weak const commits to dt there), else None"""
  if u.op is Ops.CAST and u.dtype == dt and u.src[0].op is Ops.CONST and not u.src[0].is_invalid:
    v = u.src[0].arg
    if dtypes.is_float(dt): return dt.const(v)
    return int(v) if dtypes.is_int(dt) and isinstance(v, int) and not isinstance(v, bool) else None
  if u.op is not Ops.CONST or u.is_invalid or isinstance(u.arg, bool): return None
  if u.dtype not in dtypes.weaks: return u.arg if u.dtype == dt else None
  if dtypes.is_float(dt): return dt.const(u.arg)                                  # weakint/weakfloat -> float32 rounding
  return int(u.arg) if dtypes.is_int(dt) and u.dtype is dtypes.weakint else None  # a weakfloat never commits to an int

def kval(dt:DType, x:UOp, c:UOp):
  """the value of c, the constant operand of an op in dt whose other operand x is not a constant, else None"""
  return v if (v:=cval(c, dt)) is not None and cval(x, dt) is None else None
def addend(u:UOp, x:UOp, c:UOp):
  """d for u = x + d with the constant c: c for x + c, -c for x - c (None for c - x)"""
  if u.op is Ops.SUB and u.src[0] is not x: return None
  return None if (k:=kval(u.dtype, x, c)) is None else k if u.op is Ops.ADD else -k
def buffer(p:UOp) -> UOp:
  need(p.arg.addrspace == AddrSpace.GLOBAL and p.arg.image is None and p.arg.size is not None, "not a global buffer")
  need(not isinstance(p.arg.device, tuple), "multi-device buffer")
  return p

def linear(u:UOp) -> tuple[dict[UOp, int], int]:
  """an index expression as sum(coef * RANGE) + offset; anything else (gates, ...) is no match. A floor div or mod by a constant
  d is linear where it is exactly: sum(k*r) + c = d*(sum(q*r) + q0) + sum(e*r) + e0 with k = q*d + e, 0 <= e < d, and the
  remainder part below d on the whole domain (tinygrad's windows, _pool, leave such a mod in some kernels)"""
  if u.op is Ops.RANGE: return {u: 1}, 0
  if u.op is Ops.CONST and type(u.arg) is int: return {}, u.arg
  if u.op in (Ops.ADD, Ops.MUL) and len(u.src) == 2:
    (a, ca), (b, cb) = linear(u.src[0]), linear(u.src[1])
    if u.op is Ops.ADD:
      out = dict(a)
      for r, k in b.items(): out[r] = out.get(r, 0) + k
      return {r: k for r, k in out.items() if k != 0}, ca + cb
    need(not a or not b, "index: product of ranges")
    return {r: k * (cb if a else ca) for r, k in (a or b).items() if k * (cb if a else ca) != 0}, ca * cb
  if u.op in (Ops.FLOORDIV, Ops.FLOORMOD) and u.src[1].op is Ops.CONST and type(d:=u.src[1].arg) is int and d > 0:
    a, c = linear(u.src[0])
    need(c % d + sum(k % d * (rsize(r) - 1) for r, k in a.items()) < d, f"index: {u.op} is not linear on the ranges")
    if u.op is Ops.FLOORDIV: return {r: k // d for r, k in a.items() if k // d != 0}, c // d
    return {r: k % d for r, k in a.items() if k % d != 0}, c % d
  raise NoMatch(f"index: {u.op}")

def rsize(r:UOp) -> int:
  """the size of a plain loop range (the scheduler's WEAK ranges; DEVICE ranges of sharded kernels are not loops)"""
  need(r.op is Ops.RANGE and r.arg[0] in (AxisType.WEAK, AxisType.LOOP), f"range {r.arg}")
  need(len(r.src) == 1 and r.src[0].op is Ops.CONST and type(r.src[0].arg) is int and r.src[0].arg > 0, "symbolic range")
  return r.src[0].arg

# ***** the kernel and the int32 accumulator: sum_k (x - in_zp) * (w - w_zp) + b *****

X, Y = UPat.var("x"), UPat.var("y")
def cst(name:str) -> UPat: return UPat.cvar().or_casted(name)   # a constant operand, CONST or CAST(CONST); its value is cval's
def ld(dt:DType, name:str) -> UPat: return UPat(Ops.INDEX, dt, (UPat(Ops.PARAM, dt, name=name), UPat.var(name+"_idx")))

STORE = UPat(Ops.STORE, src=(ld(U8, "out"), UPat(Ops.CAST, U8, name="val")))
KERNEL = UPat.sink(UPat.any(STORE, UPat(Ops.END, src=(STORE,), allow_any_len=True, name="end")))
SUM = UPat(Ops.REDUCE, arg=(Ops.ADD, 0), src=(UPat.var("a") * UPat.var("b"), UPat(Ops.RANGE, name="k")))
ACC = UPat.any(SUM, SUM + ld(I32, "bias"))                      # one match for each order of the product's operands
U8_LOAD = ld(U8, "p")

# an operand of the product, (v - zp) in int32 with v a cast of a uint8 load: v + (-zp), v - zp, or v when zp = 0: (v, zp)
pm_operand = PatternMatcher([
  (UPat(Ops.CAST, I32, name="v"), lambda v: (v, 0)),
  (UPat((Ops.ADD, Ops.SUB), I32, [UPat(Ops.CAST, name="v"), cst("c")], name="u"), lambda u,v,c: None if (d:=addend(u, v, c)) is None else (v, -d)),
])

def operand(u:UOp) -> tuple[UOp, UOp, int]:
  """(v - zp) in int32 with v a uint8 load: (param, index, zp)"""
  need((m:=pm_operand.rewrite(u)) is not None, f"operand {u.op} in {u.dtype} is not x.cast(int32) - zp")
  v, zp = m
  need(0 <= zp <= 255, f"zero point {zp}")
  while v.op is Ops.CAST:   # uint8 -> int32, possibly through other ints that hold [0, 255]
    need(dtypes.is_int(v.dtype) and (v.dtype.itemsize >= 2 or v.dtype == U8), f"narrowing cast to {v.dtype}")
    v = v.src[0]
  need(len(ms:=U8_LOAD.match(v, {})) == 1, f"not a load of uint8: {v.op}")
  return buffer(ms[0]["p"]), ms[0]["p_idx"], zp

# ***** the float32 epilogue *****

# float32(acc) * mult, or float32(acc) for mult = 1.0: (acc, mult)
pm_rescale = PatternMatcher([
  (UPat(Ops.CAST, F32, (UPat.var("acc", I32),)), lambda acc: (acc, 1.0)),
  (UPat(Ops.MUL, F32, [UPat(Ops.CAST, src=(UPat.var("acc", I32),), name="f"), cst("c")]),
   lambda f,acc,c: None if (m:=kval(F32, f, c)) is None or not math.isfinite(m) else (acc, float(m))),
])

# x with v == -x, exactly (ctx: the dtype): NEG(x) and x * -1 are -x, z * c is -(z * -c): tinygrad folds a negation into a
# constant factor, and fl(z * -c) == -fl(z * c). That builds z * -c, which UOp hash-consing makes the graph's node if it exists.
pm_neg = PatternMatcher([(UPat(Ops.NEG, src=(X,)), lambda x: x),
  (X * cst("c"), lambda ctx,x,c: None if (k:=kval(ctx, x, c)) is None else x if k == -1 else UOp(Ops.MUL, (x, UOp.const(-k, ctx))))])

# clamps of x at a constant: (x, (kind, dtype, c)) with kind "lo" for max(x, c) and "hi" for min(x, c)
NUMS, FLOATS = dtypes.floats + dtypes.ints + dtypes.weaks, dtypes.floats + (dtypes.weakfloat,)
def where_clamp(u:UOp, lt:UOp, x:UOp, c:UOp, e:UOp, v:UOp|None=None, d:UOp|None=None):
  """u = (x < c or c < x) ? x : e (or ? e : x) with e == c: min(x, c) if it takes x when x < c, else max(x, c). In ints the
  value may be v = x + d with e == c + d, a clamp of v at c + d: tinygrad moves the add out of the comparison. Exact: the int
  domain only exists after a cast of a bounded value (see epilogue), so x + d does not overflow."""
  dt, kind = u.dtype, "hi" if (u.src[2] is e) == (lt.src[0] is x) else "lo"
  if x.dtype != dt or (k:=kval(dt, x, c)) is None: return None
  if v is None: return (x, (kind, dt, k)) if cval(e, dt) == k else None
  return (v, (kind, dt, k+kd)) if dtypes.is_int(dt) and (kd:=kval(dt, x, d)) is not None and cval(e, dt) == k+kd else None

def neg_max(u:UOp, n:UOp, c:UOp, m:UOp|None=None):
  """u = -max(n, c) = min(-n, -c), tinygrad's float minimum (negated as NEG, or as * m with m == -1)"""
  if (m is not None and cval(m, u.dtype) != -1) or (k:=kval(u.dtype, n, c)) is None or (x:=pm_neg.rewrite(n, u.dtype)) is None: return None
  return x, ("hi", u.dtype, -k)

LT = UPat(Ops.CMPLT, src=[X, cst("c")], name="lt")             # x < c or c < x
VAL = UPat.any(X, (X + cst("d")).named("v"))                   # the clamped value: x, or x + d (ints, see where_clamp)
MAX_N = UPat(Ops.MAX, src=[UPat.var("n"), cst("c")])           # max(n, c), n = -x
clamps = [
  (UPat(Ops.MAX, NUMS, [X, cst("c")], name="u"), lambda u,x,c: None if (k:=kval(u.dtype, x, c)) is None else (x, ("lo", u.dtype, k))),
  (UPat(Ops.WHERE, NUMS, (LT, VAL, cst("e")), name="u"), where_clamp),   # x < c ? x : c = min, c < x ? x : c = max
  (UPat(Ops.WHERE, NUMS, (LT, cst("e"), VAL), name="u"), where_clamp),   # x < c ? c : x = max, c < x ? c : x = min
  (UPat(Ops.NEG, FLOATS, (MAX_N,), name="u"), neg_max),                  # -max(-x, -c) = min
  (UPat(Ops.MUL, FLOATS, [MAX_N, cst("m")], name="u"), neg_max),         # max(-x, -c) * -1 = min
]
pm_clamp = PatternMatcher(clamps)
# one step of the integer valued tail after the rounding, outermost first (a cast, + c or - c, a clamp): (inner, step)
pm_post = PatternMatcher([
  (UPat(Ops.CAST, src=(X,), name="u"), lambda u,x: (x, ("cast", x.dtype, u.dtype))),
  (UPat((Ops.ADD, Ops.SUB), src=[X, cst("c")], name="u"), lambda u,x,c: None if (d:=addend(u, x, c)) is None else (x, ("add", u.dtype, d))),
]+clamps)

# R's condition on one class of y (ctx: y, its sign, trunc(y) even): bool constants and connectives over two kinds of atoms
LOGIC = {Ops.CMPNE: operator.ne, Ops.XOR: operator.ne, Ops.CMPEQ: operator.eq, Ops.AND: operator.and_, Ops.OR: operator.or_}
def cond_value(n:UOp, ctx:tuple[UOp, int, bool]) -> bool:
  need((v:=pm_cond.rewrite(n, ctx)) is not None, f"condition: {n.op}")
  return v

def sign_atom(ctx:tuple[UOp, int, bool], n:UOp, x:UOp, c:UOp) -> bool|None:
  """y < 0, 0 < y, y != 0 or y == 0"""
  y, s, _ = ctx
  if x is not y or kval(F32, x, c) != 0: return None
  return (s < 0 if n.src[0] is y else s > 0) if n.op is Ops.CMPLT else (s != 0) == (n.op is Ops.CMPNE)

def parity_atom(ctx:tuple[UOp, int, bool], n:UOp, y:UOp, ty:UOp, half:UOp, h:UOp) -> bool|None:
  """trunc(h) != h (or ==) with h = trunc(y) * 0.5: h is an integer iff trunc(y) is even (exact for every float32)"""
  return None if y is not ctx[0] or kval(F32, ty, half) != 0.5 else ctx[2] == (n.op is Ops.CMPEQ)

H = (Y.trunc().named("ty") * cst("half")).named("h")
pm_cond = PatternMatcher([
  (UPat(Ops.CONST, dtypes.bool, name="b"), lambda b: b.arg if isinstance(b.arg, bool) else None),
  (UPat(tuple(LOGIC), src=(UPat.var("a", dtypes.bool), UPat.var("b", dtypes.bool)), name="n"),
   lambda ctx,n,a,b: LOGIC[n.op](cond_value(a, ctx), cond_value(b, ctx))),
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ), src=[X, cst("c")], name="n"), sign_atom),
  (UPat((Ops.CMPNE, Ops.CMPEQ), src=[H.trunc(), H], name="n"), parity_atom),
])

def round_half_even(r:UOp, cond:UOp, y:UOp, floor:UOp, up:UOp, up_c:UOp, down:UOp, down_c:UOp, floor_1:UOp, ceil_1:UOp, **_) -> UOp|None:
  """r = R(y), round half to even in float32, as tinygrad lowers Tensor.round():
       ((y > 0) == (trunc(y) is even)) ? ceil(y - 0.5) : floor(y + 0.5)
  With exact y +- 0.5 (|y| < 2**22) the floor branch is R on y > 0 with trunc(y) odd and on y < 0 with trunc(y) even, the
  ceil branch on the other two classes, and both are 0 at y == 0. The condition's atoms only depend on that class, so its
  truth table on the 5 classes is checked exhaustively, whatever boolean form tinygrad gives it. Returns y."""
  if y.dtype != F32 or addend(up, y, up_c) != 0.5 or addend(down, y, down_c) != -0.5 or cval(floor_1, F32) != -1 or cval(ceil_1, F32) != 1:
    return None
  for s, even in ((1, True), (1, False), (-1, True), (-1, False), (0, True)):
    floor_picked = cond_value(cond, (y, s, even)) == (r.src[1] is floor)
    need(s == 0 or floor_picked == ((s > 0) != even), "not round half to even")
  return y

# R(y): cond ? floor(y + 0.5) : ceil(y - 0.5), or the branches swapped, with floor and ceil as tinygrad lowers them
def shifted(name:str) -> UPat: return UPat((Ops.ADD, Ops.SUB), src=[Y, cst(name+"_c")], name=name)   # y + d, or y - (-d)
def floor_of(t:UPat) -> UPat:
  tr = t.trunc().named("floor_trunc")
  return (t < tr).where(tr + cst("floor_1"), tr).named("floor")
def ceil_of(t:UPat) -> UPat:
  tr = t.trunc().named("ceil_trunc")
  return (tr < t).where(tr + cst("ceil_1"), tr).named("ceil")
FLOOR, CEIL = floor_of(shifted("up")), ceil_of(shifted("down"))
pm_round = PatternMatcher([(UPat(Ops.WHERE, F32, (UPat.var("cond"), a, b), name="r"), round_half_even) for a, b in ((FLOOR, CEIL), (CEIL, FLOOR))],
                          compiled=False)   # too many alternatives for tinygrad's UPat compiler

def R(v:float) -> float:
  """the rounding as the TPU (and coral.tpu.fc_reference) does it: half to even, in float32"""
  return float(np.round(np.float32(v)))

def epilogue(v:UOp) -> tuple[UOp, float, int, int, int]:
  """uint8 out = requant(acc), v the stored cast to uint8: (acc, mult, lo, hi, out_zp)"""
  post: list[tuple] = []                                       # after the rounding, outermost first
  while (st:=pm_post.rewrite(v)) is not None:
    v, step = st
    post.append(step)
  need((y:=pm_round.rewrite(v)) is not None, "not round half to even")
  pre = []                                                     # clamps before the rounding, outermost first
  while y.dtype == F32 and (cl:=pm_clamp.rewrite(y)) is not None:
    y, (kind, _, c) = cl
    pre.append((kind, c))
  lo_b, hi_b = -math.inf, math.inf
  for kind, c in reversed(pre):
    need(abs(c) < 2**22, f"clamp at {c}")                     # where the lowered Tensor.round() is exact
    if kind == "lo":
      need(c <= hi_b, "empty clamp")
      lo_b = max(lo_b, c)
    else:
      need(c >= lo_b, "empty clamp")
      hi_b = min(hi_b, c)
  # R is monotone: R(clip(y, a, b)) == clip(R(y), R(a), R(b)). Unclamped, a huge |y| saturates at the final clamp either way
  L, H, Z, dt = (R(lo_b) if lo_b > -math.inf else -math.inf), (R(hi_b) if hi_b < math.inf else math.inf), 0, F32
  for kind, odt, c in reversed(post):                          # value = clip(R(y), L, H) + Z, in dtype dt
    if kind == "cast":
      need(odt == dt and dtypes.is_int(c) and math.isfinite(L) and math.isfinite(H), f"cast {odt} -> {c}")
      need(c.min <= L + Z and H + Z <= c.max, f"cast to {c} out of range")
      dt = c
      continue
    need(odt == dt and math.isfinite(c) and float(c) == int(c) and abs(c) < 2**16, f"{kind} {c} after the rounding")  # small ints: exact
    if kind == "add": Z += int(c)
    elif kind == "lo":
      need(c - Z <= H, "empty clamp")
      L = max(L, int(c) - Z)
    else:
      need(c - Z >= L, "empty clamp")
      H = min(H, int(c) - Z)
  need(dt == U8 and math.isfinite(L) and math.isfinite(H), "output is not clamped")
  out_zp, lo, hi = Z, int(L) + Z, int(H) + Z
  need(0 <= out_zp <= 255 and 0 <= lo <= hi <= 255, f"out_zp {out_zp}, clamp [{lo}, {hi}]")
  need((am:=pm_rescale.rewrite(y)) is not None, "requant input is not float32(int32) times a constant")
  return am[0], am[1], lo, hi, out_zp

# ***** the kernel *****

def match_fc(ast:UOp) -> FCMatch:
  need(ks:=KERNEL.match(ast, {}), "not a kernel storing a cast to uint8")
  out, out_idx, ends = buffer(ks[0]["out"]), ks[0]["out_idx"], ks[0]["end"].src[1:] if "end" in ks[0] else ()
  acc, mult, lo, hi, out_zp = epilogue(ks[0]["val"])
  need(accs:=ACC.match(acc, {}), "accumulator is not sum((x - x_zp) * (w - w_zp)) (+ bias)")
  K = rsize(k:=accs[0]["k"])
  need(K * 255 * 255 < 2**31, "K too large for an int32 accumulator")
  out_lin, out_off = linear(out_idx)
  need(out_off == 0 and set(out_lin) == set(ends) and k not in out_lin and len(set(ends)) == len(ends), "output index")
  err: NoMatch|None = None                                     # the first operand order's reason is the informative one
  for m in accs:
    try:
      (x, x_idx, in_zp), (w, w_idx, w_zp) = operand(m["a"]), operand(m["b"])
      # w[n, k]: the transposed index n*K + k (n absent when N == 1)
      w_lin, w_off = linear(w_idx)
      need(w_off == 0 and w_lin.get(k) == 1 and len(w_lin) <= 2, "w index")
      n = next((r for r in w_lin if r is not k), None)
      N = 1 if n is None else rsize(n)
      need(n is None or (w_lin[n] == K and out_lin.get(n) == 1), "w index")
      # x[m, k], out[m, n]: the m ranges step through the rows with the same row strides, flattened row-major
      rows = [r for r in ends if r is not n]
      x_lin, x_off = linear(x_idx)
      need(x_off == 0 and x_lin.get(k) == 1 and set(x_lin) == set(rows) | {k}, "x index")
      strides = []
      for r in rows:
        need(x_lin[r] % K == 0 and out_lin[r] % N == 0 and x_lin[r] // K == out_lin[r] // N, "x/out row strides")
        strides.append((x_lin[r] // K, rsize(r)))
      M = 1
      for s, sz in sorted(strides):
        need(s == M, "rows are not row-major")
        M *= sz
      b = buffer(m["bias"]) if "bias" in m else None
      need(b is None or linear(m["bias_idx"]) == ({} if n is None else {n: 1}, 0), "bias index")
      need(out not in (x, w, b), "output aliases an input")
      need(x.arg.size >= M*K and w.arg.size >= N*K and out.arg.size >= M*N and (b is None or b.arg.size >= N), "buffer sizes")
      need({u for u in ast.toposort() if u.op is Ops.PARAM} == {out, x, w} | ({b} if b is not None else set()), "other params")
      return FCMatch(FCSpec(M, N, K, in_zp, w_zp, out_zp, mult, lo, hi), out, x, w, b)
    except NoMatch as e: err = err or e
  raise err or NoMatch("no operand order matches")

def select_fc(ast:UOp) -> FCMatch|None:
  """the Edge TPU FC this kernel computes, or None (then it stays a clang kernel)"""
  try: return match_fc(ast)
  except NoMatch: return None

# ***** CONV_2D: the same requant over a convolution *****
#
#   out[n,o,oy,ox] = requant(sum_{c,ky,kx} (xp[n, c, oy*s + ky, ox*s + kx] - in_zp) * (w[o,c,ky,kx] - w_zp) + b[o])
#
# with xp = x padded by p on every side with in_zp, so the padding contributes nothing (coral.qops.qconv2d). In the kernel the
# padding is a mask: x is read at n*Cin*H*W + c*H*W + (oy*s + ky - p)*W + ox*s + kx - p through an INDEX gated by it, and
# WHERE(mask, ., fill) on the way to the product gives the padding its value. Accepted, exactly:
#   - layout: x[N,Cin,H,W], w[Cout,Cin,kh,kw] and out[N,Cout,OH,OW] contiguous, b[Cout] int32 or no bias, one stride and one
#     padding for both axes. A size 1 axis has no range, so the ranges are put on the axes by their order in the out and the w
#     index, every way; s, p, H and W are read off the x index (where it has no term for one: the output size, or the size of
#     the x buffer), and a candidate is the conv only if every index is exactly the conv's
#   - the masks (the INDEX's gate, the WHEREs' conditions) in any boolean form: their truth table on every (oy, ky) and every
#     (ox, kx) must be the image, 0 <= oy*s + ky - p < H and 0 <= ox*s + kx - p < W
#   - the x operand in any integer form whose value is x - in_zp where its masks hold and 0 where they don't, for all 256 x
#   - the w operand and the epilogue as for the FC
# No match: no reduce (Cin*kh*kw = 1), groups, dilation, other layouts (NHWC, a transposed w), a padding of another value. A
# padding no smaller than the kernel on an axis (a 1x1 conv with padding: whole output rows of padding), or a padded image of
# height or width 1, may also stay clang: tinygrad folds what the mask implies into the index there.

@dataclass(frozen=True)
class ConvSpec:
  N: int                                                   # x[N,Cin,H,W] uint8, NCHW contiguous
  Cin: int
  H: int
  W: int
  Cout: int                                                # w[Cout,Cin,kh,kw] uint8 contiguous, b[Cout] int32 (zeros without bias)
  kh: int
  kw: int
  stride: int                                              # the same on both axes; the padding reads as in_zp
  padding: int
  in_zp: int                                               # the requant, as FCSpec's
  w_zp: int
  out_zp: int
  mult: float
  lo: int = 0
  hi: int = 255
  @property
  def OH(self) -> int: return (self.H + 2 * self.padding - self.kh) // self.stride + 1
  @property
  def OW(self) -> int: return (self.W + 2 * self.padding - self.kw) // self.stride + 1
  @property
  def tflite_padding(self) -> str|None:
    """the padding as TFLite names it: VALID, SAME (OH = ceil(H / stride), pads split top/bottom as total // 2 first), or None
    where SAME splits it unevenly (e.g. stride 2 on an even size: no row on top, one at the bottom)"""
    if self.padding == 0: return "VALID"
    same = all(-(-n // self.stride) == o and max((o - 1) * self.stride + k - n, 0) // 2 == self.padding
               for n, o, k in ((self.H, self.OH, self.kh), (self.W, self.OW, self.kw)))
    return "SAME" if same else None
  def fc(self) -> FCSpec:
    """the conv as im2col + FC: a row of Cin*kh*kw window values per output position, in w.reshape(Cout, -1)'s (c, ky, kx) order"""
    return FCSpec(self.N * self.OH * self.OW, self.Cout, self.Cin * self.kh * self.kw,
                  self.in_zp, self.w_zp, self.out_zp, self.mult, self.lo, self.hi)
  def quant(self): return self.fc().quant()
  def dumps(self) -> str: return json.dumps({"conv2d": asdict(self)})
  @staticmethod
  def loads(s:str|bytes) -> ConvSpec: return ConvSpec(**json.loads(s)["conv2d"])

def im2col(spec:ConvSpec, x:np.ndarray) -> np.ndarray:
  """the windows of x[N,Cin,H,W] padded with in_zp: rows[N*OH*OW, Cin*kh*kw], the x of spec.fc()"""
  s, p = spec.stride, spec.padding
  xp = np.pad(x.reshape(spec.N, spec.Cin, spec.H, spec.W), ((0, 0), (0, 0), (p, p), (p, p)), constant_values=spec.in_zp)
  win = np.lib.stride_tricks.sliding_window_view(xp, (spec.kh, spec.kw), axis=(2, 3))[:, :, ::s, ::s]   # [N, Cin, OH, OW, kh, kw]
  return win.transpose(0, 2, 3, 1, 4, 5).reshape(spec.N * spec.OH * spec.OW, spec.Cin * spec.kh * spec.kw)

def _nchw(spec:ConvSpec, y:np.ndarray) -> np.ndarray: return y.reshape(spec.N, spec.OH, spec.OW, spec.Cout).transpose(0, 3, 1, 2)
def _bias(spec:ConvSpec, b) -> np.ndarray: return np.zeros(spec.Cout, np.int32) if b is None else np.asarray(b, np.int32).reshape(spec.Cout)

def conv_reference(spec:ConvSpec, x:np.ndarray, w:np.ndarray, b:np.ndarray|None) -> np.ndarray:
  """what the Edge TPU computes, out[N,Cout,OH,OW]: im2col + coral.tpu.fc_reference"""
  return _nchw(spec, fc_reference(spec.fc(), im2col(spec, x), w.reshape(spec.Cout, -1), _bias(spec, b)))

def run_conv(spec:ConvSpec, x:np.ndarray, w:np.ndarray, b:np.ndarray|None, wkey=None) -> np.ndarray:
  """a selected conv as im2col on the host and the Edge TPU matmul coral.tpu.run_fc (with MOCKCORAL=1 coral.tpu.fc_reference):
  coral.tpu.run_conv's fallback where its conv2d program doesn't cover the shape. x, w, b: the kernel's buffers (any shape, b
  None without bias); returns out[N,Cout,OH,OW]"""
  return _nchw(spec, tpu.run_fc(spec.fc(), im2col(spec, x), np.ascontiguousarray(w).reshape(spec.Cout, -1), _bias(spec, b), wkey))

@dataclass(frozen=True)
class ConvMatch:
  spec: ConvSpec
  out: UOp                                 # the kernel's PARAMs in Edge TPU argument order (bias None: no bias)
  x: UOp
  w: UOp
  b: UOp|None
  @property
  def params(self) -> tuple[UOp, ...]: return (self.out, self.x, self.w) + ((self.b,) if self.b is not None else ())

RED = UPat(Ops.REDUCE, arg=(Ops.ADD, 0), src=(UPat.var("a") * UPat.var("b"),), allow_any_len=True, name="red")   # over its ranges
CONV_ACC = UPat.any(RED, RED + ld(I32, "bias"))

# ***** exact evaluation (the conv's masks and x operand, the glue's uint8 maps) *****
#
# Integers and booleans are exact; an overflow (which C wraps or leaves undefined) is no match. A float is float32, and clang
# rounds every op on its own or fuses a multiply into the add after it (-O2 contracts, -ffast-math is off: nothing else
# changes), so a float is an interval [lo, hi] that holds its exact value and every such evaluation: an op's result comes
# from its operands' ends in float64, widened by 2^-23 relative (one float32 rounding is 2^-24), unless its operands are
# float32 values and so is its exact result (then every evaluation gives that value). A comparison is decided when the
# intervals are apart (a boolean interval: (False, True) is undecided), a WHERE on an undecided condition holds both branches,
# and a cast to an integer truncates both ends. A decided result is what clang's code computes.
EPS32, TINY32 = 2.0 ** -23, 2.0 ** -149     # widening: relative, and absolute for subnormals
def _is_f32(x:np.ndarray) -> np.ndarray: return x.astype(np.float32).astype(np.float64) == x

def _arith(op:Ops, a, b, fl:bool):
  (alo, ahi), (blo, bhi) = a, b
  if op is Ops.ADD: lo, hi = alo + blo, ahi + bhi
  elif op is Ops.SUB: lo, hi = alo - bhi, ahi - blo
  else:
    need(op is Ops.MUL or bool(np.all((blo > 0) | (bhi < 0))), "division by an interval that holds 0")
    ends = [np.multiply(x, y) if op is Ops.MUL else np.true_divide(x, y) for x in (alo, ahi) for y in (blo, bhi)]
    lo, hi = np.minimum.reduce(np.broadcast_arrays(*ends)), np.maximum.reduce(np.broadcast_arrays(*ends))
  if not fl: return lo, hi
  single = (alo == ahi) & (blo == bhi)
  if op is Ops.MUL: exact = single                                  # float32 * float32 fits a float64
  elif op is Ops.FDIV: exact = single & (lo * blo == alo)
  else:                                                             # TwoSum: the float64 sum is exact iff its error term is 0
    b_ = blo if op is Ops.ADD else -blo
    t = lo - alo
    exact = single & ((alo - (lo - t)) + (b_ - t) == 0)
  exact = exact & _is_f32(lo)
  return np.where(exact, lo, lo - (np.abs(lo) * EPS32 + TINY32)), np.where(exact, hi, hi + (np.abs(hi) * EPS32 + TINY32))

def _cmp(op:Ops, a, b):
  """(definitely, possibly) a < b, a == b or a != b"""
  (alo, ahi), (blo, bhi) = a, b
  if op is Ops.CMPLT: return ahi < blo, alo < bhi
  eq = (alo == ahi) & (blo == bhi) & (alo == blo), (alo <= bhi) & (blo <= ahi)
  return eq if op is Ops.CMPEQ else (~eq[1], ~eq[0])

def bounds(u:UOp, env:dict[UOp, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
  """the interval (lo, hi) of u's value, its leaves (ranges, loads, masks, a reduce) given in env as intervals; numpy broadcasts"""
  memo = dict(env)
  def ev(v:UOp) -> tuple[np.ndarray, np.ndarray]:
    if (r:=memo.get(v)) is not None: return r
    dt, fl = v.dtype, dtypes.is_float(v.dtype)
    need(dt == dtypes.bool or dtypes.is_int(dt) or dt in (dtypes.float32, dtypes.weakfloat), f"{v.op} in {dt}")
    if v.op is Ops.CONST:
      need(not v.is_invalid and isinstance(v.arg, (int, float)), f"constant {v.arg!r}")
      c = np.array(float(np.float32(v.arg)) if fl else v.arg)     # a weak float takes the float32 of the op it is in
      r = (c, c)
    elif v.op is Ops.CAST:
      (lo, hi), src = ev(v.src[0]), v.src[0].dtype
      if dt == dtypes.bool: r = ((lo > 0) | (hi < 0), (lo != 0) | (hi != 0))
      elif fl:                                                      # an int converts exactly up to 2^24, a float stays float32
        need(dtypes.is_float(src) or bool(np.all(np.abs(lo) <= 2**24) & np.all(np.abs(hi) <= 2**24)), f"cast {src} -> {dt}")
        r = (lo.astype(np.float64), hi.astype(np.float64))
      elif dtypes.is_float(src):                                    # C truncates; out of range is undefined
        lo, hi = np.trunc(lo), np.trunc(hi)
        need(bool(np.all(lo >= dt.min) & np.all(hi <= dt.max)), f"cast to {dt} out of range")
        r = (lo.astype(np.int64), hi.astype(np.int64))
      else: r = (lo.astype(np.int64), hi.astype(np.int64))
    elif v.op is Ops.WHERE and v.src[0].op is Ops.CMPLT and v.src[1] is not v.src[2] and {v.src[1], v.src[2]} == set(v.src[0].src):
      f = np.minimum if v.src[1] is v.src[0].src[0] else np.maximum           # a < b ? a : b is min(a, b), a < b ? b : a max
      r = tuple(f(x, y) for x, y in zip(ev(v.src[1]), ev(v.src[2])))           # (exactly: no hull of both branches)
    elif v.op is Ops.WHERE:
      (clo, chi), (alo, ahi), (blo, bhi) = (ev(s) for s in v.src)
      r = (np.where(clo, alo, np.where(chi, np.minimum(alo, blo), blo)), np.where(clo, ahi, np.where(chi, np.maximum(ahi, bhi), bhi)))
    elif v.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): r = _cmp(v.op, ev(v.src[0]), ev(v.src[1]))
    elif v.op is Ops.XOR and dt == dtypes.bool: r = _cmp(Ops.CMPNE, ev(v.src[0]), ev(v.src[1]))
    elif v.op in (Ops.AND, Ops.OR) and dt == dtypes.bool:
      (alo, ahi), (blo, bhi), f = ev(v.src[0]), ev(v.src[1]), np.logical_and if v.op is Ops.AND else np.logical_or
      r = (f(alo, blo), f(ahi, bhi))
    elif v.op is Ops.NEG: r = tuple(-x for x in ev(v.src[0])[::-1])
    elif v.op is Ops.MAX: r = tuple(np.maximum(x, y) for x, y in zip(ev(v.src[0]), ev(v.src[1])))
    elif v.op is Ops.TRUNC and fl: r = tuple(np.trunc(x) for x in ev(v.src[0]))
    elif v.op in (Ops.ADD, Ops.SUB, Ops.MUL) or (v.op is Ops.FDIV and fl):
      a, b = ev(v.src[0]), ev(v.src[1])
      need(not fl or all(dtypes.is_float(s.dtype) or bool(np.all(np.abs(x) <= 2**24)) for s, ab in zip(v.src, (a, b)) for x in ab),
           "int too big for float32")
      r = _arith(v.op, a, b, fl)
    elif v.op in (Ops.FLOORDIV, Ops.FLOORMOD, Ops.AND, Ops.OR, Ops.XOR):   # ints: exact operands only
      (alo, ahi), (blo, bhi) = ev(v.src[0]), ev(v.src[1])
      need(bool(np.all(alo == ahi) & np.all(blo == bhi)) and (v.op not in (Ops.FLOORDIV, Ops.FLOORMOD) or bool(np.all(blo != 0))), f"{v.op}")
      x = {Ops.FLOORDIV: np.floor_divide, Ops.FLOORMOD: np.mod, Ops.AND: np.bitwise_and, Ops.OR: np.bitwise_or,
           Ops.XOR: np.bitwise_xor}[v.op](alo, blo)
      r = (x, x)
    else: raise NoMatch(f"{v.op}")
    r = tuple(np.asarray(x) for x in r)
    if fl: need(bool(np.all(np.isfinite(r[0])) & np.all(np.isfinite(r[1]))), "not finite")
    elif dt != dtypes.bool and dt not in dtypes.weaks: need(dt.min <= int(r[0].min()) and int(r[1].max()) <= dt.max, f"{dt} overflow")
    memo[v] = r
    return r
  return ev(u)

def evaluate(u:UOp, env:dict[UOp, np.ndarray]) -> np.ndarray:
  """u's value, decided, with exact leaves"""
  lo, hi = bounds(u, {k: (np.asarray(v), np.asarray(v)) for k, v in env.items()})
  need(bool(np.array_equal(lo, hi)), "value not decided")
  return lo

def conv_operand(u:UOp) -> tuple[UOp, UOp, list[UOp], int, bool]:
  """the x operand, (x - zp) in int32 with x a uint8 load, and its masks. tinygrad builds x.pad(value=zp).cast(int32) - zp as
  CAST(WHERE(mask, WHERE(mask, x, 0), zp)) - zp, the INDEX gated by the mask too. Any integer form is accepted whose value is
  x - zp for every uint8 x where its masks hold: (param, index, masks, zp, whether it is 0 for every x where they don't)"""
  need(u.dtype == I32, f"operand in {u.dtype}")
  loads, masks, todo, seen = [], [], [u], set()
  while todo:
    if (v:=todo.pop()) in seen: continue
    seen.add(v)
    if v.op is Ops.INDEX: loads.append(v)
    elif v.op is Ops.WHERE:
      masks.append(v.src[0])
      todo += v.src[1:]
    elif v.op in (Ops.CAST, Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG): todo += v.src
    else: need(v.op is Ops.CONST, f"{v.op} in the x operand")
  need(len(loads) == 1 and len(ms:=U8_LOAD.match(loads[0], {})) == 1, "x operand is not one load of uint8")
  idx = ms[0]["p_idx"]
  if idx.op is Ops.WHERE and idx.src[2].op is Ops.CONST and idx.src[2].is_invalid:
    masks.append(idx.src[0])
    idx = idx.src[1]
  masks = list(dict.fromkeys(masks))
  need(all(m.dtype == dtypes.bool and not any(s.op in (Ops.INDEX, Ops.PARAM) for s in m.toposort()) for m in masks), "mask reads memory")
  x = np.arange(256)
  def value(inside:bool) -> np.ndarray: return np.broadcast_to(evaluate(u, {loads[0]: x, **{m: np.array(inside) for m in masks}}), x.shape)
  zp = int(x[0] - (inside:=value(True))[0])
  need(np.array_equal(inside, x - zp) and 0 <= zp <= 255, "x operand is not x - zp in int32")
  return buffer(ms[0]["p"]), idx, masks, zp, bool(masks) and not value(False).any()

def conjuncts(m:UOp) -> list[UOp]: return [c for s in m.src for c in conjuncts(s)] if m.op is Ops.AND and m.dtype == dtypes.bool else [m]

def check_masks(r:dict[str, UOp], sz:dict[str, int], s:int, p:int, H:int, W:int, masks:list[UOp], zero_pad:bool):
  """every mask is the image on every (oy, ky) and (ox, kx), each conjunct of it on one axis; where padding is read, x - zp is 0"""
  axes = {}   # image axis: (its ranges' values, the image there)
  for o, k, size in (("oy", "ky", H), ("ox", "kx", W)):
    vo, vk = np.arange(sz[o])[:, None], np.arange(sz[k])[None, :]
    axes[o] = ({r[a]: v for a, v in ((o, vo), (k, vk)) if a in r}, (0 <= vo * s + vk - p) & (vo * s + vk - p < size))
  need(all(img.any() for _, img in axes.values()), "a conv of padding only")
  need(all(img.all() for _, img in axes.values()) or zero_pad, "the padding is not the input zero point")
  for m in masks:
    got = {a: np.ones_like(img) for a, (_, img) in axes.items()}
    for c in conjuncts(m):
      rs = {u for u in c.toposort() if u.op is Ops.RANGE}
      need((a:=next((a for a, (env, _) in axes.items() if rs <= env.keys()), None)) is not None, "mask is not one per image axis")
      need((t:=evaluate(c, axes[a][0])).dtype == np.bool_, "mask is not boolean")
      got[a] = got[a] & t
    need(all(np.array_equal(got[a], img) for a, (_, img) in axes.items()), "mask is not the image")

OUT_AXES, RED_AXES = ("n", "o", "oy", "ox"), ("c", "ky", "kx")
def assignments(ranges:tuple[UOp, ...], lin:dict[UOp, int], axes:tuple[str, ...]) -> list[dict[str, UOp]]:
  """the ranges, by decreasing coefficient in lin, onto the axes in their order, every way (an axis of size 1 has no range);
  the ways that keep the later axes come first (ox before oy before o, kx before ky before c)"""
  rs = sorted(ranges, key=lambda r: -lin.get(r, 0))
  return [dict(zip(sorted(names, key=axes.index), rs)) for names in itertools.combinations(axes[::-1], len(rs))]
def terms(r:dict[str, UOp], **coef:int) -> dict[UOp, int]: return {r[a]: k for a, k in coef.items() if a in r}
def divisors(n:int) -> list[int]: return sorted({d for i in range(1, math.isqrt(n) + 1) if n % i == 0 for d in (i, n // i)}) if n > 0 else []

def geometries(r:dict[str, UOp], sz:dict[str, int], x_lin:dict[UOp, int], x_off:int, x_size:int):
  """candidates (stride, padding, H, W) for the x index n*Cin*H*W + c*H*W + oy*s*W + ky*W + ox*s + kx - p*W - p, read off its
  terms; where it has none for one: the output size, or the size of the x buffer (only hints, every candidate is verified)"""
  X = {a: x_lin.get(r[a], 0) for a in r}
  N, Cin, OH, kh, kw = sz["n"], sz["c"], sz["oy"], sz["ky"], sz["kx"]
  HW = X["c"] if "c" in r else X["n"] // Cin if "n" in r else x_size // (N * Cin)
  for W in [X["ky"]] if "ky" in r else [X["oy"] // X["ox"]] if "oy" in r and X.get("ox") else divisors(HW):
    if W < 1 or x_off > 0 or -x_off % (W + 1): continue
    p = -x_off // (W + 1)                                      # the offset is -(p*W + p)
    s = X["ox"] if "ox" in r else X["oy"] // W if "oy" in r else None   # None: one output position, any stride does
    if "c" in r or "n" in r: Hs = [HW // W]
    else:                                                      # H only bounds the rows read: the buffer's, or what the output allows
      h0 = (OH - 1) * s + kh - 2 * p if s else kh - 2 * p
      Hs = [x_size // (N * Cin * W), *range(max(1, h0), h0 + s if s else kh - p + 1)]
    for H in dict.fromkeys(h for h in Hs if h >= 1):
      if s is None or s >= 1: yield s or max(1, H + 2 * p - kh + 1, W + 2 * p - kw + 1), p, H, W

def conv_layout(ends, red, out_lin, w_lin, bias_lin, x_lin, x_off, x_size, masks, zero_pad) -> dict[str, int]:
  """N, Cin, H, W, Cout, kh, kw, stride and padding of the conv whose index expressions these are; no match if there is none.
  Where size 1 axes leave several convs with exactly these indices (a 1 x kw kernel on one row is a kw x 1 kernel on a column),
  all compute the same; the one that fills the x buffer, with the smallest stride and kernel, is taken as the conv written"""
  err: NoMatch|None = None
  found: list[tuple[tuple, dict[str, int]]] = []
  for i, r in enumerate({**a, **b} for a in assignments(ends, out_lin, OUT_AXES) for b in assignments(red, w_lin, RED_AXES)):
    sz = {a: rsize(r[a]) if a in r else 1 for a in OUT_AXES + RED_AXES}
    N, Cout, OH, OW, Cin, kh, kw = (sz[a] for a in OUT_AXES + RED_AXES)
    if out_lin != terms(r, n=Cout*OH*OW, o=OH*OW, oy=OW, ox=1) or w_lin != terms(r, o=Cin*kh*kw, c=kh*kw, ky=kw, kx=1) or \
       (bias_lin is not None and bias_lin != (terms(r, o=1), 0)): continue
    for s, p, H, W in geometries(r, sz, x_lin, x_off, x_size):
      try:
        need((H + 2 * p - kh) // s + 1 == OH and (W + 2 * p - kw) // s + 1 == OW, "output size")
        need(x_lin == terms(r, n=Cin*H*W, c=H*W, oy=s*W, ky=W, ox=s, kx=1), "x index")
        check_masks(r, sz, s, p, H, W, masks, zero_pad)
        found.append(((N * Cin * H * W != x_size, s, kh * kw, i), dict(N=N, Cin=Cin, H=H, W=W, Cout=Cout, kh=kh, kw=kw, stride=s, padding=p)))
      except NoMatch as e: err = err or e
    err = err or NoMatch("x index: no stride, padding and image size")
  if found: return min(found, key=lambda f: f[0])[1]
  raise err or NoMatch("index is not out[n, o, oy, ox], w[o, c, ky, kx], b[o]")

def match_conv(ast:UOp) -> ConvMatch:
  need(ks:=KERNEL.match(ast, {}), "not a kernel storing a cast to uint8")
  out, out_idx, ends = buffer(ks[0]["out"]), ks[0]["out_idx"], ks[0]["end"].src[1:] if "end" in ks[0] else ()
  acc, mult, lo, hi, out_zp = epilogue(ks[0]["val"])
  need(accs:=CONV_ACC.match(acc, {}), "accumulator is not sum((x - x_zp) * (w - w_zp)) (+ bias)")
  red = accs[0]["red"].src[1:]
  need(math.prod(rsize(k) for k in red) * 255 * 255 < 2**31, "K too large for an int32 accumulator")
  out_lin, out_off = linear(out_idx)
  need(out_off == 0 and set(out_lin) == set(ends) and len(set(ends)) == len(ends) and not set(ends) & set(red), "output index")
  err: NoMatch|None = None                                     # the first operand order's reason is the informative one
  for m in accs:
    try:
      x, x_idx, masks, in_zp, zero_pad = conv_operand(m["a"])
      w, w_idx, w_zp = operand(m["b"])
      b = buffer(m["bias"]) if "bias" in m else None
      (x_lin, x_off), (w_lin, w_off) = linear(x_idx), linear(w_idx)
      need(w_off == 0, "w index")
      g = conv_layout(ends, red, out_lin, w_lin, None if b is None else linear(m["bias_idx"]), x_lin, x_off, x.arg.size, masks, zero_pad)
      spec = ConvSpec(**g, in_zp=in_zp, w_zp=w_zp, out_zp=out_zp, mult=mult, lo=lo, hi=hi)
      need(out not in (x, w, b), "output aliases an input")
      need(x.arg.size >= spec.N * spec.Cin * spec.H * spec.W and w.arg.size >= spec.Cout * spec.Cin * spec.kh * spec.kw and
           out.arg.size >= spec.N * spec.Cout * spec.OH * spec.OW and (b is None or b.arg.size >= spec.Cout), "buffer sizes")
      need({u for u in ast.toposort() if u.op is Ops.PARAM} == {out, x, w} | ({b} if b is not None else set()), "other params")
      return ConvMatch(spec, out, x, w, b)
    except NoMatch as e: err = err or e
  raise err or NoMatch("no operand order matches")

def select_conv(ast:UOp) -> ConvMatch|None:
  """the Edge TPU CONV_2D this kernel computes, or None"""
  try: return match_conv(ast)
  except NoMatch: return None

# ***** glue: the uint8 kernels between two layers, for a fusion pass (coral/chain.py) *****
#
# Behind a ReLU or a max pool, coral.quantize gives the producer's output and the consumer's input one quantization (zero point
# 0 behind a ReLU), so the kernel between them maps uint8 to uint8: dequantize, ReLU, (max pool,) quantize, round, clip. Alone
# such a kernel stays clang (select() doesn't take it); select_glue recognizes the two a chain can run:
#   CopySpec(n, lo, hi)              out[i] = clip(x[i], lo, hi), i < n: elementwise, in and out indexed alike, contiguous
#   PoolSpec(N, C, H, W, k, stride)  out[N,C,OH,OW] = max of x[N,C,H,W] over k x k windows at stride, no padding, NCHW contiguous
# Its maps are evaluated exactly (bounds) on all 256 uint8 values: the value stored must be decided for each and be clip(v, lo,
# hi); a pool's is out = g(max f(v)), with f the map before the max and g the one after: f must be non-decreasing (then
# max f(v) = f(max v), so out = g(f(max v))) and g(f(v)) = v (no clamp: the chain runs a pool as a plain max pool).

@dataclass(frozen=True)
class CopySpec:
  n: int                                  # out[i] = clip(x[i], lo, hi) for i < n, uint8
  lo: int = 0
  hi: int = 255

@dataclass(frozen=True)
class PoolSpec:
  N: int                                  # x[N,C,H,W] uint8, NCHW contiguous
  C: int
  H: int
  W: int
  k: int                                  # out[N,C,OH,OW] uint8 = the max of each k x k window, no padding
  stride: int
  @property
  def OH(self) -> int: return (self.H - self.k) // self.stride + 1
  @property
  def OW(self) -> int: return (self.W - self.k) // self.stride + 1

def glue_reference(spec:CopySpec|PoolSpec, x:np.ndarray) -> np.ndarray:
  """what a glue kernel stores: clip(x[:n], lo, hi), or out[N,C,OH,OW] the max pool of x"""
  if isinstance(spec, CopySpec): return np.clip(x.reshape(-1)[:spec.n], spec.lo, spec.hi).astype(np.uint8)
  win = np.lib.stride_tricks.sliding_window_view(x.reshape(spec.N, spec.C, spec.H, spec.W), (spec.k, spec.k), axis=(2, 3))
  return win[:, :, ::spec.stride, ::spec.stride].max((4, 5))

@dataclass(frozen=True)
class CopyMatch:
  spec: CopySpec
  out: UOp                                # the kernel's PARAMs
  x: UOp
  @property
  def params(self) -> tuple[UOp, ...]: return (self.out, self.x)

@dataclass(frozen=True)
class PoolMatch:
  spec: PoolSpec
  out: UOp                                # the kernel's PARAMs
  x: UOp
  @property
  def params(self) -> tuple[UOp, ...]: return (self.out, self.x)

GLUE_STORE = UPat(Ops.STORE, src=(ld(U8, "out"), UPat.var("val", U8)))
GLUE = UPat.sink(UPat.any(GLUE_STORE, UPat(Ops.END, src=(GLUE_STORE,), allow_any_len=True, name="end")))

def uint8_map(y:tuple[np.ndarray, np.ndarray]) -> tuple[int, int]:
  """(lo, hi) where the map y on the 256 uint8 values is decided and clip(v, lo, hi)"""
  v = np.arange(256)
  need(bool(np.array_equal(*np.broadcast_arrays(*y))), "the map is not decided on every value (a rounding too close to call)")
  m = np.broadcast_to(y[0], v.shape)
  lo, hi = int(m[0]), int(m[-1])
  need(lo <= hi and bool(np.array_equal(m, np.clip(v, lo, hi))), "the map is not clip(v, lo, hi)")
  return lo, hi

def unflatten(*idxs:UOp) -> tuple[dict[UOp, list[UOp]], tuple[UOp, ...]]:
  """a range the indices take apart with // and % by constants (a flattened output j: c = j // 9, oy = j // 3 % 3, ox = j % 3)
  becomes its digits, j = 9*c + 3*oy + ox over new ranges; linear() then resolves the // and % exactly -> ({range: its digits,
  most significant first}, the indices over the digits)"""
  def place(e:UOp) -> tuple[UOp, int]|None:                   # (r, p) for e = r // p
    if e.op is Ops.RANGE: return e, 1
    if e.op is Ops.FLOORDIV and e.src[1].op is Ops.CONST and type(d:=e.src[1].arg) is int and d > 0 and (rp:=place(e.src[0])): return rp[0], rp[1] * d
    return None
  places: dict[UOp, set[int]] = {}
  for u in (u for i in idxs for u in i.toposort()):
    if u.op in (Ops.FLOORDIV, Ops.FLOORMOD) and u.src[1].op is Ops.CONST and type(d:=u.src[1].arg) is int and d > 0 and (rp:=place(u.src[0])):
      places.setdefault(rp[0], set()).update((rp[1], rp[1] * d))
  sub, digits = {}, {}
  for r, ps in places.items():
    n, ps = rsize(r), sorted({1} | {p for p in ps if p < rsize(r)})
    if not all(b % a == 0 for a, b in zip(ps, ps[1:] + [n])): continue          # not a mixed radix: leave it
    radix = [(a, b // a) for a, b in zip(ps, ps[1:] + [n]) if b // a > 1][::-1]        # (place, size), most significant first
    digits[r] = [UOp.range(m, -1000 * (1 + r.arg[1]) - i) for i, (_, m) in enumerate(radix)]   # ids no scheduler range has
    sub[r] = sum((d * a for d, (a, _) in zip(digits[r], radix)), UOp.const(0, dtypes.weakint))
  return digits, tuple(i.substitute(sub) for i in idxs) if sub else idxs

POOL_OUT, POOL_RED = ("n", "c", "oy", "ox"), ("ky", "kx")
def pool_layout(ends, red, out_lin, x_lin, x_off, x_size) -> dict[str, int]:
  """N, C, H, W, k, stride of the max pool whose index expressions these are (the conv's geometry without w and padding)"""
  err: NoMatch|None = None
  found: list[tuple[tuple, dict[str, int]]] = []
  for i, r in enumerate({**a, **b} for a in assignments(ends, out_lin, POOL_OUT) for b in assignments(red, x_lin, POOL_RED)):
    sz = {a: rsize(r[a]) if a in r else 1 for a in POOL_OUT + POOL_RED}
    N, C, OH, OW, k = sz["n"], sz["c"], sz["oy"], sz["ox"], sz["ky"]
    if sz["kx"] != k or k < 2 or out_lin != terms(r, n=C*OH*OW, c=OH*OW, oy=OW, ox=1): continue
    for s, p, H, W in geometries(r, sz, x_lin, x_off, x_size):
      try:
        need(p == 0 and (H - k) // s + 1 == OH and (W - k) // s + 1 == OW, "output size")
        need(x_lin == terms(r, n=C*H*W, c=H*W, oy=s*W, ky=W, ox=s, kx=1), "x index")
        found.append(((N * C * H * W != x_size, s, i), dict(N=N, C=C, H=H, W=W, k=k, stride=s)))
      except NoMatch as e: err = err or e
    err = err or NoMatch("x index: no stride and image size")
  if found: return min(found, key=lambda f: f[0])[1]
  raise err or NoMatch("index is not a square window of out[n, c, oy, ox]")

def match_glue(ast:UOp) -> CopyMatch|PoolMatch:
  need(ks:=GLUE.match(ast, {}), "not a kernel storing uint8")
  out, out_idx, val, ends = buffer(ks[0]["out"]), ks[0]["out_idx"], ks[0]["val"], ks[0]["end"].src[1:] if "end" in ks[0] else ()
  topo = val.toposort()
  loads, reds = [u for u in topo if u.op is Ops.INDEX], [u for u in topo if u.op is Ops.REDUCE]
  need(len(loads) == 1 and len(ms:=U8_LOAD.match(loads[0], {})) == 1, "not one load of uint8")
  x, x_idx = buffer(ms[0]["p"]), ms[0]["p_idx"]
  need(x is not out and {u for u in ast.toposort() if u.op is Ops.PARAM} == {out, x}, "other params")
  digits, (out_idx, x_idx) = unflatten(out_idx, x_idx)
  ends, red_ranges = (tuple(d for r in rs for d in digits.get(r, [r])) for rs in (ends, reds[0].src[1:] if reds else ()))
  (out_lin, out_off), (x_lin, x_off), v = linear(out_idx), linear(x_idx), np.arange(256)
  need(out_off == 0 and set(out_lin) == set(ends) and len(set(ends)) == len(ends), "output index")
  if not reds:                                                 # a copy: each output from the input at the same index
    need(x_lin == out_lin and x_off == 0, "in and out are not indexed alike")
    n = 1
    for r in sorted(ends, key=lambda r: out_lin[r]):
      need(out_lin[r] == n, "not a contiguous index")
      n *= rsize(r)
    lo, hi = uint8_map(bounds(val, {loads[0]: (v, v)}))
    need(x.arg.size >= n and out.arg.size >= n, "buffer sizes")
    return CopyMatch(CopySpec(n, lo, hi), out, x)
  need(len(reds) == 1 and (red:=reds[0]).arg == (Ops.MAX, 0), "not one max")
  f = tuple(np.broadcast_to(e, v.shape) for e in bounds(red.src[0], {loads[0]: (v, v)}))
  need(bool(np.all(f[1][:-1] <= f[0][1:])), "the map before the max is not non-decreasing")
  need(uint8_map(bounds(val, {red: f})) == (0, 255), "the maps around the max are not the identity")
  spec = PoolSpec(**pool_layout(ends, red_ranges, out_lin, x_lin, x_off, x.arg.size))
  need(x.arg.size >= spec.N * spec.C * spec.H * spec.W and out.arg.size >= spec.N * spec.C * spec.OH * spec.OW, "buffer sizes")
  return PoolMatch(spec, out, x)

def select_glue(ast:UOp) -> CopyMatch|PoolMatch|None:
  """the uint8 glue this kernel is, for a fusion pass: an exact copy (with a clamp) or an exact max pool; params (out, x).
  Not part of select(): a glue kernel alone stays a clang kernel"""
  try: return match_glue(ast)
  except NoMatch: return None

def why_not_glue(ast:UOp) -> str|None:
  """why select_glue() gives None (None if it doesn't)"""
  try: match_glue(ast)
  except NoMatch as e: return str(e)
  return None

# ***** entry points *****

def select(ast:UOp) -> FCMatch|ConvMatch|None:
  """the Edge TPU program this kernel computes: an FCMatch (spec: FCSpec) or a ConvMatch (spec: ConvSpec), both with params in
  Edge TPU argument order (out, x, w[, b]); None: it stays a clang kernel. A kernel that is both (a conv of a 1x1 image) is the FC."""
  return select_fc(ast) or select_conv(ast)

def why_not(ast:UOp) -> str|None:
  """why select() keeps a kernel on clang (None if it selects it)"""
  errs = []
  for kind, match in (("FC", match_fc), ("conv", match_conv)):
    try: match(ast)
    except NoMatch as e:
      errs.append(f"not an FC: {e}" if kind == "FC" else f"not a conv: {e}")
      continue
    return None
  return "; ".join(errs)
