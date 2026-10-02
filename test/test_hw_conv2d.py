# hardware: CONV_2D programs from our code generator (coral/codegen/conv2d.py, no compiler) vs the TPU's numpy model
# (coral.select.conv_reference), bit for bit: the MNIST convolutions first (one image, then tall stacks of images per call), then
# random shapes, strides, paddings and quantizations.
#   .venv/bin/python test/test_hw_conv2d.py [seed] [random shapes]
import sys, time, numpy as np
from coral.select import ConvSpec, conv_reference, im2col
from coral.tpu import TPURunner

def case(r:TPURunner, rng, N, Cin, H, W, Cout, k, stride=1, padding=0) -> int:
  x, w = rng.integers(0, 256, (N, Cin, H, W), dtype=np.uint8), rng.integers(0, 256, (Cout, Cin, k, k), dtype=np.uint8)
  b = rng.integers(-20000, 20000, Cout).astype(np.int32)
  in_zp, w_zp, out_zp = (int(v) for v in rng.integers(0, 256, 3))
  spec0 = ConvSpec(N, Cin, H, W, Cout, k, k, stride, padding, in_zp, w_zp, out_zp, 1.0)
  acc = (im2col(spec0, x).astype(np.int64) - in_zp) @ (w.reshape(Cout, -1).astype(np.int64) - w_zp).T + b
  lo, hi = sorted(int(v) for v in rng.integers(0, 256, 2)) if rng.random() < 0.3 else (0, 255)
  mult = float(np.float32(rng.uniform(60, 140) / max(1, int(np.abs(acc).max()))))   # outputs spread over the uint8 range
  spec = ConvSpec(N, Cin, H, W, Cout, k, k, stride, padding, in_zp, w_zp, out_zp, mult, lo, hi)
  try: y = r.conv2d(spec, x, w, b, wkey=(Cin, H, W, Cout, k, stride, padding))
  except NotImplementedError as e:
    print(f"N={N} {Cin:3d}x{H:2d}x{W:2d} -> {Cout:3d} k{k} s{stride} p{padding}: skipped ({e})")
    return -1
  ref = conv_reference(spec, x, w, b)
  d = int(np.abs(y.astype(int) - ref.astype(int)).max())
  print(f"N={N} {Cin:3d}x{H:2d}x{W:2d} -> {Cout:3d} k{k} s{stride} p{padding}: max diff {d}, "
        f"exact {np.mean(y == ref) * 100:5.1f}%, spread {len(np.unique(ref))} values", flush=True)
  return d

if __name__ == "__main__":
  rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
  n_random = int(sys.argv[2]) if len(sys.argv) > 2 else 12
  r, worst, st = TPURunner(), 0, time.perf_counter()
  for N in (1, 2, 16, 37):                                       # MNIST, one image and tall stacks of up to 32 per call
    for args in [(1, 28, 28, 32, 5), (32, 24, 24, 32, 5), (32, 10, 10, 64, 3), (64, 8, 8, 64, 3)]:
      worst = max(worst, case(r, rng, N, *args))
  for _ in range(n_random):
    k, s = int(rng.choice([1, 3, 5])), int(rng.choice([1, 2]))
    H, W = (int(v) for v in rng.integers(max(k, 8), 33, 2))
    pad = int(rng.choice([0, k // 2])) if s == 1 else 0
    worst = max(worst, case(r, rng, 1, int(rng.integers(1, 65)), H, W, int(rng.integers(8, 129)), k, s, pad))
  print(f"{'PASS' if worst == 0 else 'FAIL'} worst={worst} in {time.perf_counter() - st:.1f}s, stats {r.stats}")
