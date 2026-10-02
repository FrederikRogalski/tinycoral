# our generated programs on the device vs the reference (random weights, inputs and quantizations), at the placements the
# runtime uses: FC (one row) on tiles [s, s+T) at a parameter offset, 1x1 conv (M rows) with its weights on one tile
import numpy as np
from coral.tpu import FCSpec, fc_reference, runner
from coral.programs import fc_program, conv_program, bucket
from coral.runtime import run_executable

def case(rng, M, N, K):
  x, w = rng.integers(0, 256, (M, K), dtype=np.uint8), rng.integers(0, 256, (N, K), dtype=np.uint8)
  b = rng.integers(-20000, 20000, N).astype(np.int32)
  in_zp, w_zp, out_zp = (int(v) for v in rng.integers(0, 256, 3))
  acc = (x.astype(np.int64) - in_zp) @ (w.astype(np.int64) - w_zp).T + b
  return FCSpec(M, N, K, in_zp, w_zp, out_zp, float(np.float32(rng.uniform(60, 140) / max(1, np.abs(acc).max())))), x, w, b

if __name__ == "__main__":
  r, rng, worst = runner(), np.random.default_rng(0), 0
  r._own("test")                                  # we place things ourselves: the runner forgets what it had on chip
  runs = [(1, N, K, place) for N, K in [(1, 1), (7, 5), (100, 300), (288, 288), (864, 288), (1536, 288), (288, 768), (1500, 700), (64, 4096)]
          for place in [(0, 0), (16 - fc_program(N, K).ntiles, 256 * 100)]]
  runs += [(M, N, K, place) for M in (16, 128, 256) for (N, K), place in zip([(864, 288), (288, 288), (1536, 288), (288, 768), (40, 30)],
                                                                              [(0, 0), (5, 93440), (9, 10240), (14, 0), (15, 256)])]
  for M, N, K, place in runs:
    spec, x, w, b = case(rng, M, N, K)
    prog = fc_program(N, K) if M == 1 else conv_program(bucket(M), N, K)
    caching, exe = prog.programs(*place, spec.quant())
    run_executable(r.tpu, caching, parameters=prog.params(w, b, spec.w_zp))
    y = prog.output_array(run_executable(r.tpu, exe, prog.input_bytes(x, spec.in_zp)))[:M]
    d = np.abs(y.astype(int) - fc_reference(spec, x, w, b).astype(int))
    worst = max(worst, int(d.max()))
    print(f"{'FC  ' if M == 1 else 'conv'} M={M:3d} {N:4d}x{K:<4d} tile {place[0]:2d} offset {place[1]:6d}: max diff {d.max()}", flush=True)
  print("PASS" if worst == 0 else "FAIL", f"{len(runs)} programs, worst {worst}")
