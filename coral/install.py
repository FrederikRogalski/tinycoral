# tinygrad finds devices by their file tinygrad/runtime/ops_<name>.py: link ours in.   python -m coral.install
import pathlib, tinygrad
if __name__ == "__main__":
  dst, src = pathlib.Path(tinygrad.__file__).parent / "runtime" / "ops_coral.py", pathlib.Path(__file__).resolve().parent / "ops_coral.py"
  if dst.is_symlink() or dst.exists(): dst.unlink()
  dst.symlink_to(src)
  print(f"{dst} -> {src}")
