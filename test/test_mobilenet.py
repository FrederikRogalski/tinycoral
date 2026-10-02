# end to end: our driver runs Google's compiled MobileNet v1 (ref/edgetpu/test_data); compare with the CPU TFLite reference
# (needs pillow and ai-edge-litert)
import sys, time, pathlib, numpy as np
from PIL import Image

from coral.device import EdgeTPU
from coral.executable import load_edgetpu_tflite
from coral.runtime import Model
TD = pathlib.Path(__file__).resolve().parent.parent / "ref/edgetpu/test_data"
labels = (TD / "imagenet_labels.txt").read_text().splitlines()

def cpu_reference(img):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(str(TD / "mobilenet_v1_1.0_224_quant.tflite"))
  it.allocate_tensors()
  it.set_tensor(it.get_input_details()[0]["index"], img[None])
  it.invoke()
  return it.get_tensor(it.get_output_details()[0]["index"])[0]

if __name__ == "__main__":
  img = np.asarray(Image.open(TD / sys.argv[1] if len(sys.argv) > 1 else TD / "cat.bmp").convert("RGB").resize((224, 224)), dtype=np.uint8)
  tpu = EdgeTPU()
  model = Model(tpu, load_edgetpu_tflite(TD / "mobilenet_v1_1.0_224_quant_edgetpu.tflite"))
  st = time.perf_counter()
  out = np.frombuffer(model(img.tobytes()), np.uint8)[:1001]
  t1 = time.perf_counter() - st
  times = []
  for _ in range(10):
    st = time.perf_counter()
    out2 = np.frombuffer(model(img.tobytes()), np.uint8)[:1001]
    times.append(time.perf_counter() - st)
  ref = cpu_reference(img)
  print(f"first run (incl. 4.4MB weight upload) {t1*1e3:.1f} ms, then {min(times)*1e3:.2f} ms/inference")
  print("edgetpu top5:", [(labels[i], int(out[i])) for i in np.argsort(-out.astype(int))[:5]])
  print("cpu    top5:", [(labels[i], int(ref[i])) for i in np.argsort(-ref.astype(int))[:5]])
  print(f"max abs diff vs cpu: {np.abs(out.astype(int) - ref.astype(int)).max()}, repeat-identical: {np.array_equal(out, out2)}")
