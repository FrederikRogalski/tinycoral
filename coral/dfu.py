# load the Edge TPU firmware over USB DFU 1.1 (mirrors libedgetpu driver/usb/usb_dfu_commands.cc)
import pathlib, time
from coral.usbdev import USBDevice, find, wait_for

DFU_VID, DFU_PID = 0x1a6e, 0x089a  # Global Unichip Corp, bootloader
APP_VID, APP_PID = 0x18d1, 0x9302  # Google, after firmware
FW_PATH = pathlib.Path(__file__).resolve().parent.parent / "ref/libedgetpu/driver/usb/apex_latest_single_ep.bin"
DFU_DNLOAD, DFU_UPLOAD, DFU_GETSTATUS = 1, 2, 3
DFU_STATE_IDLE, DFU_STATE_DNLOAD_IDLE = 2, 5

def parse_dfu_config(cfg:bytes) -> tuple[int, int]:
  iface, xfer, i = None, None, cfg[0]
  while i + 1 < len(cfg) and cfg[i] != 0:
    ln, ty = cfg[i], cfg[i+1]
    if ty == 4 and cfg[i+4] == 0 and cfg[i+5] == 0xFE and cfg[i+6] == 1: iface = cfg[i+2]  # interface, 0 eps, app-specific, DFU
    if ty == 0x21: xfer = int.from_bytes(cfg[i+5:i+7], "little")                             # DFU functional descriptor
    i += ln
  assert iface is not None and xfer is not None, f"no DFU interface in {cfg.hex()}"
  return iface, xfer

def dfu_load(fw:bytes|None=None, verify=True):
  fw = fw if fw is not None else FW_PATH.read_bytes()
  dev = USBDevice(DFU_VID, DFU_PID)
  iface, xfer = parse_dfu_config(dev.ctrl_in(0x80, 6, 0x0200, 0, 512))
  dev.claim(iface)
  st = time.perf_counter()
  for blk, off in enumerate(list(range(0, len(fw), xfer)) + [len(fw)]):
    dev.ctrl_out(0x21, DFU_DNLOAD, blk & 0xFFFF, iface, fw[off:off+xfer])
    status = dev.ctrl_in(0xA1, DFU_GETSTATUS, 0, iface, 6)
    want = DFU_STATE_DNLOAD_IDLE if off < len(fw) else DFU_STATE_IDLE
    assert status[0] == 0 and status[4] == want, f"DFU block {blk}: status {status.hex()}"
  if verify:
    up, blk = b"", 0
    while len(chunk:=dev.ctrl_in(0xA1, DFU_UPLOAD, blk, iface, xfer)) == xfer: up, blk = up + chunk, blk + 1
    up += chunk
    assert up[:len(fw)] == fw, "firmware readback mismatch"
  print(f"firmware: {len(fw)} bytes in {blk+1} blocks of {xfer}, {time.perf_counter()-st:.2f}s, resetting")
  dev.reset()
  dev.close()

def ensure_firmware():
  if find(APP_VID, APP_PID): return
  dfu_load()
  wait_for(APP_VID, APP_PID)

if __name__ == "__main__":
  ensure_firmware()
  print("device is in app mode:", find(APP_VID, APP_PID))
