# Edge TPU (Coral USB, "beagle") driver in pure python, mirrors libedgetpu driver/usb/usb_driver.cc DoOpen() in single-endpoint mode
from __future__ import annotations
import struct, time, os, fcntl, pathlib
from coral.usbdev import USBDevice
from coral.dfu import ensure_firmware, APP_VID, APP_PID
from coral.regs import REGS
from tinygrad.helpers import DEBUG

TAG_INSTRUCTIONS, TAG_INPUT, TAG_PARAMETERS, TAG_OUTPUT, TAG_INT0 = 0, 1, 2, 3, 4
EP_OUT, EP_DATA_IN, EP_EVENT_IN = 0x01, 0x81, 0x82
TILE_RUN_CONTROLS = ["opRunControl", "narrowToWideRunControl", "wideToNarrowRunControl", "meshBus0RunControl", "meshBus1RunControl",
                     "meshBus2RunControl", "meshBus3RunControl", "ringBusConsumer0RunControl", "ringBusConsumer1RunControl",
                     "ringBusProducerRunControl"]
SC_RUN_CONTROLS = ["scalarCoreRunControl", "avDataPopRunControl", "parameterPopRunControl", "infeedRunControl", "outfeedRunControl"]
# gcb clock divider (scu_ctrl_3[29:28]): 0=500MHz ("max"), 1=250MHz (libedgetpu "std"), axi clock bit 30: 1=125MHz
CLOCKS = {"max": (0, 0), "high": (1, 1), "medium": (2, 1), "low": (3, 1)}

def bits(v:int, lo:int, n:int) -> int: return (v >> lo) & ((1 << n) - 1)
def setbits(v:int, lo:int, n:int, x:int) -> int: return (v & ~(((1 << n) - 1) << lo)) | ((x & ((1 << n) - 1)) << lo)

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCK, HUNG = ROOT / ".coral.lock", ROOT / ".coral.hung"

def mark_hung(why:str):
  """a transfer failed: the device may be hung. The next EdgeTPU() re-initializes the chip and runs a canary program
  (coral/recover.py); if the firmware doesn't answer or the canary fails, it refuses until a person replugs the stick"""
  HUNG.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}: {why}\n")

class EdgeTPU:
  def __init__(self, clock:str=os.getenv("CORAL_CLOCK", "high")):
    # one process at a time: two drivers on the same endpoints hang it. After a failed transfer: no USB port reset (it makes a
    # wedged stick vanish until it is replugged), only a chip re-initialization that a canary program has to confirm
    self.lock = open(LOCK, "a+")
    try: fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError(f"the Coral is in use by another process ({LOCK.read_text().strip()})") from None
    self.lock.seek(0)
    self.lock.truncate()
    self.lock.write(f"pid {os.getpid()}")
    self.lock.flush()
    hung = HUNG.read_text().strip() if HUNG.exists() else None
    ensure_firmware()
    self.clock = clock
    for attempt in range(50):   # another process (e.g. libedgetpu) may just have released or reset the device
      try:
        self.usb = USBDevice(APP_VID, APP_PID)
        self.usb.claim(0)
        break
      except RuntimeError:
        if attempt == 49: raise
        time.sleep(0.1)
    try: self.usb.ctrl_in(0xC0, 1, REGS["scu_ctrl_0"] & 0xffff, REGS["scu_ctrl_0"] >> 16, 4, timeout=500)
    except RuntimeError:
      mark_hung("no answer to a control transfer at open")
      raise RuntimeError(f"the Coral does not answer: replug it, then delete {HUNG} (no USB reset: it makes a wedged stick vanish)") from None
    self.open(clock)
    from coral.usbdev import AsyncBatch
    self.batch = AsyncBatch(self.usb, 512)
    if hung is not None:            # the last program stalled, but the firmware answers: re-initialize, and only a canary clears it
      from coral.recover import recover
      if not recover(self): raise RuntimeError(f"the Coral is hung ({hung}) and did not recover: replug it, then delete {HUNG}")
      print(f"edgetpu: recovered after a stalled program ({hung}), canary bit-exact")
      HUNG.unlink()

  def recover(self):
    """after a hung program: drain the usb pipes and reset the chip (the usb firmware keeps running)"""
    for ep, n in [(EP_DATA_IN, 1 << 16), (EP_EVENT_IN, 16)]:
      try:
        while self.usb.bulk_in(ep, n, timeout=50): pass
      except RuntimeError: pass
    self.open(self.clock)

  # *** CSR access: vendor control requests, bRequest 0 = 64-bit, 1 = 32-bit, address split over wValue/wIndex ***
  def _addr(self, reg:str|int) -> int: return REGS[reg] if isinstance(reg, str) else reg
  def read32(self, reg) -> int:
    a = self._addr(reg)
    return struct.unpack("<I", self.usb.ctrl_in(0xC0, 1, a & 0xffff, a >> 16, 4))[0]
  def write32(self, reg, v:int):
    a = self._addr(reg)
    self.usb.ctrl_out(0x40, 1, a & 0xffff, a >> 16, struct.pack("<I", v & 0xffffffff))
  def read64(self, reg) -> int:
    a = self._addr(reg)
    return struct.unpack("<Q", self.usb.ctrl_in(0xC0, 0, a & 0xffff, a >> 16, 8))[0]
  def write64(self, reg, v:int):
    a = self._addr(reg)
    self.usb.ctrl_out(0x40, 0, a & 0xffff, a >> 16, struct.pack("<Q", v & 0xffffffffffffffff))
  def poll(self, reg, want:int, mask:int=~0, timeout:float=1.0, read=None) -> int:
    read, st = read or self.read64, time.perf_counter()
    while ((v:=read(reg)) & mask) != (want & mask):
      if time.perf_counter() - st > timeout: raise TimeoutError(f"poll {reg}: {v:#x} != {want:#x}")
    return v

  def open(self, clock:str):
    # BeagleTopLevelHandler::Open: phy modes, and note whether the gcb is hardware clock gated
    self.write32("scu_ctrl_0", setbits(setbits(self.read32("scu_ctrl_0"), 8, 3, 0), 11, 3, 0))
    gated = bits(self.read32("scu_ctrl_2"), 18, 2) == 1
    # DisableHardwareClockGate
    if gated: self.write32("scu_ctrl_2", setbits(self.read32("scu_ctrl_2"), 18, 2, 2))
    # EnableReset: rg_force_sleep=3, wait cur_pwr_state==2, pulse gcbb_credit0
    s3 = self.read32("scu_ctrl_3")
    if bits(s3, 22, 2) != 3:
      self.write32("scu_ctrl_3", setbits(s3, 22, 2, 3))
      self.poll("scu_ctrl_3", 2 << 8, 3 << 8, read=self.read32)
      self.write32("gcbb_credit0", 0xF)
      self.write32("gcbb_credit0", 0x0)
    # QuitReset: rg_force_sleep=2 + clocks, wait cur_pwr_state==0
    clkdiv, axi125 = CLOCKS[clock]
    s3 = setbits(setbits(setbits(setbits(self.read32("scu_ctrl_3"), 22, 2, 2), 28, 2, clkdiv), 30, 1, axi125), 31, 1, 0)
    self.write32("scu_ctrl_3", s3)
    self.poll("scu_ctrl_3", 0, 3 << 8, read=self.read32)
    self.poll("scalarCoreRunControl", 0)
    self.write64("idleRegister", 1)                     # idle enabled, counter=1
    self.write64("tileconfig0", 0x7f)                   # broadcast to all tiles
    self.poll("tileconfig0", 0x7f)
    self.write64("deepSleep", (30 << 8) | 2)            # to_wake_delay=30, to_sleep_delay=2
    # EnableHardwareClockGate
    self.write32("scu_ctrl_2", setbits(self.read32("scu_ctrl_2"), 18, 2, 1))
    # InitializeChip
    self.efuse_rev = bits(self.read32("omc0_00"), 24, 8)
    self.write64("descr_ep", 0xF0)                      # only sc host interrupt descriptors on the event endpoint
    self.write64("multi_bo_ep", 0)                      # single bulk-out endpoint mode
    self.write64("outfeed_chunk_length", 0x80)          # 1KB bulk-in chunks (superspeed)
    # RunControl::kMoveToRun
    for r in SC_RUN_CONTROLS: self.write64(r, 1)
    self.write64("tileconfig0", 0x7f)
    for r in TILE_RUN_CONTROLS: self.write64(r, 1)
    # RegisterAndEnableAllInterrupts
    for r in ["fatal_err_int_control", "top_level_int_0_control", "top_level_int_1_control", "top_level_int_2_control", "top_level_int_3_control"]:
      self.write64(r, 1)
    # BeagleTopLevelInterruptManager::DoEnableInterrupts (thermal warning, mbist, pcie error, thermal shutdown)
    self.write32("omc0_d4", self.read32("omc0_d4") | 0x80000001)
    self.write32("rambist_ctrl_1", 0x7f)
    self.write32("scu_ctr_7", 0x3f)
    for r, v in [("slv_abm_en", 1), ("mst_abm_en", 1), ("slv_err_resp_isr_mask", 3), ("mst_err_resp_isr_mask", 3)]: self.write32(r, v)
    self.write32("omc0_d8", self.read32("omc0_d8") | 0x80000000)
    if DEBUG >= 1: print(f"edgetpu: opened, efuse rev {self.efuse_rev}, clock {clock}, usb speed {self.usb.speed()}")

  # *** data movement: single bulk-out endpoint, every transfer is prefixed by an 8 byte header (u32 length, u8 tag) ***
  def send(self, tag:int, data:bytes|memoryview, chunk:int=1<<20, timeout:int=6000):
    # header and payload in one bulk transfer (one dma per transfer, two dmas in one transfer is misparsed by the firmware)
    self.usb.bulk_out(EP_OUT, struct.pack("<II", len(data), tag) + bytes(data[:chunk]), timeout)
    for i in range(chunk, len(data), chunk): self.usb.bulk_out(EP_OUT, data[i:i+chunk], timeout)

  def read_output(self, n:int, timeout:int=6000) -> bytes:
    out = b""
    while len(out) < n: out += self.usb.bulk_in(EP_DATA_IN, min(n - len(out), 1 << 20), timeout)
    return out

  def read_event(self, timeout:int=6000) -> tuple[int, int, int]:
    ev = self.usb.bulk_in(EP_EVENT_IN, 16, timeout)
    assert len(ev) == 16, f"short event {ev.hex()}"
    offset, length, tag = struct.unpack_from("<QIB", ev)
    return tag & 0xF, offset, length
