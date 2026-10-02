# libusb through tinygrad's bindings: control and bulk transfers, and AsyncBatch, which queues all of a program's transfers at once
import ctypes, functools, time
from tinygrad.runtime.autogen import libusb
from tinygrad.runtime.support import c
from tinygrad.helpers import DEBUG

def checked(fn, msg=None):
  @functools.wraps(fn)
  def wrapper(*args):
    if (rc:=fn(*args)) < 0: raise RuntimeError(f"{msg or fn.__name__}: {ctypes.string_at(libusb.libusb_strerror(rc)).decode()}")
    return rc
  return wrapper

@functools.cache
def usb_ctx(): return c.init_c_var(ctypes.POINTER(libusb.struct_libusb_context), checked(libusb.libusb_init))

def find(vid:int, pid:int) -> bool:
  h = libusb.libusb_open_device_with_vid_pid(usb_ctx(), vid, pid)
  if h: libusb.libusb_close(h)
  return bool(h)

class USBDevice:
  def __init__(self, vid:int, pid:int):
    self.handle = libusb.libusb_open_device_with_vid_pid(usb_ctx(), vid, pid)
    if not self.handle: raise RuntimeError(f"usb device {vid:04x}:{pid:04x} not found")
    self.buf = (ctypes.c_ubyte * (1 << 16))()
    self.xfer = ctypes.c_int32(0)
    self.claimed: set[int] = set()

  def speed(self) -> int: return libusb.libusb_get_device_speed(libusb.libusb_get_device(self.handle))
  def claim(self, iface:int):
    checked(libusb.libusb_claim_interface)(self.handle, iface)
    self.claimed.add(iface)

  def _buf(self, n:int):
    if n > len(self.buf): self.buf = (ctypes.c_ubyte * n)()
    return self.buf

  def ctrl_out(self, rtype:int, req:int, val:int, idx:int, data:bytes=b'', timeout:int=1000):
    buf = self._buf(len(data))
    ctypes.memmove(buf, data, len(data))
    rc = checked(libusb.libusb_control_transfer)(self.handle, rtype, req, val, idx, buf, len(data), timeout)
    assert rc == len(data), f"ctrl_out short {rc}/{len(data)}"

  def ctrl_in(self, rtype:int, req:int, val:int, idx:int, n:int, timeout:int=1000) -> bytes:
    buf = self._buf(n)
    rc = checked(libusb.libusb_control_transfer)(self.handle, rtype, req, val, idx, buf, n, timeout)
    return bytes(buf[:rc])

  def bulk_out(self, ep:int, data:bytes|memoryview, timeout:int=6000):
    buf = self._buf(len(data))
    ctypes.memmove(buf, bytes(data), len(data))
    checked(libusb.libusb_bulk_transfer)(self.handle, ep, buf, len(data), ctypes.byref(self.xfer), timeout)
    assert self.xfer.value == len(data), f"bulk_out short {self.xfer.value}/{len(data)}"

  def bulk_in(self, ep:int, n:int, timeout:int=6000) -> bytes:
    buf = self._buf(n)
    checked(libusb.libusb_bulk_transfer)(self.handle, ep, buf, n, ctypes.byref(self.xfer), timeout)
    return bytes(buf[:self.xfer.value])

  def reset(self): libusb.libusb_reset_device(self.handle)
  def close(self):
    for i in self.claimed: libusb.libusb_release_interface(self.handle, i)
    libusb.libusb_close(self.handle)
    self.handle = None

def wait_for(vid:int, pid:int, timeout:float=10.0) -> float:
  st = time.perf_counter()
  while not find(vid, pid):
    if time.perf_counter() - st > timeout: raise TimeoutError(f"usb device {vid:04x}:{pid:04x} did not appear")
    time.sleep(0.05)
  if DEBUG >= 1: print(f"{vid:04x}:{pid:04x} appeared after {time.perf_counter()-st:.2f}s")
  return time.perf_counter() - st

# *** async transfers: submit all bulk INs first, then the OUTs, and let libusb complete them concurrently ***
class _Transfer(ctypes.Structure):
  _fields_ = [("dev_handle", ctypes.c_void_p), ("flags", ctypes.c_uint8), ("endpoint", ctypes.c_uint8), ("type", ctypes.c_uint8),
              ("timeout", ctypes.c_uint32), ("status", ctypes.c_int32), ("length", ctypes.c_int32), ("actual_length", ctypes.c_int32),
              ("callback", ctypes.c_void_p), ("user_data", ctypes.c_void_p), ("buffer", ctypes.c_void_p), ("num_iso_packets", ctypes.c_int32)]
_CB = ctypes.CFUNCTYPE(None, ctypes.POINTER(_Transfer))
_lib = ctypes.CDLL(libusb.dll._name)
_lib.libusb_alloc_transfer.restype = ctypes.POINTER(_Transfer)
_lib.libusb_alloc_transfer.argtypes = [ctypes.c_int]
_lib.libusb_submit_transfer.argtypes = [ctypes.POINTER(_Transfer)]
_lib.libusb_cancel_transfer.argtypes = [ctypes.POINTER(_Transfer)]
_lib.libusb_handle_events_completed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

class AsyncBatch:
  """reusable set of transfer structs; run([(ep, data_or_len), ...]) -> list of received bytes (None for OUTs)"""
  def __init__(self, dev:USBDevice, n:int=8):
    self.dev, self.pending = dev, ctypes.c_int(0)
    self.xfers = [_lib.libusb_alloc_transfer(0) for _ in range(n)]
    self.bufs: list = [None] * n
    self.done_flags = [False] * n
    def cb(t):
      self.pending.value -= 1
      self.done_flags[self.index[ctypes.addressof(t.contents)]] = True
    self.cb = _CB(cb)
    self.ctx = ctypes.cast(usb_ctx(), ctypes.c_void_p)
    self.index = {ctypes.addressof(x.contents): i for i, x in enumerate(self.xfers)}
  def _finished(self, i:int) -> bool: return self.done_flags[i]

  def run(self, ops:list[tuple[int, bytes|int]], timeout:int=6000) -> list[bytes|None]:
    assert len(ops) <= len(self.xfers)
    order = [i for i, (ep, _) in enumerate(ops) if ep & 0x80] + [i for i, (ep, _) in enumerate(ops) if not ep & 0x80]
    for i in order:
      ep, d = ops[i]
      n = d if isinstance(d, int) else len(d)
      if self.bufs[i] is None or len(self.bufs[i]) < n: self.bufs[i] = (ctypes.c_ubyte * max(n, 1024))()
      if not isinstance(d, int): ctypes.memmove(self.bufs[i], d, n)
      t = self.xfers[i].contents
      t.dev_handle, t.endpoint, t.type, t.timeout, t.length = ctypes.cast(self.dev.handle, ctypes.c_void_p), ep, 2, timeout, n
      t.flags, t.status, t.actual_length, t.num_iso_packets = 0, 0, 0, 0
      t.buffer, t.callback = ctypes.cast(self.bufs[i], ctypes.c_void_p), ctypes.cast(self.cb, ctypes.c_void_p)
    self.pending.value = len(ops)
    for i in range(len(ops)): self.done_flags[i] = False
    for i in order:
      if (rc:=_lib.libusb_submit_transfer(self.xfers[i])) < 0: raise RuntimeError(f"submit failed {rc}")
    done, got, counted = ctypes.c_int(0), [0] * len(ops), [False] * len(ops)
    while self.pending.value > 0:
      _lib.libusb_handle_events_completed(self.ctx, ctypes.byref(done))
      # one transfer failed (e.g. timed out): cancel everything still in flight before anybody touches the device again,
      # a pending bulk OUT the chip can't drain otherwise blocks the usb firmware (and control transfers) for good
      if any(self.xfers[i].contents.status != 0 for i in range(len(ops)) if self._finished(i)):
        for i in range(len(ops)): _lib.libusb_cancel_transfer(self.xfers[i])
        while self.pending.value > 0: _lib.libusb_handle_events_completed(self.ctx, ctypes.byref(done))
        break
      # an IN that ended short (a short packet ends every device DMA): the device may send one output in several DMAs, e.g.
      # through scalar memory. Keep reading until the requested size, before anything else goes out (else: deadlock)
      for i, (ep, d) in enumerate(ops):
        if not (ep & 0x80) or not self._finished(i) or counted[i]: continue
        t, counted[i] = self.xfers[i].contents, True
        got[i] += t.actual_length
        if got[i] < d:
          t.buffer, t.length, t.status, t.actual_length = ctypes.cast(ctypes.byref(self.bufs[i], got[i]), ctypes.c_void_p), d - got[i], 0, 0
          self.done_flags[i], counted[i] = False, False
          self.pending.value += 1
          if (rc:=_lib.libusb_submit_transfer(self.xfers[i])) < 0: raise RuntimeError(f"resubmit failed {rc}")
    out = []
    for i, (ep, d) in enumerate(ops):
      t = self.xfers[i].contents
      if t.status != 0: raise RuntimeError(f"transfer on ep {ep:#x} failed with status {t.status}")
      out.append(bytes(self.bufs[i][:got[i]]) if ep & 0x80 else None)
    return out
