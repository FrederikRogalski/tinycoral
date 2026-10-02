# inter-tile data movement DMAs (docs/isa/ring_mesh.md):
#   0x10 ringProducer   4 words  narrow/wide memory -> ring, multicast to a destination bitmap
#   0x11 ringConsumer   4 words  ring -> memory (parameters in caching programs, activations in execution programs)
#   0x12 ringConsumer1  4 words  same layout; receives the packets sent with to_c1=1
#   0x15 / 0x16 / 0x17 / 0x18 meshBus, 6 words: the opcode is the direction the data moves, south (t -> t+4), west (t -> t-1),
#        north (t -> t-4), east (t -> t+1), with tile t at row t//4, column t%4. An outbound (o_) and an inbound (i_) half.
from __future__ import annotations
from coral.isa import Layout

SCALAR_CORE = 16                                   # bit 16 of the ring destination bitmap = the scalar core (outfeed)

def _desc(p:str, off:int) -> list[tuple]:
  """one address generator: 18-bit address, 4 loop dims (signed increment, count - 1), the buffer / sync block"""
  dims = [f for j in range(4) for f in ((f"{p}inc{j}", 78 + off + 33 * j, 17, True), (f"{p}cnt{j}", 95 + off + 33 * j, 16))]
  return [(f"{p}addr", 60 + off, 18), *dims,
          (f"{p}sdims", 210 + off, 4),     # 1 = plain transfer; a mask of the used levels (1/3/7) for multi-dim mesh moves
          (f"{p}cbuf", 214 + off, 2),      # 1 = circular staging buffer in use (with slots)
          (f"{p}slots", 216 + off, 11),    # buffer depth: ring = packets in stream, mesh forwarder = 4
          (f"{p}rsv_s", 227 + off, 6),
          (f"{p}grp", 233 + off, 10)]      # ring consumer: packets per tile group - 1 (multi-packet groups)

def _rec(p:str, lo:int, idw:int, valw:int) -> list[tuple]:
  """41-bit sync record: id (ring: op<<5 | tile SyncCounter, mesh: counter<<1 | op), signed value, enable a, enable b"""
  q = lo + idw + valw
  return [(f"{p}id", lo, idw), (f"{p}val", lo + idw, valw, True), (f"{p}en_a", q, 1), (f"{p}x", q + 1, 15), (f"{p}en_b", q + 16, 1)] + \
         ([(f"{p}y", q + 17, lo + 24 - q)] if q + 17 < lo + 41 else [])

RING_PRODUCER = Layout("ringProducer", (0x10,), 4, [("tile_mask", 12, 16), ("seq", 46, 14), *_desc("", 0),
  ("mode", 335, 2),                                # 3 = to the scalar core (output), 1 = tile -> tile activation forward, 0 = other
  ("pcfg", 337, 6),                                # 20 output, 0 forward, 4 / 12 parameter broadcast
  *_rec("r0_", 343, 7, 16), *_rec("r1_", 384, 7, 16), *_rec("r2_", 425, 7, 16),
  ("to_c1", 466, 1),                               # 1 = the packets go to ringBusConsumer1 (0x12)
  ("dest", 467, 17)])                              # destination bitmap: bit t = tile t, bit 16 = the scalar core
RING_CONSUMER = Layout("ringConsumer", (0x11, 0x12), 4, [("tile_mask", 12, 16), ("seq", 46, 14), *_desc("", 0),
  ("gstride", 249, 18),                            # multi-packet groups: 4*(G-1)*p + 1 (G groups of p packets)
  ("aux_en0", 267, 1), ("aux_addr0", 268, 18), ("aux_addr1", 300, 18), ("aux_en1", 318, 1),   # caching: parameter relocation fields
  ("mode", 335, 2),                                # 3 = infeed-fed / parameter, 1 = fed by a tile's ringProducer
  ("s_id", 339, 7), ("s_val", 346, 16, True), ("s_en_a", 362, 1), ("s_cnt", 368, 10), ("s_en_b", 378, 1), ("s_y", 379, 1)])
MESH = Layout("mesh", (0x16, 0x17, 0x15, 0x18), 6, [("tile_mask", 12, 16), ("seq", 46, 14), *_desc("o_", 0), *_desc("i_", 205),
  ("out_mode", 473, 2), ("in_mode", 477, 3), *[f for k in range(5) for f in _rec(f"s{k}_", 480 + 41 * k, 6, 18)],
  ("fill_en", 727, 1),                             # the inbound half generates a constant instead of receiving (edge padding)
  ("fill", 728, 32)])                              # 32-bit fill pattern, e.g. 0x80808080 = zero point 128 in every byte
encode_ringProducer, decode_ringProducer = RING_PRODUCER.encode, RING_PRODUCER.decode
encode_ringConsumer, decode_ringConsumer = RING_CONSUMER.encode, RING_CONSUMER.decode
encode_mesh, decode_mesh = MESH.encode, MESH.decode
