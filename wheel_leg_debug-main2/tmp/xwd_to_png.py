from pathlib import Path
import struct

from PIL import Image

source = Path("tmp/slides/robot_motion_interview_v2/assets/mujoco_suspended.xwd")
target = source.with_name("mujoco_suspended_raw.png")
data = source.read_bytes()

header = struct.unpack(">25I", data[:100])
(
    header_size,
    _file_version,
    _pixmap_format,
    _pixmap_depth,
    width,
    height,
    _xoffset,
    byte_order,
    _bitmap_unit,
    _bitmap_bit_order,
    _bitmap_pad,
    bits_per_pixel,
    bytes_per_line,
    _visual_class,
    red_mask,
    green_mask,
    blue_mask,
    _bits_per_rgb,
    _colormap_entries,
    ncolors,
    *_rest,
) = header

if bits_per_pixel != 32:
    raise RuntimeError(f"Unsupported XWD pixel format: {bits_per_pixel} bpp")

pixel_offset = header_size + ncolors * 12
raw = memoryview(data)[pixel_offset : pixel_offset + bytes_per_line * height]

def shift_for(mask: int) -> int:
    shift = 0
    while mask and not (mask & 1):
        mask >>= 1
        shift += 1
    return shift

rs = shift_for(red_mask)
gs = shift_for(green_mask)
bs = shift_for(blue_mask)
endianness = "little" if byte_order == 0 else "big"

rgb = bytearray(width * height * 3)
out_index = 0
for y in range(height):
    row = raw[y * bytes_per_line : y * bytes_per_line + width * 4]
    for x in range(width):
        value = int.from_bytes(row[x * 4 : x * 4 + 4], endianness)
        rgb[out_index] = (value & red_mask) >> rs
        rgb[out_index + 1] = (value & green_mask) >> gs
        rgb[out_index + 2] = (value & blue_mask) >> bs
        out_index += 3

Image.frombytes("RGB", (width, height), bytes(rgb)).save(target)
print(target, width, height)
