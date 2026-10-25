"""Byte-level helpers for the scripted model's tests: a picture made byte-unique
without changing its pixels, the way the load generator makes every upload its
own."""

import struct
import zlib


def png_with_text_chunk(png: bytes, text: bytes) -> bytes:
    """A PNG with a tEXt chunk before IEND: the same picture, different bytes."""
    data = b"Comment\0" + text
    chunk = struct.pack(">I", len(data)) + b"tEXt" + data
    chunk += struct.pack(">I", zlib.crc32(b"tEXt" + data))
    iend = png.rindex(b"IEND") - 4
    return png[:iend] + chunk + png[iend:]


def jpeg_with_comment(jpeg: bytes, text: bytes) -> bytes:
    """A JPEG with a COM segment after SOI."""
    return jpeg[:2] + b"\xff\xfe" + struct.pack(">H", len(text) + 2) + text + jpeg[2:]
