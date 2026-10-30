"""Validate uploaded page files by content, never by client-supplied name or type."""

from dataclasses import dataclass
import hashlib
import struct

MAX_SIDE = 10000
MAX_PIXELS = 50_000_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"
JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


class UploadError(ValueError):
    pass


@dataclass(frozen=True)
class UploadInfo:
    kind: str
    extension: str
    content_type: str
    size: int
    sha256: str
    width: int | None = None
    height: int | None = None


def _check_dimensions(width, height):
    if width < 1 or height < 1 or max(width, height) > MAX_SIDE or width * height > MAX_PIXELS:
        raise UploadError(f"Unsupported image dimensions {width}x{height}")


def _png(data):
    if len(data) < 33 or data[12:16] != b"IHDR" or data[8:12] != b"\x00\x00\x00\x0d":
        raise UploadError("Truncated or malformed PNG")
    width, height = struct.unpack(">II", data[16:24])
    _check_dimensions(width, height)
    if not data.endswith(PNG_IEND):
        raise UploadError("Truncated PNG (missing IEND)")
    return width, height


def _jpeg(data):
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            raise UploadError("Malformed JPEG marker")
        marker = data[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue
        if marker in (0x01, *range(0xD0, 0xD8)):
            offset += 2
            continue
        if marker in (0xD9, 0xDA):
            break
        (length,) = struct.unpack(">H", data[offset + 2 : offset + 4])
        if length < 2 or offset + 2 + length > len(data):
            break
        if marker in JPEG_SOF:
            if length < 7:
                break
            height, width = struct.unpack(">HH", data[offset + 5 : offset + 9])
            _check_dimensions(width, height)
            return width, height
        offset += 2 + length
    raise UploadError("Truncated or malformed JPEG (no frame header)")


def validate_upload(data: bytes, *, max_bytes: int) -> UploadInfo:
    if not data:
        raise UploadError("Empty upload")
    if len(data) > max_bytes:
        raise UploadError(f"Upload too large ({len(data)} > {max_bytes} bytes)")
    digest = hashlib.sha256(data).hexdigest()
    if data.startswith(PNG_SIGNATURE):
        width, height = _png(data)
        return UploadInfo("png", ".png", "image/png", len(data), digest, width, height)
    if data.startswith(b"\xff\xd8\xff"):
        width, height = _jpeg(data)
        return UploadInfo("jpeg", ".jpg", "image/jpeg", len(data), digest, width, height)
    if data.startswith(b"%PDF-"):
        if b"%%EOF" not in data[-2048:]:
            raise UploadError("Truncated PDF (missing %%EOF)")
        return UploadInfo("pdf", ".pdf", "application/pdf", len(data), digest)
    raise UploadError("Unsupported file type: upload a PNG, JPEG, or single-page PDF")
