import unittest
from pathlib import Path

from capstone_demo.uploads import UploadError, validate_upload
from helpers import jpeg_bytes, png_bytes, png_header


class UploadValidationTests(unittest.TestCase):
    def test_png_is_sniffed_with_dimensions(self):
        info = validate_upload(png_bytes(4, 3), max_bytes=1024)
        self.assertEqual((info.kind, info.extension), ("png", ".png"))
        self.assertEqual((info.width, info.height), (4, 3))
        self.assertEqual(len(info.sha256), 64)

    def test_jpeg_is_sniffed_with_dimensions(self):
        info = validate_upload(jpeg_bytes(5, 7), max_bytes=1024)
        self.assertEqual((info.kind, info.width, info.height), ("jpeg", 5, 7))

    def test_pdf_is_accepted_without_dimensions(self):
        info = validate_upload((Path(__file__).parent / "fixtures/fixture-one.pdf").read_bytes(), max_bytes=10000)
        self.assertEqual((info.kind, info.extension), ("pdf", ".pdf"))

    def test_multi_page_pdf_rejected_before_starting_gpu(self):
        data = (Path(__file__).parent / "fixtures/fixture-two.pdf").read_bytes()
        with self.assertRaisesRegex(UploadError, "one page"):
            validate_upload(data, max_bytes=10000)

    def test_rejects_empty_and_oversize(self):
        with self.assertRaises(UploadError):
            validate_upload(b"", max_bytes=10)
        with self.assertRaisesRegex(UploadError, "too large"):
            validate_upload(png_bytes(), max_bytes=10)

    def test_rejects_html_svg_and_unknown(self):
        for payload in (
            b"<html><script>alert(1)</script>",
            b'<svg xmlns="http://www.w3.org/2000/svg"/>',
            b"GIF89a....",
        ):
            with self.assertRaisesRegex(UploadError, "Unsupported"):
                validate_upload(payload, max_bytes=1024)

    def test_rejects_decompression_bomb_dimensions(self):
        with self.assertRaisesRegex(UploadError, "dimensions"):
            validate_upload(png_header(20000, 20000) + b"rest", max_bytes=10**6)

    def test_rejects_truncated_png_and_jpeg(self):
        with self.assertRaises(UploadError):
            validate_upload(png_bytes()[:20], max_bytes=1024)
        with self.assertRaises(UploadError):
            validate_upload(b"\xff\xd8\xff\xe0\x00", max_bytes=1024)

    def test_rejects_pdf_without_trailer(self):
        with self.assertRaisesRegex(UploadError, "PDF"):
            validate_upload(b"%PDF-1.7\nno end", max_bytes=1024)


if __name__ == "__main__":
    unittest.main()
