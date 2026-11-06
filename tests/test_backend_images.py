import io
import unittest

from PIL import Image

from backend.errors import InvalidImage
from backend.images import assess_front_view, validate_image
from tests.backend_fakes import blank_png_bytes, mirrored_photo_bytes, photo_bytes, side_view_bytes

MB = 1024 * 1024


def validate(data, **kwargs):
    return validate_image(data, max_bytes=kwargs.pop("max_bytes", 10 * MB),
                          max_pixels=kwargs.pop("max_pixels", 40_000_000), **kwargs)


def encode(format, size=(200, 120), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size, "white").save(buffer, format=format)
    return buffer.getvalue()


class ValidateImageTests(unittest.TestCase):
    def test_accepts_png_and_jpeg(self):
        jpeg = validate(photo_bytes())
        self.assertEqual((jpeg.media_type, jpeg.width, jpeg.height, jpeg.extension), ("image/jpeg", 1024, 768, ".jpg"))
        self.assertEqual(validate(encode("PNG")).media_type, "image/png")

    def test_rejects_other_formats_and_garbage(self):
        for data in [encode("GIF"), encode("BMP"), b"<svg xmlns='http://www.w3.org/2000/svg'/>", b"hello",
                     b"%PDF-1.4"]:
            with self.subTest(data=data[:8]):
                with self.assertRaises(InvalidImage):
                    validate(data)

    def test_webp_only_when_allowed_and_converted_to_png(self):
        webp = encode("WEBP")
        with self.assertRaises(InvalidImage):
            validate(webp)
        converted = validate(webp, allow_webp=True)
        self.assertEqual(converted.media_type, "image/png")
        self.assertTrue(converted.data.startswith(b"\x89PNG"))

    def test_rejects_empty_truncated_tiny_and_oversized(self):
        cases = {"empty_image": (b"", {}), "corrupt_image": (photo_bytes()[:2000], {}),
                 "image_too_small": (encode("PNG", (40, 40)), {}),
                 "image_too_large": (photo_bytes(), {"max_bytes": 1000})}
        for code, (data, limits) in cases.items():
            with self.subTest(code=code):
                with self.assertRaises(InvalidImage) as caught:
                    validate(data, **limits)
                self.assertEqual(caught.exception.code, code)

    def test_rejects_too_many_pixels_before_decoding(self):
        with self.assertRaises(InvalidImage) as caught:
            validate(encode("PNG", (5000, 5000)), max_pixels=10_000_000)
        self.assertEqual(caught.exception.code, "image_too_large")


class FrontViewTests(unittest.TestCase):
    def test_product_photos_are_front_views(self):
        for data in (photo_bytes(), mirrored_photo_bytes()):
            assessment = assess_front_view(validate(data))
            self.assertTrue(assessment.suitable)
            self.assertEqual(assessment.openings, 2)

    def test_side_views_and_blank_images_are_not(self):
        self.assertFalse(assess_front_view(validate(side_view_bytes())).suitable)
        self.assertFalse(assess_front_view(validate(blank_png_bytes())).suitable)

    def test_busy_backgrounds_are_not_front_views(self):
        import cv2
        import numpy as np
        photo = cv2.imdecode(np.frombuffer(photo_bytes(), np.uint8), cv2.IMREAD_COLOR)
        noisy = photo.copy()
        noisy[:40] = (40, 90, 160)
        noisy[-40:] = (40, 90, 160)
        self.assertFalse(assess_front_view(validate(cv2.imencode(".jpg", noisy)[1].tobytes())).suitable)

    def test_assessment_is_deterministic(self):
        source = validate(photo_bytes())
        self.assertEqual(assess_front_view(source), assess_front_view(source))


if __name__ == "__main__":
    unittest.main()
