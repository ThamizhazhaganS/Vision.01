"""Unit tests for src/image_processor.py"""

import unittest
import io
import numpy as np
from PIL import Image

from src.image_processor import (
    validate_image_bytes,
    bytes_to_numpy_rgb,
    numpy_rgb_to_bgr,
    numpy_bgr_to_rgb,
    get_image_metadata,
)


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 100x100 red test image in PNG format
        self.test_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        self.test_img.save(buf, format="PNG")
        self.valid_png_bytes = buf.getvalue()

    def test_validate_valid_image(self):
        is_valid, msg = validate_image_bytes(self.valid_png_bytes, "test.png")
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_validate_empty_bytes(self):
        is_valid, msg = validate_image_bytes(b"", "test.png")
        self.assertFalse(is_valid)
        self.assertIn("empty", msg.lower())

    def test_validate_corrupted_bytes(self):
        corrupted = b"NOT_AN_IMAGE_HEADER_DATA_12345"
        is_valid, msg = validate_image_bytes(corrupted, "test.png")
        self.assertFalse(is_valid)
        self.assertIn("corrupt", msg.lower())

    def test_validate_unsupported_extension(self):
        is_valid, msg = validate_image_bytes(self.valid_png_bytes, "test.txt")
        self.assertFalse(is_valid)
        self.assertIn("unsupported", msg.lower())

    def test_bytes_to_numpy_rgb(self):
        arr = bytes_to_numpy_rgb(self.valid_png_bytes)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (100, 100, 3))
        # Top-left pixel should be pure red (255, 0, 0)
        np.testing.assert_array_equal(arr[0, 0], [255, 0, 0])

    def test_color_space_conversions(self):
        # Red in RGB: (255, 0, 0)
        rgb_arr = np.array([[[255, 0, 0]]], dtype=np.uint8)
        # Red in BGR should be (0, 0, 255)
        bgr_arr = numpy_rgb_to_bgr(rgb_arr)
        self.assertEqual(bgr_arr[0, 0, 0], 0)    # Blue
        self.assertEqual(bgr_arr[0, 0, 1], 0)    # Green
        self.assertEqual(bgr_arr[0, 0, 2], 255)  # Red

        # Convert back to RGB
        recovered_rgb = numpy_bgr_to_rgb(bgr_arr)
        np.testing.assert_array_equal(recovered_rgb, rgb_arr)

    def test_get_image_metadata(self):
        arr = np.zeros((120, 160, 3), dtype=np.uint8)
        meta = get_image_metadata(arr)
        self.assertEqual(meta["height"], 120)
        self.assertEqual(meta["width"], 160)
        self.assertEqual(meta["channels"], 3)
        self.assertEqual(meta["aspect_ratio"], "4:3")
        self.assertGreater(meta["memory_mb"], 0)


if __name__ == "__main__":
    unittest.main()
