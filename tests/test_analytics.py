"""Unit tests for src/analytics.py and src/export_utils.py"""

import unittest
import numpy as np
import pandas as pd

from src.analytics import (
    calculate_detection_summary,
    create_detections_dataframe,
    create_class_distribution_df,
)
from src.export_utils import (
    detections_to_csv_bytes,
    image_to_png_bytes,
    generate_timestamped_filename,
)


class TestAnalyticsAndExport(unittest.TestCase):
    def setUp(self):
        self.sample_detections = [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.85,
                "bbox": (10, 20, 50, 80),
            },
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox": (60, 20, 100, 80),
            },
            {
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.75,
                "bbox": (120, 40, 200, 110),
            },
        ]

    def test_summary_empty(self):
        summary = calculate_detection_summary([])
        self.assertEqual(summary["total_objects"], 0)
        self.assertEqual(summary["unique_classes"], 0)
        self.assertEqual(summary["avg_confidence"], 0.0)

    def test_summary_with_detections(self):
        summary = calculate_detection_summary(self.sample_detections)
        self.assertEqual(summary["total_objects"], 3)
        self.assertEqual(summary["unique_classes"], 2)
        self.assertEqual(summary["class_counts"]["person"], 2)
        self.assertEqual(summary["class_counts"]["car"], 1)
        self.assertAlmostEqual(summary["avg_confidence"], (0.85 + 0.95 + 0.75) / 3, places=3)
        self.assertEqual(summary["min_confidence"], 0.75)
        self.assertEqual(summary["max_confidence"], 0.95)

    def test_create_detections_dataframe(self):
        df = create_detections_dataframe(self.sample_detections)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("Class", df.columns)
        self.assertIn("Confidence", df.columns)
        self.assertIn("Width", df.columns)
        self.assertIn("Height", df.columns)
        # Check computed width for first box: 50 - 10 = 40
        self.assertEqual(df.iloc[0]["Width"], 40)
        self.assertEqual(df.iloc[0]["Height"], 60)

    def test_create_class_distribution_df(self):
        df = create_class_distribution_df(self.sample_detections)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["Class"], "person")
        self.assertEqual(df.iloc[0]["Count"], 2)

    def test_export_csv_bytes(self):
        csv_bytes = detections_to_csv_bytes(self.sample_detections, "test_pic.jpg")
        self.assertIsInstance(csv_bytes, bytes)
        csv_str = csv_bytes.decode("utf-8")
        self.assertIn("person", csv_str)
        self.assertIn("car", csv_str)
        self.assertIn("test_pic.jpg", csv_str)

    def test_export_png_bytes(self):
        dummy_rgb = np.full((50, 50, 3), 128, dtype=np.uint8)
        png_bytes = image_to_png_bytes(dummy_rgb)
        self.assertIsInstance(png_bytes, bytes)
        # PNG signature check (first 8 bytes)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_timestamped_filename(self):
        name = generate_timestamped_filename("report", "csv")
        self.assertTrue(name.startswith("report_"))
        self.assertTrue(name.endswith(".csv"))


if __name__ == "__main__":
    unittest.main()
