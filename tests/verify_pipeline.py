"""End-to-End Pipeline Verification Script

Verifies:
1. All core library imports.
2. YOLO model loading and automatic weight download.
3. Real inference execution.
4. OpenCV annotation rendering.
5. CSV report and PNG export generation.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2
from PIL import Image

print("=" * 60)
print("1. Checking Imports...")
try:
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO
    import streamlit
    import pandas as pd
    from src.image_processor import bytes_to_numpy_rgb, get_image_metadata
    from src.detector import YOLODetector
    from src.analytics import calculate_detection_summary, create_detections_dataframe
    from src.export_utils import detections_to_csv_bytes, image_to_png_bytes

    print(f"   [OK] Torch version: {torch.__version__}")
    print(f"   [OK] Ultralytics version: {ultralytics.__version__}")
    print(f"   [OK] OpenCV version: {cv2.__version__}")
    print(f"   [OK] Pandas version: {pd.__version__}")
except Exception as e:
    print(f"   [FAIL] Import error: {e}")
    sys.exit(1)

print("\n2. Initializing YOLO Detector...")
try:
    detector = YOLODetector(model_name="yolo11n.pt", device="cpu")
    print(f"   [OK] Loaded model: {detector.model_name}")
    print(f"   [OK] Total classes in taxonomy: {len(detector.classes_map)}")
except Exception as e:
    print(f"   [FAIL] Model load error: {e}")
    sys.exit(1)

print("\n3. Generating Verification Test Image...")
# Create a test image with high-contrast colored regions
test_img = np.full((480, 640, 3), 230, dtype=np.uint8)
# Add geometric patterns
cv2.rectangle(test_img, (50, 50), (250, 400), (0, 100, 255), -1)
cv2.putText(test_img, "VisionGuard Test", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
print("   [OK] Synthetic test image prepared (480x640x3)")

print("\n4. Running Inference...")
start_time = time.perf_counter()
result = detector.detect(test_img, conf_threshold=0.20, iou_threshold=0.45)
elapsed_ms = (time.perf_counter() - start_time) * 1000

print(f"   [OK] Total execution time: {elapsed_ms:.1f} ms")
print(f"   [OK] Model forward pass time: {result['inference_time_ms']:.1f} ms")
print(f"   [OK] Objects detected: {len(result['detections'])}")

print("\n5. Testing Analytics & Export...")
summary = calculate_detection_summary(result["detections"])
df = create_detections_dataframe(result["detections"])
csv_data = detections_to_csv_bytes(result["detections"], "test.png")
png_data = image_to_png_bytes(result["annotated_image_rgb"])

print(f"   [OK] Analytics summary generated: {summary['total_objects']} objects")
print(f"   [OK] CSV export size: {len(csv_data)} bytes")
print(f"   [OK] Annotated PNG export size: {len(png_data)} bytes")

print("\n" + "=" * 60)
print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
print("=" * 60)
