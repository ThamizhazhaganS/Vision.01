"""VisionGuard AI - Export Utilities

This module handles serialization and export formats:
1. Converting detection metadata into CSV bytes ready for user download.
2. Encoding annotated RGB NumPy images into PNG / JPEG bytes.
3. Safe filename generation with ISO timestamps.
"""

from typing import List, Dict, Any
from datetime import datetime
import io
import pandas as pd
from PIL import Image
import numpy as np


def generate_timestamped_filename(prefix: str, extension: str) -> str:
    """Generates a standardized filename containing an ISO timestamp.

    Args:
        prefix: Base name (e.g. 'visionguard_annotated', 'detection_report').
        extension: File extension without leading dot (e.g. 'png', 'csv').

    Returns:
        str: Timestamped filename like 'visionguard_annotated_20261019_143000.png'.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_ext = extension.lstrip(".")
    return f"{prefix}_{timestamp}.{clean_ext}"


def detections_to_csv_bytes(
    detections: List[Dict[str, Any]],
    source_filename: str = "uploaded_image.jpg",
) -> bytes:
    """Serializes detection records into CSV byte buffer.

    Args:
        detections: List of detection dictionaries.
        source_filename: Name of the original source image.

    Returns:
        bytes: Encoded UTF-8 CSV content.
    """
    rows = []
    for idx, d in enumerate(detections, start=1):
        x1, y1, x2, y2 = d["bbox"]
        width = x2 - x1
        height = y2 - y1
        conf = d["confidence"]

        rows.append(
            {
                "detection_id": idx,
                "source_file": source_filename,
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "confidence": conf,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_width": width,
                "bbox_height": height,
            }
        )

    df = pd.DataFrame(rows)
    # If empty, supply expected column headers
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "detection_id",
                "source_file",
                "class_id",
                "class_name",
                "confidence",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "bbox_width",
                "bbox_height",
            ]
        )

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode("utf-8")


def image_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    """Encodes a NumPy RGB array into PNG byte buffer for direct browser download.

    Args:
        image_rgb: NumPy array with shape (H, W, 3) in RGB format.

    Returns:
        bytes: Binary PNG representation.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Cannot encode an empty image to PNG.")

    pil_img = Image.fromarray(image_rgb)
    byte_buf = io.BytesIO()
    pil_img.save(byte_buf, format="PNG", optimize=True)
    return byte_buf.getvalue()
