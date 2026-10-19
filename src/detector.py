"""VisionGuard AI - YOLO Object Detection Engine

This module wraps Ultralytics YOLO models (YOLO11 / YOLOv8) to provide:
1. Deterministic CPU-friendly model loading and weight caching.
2. Real-time inference with configurable Confidence and IoU (Intersection-over-Union) thresholds.
3. Class filtering and structured detection extraction.
4. Professional OpenCV bounding-box rendering with high-contrast text badges.
5. Accurate inference latency timing (distinguishing model forward-pass time from post-processing).

COMPUTER VISION CONCEPTS:
- Object Detection vs Image Classification:
  Classification assigns a single label to an entire image ("a dog").
  Detection predicts *where* each object is (bounding box coords [x1, y1, x2, y2])
  and *what* it is (class label + confidence score) for multiple objects simultaneously.
- Non-Maximum Suppression (NMS):
  YOLO predicts tens of thousands of candidate bounding boxes. NMS filters overlapping boxes
  by calculating their IoU (Intersection over Union). Boxes with IoU > iou_threshold
  belonging to the same class are suppressed, leaving only the most confident detection.
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# A curated, modern 20-color palette for distinct visual separation in annotations
COLOR_PALETTE = [
    (0, 165, 255),    # Vibrant Orange
    (34, 197, 94),    # Emerald Green
    (59, 130, 246),   # Royal Blue
    (239, 68, 68),    # Crimson Red
    (168, 85, 247),   # Purple
    (234, 179, 8),    # Amber
    (6, 182, 212),    # Cyan
    (244, 63, 94),    # Rose Pink
    (16, 185, 129),   # Mint
    (249, 115, 22),   # Bright Orange
    (139, 92, 246),   # Violet
    (14, 165, 233),   # Sky Blue
    (132, 204, 22),   # Lime
    (236, 72, 153),   # Magenta
    (20, 184, 166),   # Teal
    (245, 158, 11),   # Gold
    (99, 102, 241),   # Indigo
    (225, 29, 72),    # Ruby
    (4, 120, 87),     # Forest Green
    (124, 58, 237),   # Deep Purple
]


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Returns a deterministic RGB color for a given class ID."""
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]


class YOLODetector:
    """Wrapper around Ultralytics YOLO models for modular inference and visualization."""

    def __init__(self, model_name: str = "yolo11n.pt", device: str = "cpu"):
        """Initializes the detector and loads model weights.

        Args:
            model_name: Path or identifier of the pretrained weights (e.g. 'yolo11n.pt', 'yolov8n.pt').
            device: Computing device ('cpu', 'cuda', etc.). Defaults to CPU.
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "The 'ultralytics' package is not installed. Please install it via 'pip install ultralytics'."
            )

        self.model_name = model_name
        self.device = device
        self.model = None
        self.classes_map: Dict[int, str] = {}
        self._load_model()

    def _load_model(self) -> None:
        """Loads weights with graceful fallback from YOLO11 to YOLOv8 if needed."""
        try:
            self.model = YOLO(self.model_name)
            self.classes_map = self.model.names if hasattr(self.model, "names") else {}
        except Exception as e:
            # Fallback to yolov8n.pt if yolo11n fails to download or load
            if "11" in self.model_name:
                fallback_name = "yolov8n.pt"
                try:
                    self.model = YOLO(fallback_name)
                    self.model_name = fallback_name
                    self.classes_map = self.model.names if hasattr(self.model, "names") else {}
                    return
                except Exception as inner_e:
                    raise RuntimeError(
                        f"Failed to load primary model '{self.model_name}' and fallback '{fallback_name}': {str(inner_e)}"
                    ) from inner_e
            raise RuntimeError(f"Error loading YOLO model '{self.model_name}': {str(e)}") from e

    def detect(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Runs object detection inference on an RGB NumPy image.

        Args:
            image_rgb: Input image array of shape (H, W, 3) in RGB format.
            conf_threshold: Minimum confidence score (0.0 to 1.0) to retain a detection.
            iou_threshold: Intersection over Union threshold for Non-Maximum Suppression.
            target_classes: Optional list of class names to filter by (e.g. ['person', 'car']).

        Returns:
            dict containing:
                - detections: list of detection dicts (class_id, class_name, confidence, bbox)
                - annotated_image_rgb: RGB image with bounding boxes drawn
                - inference_time_ms: forward pass time in milliseconds
                - total_time_ms: total pipeline time in milliseconds
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized.")

        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("Input image is empty.")

        start_total = time.perf_counter()

        # Run inference using Ultralytics
        # Note: Ultralytics handles RGB input natively or converts it internally
        results = self.model.predict(
            source=image_rgb,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        inference_time_ms = 0.0
        detections: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            result = results[0]

            # Extract native speed metrics if reported by Ultralytics
            if hasattr(result, "speed") and isinstance(result.speed, dict):
                inference_time_ms = float(result.speed.get("inference", 0.0))

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = self.classes_map.get(cls_id, f"class_{cls_id}")

                    # Apply optional target class filter
                    if target_classes and cls_name not in target_classes:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = [int(coord) for coord in xyxy]

                    detections.append(
                        {
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                            "bbox": (x1, y1, x2, y2),
                        }
                    )

        # Draw custom annotations on a fresh copy of the image
        annotated_image_rgb = self.draw_annotations(image_rgb, detections)

        total_time_ms = round((time.perf_counter() - start_total) * 1000, 2)
        if inference_time_ms == 0.0:
            inference_time_ms = total_time_ms

        return {
            "detections": detections,
            "annotated_image_rgb": annotated_image_rgb,
            "inference_time_ms": round(inference_time_ms, 2),
            "total_time_ms": total_time_ms,
        }

    def draw_annotations(
        self,
        image_rgb: np.ndarray,
        detections: List[Dict[str, Any]],
        box_thickness: int = 2,
    ) -> np.ndarray:
        """Renders bounding boxes and label badges onto a copy of the RGB image.

        Args:
            image_rgb: Base image array (RGB).
            detections: List of detection records.
            box_thickness: Line thickness for the bounding boxes.

        Returns:
            np.ndarray: Annotated RGB image array.
        """
        output = image_rgb.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_id = det["class_id"]
            class_name = det["class_name"]
            confidence = det["confidence"]

            color = get_class_color(class_id)
            label = f"{class_name} {confidence * 100:.1f}%"

            # Draw outer rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)

            # Calculate text size for background badge
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            text_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

            # Draw solid background badge behind text for high legibility
            badge_y1 = max(0, y1 - text_h - baseline - 6)
            badge_y2 = y1
            badge_x2 = min(output.shape[1], x1 + text_w + 8)

            cv2.rectangle(output, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

            # Text is rendered in white (or black if color is very light)
            cv2.putText(
                output,
                label,
                (x1 + 4, y1 - baseline - 3 if y1 - text_h - baseline - 6 >= 0 else y1 + text_h + 3),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        return output
