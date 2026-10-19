"""VisionGuard AI - Image Processing Module

This module handles:
1. Image byte validation (detecting truncated/corrupt/unsupported files).
2. Decoding image bytes into standard NumPy RGB / BGR arrays.
3. Safe color-space transformations (OpenCV BGR <-> Web/Pillow RGB).
4. Image metadata extraction (width, height, aspect ratio, channel depth).

COMPUTER VISION NOTE:
OpenCV traditionally decodes images in BGR (Blue-Green-Red) channel order
due to legacy camera sensors and early PC frame grabbers.
Conversely, Pillow, modern web browsers, and Streamlit expect RGB order.
Passing BGR directly to Streamlit causes images to look blue-shifted/inverted.
This module guarantees explicit, deterministic color management.
"""

from typing import Tuple, Dict, Any, Optional
import io
import numpy as np
from PIL import Image
import cv2

# Maximum allowed image upload size: 25 Megabytes
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024

# Allowed file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def validate_image_bytes(data: bytes, filename: str = "") -> Tuple[bool, str]:
    """Validates raw image bytes for size, non-emptiness, and decodability.

    Args:
        data: The raw binary content of the uploaded file.
        filename: Optional filename for extension verification.

    Returns:
        (is_valid, error_message): A boolean indicating validity and an error string if invalid.
    """
    if not data or len(data) == 0:
        return False, "The uploaded file is empty (0 bytes)."

    if len(data) > MAX_IMAGE_SIZE_BYTES:
        size_mb = len(data) / (1024 * 1024)
        return False, f"File size ({size_mb:.2f} MB) exceeds the maximum limit of 25 MB."

    if filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext and ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

    # Verify Pillow can open and verify the integrity of the image headers
    try:
        with io.BytesIO(data) as buf:
            with Image.open(buf) as img:
                img.verify()  # Check for header corruption without loading full raster
    except Exception as e:
        return False, f"Corrupted or invalid image file: {str(e)}"

    return True, ""


def bytes_to_numpy_rgb(data: bytes) -> np.ndarray:
    """Decodes raw image bytes directly into a 3-channel NumPy RGB array.

    Args:
        data: Raw image file bytes.

    Returns:
        np.ndarray: Image array with shape (H, W, 3) in uint8 RGB format.

    Raises:
        ValueError: If the bytes cannot be decoded into a valid image.
    """
    try:
        with io.BytesIO(data) as buf:
            with Image.open(buf) as pil_img:
                # Convert RGBA or grayscale to standard 3-channel RGB
                converted = pil_img.convert("RGB")
                return np.array(converted, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Failed to decode image bytes into RGB array: {str(e)}") from e


def numpy_rgb_to_bgr(rgb_img: np.ndarray) -> np.ndarray:
    """Converts a NumPy array from RGB color space to OpenCV BGR color space.

    Args:
        rgb_img: Image array with shape (H, W, 3) in RGB format.

    Returns:
        np.ndarray: Image array in BGR format.
    """
    if rgb_img is None or rgb_img.size == 0:
        raise ValueError("Cannot convert an empty image array.")
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)


def numpy_bgr_to_rgb(bgr_img: np.ndarray) -> np.ndarray:
    """Converts a NumPy array from OpenCV BGR color space to RGB color space.

    Args:
        bgr_img: Image array with shape (H, W, 3) in BGR format.

    Returns:
        np.ndarray: Image array in RGB format.
    """
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("Cannot convert an empty image array.")
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def get_image_metadata(img: np.ndarray) -> Dict[str, Any]:
    """Extracts fundamental geometric and memory metadata from an image array.

    Args:
        img: Image array with shape (H, W, C) or (H, W).

    Returns:
        dict: Containing height, width, channels, aspect_ratio, and memory_mb.
    """
    if img is None or img.size == 0:
        return {
            "height": 0,
            "width": 0,
            "channels": 0,
            "aspect_ratio": "0:0",
            "memory_mb": 0.0,
        }

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    gcd = np.gcd(w, h) if w > 0 and h > 0 else 1
    aspect_ratio = f"{w // gcd}:{h // gcd}" if gcd > 0 else f"{w}:{h}"
    memory_mb = round(img.nbytes / (1024 * 1024), 3)

    return {
        "height": h,
        "width": w,
        "channels": channels,
        "aspect_ratio": aspect_ratio,
        "memory_mb": memory_mb,
    }
