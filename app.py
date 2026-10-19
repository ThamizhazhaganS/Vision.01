"""VisionGuard AI - Real-Time Object Detection System

Main Streamlit Application Entrypoint.
Provides a modern computer vision dashboard for image upload,
real-time YOLO inference, OpenCV annotation, performance analytics, and export.
"""

import sys
from pathlib import Path
from typing import Optional, List
import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure root directory is on Python path for modular src imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.image_processor import (
    validate_image_bytes,
    bytes_to_numpy_rgb,
    get_image_metadata,
)
from src.detector import YOLODetector
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

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionGuard AI — Real-Time Object Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Modern UI Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Typography & Background adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        padding: 1.5rem 1.8rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #60A5FA 0%, #38BDF8 50%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        margin-bottom: 0.6rem;
    }
    .hero-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.25);
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(96, 165, 250, 0.3);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 0.2rem;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Image Display Containers */
    .image-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.75rem;
    }

    /* Clean divider */
    hr {
        border-color: rgba(255, 255, 255, 0.08);
        margin: 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_detector(model_name: str) -> YOLODetector:
    """Loads and caches the YOLO detector model to prevent redundant reloads."""
    return YOLODetector(model_name=model_name, device="cpu")


# Helper to load demo image for immediate exploration
def get_demo_image_bytes() -> bytes:
    """Loads a real demo image from assets/demo.jpg or generates a fallback image."""
    demo_path = ROOT_DIR / "assets" / "demo.jpg"
    if demo_path.exists():
        with open(demo_path, "rb") as f:
            return f.read()

    # Fallback synthetic image if file is absent
    import cv2
    img = np.full((400, 600, 3), 240, dtype=np.uint8)
    img[:250, :] = [235, 206, 135]
    img[250:, :] = [100, 180, 100]
    cv2.rectangle(img, (120, 180), (320, 290), (40, 80, 200), -1)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    # --- HERO HEADER ---
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">VisionGuard AI</h1>
            <div class="hero-subtitle">Real-Time Object Detection System</div>
            <p style="color: #CBD5E1; font-size: 0.95rem; margin: 0; max-width: 850px;">
                High-performance object detection suite powered by state-of-the-art YOLO architectures, 
                OpenCV image processing, and real-time analytical reporting.
            </p>
            <div class="hero-badges">
                <span class="badge">YOLO Vision Engine</span>
                <span class="badge">OpenCV 4.x Pipeline</span>
                <span class="badge badge-purple">CPU-Optimized Inference</span>
                <span class="badge">COCO 80-Class Taxonomy</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.markdown("### ⚙️ Detection Settings")
        
        # Model Selection
        model_choice = st.selectbox(
            "YOLO Architecture",
            options=["yolo11n.pt", "yolov8n.pt"],
            index=0,
            help="Ultralytics pretrained weights. 'yolo11n.pt' is the latest compact nano model optimized for speed.",
        )

        st.markdown("---")
        st.markdown("#### 🎯 Inference Parameters")
        
        # Confidence Threshold
        conf_thresh = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=1.00,
            value=0.25,
            step=0.05,
            help="Minimum confidence score required for an object proposal to be retained.",
        )

        # IoU Threshold for NMS
        iou_thresh = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Intersection-over-Union threshold for Non-Maximum Suppression (filtering redundant overlapping boxes).",
        )

        # Visual Preferences
        st.markdown("---")
        st.markdown("#### 🎨 Display Preferences")
        box_thickness = st.slider("Bounding Box Thickness", min_value=1, max_value=5, value=2)

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #94A3B8;">
                <strong>Architecture:</strong> Single-Stage Detector<br>
                <strong>Execution Mode:</strong> CPU Float32<br>
                <strong>Color Pipeline:</strong> PIL RGB ↔ OpenCV BGR
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- MODEL INITIALIZATION ---
    detector = None
    try:
        with st.spinner(f"Loading {model_choice} model weights..."):
            detector = load_cached_detector(model_choice)
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {str(e)}")
        st.info("Ensure an active internet connection is available to download the pretrained weights on first run.")
        return

    # --- INPUT SELECTION ---
    st.markdown("### 📤 Upload Image")
    
    col_upload, col_demo = st.columns([3, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Select an image file (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Upload an image from your computer to run real-time object detection.",
        )

    use_sample = False
    with col_demo:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("✨ Load Demo Image", use_container_width=True, help="Load an immediate built-in test image."):
            use_sample = True

    # Obtain image bytes from upload or demo
    raw_bytes: Optional[bytes] = None
    filename: str = "uploaded_image.jpg"

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
    elif use_sample:
        raw_bytes = get_demo_image_bytes()
        filename = "demo.jpg"

    if raw_bytes is None:
        # Prompt user to provide input
        st.info("👆 Please upload an image or click **'Load Demo Image'** to begin object detection.")
        
        # Educational CV architecture guide displayed while waiting for user input
        with st.expander("📚 How VisionGuard AI's Computer Vision Pipeline Works", expanded=True):
            st.markdown(
                """
                1. **Image Ingestion & Preprocessing:**
                   The raw binary bytes are validated to prevent header corruption. The image is decoded into a 3-channel RGB NumPy array.
                2. **Color Space Management:**
                   OpenCV natively works with BGR channel ordering. VisionGuard strictly enforces explicit transformations between BGR and RGB to guarantee no channel inversions occur.
                3. **Deep Feature Extraction & Prediction:**
                   The input tensor is passed through the YOLO backbone (CSPNet/C3k2) and neck (PANet) to generate bounding box coordinates $(x_1, y_1, x_2, y_2)$ and class confidence distributions.
                4. **Non-Maximum Suppression (NMS):**
                   Thousands of raw candidates are filtered using the configured **IoU threshold** to remove duplicate overlapping predictions.
                5. **Annotation & Metric Extraction:**
                   OpenCV renders high-contrast bounding boxes with class badges directly onto the frame, while latency metrics are gathered for analytical reporting.
                """
            )
        return

    # --- VALIDATE & DECODE IMAGE ---
    is_valid, error_msg = validate_image_bytes(raw_bytes, filename=filename)
    if not is_valid:
        st.error(f"❌ Image Validation Error: {error_msg}")
        return

    try:
        image_rgb = bytes_to_numpy_rgb(raw_bytes)
    except Exception as e:
        st.error(f"❌ Failed to decode image: {str(e)}")
        return

    meta = get_image_metadata(image_rgb)

    # --- CLASS FILTERING SELECTION ---
    all_class_names = sorted(list(detector.classes_map.values())) if detector.classes_map else []
    selected_classes = None
    if all_class_names:
        with st.sidebar:
            st.markdown("---")
            st.markdown("#### 🔍 Filter Specific Classes")
            target_filter = st.multiselect(
                "Detect Only Selected Classes",
                options=all_class_names,
                default=[],
                help="Leave empty to detect all 80 COCO classes.",
            )
            if target_filter:
                selected_classes = target_filter

    # --- INFERENCE EXECUTION ---
    with st.spinner("🔍 Running YOLO neural inference..."):
        detection_result = detector.detect(
            image_rgb=image_rgb,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            target_classes=selected_classes,
        )

    detections = detection_result["detections"]
    annotated_rgb = detection_result["annotated_image_rgb"]
    inference_time = detection_result["inference_time_ms"]
    total_time = detection_result["total_time_ms"]

    # Compute summary analytics
    summary = calculate_detection_summary(detections)

    # --- KPI METRICS ROW ---
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Objects Detected</div>
                <div class="metric-value">{summary["total_objects"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Unique Classes</div>
                <div class="metric-value">{summary["unique_classes"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Confidence</div>
                <div class="metric-value">{summary["avg_confidence"] * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Forward Pass</div>
                <div class="metric-value">{inference_time:.1f} <span style="font-size: 1rem; color: #94A3B8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Latency</div>
                <div class="metric-value">{total_time:.1f} <span style="font-size: 1rem; color: #94A3B8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # --- SIDE-BY-SIDE VISUAL COMPARISON ---
    col_orig, col_annotated = st.columns(2)

    with col_orig:
        st.markdown("#### 📷 Original Image")
        st.image(image_rgb, use_container_width=True)
        st.caption(
            f"Dimensions: {meta['width']}×{meta['height']} px | Ratio: {meta['aspect_ratio']} | Memory: {meta['memory_mb']} MB"
        )

    with col_annotated:
        st.markdown("#### 🎯 Annotated Detections")
        st.image(annotated_rgb, use_container_width=True)
        st.caption(
            f"Active Model: {model_choice} | Threshold: {conf_thresh * 100:.0f}% | IoU: {iou_thresh * 100:.0f}%"
        )

    # --- DETECTION DETAILS & EXPORT SECTION ---
    st.markdown("---")

    if summary["total_objects"] == 0:
        st.warning(
            f"⚠️ **No objects detected** meeting the {conf_thresh * 100:.0f}% confidence threshold. "
            "Try lowering the **Confidence Threshold** in the sidebar or verifying the image content."
        )
    else:
        st.success(
            f"✅ Successfully detected **{summary['total_objects']} object(s)** belonging to **{summary['unique_classes']} class(es)**."
        )

        col_table, col_chart = st.columns([3, 2])

        with col_table:
            st.markdown("#### 📋 Detailed Detections")
            df_detections = create_detections_dataframe(detections)
            st.dataframe(
                df_detections,
                use_container_width=True,
                hide_index=True,
            )

        with col_chart:
            st.markdown("#### 📊 Class Distribution")
            df_dist = create_class_distribution_df(detections)
            if not df_dist.empty:
                st.bar_chart(
                    data=df_dist.set_index("Class")["Count"],
                    color="#38BDF8",
                    use_container_width=True,
                )
                st.dataframe(df_dist, use_container_width=True, hide_index=True)

        # --- EXPORT DOWNLOAD ACTIONS ---
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col_dl_img, col_dl_csv = st.columns(2)

        with col_dl_img:
            annotated_png_bytes = image_to_png_bytes(annotated_rgb)
            img_filename = generate_timestamped_filename("visionguard_annotated", "png")
            st.download_button(
                label="📥 Download Annotated Image (PNG)",
                data=annotated_png_bytes,
                file_name=img_filename,
                mime="image/png",
                use_container_width=True,
            )

        with col_dl_csv:
            csv_bytes = detections_to_csv_bytes(detections, source_filename=filename)
            csv_filename = generate_timestamped_filename("visionguard_report", "csv")
            st.download_button(
                label="📊 Download Detection Report (CSV)",
                data=csv_bytes,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
