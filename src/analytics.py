"""VisionGuard AI - Detection Analytics Module

This module processes raw detection outputs into structured metrics,
tabular summaries, and distributions for user display and export.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def calculate_detection_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes aggregate metrics from a list of detections.

    Args:
        detections: List of detection dictionaries.

    Returns:
        dict: Aggregated summary including count, unique_classes, class_counts,
              avg_confidence, min_confidence, max_confidence.
    """
    total = len(detections)
    if total == 0:
        return {
            "total_objects": 0,
            "unique_classes": 0,
            "class_counts": {},
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
        }

    confs = [d["confidence"] for d in detections]
    class_counts: Dict[str, int] = {}

    for d in detections:
        c_name = d["class_name"]
        class_counts[c_name] = class_counts.get(c_name, 0) + 1

    return {
        "total_objects": total,
        "unique_classes": len(class_counts),
        "class_counts": class_counts,
        "avg_confidence": round(float(np.mean(confs)), 4),
        "min_confidence": round(float(np.min(confs)), 4),
        "max_confidence": round(float(np.max(confs)), 4),
    }


def create_detections_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts a detection list into a tabular DataFrame for display and reporting.

    Args:
        detections: List of detection dictionaries.

    Returns:
        pd.DataFrame: Formatted table of detections with index, class, confidence, coordinates.
    """
    if not detections:
        return pd.DataFrame(
            columns=["#", "Class", "Confidence", "Confidence (%)", "X1", "Y1", "X2", "Y2", "Width", "Height"]
        )

    rows = []
    for idx, d in enumerate(detections, start=1):
        x1, y1, x2, y2 = d["bbox"]
        width = x2 - x1
        height = y2 - y1
        conf = d["confidence"]

        rows.append(
            {
                "#": idx,
                "Class": d["class_name"],
                "Confidence": conf,
                "Confidence (%)": f"{conf * 100:.1f}%",
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2,
                "Width": width,
                "Height": height,
            }
        )

    return pd.DataFrame(rows)


def create_class_distribution_df(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """Creates a aggregated class count and confidence breakdown DataFrame.

    Args:
        detections: List of detection dictionaries.

    Returns:
        pd.DataFrame: Sorted by frequency with columns [Class, Count, Avg Confidence (%)].
    """
    if not detections:
        return pd.DataFrame(columns=["Class", "Count", "Avg Confidence (%)"])

    df = create_detections_dataframe(detections)
    grouped = (
        df.groupby("Class")
        .agg(
            Count=("Confidence", "count"),
            Avg_Conf=("Confidence", lambda x: round(float(x.mean()) * 100, 1)),
        )
        .reset_index()
    )

    grouped.rename(columns={"Avg_Conf": "Avg Confidence (%)"}, inplace=True)
    grouped.sort_values(by="Count", ascending=False, inplace=True)
    return grouped
