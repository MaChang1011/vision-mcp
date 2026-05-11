"""Object detection tools backed by YOLO."""

from __future__ import annotations

import time
from typing import Any

from vision_mcp.backends.yolo_backend import YOLOBackend

_default_backend: YOLOBackend | None = None


def _get_backend() -> YOLOBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = YOLOBackend()
    return _default_backend


def detect_objects(
    image_path: str,
    labels: list[str] | None = None,
    confidence_threshold: float = 0.25,
    max_detections: int = 100,
) -> dict[str, Any]:
    """Detect common objects in an image using YOLOv8.

    Args:
        image_path: Absolute path to the image file.
        labels: Optional list of target labels to filter (e.g. ["person", "car"]).
        confidence_threshold: Minimum confidence (0.0-1.0) to keep a detection.
        max_detections: Maximum number of detections to return.

    Returns:
        ToolResponse-compatible dict with `data.objects[]`.
    """
    t0 = time.perf_counter()
    backend = _get_backend()
    result = backend.run(
        image_path=image_path,
        labels=labels,
        confidence_threshold=confidence_threshold,
        max_detections=max_detections,
    )
    result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result
