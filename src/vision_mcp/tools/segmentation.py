"""Segmentation tools backed by SAM."""

from __future__ import annotations

import time
from typing import Any

from vision_mcp.backends.sam_backend import SAMBackend

_default_backend: SAMBackend | None = None


def _get_backend() -> SAMBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = SAMBackend()
    return _default_backend


def segment_by_box(
    image_path: str,
    bbox: list[int],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate a segmentation mask for a bbox region.

    Args:
        image_path: Absolute path to the image file.
        bbox: [x1, y1, x2, y2] defining the region to segment.
        output_path: Optional output path for the mask PNG. Auto-generated if omitted.

    Returns:
        ToolResponse-compatible dict with mask_path and bbox.
    """
    t0 = time.perf_counter()
    backend = _get_backend()
    result = backend.segment_by_box(
        image_path=image_path,
        bbox=bbox,
        output_path=output_path,
    )
    result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result
