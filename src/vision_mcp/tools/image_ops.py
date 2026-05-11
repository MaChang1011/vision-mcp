"""Image operations tools — preprocess and crop, backed by OpenCV."""

from __future__ import annotations

import time
from typing import Any

from vision_mcp.backends.opencv_backend import OpenCVBackend

_default_backend: OpenCVBackend | None = None


def _get_backend() -> OpenCVBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = OpenCVBackend()
    return _default_backend


def preprocess_image(
    image_path: str,
    operations: list[str],
    output_path: str | None = None,
    resize_width: int | None = None,
    resize_height: int | None = None,
    denoise_strength: int = 10,
) -> dict[str, Any]:
    """Apply image preprocessing operations.

    Supported operations: grayscale, denoise, resize.

    Returns:
        ToolResponse-compatible dict with output_path and applied operations.
    """
    t0 = time.perf_counter()
    backend = _get_backend()
    result = backend.preprocess(
        image_path=image_path,
        operations=operations,
        output_path=output_path,
        resize_width=resize_width,
        resize_height=resize_height,
        denoise_strength=denoise_strength,
    )
    result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


def crop_image(
    image_path: str,
    bbox: list[int],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Crop an image region defined by bbox [x1, y1, x2, y2].

    Returns:
        ToolResponse-compatible dict with output_path and the applied bbox.
    """
    t0 = time.perf_counter()
    backend = _get_backend()
    result = backend.crop(
        image_path=image_path,
        bbox=bbox,
        output_path=output_path,
    )
    result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result
