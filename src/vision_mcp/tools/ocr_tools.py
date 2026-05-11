"""OCR tool implementations backed by PaddleOCR."""

from __future__ import annotations

import time
from typing import Any

from vision_mcp.backends.paddleocr_backend import PaddleOCRBackend

_default_backend: PaddleOCRBackend | None = None


def _get_backend() -> PaddleOCRBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = PaddleOCRBackend()
    return _default_backend


def ocr_image(image_path: str, lang_hint: str | None = None) -> dict[str, Any]:
    """Extract all text from an image.

    Returns:
        ToolResponse-compatible dict with `data.text` and `data.confidence`.
    """
    t0 = time.perf_counter()
    backend = _get_backend()
    result = backend.run(image_path=image_path, lang_hint=lang_hint, with_boxes=False)
    result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


def ocr_image_with_boxes(image_path: str, lang_hint: str | None = None) -> dict[str, Any]:
    """Extract text with bounding boxes from an image.

    Returns:
        ToolResponse-compatible dict with `data.lines[]`.
    """
    t0 = time.perf_counter()
    backend = _get_backend()
    result = backend.run(image_path=image_path, lang_hint=lang_hint, with_boxes=True)
    result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result
