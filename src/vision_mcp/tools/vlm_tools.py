"""VLM tools powered by Moondream — describe images and answer questions."""

from __future__ import annotations

import time
from typing import Any

from vision_mcp.backends.moondream_backend import MoondreamBackend

_default_backend: MoondreamBackend | None = None


def _get_backend() -> MoondreamBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = MoondreamBackend()
    return _default_backend


def describe_image(
    image_path: str,
    length: str = "normal",
) -> dict[str, Any]:
    """Generate a textual description of an image.

    Args:
        image_path: Absolute path to the image file.
        length: Detail level — 'short', 'normal' (default), or 'long'.

    Returns:
        ToolResponse-compatible dict with `data.summary`.
    """
    t0 = time.perf_counter()
    backend = _get_backend()
    result = backend.caption(image_path=image_path, length=length)
    result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


def answer_about_image(
    image_path: str,
    question: str,
) -> dict[str, Any]:
    """Answer a natural-language question about the content of an image.

    Args:
        image_path: Absolute path to the image file.
        question: The question to answer (e.g., "What color is the car?").

    Returns:
        ToolResponse-compatible dict with `data.answer`.
    """
    t0 = time.perf_counter()
    backend = _get_backend()
    result = backend.query(image_path=image_path, question=question)
    result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result
