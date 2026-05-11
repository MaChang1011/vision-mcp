"""Open-vocabulary detection tools backed by Grounding DINO."""

from __future__ import annotations

import time
from typing import Any

from vision_mcp.backends.grounding_backend import GroundingDINOBackend

_default_backend: GroundingDINOBackend | None = None


def _get_backend() -> GroundingDINOBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = GroundingDINOBackend()
    return _default_backend


def detect_by_prompt(
    image_path: str,
    prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> dict[str, Any]:
    """Detect objects in an image by natural language prompt.

    Args:
        image_path: Absolute path to the image file.
        prompt: Natural language description of what to find, e.g. "power strip . charger . cable".
        box_threshold: Minimum confidence for bounding box proposals (0.0-1.0).
        text_threshold: Minimum confidence for text-phrase matching (0.0-1.0).

    Returns:
        ToolResponse-compatible dict with `data.objects[]`.
    """
    t0 = time.perf_counter()
    backend = _get_backend()
    result = backend.run(
        image_path=image_path,
        prompt=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result
