"""Tests for VLM tools — describe_image and answer_about_image."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_has_api_key = bool(os.getenv("MOONDREAM_API_KEY"))
needs_api_key = pytest.mark.skipif(not _has_api_key, reason="MOONDREAM_API_KEY not set")


def _test_image() -> str:
    """Return path to a small test image, creating one if needed."""
    p = Path("/tmp/vision_mcp_test_vlm.png")
    if not p.exists():
        import numpy as np
        from PIL import Image

        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        arr[50:150, 80:220] = [0, 100, 200]  # blue rectangle
        Image.fromarray(arr).save(p)
    return str(p)


def _valid_response(result: dict) -> None:
    """Verify basic ToolResponse shape."""
    assert isinstance(result, dict)
    assert "ok" in result
    assert "engine" in result
    assert "data" in result
    assert "meta" in result
    assert "error" in result


# ------------------------------------------------------------------
# describe_image
# ------------------------------------------------------------------

@needs_api_key
def test_describe_image_ok() -> None:
    """Smoke test — describe an image and check response shape."""
    from vision_mcp.tools.vlm_tools import describe_image

    result = describe_image(image_path=_test_image(), length="short")

    _valid_response(result)
    assert result["ok"] is True
    assert "summary" in result["data"]
    assert isinstance(result["data"]["summary"], str)
    assert len(result["data"]["summary"]) > 0
    assert result["meta"].get("latency_ms", 0) > 0


@needs_api_key
def test_describe_image_lengths() -> None:
    """Ensure all three length options work."""
    from vision_mcp.tools.vlm_tools import describe_image

    for length in ("short", "normal", "long"):
        result = describe_image(image_path=_test_image(), length=length)
        assert result["ok"], f"failed for length={length}: {result.get('error')}"


def test_describe_image_missing_file() -> None:
    """Missing image path returns ok=False (no API call needed)."""
    from vision_mcp.tools.vlm_tools import describe_image

    result = describe_image(image_path="/nonexistent/image.jpg")
    assert result["ok"] is False
    assert result["error"] is not None


def test_describe_image_no_api_key() -> None:
    """Without API key, returns ok=False with a helpful error."""
    from vision_mcp.tools.vlm_tools import describe_image

    # Ensure no inherited key from env
    old = os.environ.pop("MOONDREAM_API_KEY", None)
    try:
        # Force a fresh backend so it reads the (missing) env var
        import vision_mcp.tools.vlm_tools as vt
        vt._default_backend = None

        result = describe_image(image_path=_test_image())
        assert result["ok"] is False
        assert "API key" in (result.get("error") or "")
    finally:
        if old is not None:
            os.environ["MOONDREAM_API_KEY"] = old
        # Reset cached backend
        import vision_mcp.tools.vlm_tools as vt
        vt._default_backend = None


# ------------------------------------------------------------------
# answer_about_image
# ------------------------------------------------------------------

@needs_api_key
def test_answer_about_image_ok() -> None:
    """Smoke test — ask about an image and check response shape."""
    from vision_mcp.tools.vlm_tools import answer_about_image

    result = answer_about_image(
        image_path=_test_image(),
        question="What shape is in the image?",
    )

    _valid_response(result)
    assert result["ok"] is True
    assert "answer" in result["data"]
    assert isinstance(result["data"]["answer"], str)
    assert result["meta"].get("latency_ms", 0) > 0


def test_answer_about_image_empty_question() -> None:
    """Empty question should return error (no API call needed)."""
    from vision_mcp.tools.vlm_tools import answer_about_image

    result = answer_about_image(image_path=_test_image(), question="")
    assert result["ok"] is False


# ------------------------------------------------------------------
# backend lazy-loading (import coverage)
# ------------------------------------------------------------------


def test_moondream_backend_import() -> None:
    """Ensure the backend module is importable."""
    from vision_mcp.backends.moondream_backend import MoondreamBackend

    backend = MoondreamBackend()
    assert backend.engine_name == "moondream"
