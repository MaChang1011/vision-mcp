from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision_mcp.tools.detection import detect_objects


def _make_scene_image(tmp_path: Path) -> Path:
    """Create a synthetic scene with colored blocks that look like common objects."""
    image_path = tmp_path / "detection_test.png"
    img = np.full((400, 600, 3), 200, dtype=np.uint8)

    # Red rectangle that looks like a "person" block
    cv2.rectangle(img, (50, 50), (150, 350), (0, 0, 200), -1)
    cv2.putText(img, "P", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Blue rectangle
    cv2.rectangle(img, (200, 100), (350, 300), (200, 100, 0), -1)
    cv2.putText(img, "B", (255, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Green square
    cv2.rectangle(img, (420, 80), (560, 220), (0, 180, 0), -1)
    cv2.putText(img, "G", (465, 175), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Yellow circle
    cv2.circle(img, (500, 320), 60, (0, 220, 220), -1)

    cv2.imwrite(str(image_path), img)
    return image_path


def test_detect_objects_returns_results(tmp_path: Path):
    image_path = _make_scene_image(tmp_path)

    result = detect_objects(str(image_path), confidence_threshold=0.01)

    assert result["ok"] is True
    assert result["engine"] == "yolo"
    assert "objects" in result["data"]
    assert result["meta"]["image_width"] == 600
    assert result["meta"]["image_height"] == 400
    assert result["meta"]["object_count"] >= 1

    first_obj = result["data"]["objects"][0]
    assert "label" in first_obj
    assert "confidence" in first_obj
    assert "bbox" in first_obj
    assert set(first_obj["bbox"].keys()) == {"x1", "y1", "x2", "y2"}


def test_detect_objects_missing_file():
    result = detect_objects("/tmp/not-exists-for-detection.png")

    assert result["ok"] is False
    assert result["data"] is None
    assert "Image not found" in result["error"]
