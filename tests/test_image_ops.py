from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision_mcp.tools.image_ops import crop_image, preprocess_image


def _make_test_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "image_ops_test.png"
    image = np.full((120, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (180, 90), (0, 0, 0), thickness=2)
    cv2.putText(image, "ABC", (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imwrite(str(image_path), image)
    return image_path


def test_preprocess_image_grayscale_and_resize(tmp_path: Path):
    image_path = _make_test_image(tmp_path)
    output_path = tmp_path / "preprocessed.png"

    result = preprocess_image(
        str(image_path),
        operations=["grayscale", "resize"],
        output_path=str(output_path),
        resize_width=100,
    )

    assert result["ok"] is True
    assert result["engine"] == "opencv"
    assert result["data"]["output_path"] == str(output_path)
    assert result["data"]["applied"] == ["grayscale", "resize"]
    assert output_path.exists()
    assert result["meta"]["image_width"] == 200
    assert result["meta"]["output_width"] == 100

    processed = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert processed is not None
    assert processed.shape[1] == 100


def test_crop_image_success(tmp_path: Path):
    image_path = _make_test_image(tmp_path)
    output_path = tmp_path / "cropped.png"

    result = crop_image(
        str(image_path),
        bbox=[20, 20, 180, 90],
        output_path=str(output_path),
    )

    assert result["ok"] is True
    assert result["engine"] == "opencv"
    assert result["data"]["output_path"] == str(output_path)
    assert result["data"]["bbox"] == [20, 20, 180, 90]
    assert output_path.exists()
    assert result["meta"]["crop_width"] == 160
    assert result["meta"]["crop_height"] == 70


def test_crop_image_invalid_bbox(tmp_path: Path):
    image_path = _make_test_image(tmp_path)

    result = crop_image(str(image_path), bbox=[-1, 0, 10, 10])

    assert result["ok"] is False
    assert "bbox is invalid" in result["error"]


def test_preprocess_image_invalid_operation(tmp_path: Path):
    image_path = _make_test_image(tmp_path)

    result = preprocess_image(str(image_path), operations=["deskew"])

    assert result["ok"] is False
    assert "not implemented" in result["error"]
