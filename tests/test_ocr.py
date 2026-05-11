from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vision_mcp.tools.ocr_tools import ocr_image, ocr_image_with_boxes


def _make_test_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "ocr_test.png"
    img = Image.new("RGB", (640, 240), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    except Exception:
        font = None
    draw.multiline_text(
        (30, 40),
        "Hello OCR\n123456\nHermes Vision",
        fill="black",
        font=font,
        spacing=12,
    )
    img.save(image_path)
    return image_path


def test_ocr_image_success(tmp_path: Path):
    image_path = _make_test_image(tmp_path)
    result = ocr_image(str(image_path), lang_hint="en")

    assert result["ok"] is True
    assert result["engine"] == "paddleocr"
    assert "Hello OCR" in result["data"]["text"]
    assert "Hermes Vision" in result["data"]["text"]
    assert result["data"]["confidence"] > 0.8
    assert result["meta"]["image_width"] == 640
    assert result["meta"]["image_height"] == 240
    assert result["meta"]["line_count"] >= 3


def test_ocr_image_with_boxes_success(tmp_path: Path):
    image_path = _make_test_image(tmp_path)
    result = ocr_image_with_boxes(str(image_path), lang_hint="en")

    assert result["ok"] is True
    assert result["engine"] == "paddleocr"
    assert len(result["data"]["lines"]) >= 3

    first = result["data"]["lines"][0]
    assert "text" in first
    assert "confidence" in first
    assert "bbox" in first
    assert set(first["bbox"].keys()) == {"x1", "y1", "x2", "y2"}


def test_ocr_image_missing_file():
    result = ocr_image("/tmp/not-exists-image.png", lang_hint="en")

    assert result["ok"] is False
    assert result["data"] is None
    assert "Image not found" in result["error"]
