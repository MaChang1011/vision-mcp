from vision_mcp.tools.ocr_tools import ocr_image


def test_ocr_image_placeholder():
    result = ocr_image("dummy.jpg")
    assert result["ok"] is False
    assert result["engine"] == "paddleocr"
