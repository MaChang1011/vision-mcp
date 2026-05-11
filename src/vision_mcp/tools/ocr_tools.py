def ocr_image(image_path: str, lang_hint: str | None = None):
    return {
        "ok": False,
        "engine": "paddleocr",
        "data": None,
        "meta": {"image_path": image_path, "lang_hint": lang_hint},
        "error": "not implemented",
    }
