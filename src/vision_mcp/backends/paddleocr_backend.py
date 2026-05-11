"""PaddleOCR backend implementation (PaddleOCR 2.x compatible)."""

from __future__ import annotations

from threading import Lock
from typing import Any

from paddleocr import PaddleOCR

from vision_mcp.backends.base import BackendError, BaseBackend
from vision_mcp.schema import BBox, OCRLine


_LANG_MAP = {
    None: "ch",
    "": "ch",
    "zh": "ch",
    "zh-cn": "ch",
    "zh-hans": "ch",
    "zh-hant": "chinese_cht",
    "en": "en",
    "english": "en",
    "japan": "japan",
    "ja": "japan",
    "korean": "korean",
    "ko": "korean",
    "fr": "fr",
    "german": "german",
    "de": "german",
}


class PaddleOCRBackend(BaseBackend):
    engine_name = "paddleocr"

    def __init__(self, use_angle_cls: bool = True, device: str = "cpu") -> None:
        self.use_angle_cls = use_angle_cls
        self.device = device
        self._ocr_by_lang: dict[str, PaddleOCR] = {}
        self._lock = Lock()

    def _normalize_lang(self, lang_hint: str | None) -> str:
        if not lang_hint:
            return "ch"
        key = lang_hint.strip().lower()
        return _LANG_MAP.get(key, key)

    def _get_ocr(self, lang: str) -> PaddleOCR:
        cached = self._ocr_by_lang.get(lang)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._ocr_by_lang.get(lang)
            if cached is not None:
                return cached
            ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=lang,
                use_gpu=False,
                show_log=False,
            )
            self._ocr_by_lang[lang] = ocr
            return ocr

    @staticmethod
    def _quad_to_bbox(points: list[list[float]]) -> BBox:
        xs = [int(round(p[0])) for p in points]
        ys = [int(round(p[1])) for p in points]
        return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

    def run(
        self,
        image_path: str,
        lang_hint: str | None = None,
        with_boxes: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            image = self.load_image(image_path)
            lang = self._normalize_lang(lang_hint)
            ocr = self._get_ocr(lang)
            result = ocr.ocr(image_path, cls=self.use_angle_cls)

            lines: list[OCRLine] = []
            texts: list[str] = []
            confidences: list[float] = []

            if result:
                page = result[0] or []
                for item in page:
                    if not item or len(item) < 2:
                        continue
                    points, rec = item
                    if not rec or len(rec) < 2:
                        continue
                    text = str(rec[0]).strip()
                    confidence = float(rec[1])
                    if not text:
                        continue
                    bbox = self._quad_to_bbox(points)
                    lines.append(OCRLine(text=text, confidence=confidence, bbox=bbox))
                    texts.append(text)
                    confidences.append(confidence)

            avg_conf = (
                round(sum(confidences) / len(confidences), 4) if confidences else 0.0
            )
            meta = {
                **self.get_image_meta(image),
                "lang": lang,
                "line_count": len(lines),
            }

            if with_boxes:
                data = {"lines": [line.model_dump() for line in lines]}
            else:
                data = {
                    "text": "\n".join(texts),
                    "confidence": avg_conf,
                }

            return {
                "ok": True,
                "engine": self.engine_name,
                "data": data,
                "meta": meta,
                "error": None,
            }
        except FileNotFoundError as exc:
            return {
                "ok": False,
                "engine": self.engine_name,
                "data": None,
                "meta": {},
                "error": str(exc),
            }
        except BackendError as exc:
            return {
                "ok": False,
                "engine": self.engine_name,
                "data": None,
                "meta": {},
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "ok": False,
                "engine": self.engine_name,
                "data": None,
                "meta": {},
                "error": f"PaddleOCR inference failed: {exc}",
            }
