"""PaddleOCR backend implementation (PaddleOCR 2.x compatible).

For Chinese models that trigger SIGILL on certain CPU instruction sets,
we run inference in a subprocess to avoid crashing the main process,
and automatically fallback to alternative language models.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from threading import Lock
from typing import Any

from vision_mcp.backends.base import BackendError, BaseBackend
from vision_mcp.schema import BBox, OCRLine

logger = logging.getLogger(__name__)

_LANG_MAP = {
    None: "ch",
    "": "ch",
    "zh": "ch",
    "zh-cn": "ch",
    "zh-hans": "ch",
    "zh-hant": "chinese_cht",
    "zh-tw": "chinese_cht",
    "zh-hk": "chinese_cht",
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

# Languages that may SIGILL on this CPU — run in subprocess with fallback chain
_SUBPROCESS_LANGS = {"ch"}

# Fallback chain: if primary lang crashes, try these in order
_LANG_FALLBACK = {
    "ch": ["chinese_cht", "en"],
    "chinese_cht": ["en"],
}


def _run_ocr_in_subprocess(image_path: str, lang: str, use_angle_cls: bool) -> dict[str, Any]:
    """Run PaddleOCR in a subprocess to isolate SIGILL crashes."""
    script = f"""
import json, sys
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls={use_angle_cls}, lang={lang!r}, use_gpu=False, show_log=False)
    result = ocr.ocr({image_path!r}, cls={use_angle_cls})
    lines = []
    texts = []
    confidences = []
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
            xs = [int(round(p[0])) for p in points]
            ys = [int(round(p[1])) for p in points]
            lines.append({{"text": text, "confidence": confidence, "bbox": {{"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}}}})
            texts.append(text)
            confidences.append(confidence)
    avg = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    json.dump({{"ok": True, "lang": {lang!r}, "lines": lines, "text": "\\n".join(texts), "confidence": avg, "line_count": len(lines)}}, sys.stdout)
except Exception as exc:
    json.dump({{"ok": False, "error": str(exc)}}, sys.stdout)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            error_msg = proc.stderr.strip()[-500:] if proc.stderr else "subprocess crashed"
            return {"ok": False, "error": f"OCR subprocess exited {proc.returncode}: {error_msg}"}
        return json.loads(proc.stdout.strip())
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "OCR subprocess timed out (120s)"}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"OCR subprocess returned invalid JSON: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"OCR subprocess failed: {exc}"}


class PaddleOCRBackend(BaseBackend):
    engine_name = "paddleocr"

    def __init__(self, use_angle_cls: bool = True, device: str = "cpu") -> None:
        self.use_angle_cls = use_angle_cls
        self.device = device
        self._ocr_by_lang: dict[str, Any] = {}
        self._lock = Lock()

    def _normalize_lang(self, lang_hint: str | None) -> str:
        if not lang_hint:
            return "ch"
        key = lang_hint.strip().lower()
        return _LANG_MAP.get(key, key)

    def _get_ocr(self, lang: str):
        cached = self._ocr_by_lang.get(lang)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._ocr_by_lang.get(lang)
            if cached is not None:
                return cached

            if lang in _SUBPROCESS_LANGS:
                return None  # signal: use subprocess path

            from paddleocr import PaddleOCR
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

    def _run_subprocess_with_fallback(self, image_path: str, lang: str) -> dict[str, Any]:
        """Try lang, then fallback languages if it crashes."""
        candidates = [lang] + _LANG_FALLBACK.get(lang, [])
        for candidate in candidates:
            result = _run_ocr_in_subprocess(
                image_path=image_path,
                lang=candidate,
                use_angle_cls=self.use_angle_cls,
            )
            if result.get("ok"):
                if candidate != lang:
                    logger.warning(
                        "PaddleOCR lang=%s crashed, falling back to lang=%s",
                        lang,
                        candidate,
                    )
                result["requested_lang"] = lang
                result["actual_lang"] = candidate
                return result
            else:
                logger.warning("PaddleOCR lang=%s failed: %s", candidate, result.get("error"))
        return {"ok": False, "error": f"All OCR languages failed: {candidates}"}

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
            use_subprocess = lang in _SUBPROCESS_LANGS

            if use_subprocess:
                sub_result = self._run_subprocess_with_fallback(image_path, lang)
                if not sub_result.get("ok"):
                    return {
                        "ok": False,
                        "engine": self.engine_name,
                        "data": None,
                        "meta": {**self.get_image_meta(image), "lang": lang},
                        "error": sub_result.get("error", "unknown subprocess error"),
                    }

            else:
                ocr = self._get_ocr(lang)
                if ocr is None:
                    raise BackendError(f"PaddleOCR init failed for lang={lang}")

                result = ocr.ocr(image_path, cls=self.use_angle_cls)

                lines_data: list[dict] = []
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
                        lines_data.append({"text": text, "confidence": confidence, "bbox": bbox.model_dump()})
                        texts.append(text)
                        confidences.append(confidence)

                sub_result = {
                    "ok": True,
                    "lines": lines_data,
                    "text": "\n".join(texts),
                    "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
                    "line_count": len(lines_data),
                    "actual_lang": lang,
                }

            lines = [
                OCRLine(
                    text=item["text"],
                    confidence=item["confidence"],
                    bbox=BBox(**item["bbox"]),
                )
                for item in sub_result.get("lines", [])
            ]
            meta = {
                **self.get_image_meta(image),
                "lang": sub_result.get("actual_lang", lang),
                "line_count": sub_result.get("line_count", len(lines)),
            }

            if with_boxes:
                data = {"lines": [line.model_dump() for line in lines]}
            else:
                data = {
                    "text": sub_result.get("text", ""),
                    "confidence": sub_result.get("confidence", 0.0),
                }

            return {
                "ok": True,
                "engine": self.engine_name,
                "data": data,
                "meta": meta,
                "error": None,
            }
        except FileNotFoundError as exc:
            return {"ok": False, "engine": self.engine_name, "data": None, "meta": {}, "error": str(exc)}
        except BackendError as exc:
            return {"ok": False, "engine": self.engine_name, "data": None, "meta": {}, "error": str(exc)}
        except Exception as exc:
            return {
                "ok": False,
                "engine": self.engine_name,
                "data": None,
                "meta": {},
                "error": f"PaddleOCR inference failed: {exc}",
            }
