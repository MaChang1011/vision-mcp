"""OpenCV backend for image preprocessing and cropping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from vision_mcp.backends.base import BackendError, BaseBackend


class OpenCVBackend(BaseBackend):
    engine_name = "opencv"

    def run(self, image_path: str, operation: str, output_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        if operation == "preprocess":
            return self.preprocess(image_path=image_path, output_path=output_path, **kwargs)
        if operation == "crop":
            return self.crop(image_path=image_path, output_path=output_path, **kwargs)
        return {
            "ok": False,
            "engine": self.engine_name,
            "data": None,
            "meta": {},
            "error": f"Unsupported operation: {operation}",
        }

    def preprocess(
        self,
        image_path: str,
        operations: list[str],
        output_path: str | None = None,
        resize_width: int | None = None,
        resize_height: int | None = None,
        denoise_strength: int = 10,
    ) -> dict[str, Any]:
        try:
            image = self.load_image(image_path)
            original_meta = self.get_image_meta(image)
            applied: list[str] = []
            result = image.copy()

            for op in operations:
                normalized = op.strip().lower()
                if normalized == "grayscale":
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    applied.append("grayscale")
                elif normalized == "denoise":
                    if len(result.shape) == 2:
                        result = cv2.fastNlMeansDenoising(result, None, denoise_strength, 7, 21)
                    else:
                        result = cv2.fastNlMeansDenoisingColored(result, None, denoise_strength, denoise_strength, 7, 21)
                    applied.append("denoise")
                elif normalized == "resize":
                    if not resize_width and not resize_height:
                        raise BackendError("resize requires resize_width or resize_height")
                    h, w = result.shape[:2]
                    if resize_width and resize_height:
                        new_w, new_h = resize_width, resize_height
                    elif resize_width:
                        scale = resize_width / w
                        new_w, new_h = resize_width, max(1, int(round(h * scale)))
                    else:
                        scale = resize_height / h
                        new_h, new_w = resize_height, max(1, int(round(w * scale)))
                    result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    applied.append("resize")
                elif normalized == "deskew":
                    raise BackendError("deskew is planned but not implemented yet")
                else:
                    raise BackendError(f"Unsupported preprocess operation: {op}")

            out_path = self._resolve_output_path(image_path, output_path, suffix="_preprocessed")
            self._save_image(out_path, result)

            final_meta = self.get_image_meta(result if len(result.shape) == 3 else cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))
            return {
                "ok": True,
                "engine": self.engine_name,
                "data": {
                    "output_path": str(out_path),
                    "applied": applied,
                },
                "meta": {
                    **original_meta,
                    "output_width": final_meta["image_width"],
                    "output_height": final_meta["image_height"],
                },
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
                "error": f"OpenCV preprocess failed: {exc}",
            }

    def crop(self, image_path: str, bbox: list[int], output_path: str | None = None) -> dict[str, Any]:
        try:
            if len(bbox) != 4:
                raise BackendError("bbox must be [x1, y1, x2, y2]")

            image = self.load_image(image_path)
            meta = self.get_image_meta(image)
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in bbox]

            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                raise BackendError("bbox is invalid")
            if x2 > w or y2 > h:
                raise BackendError("bbox exceeds image boundary")

            cropped = image[y1:y2, x1:x2]
            out_path = self._resolve_output_path(image_path, output_path, suffix="_cropped")
            self._save_image(out_path, cropped)

            crop_meta = self.get_image_meta(cropped)
            return {
                "ok": True,
                "engine": self.engine_name,
                "data": {
                    "output_path": str(out_path),
                    "bbox": [x1, y1, x2, y2],
                },
                "meta": {
                    **meta,
                    "crop_width": crop_meta["image_width"],
                    "crop_height": crop_meta["image_height"],
                },
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
                "error": f"OpenCV crop failed: {exc}",
            }

    @staticmethod
    def _resolve_output_path(image_path: str, output_path: str | None, suffix: str) -> Path:
        if output_path:
            out = Path(output_path)
        else:
            source = Path(image_path)
            out = source.with_name(f"{source.stem}{suffix}{source.suffix}")
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def _save_image(output_path: Path, image) -> None:
        ok = cv2.imwrite(str(output_path), image)
        if not ok:
            raise BackendError(f"Failed to write image: {output_path}")
