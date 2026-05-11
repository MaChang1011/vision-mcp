"""Grounding DINO backend for open-vocabulary object detection."""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

import cv2
import groundingdino
import numpy as np
from groundingdino.util.inference import Model

from vision_mcp.backends.base import BackendError, BaseBackend
from vision_mcp.schema import BBox, DetectedObject

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = os.path.expanduser(
    os.environ.get(
        "GROUNDING_DINO_WEIGHTS",
        "~/.cache/groundingdino/groundingdino_swint_ogc.pth",
    )
)

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(groundingdino.__file__),
    "config",
    "GroundingDINO_SwinT_OGC.py",
)


class GroundingDINOBackend(BaseBackend):
    engine_name = "grounding_dino"

    def __init__(
        self,
        model_checkpoint_path: str | None = None,
        model_config_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_checkpoint_path = os.path.expanduser(
            model_checkpoint_path or _DEFAULT_WEIGHTS
        )
        self.model_config_path = model_config_path or _DEFAULT_CONFIG
        self.device = device
        self._model: Model | None = None
        self._lock = Lock()

    def _get_model(self) -> Model:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            if not os.path.exists(self.model_checkpoint_path):
                raise BackendError(
                    f"Grounding DINO weights not found: {self.model_checkpoint_path}\n"
                    "Download from: https://huggingface.co/ShilongLiu/GroundingDINO/"
                )

            logger.info(
                "Loading Grounding DINO (device=%s)...", self.device
            )
            self._model = Model(
                model_config_path=self.model_config_path,
                model_checkpoint_path=self.model_checkpoint_path,
                device=self.device,
            )
            logger.info("Grounding DINO loaded")
            return self._model

    def run(
        self,
        image_path: str,
        prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            image = self.load_image(image_path)
            model = self._get_model()

            detections, phrases = model.predict_with_caption(
                image=image,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            h, w = image.shape[:2]
            objects: list[DetectedObject] = []

            if detections is not None and hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                xyxy = detections.xyxy
                confidences = detections.confidence if hasattr(detections, 'confidence') else [1.0] * len(xyxy)
                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    conf = float(confidences[i]) if i < len(confidences) else 1.0
                    label = phrases[i] if i < len(phrases) else "object"

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    objects.append(
                        DetectedObject(
                            label=label,
                            confidence=conf,
                            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        )
                    )

            meta = {
                **self.get_image_meta(image),
                "prompt": prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "object_count": len(objects),
            }

            return {
                "ok": True,
                "engine": self.engine_name,
                "data": {"objects": [obj.model_dump() for obj in objects]},
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
                "error": f"Grounding DINO inference failed: {exc}",
            }
