"""SAM (Segment Anything Model) backend for image segmentation."""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from vision_mcp.backends.base import BackendError, BaseBackend
from vision_mcp.schema import BBox

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = os.path.expanduser(
    os.environ.get("SAM_WEIGHTS", "~/.cache/sam/sam_vit_b_01ec64.pth")
)
_DEFAULT_MODEL_TYPE = "vit_b"


class SAMBackend(BaseBackend):
    engine_name = "sam"

    def __init__(
        self,
        model_checkpoint: str | None = None,
        model_type: str = _DEFAULT_MODEL_TYPE,
        device: str = "cpu",
    ) -> None:
        self.model_checkpoint = os.path.expanduser(model_checkpoint or _DEFAULT_WEIGHTS)
        self.model_type = model_type
        self.device = device
        self._predictor: SamPredictor | None = None
        self._lock = Lock()

    def _get_predictor(self) -> SamPredictor:
        if self._predictor is not None:
            return self._predictor

        with self._lock:
            if self._predictor is not None:
                return self._predictor

            if not os.path.exists(self.model_checkpoint):
                raise BackendError(
                    f"SAM weights not found: {self.model_checkpoint}\n"
                    "Download: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                )

            logger.info("Loading SAM %s from %s (device=%s)...", self.model_type, self.model_checkpoint, self.device)
            sam = sam_model_registry[self.model_type](checkpoint=self.model_checkpoint)
            sam.to(device=self.device)
            self._predictor = SamPredictor(sam)
            logger.info("SAM loaded")
            return self._predictor

    def _save_mask(self, mask: np.ndarray, output_path: str) -> str:
        mask_uint8 = (mask.astype(np.uint8) * 255)
        ok = cv2.imwrite(output_path, mask_uint8)
        if not ok:
            raise BackendError(f"Failed to write mask: {output_path}")
        return output_path

    def segment_by_box(
        self,
        image_path: str,
        bbox: list[int],
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate a mask for the region defined by bbox [x1, y1, x2, y2]."""
        try:
            if len(bbox) != 4:
                raise BackendError("bbox must be [x1, y1, x2, y2]")

            image = self.load_image(image_path)
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in bbox]

            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > w or y2 > h:
                raise BackendError("bbox is invalid or exceeds image boundary")

            predictor = self._get_predictor()
            predictor.set_image(image)

            input_box = np.array([x1, y1, x2, y2])
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            mask = masks[0]
            score = float(scores[0])

            if output_path is None:
                source = os.path.basename(image_path)
                name, _ = os.path.splitext(source)
                output_path = f"/tmp/sam_mask_{name}.png"

            self._save_mask(mask, output_path)

            meta = {
                **self.get_image_meta(image),
                "bbox": [x1, y1, x2, y2],
                "mask_score": round(score, 4),
                "mask_path": output_path,
            }

            return {
                "ok": True,
                "engine": self.engine_name,
                "data": {
                    "mask_path": output_path,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(score, 4),
                },
                "meta": meta,
                "error": None,
            }
        except FileNotFoundError as exc:
            return {"ok": False, "engine": self.engine_name, "data": None, "meta": {}, "error": str(exc)}
        except BackendError as exc:
            return {"ok": False, "engine": self.engine_name, "data": None, "meta": {}, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "engine": self.engine_name, "data": None, "meta": {}, "error": f"SAM inference failed: {exc}"}

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Use segment_by_box() directly")
