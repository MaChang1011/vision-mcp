"""YOLO backend for common object detection."""

from __future__ import annotations

from threading import Lock
from typing import Any

from ultralytics import YOLO

from vision_mcp.backends.base import BackendError, BaseBackend
from vision_mcp.schema import BBox, DetectedObject


class YOLOBackend(BaseBackend):
    engine_name = "yolo"

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: YOLO | None = None
        self._lock = Lock()

    def _get_model(self) -> YOLO:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                self._model = YOLO(self.model_name)
            return self._model

    def run(
        self,
        image_path: str,
        labels: list[str] | None = None,
        confidence_threshold: float = 0.25,
        max_detections: int = 100,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            image = self.load_image(image_path)
            model = self._get_model()
            results = model.predict(
                source=image_path,
                conf=confidence_threshold,
                max_det=max_detections,
                device=self.device,
                verbose=False,
            )

            normalized_labels = {label.strip().lower() for label in labels or [] if label.strip()}
            objects: list[DetectedObject] = []

            for result in results:
                names = result.names
                boxes = result.boxes
                if boxes is None:
                    continue

                xyxy_list = boxes.xyxy.cpu().tolist()
                conf_list = boxes.conf.cpu().tolist()
                cls_list = boxes.cls.cpu().tolist()

                for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
                    label = str(names[int(cls_id)])
                    if normalized_labels and label.lower() not in normalized_labels:
                        continue
                    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                    objects.append(
                        DetectedObject(
                            label=label,
                            confidence=float(conf),
                            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        )
                    )

            meta = {
                **self.get_image_meta(image),
                "model_name": self.model_name,
                "object_count": len(objects),
                "confidence_threshold": confidence_threshold,
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
                "error": f"YOLO inference failed: {exc}",
            }
