"""Base class for all vision backends."""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class BackendError(Exception):
    """Raised when a backend fails to process an image."""


class BaseBackend(ABC):
    """Abstract base for all vision backends.

    Subclasses must implement `run()`. This base class provides:
    - image loading with validation
    - timing / latency measurement
    - common error handling patterns
    """

    engine_name: str = "base"

    def load_image(self, image_path: str) -> np.ndarray:
        """Load and validate an image file.

        Returns:
            BGR numpy array (OpenCV convention).

        Raises:
            FileNotFoundError: if path doesn't exist.
            BackendError: if file is not a supported image or cannot be decoded.
        """
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise BackendError(
                f"Unsupported image format '{p.suffix}' "
                f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
            )

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise BackendError(f"Failed to decode image: {image_path}")

        return img

    def get_image_meta(self, img: np.ndarray) -> dict:
        """Return common image metadata."""
        h, w = img.shape[:2]
        return {"image_width": w, "image_height": h}

    def timed_run(self, image_path: str, **kwargs) -> dict[str, Any]:
        """Run with timing. Returns dict with result + latency_ms."""
        t0 = time.perf_counter()
        result = self.run(image_path=image_path, **kwargs)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {**result, "latency_ms": latency_ms}

    @abstractmethod
    def run(self, image_path: str, **kwargs) -> dict[str, Any]:
        """Execute the backend on the given image.

        Returns:
            Dict with at least: ok, engine, data, meta, error.
        """
        ...
