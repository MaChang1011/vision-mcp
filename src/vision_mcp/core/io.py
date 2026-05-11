"""Core image IO helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def validate_image_path(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{p.suffix}' "
            f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )
    return p


def load_image(path: str) -> np.ndarray:
    p = validate_image_path(path)
    image = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image
