"""Moondream VLM backend for image description and visual QA.

Uses the Moondream cloud API (moondream>=1.2).  Set MOONDREAM_API_KEY
in the environment, or pass api_key / endpoint to the constructor.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from vision_mcp.backends.base import BackendError, BaseBackend

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "moondream-2b-vl-2025-04-29"  # recommended cloud model id


class MoondreamBackend(BaseBackend):
    """VLM backend powered by the Moondream cloud API.

    Requires one of:
    - environment variable ``MOONDREAM_API_KEY``
    - ``api_key`` passed to the constructor
    - ``MOONDREAM_ENDPOINT`` for a self-hosted endpoint (optional)

    Lazy-loads the client on first use; thread-safe.
    """

    engine_name = "moondream"

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._api_key: str | None = api_key or os.getenv("MOONDREAM_API_KEY")
        self._endpoint: str | None = endpoint or os.getenv("MOONDREAM_ENDPOINT")
        self.model_name: str = model_name
        self._client: Any = None
        self._lock: Lock = Lock()

    def _get_client(self) -> Any:
        """Lazy-load the Moondream cloud client and return it."""
        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is not None:
                return self._client

            if not self._api_key:
                raise BackendError(
                    "Moondream API key not set — export MOONDREAM_API_KEY "
                    "or pass api_key= to the constructor"
                )

            import moondream as md

            logger.info(
                "Creating Moondream cloud client (model=%s, endpoint=%s) ...",
                self.model_name,
                self._endpoint,
            )
            self._client = md.vl(
                api_key=self._api_key,
                endpoint=self._endpoint,
                model=self.model_name,
            )
            logger.info("Moondream client ready")
            return self._client

    # ------------------------------------------------------------------
    # public API used by tools
    # ------------------------------------------------------------------

    def caption(
        self,
        image_path: str,
        length: str = "normal",
    ) -> dict[str, Any]:
        """Generate a textual description of the image.

        Args:
            image_path: Path to the image file.
            length: One of 'short', 'normal', 'long'.

        Returns:
            ToolResponse-compatible dict with `data.summary`.
        """
        try:
            img = self._load_pil(image_path)
            client = self._get_client()

            encoded = client.encode_image(img)

            summary = client.caption(encoded, length=length)
            if isinstance(summary, dict) and "caption" in summary:
                summary = summary["caption"]

            meta = self._image_meta(image_path)

            return {
                "ok": True,
                "engine": self.engine_name,
                "data": {"summary": str(summary).strip()},
                "meta": meta,
                "error": None,
            }

        except FileNotFoundError as exc:
            return _error(self.engine_name, str(exc))
        except BackendError as exc:
            return _error(self.engine_name, str(exc))
        except Exception as exc:
            logger.exception("Moondream caption failed")
            return _error(self.engine_name, f"Moondream caption failed: {exc}")

    def query(
        self,
        image_path: str,
        question: str,
    ) -> dict[str, Any]:
        """Answer a natural-language question about the image.

        Args:
            image_path: Path to the image file.
            question: Natural-language question.

        Returns:
            ToolResponse-compatible dict with `data.answer`.
        """
        try:
            if not question or not question.strip():
                raise BackendError("question must be a non-empty string")

            img = self._load_pil(image_path)
            client = self._get_client()

            encoded = client.encode_image(img)

            answer = client.query(encoded, question.strip())
            if isinstance(answer, dict) and "answer" in answer:
                answer = answer["answer"]

            meta = self._image_meta(image_path)

            return {
                "ok": True,
                "engine": self.engine_name,
                "data": {"answer": str(answer).strip()},
                "meta": meta,
                "error": None,
            }

        except FileNotFoundError as exc:
            return _error(self.engine_name, str(exc))
        except BackendError as exc:
            return _error(self.engine_name, str(exc))
        except Exception as exc:
            logger.exception("Moondream query failed")
            return _error(self.engine_name, f"Moondream query failed: {exc}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _load_pil(self, image_path: str) -> Image.Image:
        """Load an image as a PIL RGB image, after validating the path."""
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            img = Image.open(p).convert("RGB")
        except Exception as exc:
            raise BackendError(f"Cannot open image: {image_path} ({exc})") from exc
        return img

    def _image_meta(self, image_path: str) -> dict[str, Any]:
        """Return image width / height using PIL (avoids OpenCV dependency)."""
        try:
            with Image.open(image_path) as im:
                w, h = im.size
            return {"image_width": w, "image_height": h}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Abstract method stub
    # ------------------------------------------------------------------

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Use caption() or query() directly")


# ------------------------------------------------------------------
# module helpers
# ------------------------------------------------------------------

def _error(engine: str, msg: str) -> dict[str, Any]:
    return {"ok": False, "engine": engine, "data": None, "meta": {}, "error": msg}
