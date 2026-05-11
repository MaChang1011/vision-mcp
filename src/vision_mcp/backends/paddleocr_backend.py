from .base import BaseBackend

class PaddleOCRBackend(BaseBackend):
    def run(self, **kwargs):
        return {"text": "", "confidence": 0.0}
