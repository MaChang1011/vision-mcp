from pydantic import BaseModel
from typing import List, Optional, Any

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class OCRLine(BaseModel):
    text: str
    confidence: float
    bbox: BBox

class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: BBox

class ToolResponse(BaseModel):
    ok: bool
    engine: str
    data: Any = None
    meta: dict = {}
    error: Optional[str] = None
