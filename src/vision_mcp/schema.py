from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


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
    meta: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class OCRTextData(BaseModel):
    text: str
    confidence: Optional[float] = None


class OCRLinesData(BaseModel):
    lines: List[OCRLine]
