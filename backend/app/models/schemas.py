from pydantic import BaseModel, Field
from typing import List, Optional


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    x1_norm: float
    y1_norm: float
    x2_norm: float
    y2_norm: float


class Detection(BaseModel):
    id: int
    class_id: int
    class_name: str
    class_type: str       # person | vehicle | bicycle | other
    color: str            # hex colour for UI canvas
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class DetectionSummary(BaseModel):
    total: int
    persons: int
    vehicles: int
    bicycles: int
    other: int
    threat_detected: bool


class DetectionResponse(BaseModel):
    success: bool
    image_width: int
    image_height: int
    inference_ms: float
    conf_threshold: float
    iou_threshold: float
    summary: DetectionSummary
    detections: List[Detection]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    version: str = "1.0.0"
