"""Data structures for object detection predictions, bounding boxes, and frame records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def x_center(self) -> float:
        return self.x1 + self.width / 2.0

    @property
    def y_center(self) -> float:
        return self.y1 + self.height / 2.0

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, width, height)."""
        return (self.x1, self.y1, self.width, self.height)

    def to_normalized_yolo(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Return normalized YOLO (xc, yc, w, h) in [0, 1]."""
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {img_width}x{img_height}")
        return (
            self.x_center / img_width,
            self.y_center / img_height,
            self.width / img_width,
            self.height / img_height,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }


@dataclass
class Detection:
    """Single detected object instance."""

    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "box": self.box.to_dict(),
        }


@dataclass
class FrameDetections:
    """Structured detection predictions for a single video frame or image."""

    source: str
    frame_index: Optional[int] = None
    timestamp_ms: Optional[float] = None
    image_width: int = 0
    image_height: int = 0
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    model_name: str = ""
    device: str = ""

    @property
    def total_detections(self) -> int:
        return len(self.detections)

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "frame_index": self.frame_index,
            "timestamp_ms": round(self.timestamp_ms, 2) if self.timestamp_ms is not None else None,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "model_name": self.model_name,
            "device": self.device,
            "total_detections": self.total_detections,
            "class_counts": self.class_counts,
            "detections": [det.to_dict() for det in self.detections],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def append_to_jsonl(self, jsonl_path: Union[str, Path]) -> None:
        """Append this frame detection record to a JSONL file."""
        target_path = Path(jsonl_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as f:
            f.write(self.to_json() + "\n")
