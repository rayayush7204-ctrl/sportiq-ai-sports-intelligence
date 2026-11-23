"""Configuration dataclass for YOLO object detection."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure Ultralytics settings/cache stays inside an accessible directory
if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = str(Path(tempfile.gettempdir()) / "Ultralytics")


@dataclass
class DetectorConfig:
    """Settings and hyperparameters for YOLO detector inference."""

    model_path: str = "models/weights/yolo11n.pt"
    device: str = "cuda:0"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    half: bool = True
    batch_size: int = 1
    classes: Optional[List[int]] = None
    max_det: int = 300

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.imgsz <= 0 or self.imgsz % 32 != 0:
            raise ValueError(f"imgsz must be a positive multiple of 32, got {self.imgsz}")
        if not (0.0 <= self.conf <= 1.0):
            raise ValueError(f"conf threshold must be in [0.0, 1.0], got {self.conf}")
        if not (0.0 <= self.iou <= 1.0):
            raise ValueError(f"iou threshold must be in [0.0, 1.0], got {self.iou}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "half": self.half,
            "batch_size": self.batch_size,
            "classes": self.classes,
            "max_det": self.max_det,
        }
