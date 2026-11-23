"""SportIQ Object Detection Module.

Provides modular YOLO-based object detection, prediction structures, lazy model loading,
and visual rendering for football computer-vision pipelines.
"""

from __future__ import annotations

from sportiq.detection.config import DetectorConfig
from sportiq.detection.detector import (
    DetectionError,
    InferenceError,
    ModelLoadError,
    YOLODetector,
)
from sportiq.detection.prediction import BoundingBox, Detection, FrameDetections
from sportiq.detection.visualizer import DetectionVisualizer

__all__ = [
    "DetectorConfig",
    "BoundingBox",
    "Detection",
    "FrameDetections",
    "YOLODetector",
    "DetectionError",
    "ModelLoadError",
    "InferenceError",
    "DetectionVisualizer",
]
