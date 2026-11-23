"""Unit tests for object detection configuration, bounding boxes, predictions, and detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from sportiq.detection.config import DetectorConfig
from sportiq.detection.detector import (
    DetectionError,
    InferenceError,
    ModelLoadError,
    YOLODetector,
)
from sportiq.detection.prediction import BoundingBox, Detection, FrameDetections
from sportiq.detection.visualizer import DetectionVisualizer


def test_detector_config_valid() -> None:
    """Test DetectorConfig default values and serialization."""
    cfg = DetectorConfig()
    assert cfg.imgsz == 640
    assert cfg.conf == 0.25
    assert cfg.iou == 0.45
    assert cfg.half is True
    assert cfg.batch_size == 1

    d = cfg.to_dict()
    assert d["imgsz"] == 640
    assert d["device"] == "cuda:0"


def test_detector_config_invalid_parameters() -> None:
    """Test DetectorConfig parameter validation."""
    with pytest.raises(ValueError, match="multiple of 32"):
        DetectorConfig(imgsz=641)

    with pytest.raises(ValueError, match="multiple of 32"):
        DetectorConfig(imgsz=-640)

    with pytest.raises(ValueError, match="conf threshold"):
        DetectorConfig(conf=1.5)

    with pytest.raises(ValueError, match="iou threshold"):
        DetectorConfig(iou=-0.1)

    with pytest.raises(ValueError, match="batch_size"):
        DetectorConfig(batch_size=0)


def test_bounding_box_math_and_conversions() -> None:
    """Test BoundingBox dimension calculations and format conversions."""
    box = BoundingBox(x1=100.0, y1=200.0, x2=300.0, y2=450.0)
    assert box.width == 200.0
    assert box.height == 250.0
    assert box.x_center == 200.0
    assert box.y_center == 325.0

    assert box.to_xyxy() == (100.0, 200.0, 300.0, 450.0)
    assert box.to_xywh() == (100.0, 200.0, 200.0, 250.0)

    # Test normalized YOLO coordinate calculation on 1000x1000 image
    norm = box.to_normalized_yolo(img_width=1000, img_height=1000)
    assert norm == (0.2, 0.325, 0.2, 0.25)

    with pytest.raises(ValueError, match="must be positive"):
        box.to_normalized_yolo(img_width=0, img_height=1000)

    box_dict = box.to_dict()
    assert box_dict["x1"] == 100.0
    assert box_dict["width"] == 200.0


def test_detection_and_frame_detections_serialization(tmp_path: Path) -> None:
    """Test Detection and FrameDetections serialization and JSONL export."""
    box1 = BoundingBox(x1=50.0, y1=60.0, x2=100.0, y2=180.0)
    box2 = BoundingBox(x1=300.0, y1=320.0, x2=320.0, y2=340.0)

    det1 = Detection(box=box1, confidence=0.88, class_id=0, class_name="person")
    det2 = Detection(box=box2, confidence=0.72, class_id=32, class_name="sports ball")

    frame_dets = FrameDetections(
        source="frame_000001.jpg",
        frame_index=1,
        timestamp_ms=40.0,
        image_width=1920,
        image_height=1080,
        detections=[det1, det2],
        inference_time_ms=18.5,
        model_name="yolo11n.pt",
        device="cuda:0",
    )

    assert frame_dets.total_detections == 2
    assert frame_dets.class_counts == {"person": 1, "sports ball": 1}

    d = frame_dets.to_dict()
    assert d["source"] == "frame_000001.jpg"
    assert d["frame_index"] == 1
    assert d["timestamp_ms"] == 40.0
    assert len(d["detections"]) == 2

    # Test JSONL export
    out_jsonl = tmp_path / "preds.jsonl"
    frame_dets.append_to_jsonl(out_jsonl)
    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "frame_000001.jpg" in lines[0]


def test_detector_lazy_loading() -> None:
    """Test that YOLODetector does not load weights until explicitly needed."""
    cfg = DetectorConfig(model_path="models/weights/yolo11n.pt")
    detector = YOLODetector(config=cfg)
    assert detector.is_loaded is False


def test_detector_missing_image_file_raises() -> None:
    """Test that predicting on a non-existent image file raises FileNotFoundError."""
    detector = YOLODetector(config=DetectorConfig(device="cpu"))
    with pytest.raises(FileNotFoundError, match="Input image not found"):
        detector.predict_image("C:/nonexistent/fake_image.jpg")


def test_detector_missing_frames_directory_raises() -> None:
    """Test that predicting on a non-existent directory raises NotADirectoryError."""
    detector = YOLODetector(config=DetectorConfig(device="cpu"))
    with pytest.raises(NotADirectoryError, match="Frames directory not found"):
        detector.predict_frames("C:/nonexistent/fake_dir")


def test_detector_invalid_cuda_device_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that requesting CUDA when unavailable raises ModelLoadError."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = DetectorConfig(device="cuda:0")
    detector = YOLODetector(config=cfg)

    with pytest.raises(ModelLoadError, match="torch.cuda.is_available.. is False"):
        detector.load_model()


def test_detector_predict_image_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test end-to-end predict_image pipeline using a mocked Ultralytics model."""
    detector = YOLODetector(config=DetectorConfig(device="cpu", half=False))

    # Construct mock YOLO results
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu().numpy.return_value = np.array([[100.0, 150.0, 200.0, 350.0]])
    mock_boxes.conf.cpu().numpy.return_value = np.array([0.89])
    mock_boxes.cls.cpu().numpy.return_value = np.array([0])

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_model = MagicMock()
    mock_model.names = {0: "person", 32: "sports ball"}
    mock_model.return_value = [mock_result]

    detector._model = mock_model
    detector._class_names = {0: "person", 32: "sports ball"}

    # Run inference on dummy 720p image array
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.predict_image(dummy_img, frame_index=10, timestamp_ms=400.0)

    assert isinstance(res, FrameDetections)
    assert res.total_detections == 1
    assert res.image_width == 1280
    assert res.image_height == 720
    assert res.frame_index == 10
    assert res.timestamp_ms == 400.0

    det = res.detections[0]
    assert det.class_name == "person"
    assert det.confidence == 0.89
    assert det.box.to_xyxy() == (100.0, 150.0, 200.0, 350.0)


def test_detector_predict_frames_with_manifest_mocked(tmp_path: Path) -> None:
    """Test predict_frames correctly parses frames.jsonl and extracts metadata."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    import cv2
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "frame_000001.jpg"), dummy_img)
    cv2.imwrite(str(frames_dir / "frame_000002.jpg"), dummy_img)

    # Write frames.jsonl manifest
    manifest_file = frames_dir / "frames.jsonl"
    manifest_lines = [
        '{"frame_filename": "frame_000001.jpg", "original_frame_index": 1, "timestamp_ms": 40.0}',
        '{"frame_filename": "frame_000002.jpg", "original_frame_index": 2, "timestamp_ms": 80.0}',
    ]
    manifest_file.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    detector = YOLODetector(config=DetectorConfig(device="cpu", half=False))

    mock_model = MagicMock()
    mock_model.names = {0: "person"}
    mock_result = MagicMock()
    mock_result.boxes = None  # No detections
    mock_model.return_value = [mock_result]

    detector._model = mock_model
    detector._class_names = {0: "person"}

    out_jsonl = tmp_path / "predictions.jsonl"
    results = detector.predict_frames(frames_dir, output_jsonl=out_jsonl)

    assert len(results) == 2
    assert results[0].frame_index == 1
    assert results[0].timestamp_ms == 40.0
    assert results[1].frame_index == 2
    assert results[1].timestamp_ms == 80.0
    assert out_jsonl.exists()


def test_detection_visualizer_rendering(tmp_path: Path) -> None:
    """Test DetectionVisualizer draws boxes and saves annotated image."""
    box = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=80.0)
    det = Detection(box=box, confidence=0.85, class_id=0, class_name="person")
    frame_dets = FrameDetections(
        source="test.jpg",
        image_width=200,
        image_height=200,
        detections=[det],
        model_name="yolo11n.pt",
        device="cuda:0",
    )

    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    rendered = DetectionVisualizer.draw_detections(canvas, frame_dets)

    assert rendered.shape == (200, 200, 3)
    # The canvas should now have non-zero pixels from bounding box and text
    assert np.any(rendered > 0)

    out_img = tmp_path / "vis.jpg"
    DetectionVisualizer.visualize_and_save(canvas, frame_dets, out_img)
    assert out_img.exists()
