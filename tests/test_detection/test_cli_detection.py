"""Unit and integration CLI tests for sportiq detect-frames command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from sportiq.cli import main


def test_cli_detect_frames_missing_input(capsys: pytest.CaptureFixture) -> None:
    """Test detect-frames fails gracefully when input path is missing."""
    ret = main(["detect-frames", "C:/nonexistent/fake_image.jpg"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Input path not found" in captured.err


def test_cli_detect_frames_single_image(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test detect-frames CLI command on a single image with mocked detector."""
    img_path = tmp_path / "sample.jpg"
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy)

    with patch("sportiq.detection.detector.YOLODetector.load_model") as mock_load:
        mock_detector_model = MagicMock()
        mock_detector_model.names = {0: "person"}
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu().numpy.return_value = np.array([[10.0, 20.0, 50.0, 100.0]])
        mock_boxes.conf.cpu().numpy.return_value = np.array([0.91])
        mock_boxes.cls.cpu().numpy.return_value = np.array([0])
        mock_result.boxes = mock_boxes
        mock_detector_model.return_value = [mock_result]

        with patch("sportiq.detection.detector.YOLODetector.is_loaded", True):
            with patch("sportiq.detection.detector.YOLODetector.predict_image") as mock_pred:
                from sportiq.detection.prediction import BoundingBox, Detection, FrameDetections

                mock_pred.return_value = FrameDetections(
                    source=str(img_path),
                    image_width=320,
                    image_height=240,
                    detections=[
                        Detection(
                            box=BoundingBox(10.0, 20.0, 50.0, 100.0),
                            confidence=0.91,
                            class_id=0,
                            class_name="person",
                        )
                    ],
                    inference_time_ms=25.0,
                    model_name="yolo11n.pt",
                    device="cpu",
                )

                out_jsonl = tmp_path / "preds.jsonl"
                out_vis = tmp_path / "vis.jpg"

                ret = main([
                    "detect-frames",
                    str(img_path),
                    "--device", "cpu",
                    "--no-half",
                    "--output-jsonl", str(out_jsonl),
                    "--visualize", str(out_vis),
                    "--json",
                ])

                assert ret == 0
                captured = capsys.readouterr()
                data = json.loads(captured.out)
                assert data["total_frames_processed"] == 1
                assert data["total_detections"] == 1
                assert data["class_counts"] == {"person": 1}
                assert out_jsonl.exists()
                assert out_vis.exists()


def test_cli_detect_frames_directory(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test detect-frames CLI command on an extracted-frames directory with mocked detector."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "frame_000000.jpg"), dummy)
    cv2.imwrite(str(frames_dir / "frame_000030.jpg"), dummy)

    with patch("sportiq.detection.detector.YOLODetector.predict_frames") as mock_predict_frames:
        from sportiq.detection.prediction import BoundingBox, Detection, FrameDetections

        mock_predict_frames.return_value = [
            FrameDetections(
                source="frame_000000.jpg",
                image_width=320,
                image_height=240,
                detections=[
                    Detection(
                        box=BoundingBox(10.0, 20.0, 50.0, 100.0),
                        confidence=0.88,
                        class_id=0,
                        class_name="person",
                    )
                ],
                inference_time_ms=20.0,
                model_name="yolo11n.pt",
                device="cpu",
            ),
            FrameDetections(
                source="frame_000030.jpg",
                image_width=320,
                image_height=240,
                detections=[],
                inference_time_ms=18.0,
                model_name="yolo11n.pt",
                device="cpu",
            ),
        ]

        vis_dir = tmp_path / "vis_out"
        ret = main([
            "detect-frames",
            str(frames_dir),
            "--device", "cpu",
            "--no-half",
            "--visualize", str(vis_dir),
            "--json",
        ])

        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_frames_processed"] == 2
        assert data["total_detections"] == 1
        assert data["class_counts"] == {"person": 1}
        assert (vis_dir / "pred_frame_000000.jpg").exists()
        assert (vis_dir / "pred_frame_000030.jpg").exists()
