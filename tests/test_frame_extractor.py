"""Unit tests for FrameExtractor, manifest generation, and overwrite protection."""

import json
from pathlib import Path

import cv2
import pytest

from sportiq.video.frame_extractor import FrameExtractor, extract_frames


def test_extract_frames_every_seconds(synthetic_video_path: Path, tmp_path: Path):
    """Verify extraction sampled by seconds (1.0s interval from 2.0s 30FPS video)."""
    output_dir = tmp_path / "output_sec"
    summary = extract_frames(
        video_path=synthetic_video_path,
        output_dir=output_dir,
        every_seconds=1.0,
    )

    # In a 60 frame (2.0s) video sampled every 1.0s (30 frames step):
    # Frames extracted: index 0 (0.00s) and index 30 (1.00s) -> 2 frames
    assert summary["extracted_frames_count"] == 2
    assert summary["sampling_step_frames"] == 30

    frames_dir = output_dir / "frames"
    assert frames_dir.exists()
    assert (frames_dir / "frame_000000.jpg").exists()
    assert (frames_dir / "frame_000030.jpg").exists()

    # Verify extracted image can be read and matches dimensions
    img0 = cv2.imread(str(frames_dir / "frame_000000.jpg"))
    assert img0 is not None
    assert img0.shape == (240, 320, 3)

    # Verify manifest JSONL
    manifest_path = output_dir / "frames.jsonl"
    assert manifest_path.exists()
    lines = manifest_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    m0 = json.loads(lines[0])
    assert m0["original_frame_index"] == 0
    assert m0["timestamp_ms"] == 0.0
    assert m0["extracted_frame_filename"] == "frames/frame_000000.jpg"
    assert m0["frame_width"] == 320
    assert m0["frame_height"] == 240

    m1 = json.loads(lines[1])
    assert m1["original_frame_index"] == 30
    assert abs(m1["timestamp_ms"] - 1000.0) < 2.0

    # Verify video_metadata.json
    metadata_path = output_dir / "video_metadata.json"
    assert metadata_path.exists()
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert meta["total_frames"] == 60
    assert meta["width"] == 320
    assert meta["height"] == 240


def test_extract_frames_every_n_frames(synthetic_video_path: Path, tmp_path: Path):
    """Verify extraction sampled every N frames."""
    output_dir = tmp_path / "output_frames"
    summary = extract_frames(
        video_path=synthetic_video_path,
        output_dir=output_dir,
        every_n_frames=20,
    )

    # 60 frames / step 20 -> indices 0, 20, 40 (3 frames)
    assert summary["extracted_frames_count"] == 3
    manifest_path = output_dir / "frames.jsonl"
    lines = manifest_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    indices = [json.loads(line)["original_frame_index"] for line in lines]
    assert indices == [0, 20, 40]


def test_overwrite_protection(synthetic_video_path: Path, tmp_path: Path):
    """Verify FrameExtractor prevents accidental overwriting unless explicit."""
    output_dir = tmp_path / "output_protected"

    # First extraction creates files
    extract_frames(
        video_path=synthetic_video_path,
        output_dir=output_dir,
        every_n_frames=30,
    )

    # Second extraction without overwrite MUST raise FileExistsError
    with pytest.raises(FileExistsError, match="Output directory exists and is not empty"):
        extract_frames(
            video_path=synthetic_video_path,
            output_dir=output_dir,
            every_n_frames=30,
            overwrite=False,
        )

    # Third extraction with overwrite=True succeeds
    summary = extract_frames(
        video_path=synthetic_video_path,
        output_dir=output_dir,
        every_n_frames=30,
        overwrite=True,
    )
    assert summary["extracted_frames_count"] == 2


def test_parameter_validation(synthetic_video_path: Path, tmp_path: Path):
    """Verify input validation rules for sampling intervals and quality."""
    # Mutually exclusive interval specification
    with pytest.raises(ValueError, match="Specify either 'every_seconds' or 'every_n_frames'"):
        FrameExtractor(synthetic_video_path, every_seconds=1.0, every_n_frames=30)

    # Negative seconds
    with pytest.raises(ValueError, match="every_seconds must be positive"):
        FrameExtractor(synthetic_video_path, every_seconds=-0.5)

    # Zero/negative frames
    with pytest.raises(ValueError, match="every_n_frames must be >= 1"):
        FrameExtractor(synthetic_video_path, every_n_frames=0)

    # Invalid JPEG quality
    with pytest.raises(ValueError, match="jpeg_quality must be between 1 and 100"):
        FrameExtractor(synthetic_video_path, jpeg_quality=105)
