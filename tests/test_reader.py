"""Unit tests for VideoReader and metadata inspection."""

from pathlib import Path

import pytest

from sportiq.video.metadata import FrameMetadata, VideoMetadata
from sportiq.video.reader import InvalidVideoError, VideoReader


def test_video_metadata_serialization():
    """Verify VideoMetadata and FrameMetadata serialization methods."""
    meta = VideoMetadata(
        source_path="/path/test.mp4",
        filename="test.mp4",
        file_size_bytes=1048576,
        file_size_mb=1.0,
        codec="mp4v",
        width=1920,
        height=1080,
        fps=30.0,
        total_frames=300,
        duration_seconds=10.0,
    )
    d = meta.to_dict()
    assert d["width"] == 1920
    assert d["height"] == 1080
    assert d["fps"] == 30.0
    assert d["duration_seconds"] == 10.0
    assert '"width": 1920' in meta.to_json()

    frame_meta = FrameMetadata(
        source_filename="test.mp4",
        original_frame_index=15,
        timestamp_ms=500.0,
        extracted_frame_filename="frames/frame_000015.jpg",
        frame_width=1920,
        frame_height=1080,
    )
    fd = frame_meta.to_dict()
    assert fd["original_frame_index"] == 15
    assert fd["timestamp_ms"] == 500.0
    assert '"original_frame_index": 15' in frame_meta.to_json()


def test_valid_video_inspection(synthetic_video_path: Path):
    """Verify VideoReader correctly parses dimensions, FPS, and frame count."""
    reader = VideoReader(synthetic_video_path)
    metadata = reader.get_metadata()

    assert metadata.filename == synthetic_video_path.name
    # Ensure source_path is safe and does not leak Windows drive paths (e.g. C:\)
    assert not metadata.source_path.startswith("C:")
    assert "\\" not in metadata.source_path
    assert metadata.width == 320
    assert metadata.height == 240
    assert metadata.fps == 30.0
    assert metadata.total_frames == 60
    assert round(metadata.duration_seconds, 1) == 2.0
    assert metadata.file_size_bytes > 0
    assert metadata.file_size_mb >= 0.01


def test_missing_video_file(tmp_path: Path):
    """Verify VideoReader raises FileNotFoundError when file does not exist."""
    missing_file = tmp_path / "nonexistent.mp4"
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        VideoReader(missing_file)


def test_empty_video_file(empty_file_path: Path):
    """Verify VideoReader raises InvalidVideoError when video is empty (0 bytes)."""
    with pytest.raises(InvalidVideoError, match="Video file is empty"):
        VideoReader(empty_file_path)


def test_corrupt_video_file(corrupt_file_path: Path):
    """Verify VideoReader raises InvalidVideoError for corrupt/unreadable files."""
    reader = VideoReader(corrupt_file_path)
    with pytest.raises(InvalidVideoError, match="(Failed to open video file|contains no readable frames)"):
        reader.get_metadata()


def test_directory_passed_as_video(tmp_path: Path):
    """Verify VideoReader raises InvalidVideoError when given a directory path."""
    with pytest.raises(InvalidVideoError, match="Specified path is not a file"):
        VideoReader(tmp_path)


def test_sequential_frame_decoding(synthetic_video_path: Path):
    """Verify read_frames yields all frames with sequential indices and timestamps."""
    reader = VideoReader(synthetic_video_path)
    frames_read = list(reader.read_frames(sample_every_n_frames=1))

    assert len(frames_read) == 60
    for expected_idx, (frame_idx, timestamp_ms, frame_bgr) in enumerate(frames_read):
        assert frame_idx == expected_idx
        expected_ms = round((expected_idx / 30.0) * 1000.0, 2)
        assert abs(timestamp_ms - expected_ms) < 2.0  # within 2ms tolerance
        assert frame_bgr.shape == (240, 320, 3)


def test_stepped_frame_decoding(synthetic_video_path: Path):
    """Verify read_frames respects sample_every_n_frames stepping."""
    reader = VideoReader(synthetic_video_path)
    frames_read = list(reader.read_frames(sample_every_n_frames=15))

    # 60 frames sampled every 15 -> indices 0, 15, 30, 45 (4 frames total)
    assert len(frames_read) == 4
    indices = [f[0] for f in frames_read]
    assert indices == [0, 15, 30, 45]


def test_invalid_sampling_step(synthetic_video_path: Path):
    """Verify read_frames rejects step < 1."""
    reader = VideoReader(synthetic_video_path)
    with pytest.raises(ValueError, match="sample_every_n_frames must be >= 1"):
        list(reader.read_frames(sample_every_n_frames=0))


def test_source_path_inside_workspace(make_synthetic_video):
    """Verify source file inside project workspace produces a relative POSIX path."""
    local_video_path = Path("data/raw/test_regression_inside.mp4")
    try:
        make_synthetic_video(local_video_path, width=160, height=120, fps=10.0, num_frames=5)
        reader = VideoReader(local_video_path)
        meta = reader.get_metadata()
        assert meta.source_path == "data/raw/test_regression_inside.mp4"
        assert not meta.source_path.startswith("C:")
        assert "\\" not in meta.source_path
    finally:
        if local_video_path.exists():
            local_video_path.unlink()


def test_source_path_outside_workspace(tmp_path: Path, make_synthetic_video):
    """Verify source file outside project workspace falls back to its filename."""
    outside_video = tmp_path / "outside_regression_clip.mp4"
    make_synthetic_video(outside_video, width=160, height=120, fps=10.0, num_frames=5)
    reader = VideoReader(outside_video)
    meta = reader.get_metadata()
    assert meta.source_path == "outside_regression_clip.mp4"
    assert "\\" not in meta.source_path
    assert not meta.source_path.startswith("C:")


def test_no_windows_host_leakage_in_exported_metadata(synthetic_video_path: Path):
    """Verify exported JSON never leaks Windows username or absolute host drive paths."""
    import os

    reader = VideoReader(synthetic_video_path)
    meta = reader.get_metadata()
    json_str = meta.to_json()
    username = os.environ.get("USERNAME", "")
    if username:
        assert username not in json_str
    assert "C:\\" not in json_str
    assert "c:\\" not in json_str
    assert "Users\\" not in json_str
    assert "Users/" not in json_str
