"""Pytest fixtures for SportIQ test suite."""

import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest


def create_synthetic_video(
    output_path: Path,
    width: int = 320,
    height: int = 240,
    fps: float = 30.0,
    num_frames: int = 60,
) -> Path:
    """Generate a valid synthetic MP4 video file for testing.

    Draws a bouncing rectangle and frame counter to guarantee varying frame content.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        # Fallback to MJPG if mp4v is not supported in the environment
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to create OpenCV VideoWriter for {output_path}")

    for idx in range(num_frames):
        # Create a green background (simulating pitch)
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)

        # Draw moving white circle (simulating ball)
        cx = int(50 + (idx * 3) % (width - 100))
        cy = int(height / 2 + int(np.sin(idx / 5.0) * 40))
        cv2.circle(frame, (cx, cy), 12, (255, 255, 255), -1)

        # Draw frame number
        cv2.putText(
            frame,
            f"F:{idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)

    out.release()
    return output_path


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> Path:
    """Fixture providing a 60-frame, 30-FPS synthetic video (2.0s duration)."""
    video_file = tmp_path / "synthetic_test_video.mp4"
    return create_synthetic_video(
        video_file, width=320, height=240, fps=30.0, num_frames=60
    )


@pytest.fixture
def empty_file_path(tmp_path: Path) -> Path:
    """Fixture providing a 0-byte file."""
    empty_file = tmp_path / "empty_video.mp4"
    empty_file.write_bytes(b"")
    return empty_file


@pytest.fixture
def corrupt_file_path(tmp_path: Path) -> Path:
    """Fixture providing a corrupt non-video file with an .mp4 extension."""
    corrupt_file = tmp_path / "corrupt_video.mp4"
    corrupt_file.write_text("This is not a real video stream header content.")
    return corrupt_file
