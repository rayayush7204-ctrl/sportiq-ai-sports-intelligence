"""Unit tests for annotation visualization and frame rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sportiq.dataset.annotation_parser import MOTAnnotation, load_sequence
from sportiq.dataset.visualizer import draw_annotations, visualize_sequence_frame


def test_draw_annotations_on_image() -> None:
    """Test drawing bounding boxes and labels onto an image array."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    anns = [
        MOTAnnotation(1, 1, 50.0, 50.0, 100.0, 100.0, 1.0),
        MOTAnnotation(1, 2, 200.0, 200.0, 50.0, 50.0, 1.0),
    ]

    result = draw_annotations(img, anns)

    assert result.shape == (500, 500, 3)
    # Result must not be completely black (bounding boxes were drawn)
    assert np.any(result > 0)


def test_visualize_sequence_frame(synthetic_mot_sequence: Path, tmp_path: Path) -> None:
    """Test visualizer renders and saves image to disk."""
    seq = load_sequence(synthetic_mot_sequence)
    out_img = tmp_path / "vis_frame_1.jpg"

    vis = visualize_sequence_frame(
        sequence=seq,
        frame_id=1,
        output_path=out_img,
    )

    assert vis.shape == (1080, 1920, 3)
    assert out_img.exists()
    assert out_img.stat().st_size > 0
