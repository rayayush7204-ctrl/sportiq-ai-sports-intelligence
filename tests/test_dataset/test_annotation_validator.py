"""Unit tests for annotation bounding box validator and coordinate sanitization."""

from __future__ import annotations

import pytest

from sportiq.dataset.annotation_parser import MOTAnnotation
from sportiq.dataset.annotation_validator import AnnotationValidator, ValidationResult


def test_validate_annotation_valid() -> None:
    """Test standard bounding box entirely within image canvas."""
    val = AnnotationValidator(clip_boxes=True)
    ann = MOTAnnotation(
        frame_id=1, track_id=1,
        bb_left=100.0, bb_top=200.0, bb_width=50.0, bb_height=100.0,
        conf=1.0,
    )

    is_valid, sanitized, status = val.validate_annotation(ann, image_width=1920, image_height=1080)
    assert is_valid is True
    assert status == "valid"
    assert sanitized == (100.0, 200.0, 50.0, 100.0)


def test_validate_annotation_non_positive_dims() -> None:
    """Test rejection of non-positive widths or heights."""
    val = AnnotationValidator(clip_boxes=True)
    ann_zero_w = MOTAnnotation(frame_id=1, track_id=1, bb_left=10.0, bb_top=10.0, bb_width=0.0, bb_height=20.0, conf=1.0)
    ann_neg_h = MOTAnnotation(frame_id=1, track_id=1, bb_left=10.0, bb_top=10.0, bb_width=20.0, bb_height=-5.0, conf=1.0)

    is_valid_1, _, status_1 = val.validate_annotation(ann_zero_w, 1920, 1080)
    assert is_valid_1 is False
    assert status_1 == "non_positive_dimensions"

    is_valid_2, _, status_2 = val.validate_annotation(ann_neg_h, 1920, 1080)
    assert is_valid_2 is False
    assert status_2 == "non_positive_dimensions"


def test_validate_annotation_completely_out_of_bounds() -> None:
    """Test rejection of bounding boxes entirely outside canvas."""
    val = AnnotationValidator(clip_boxes=True)
    ann_outside_right = MOTAnnotation(frame_id=1, track_id=1, bb_left=2000.0, bb_top=100.0, bb_width=50.0, bb_height=50.0, conf=1.0)
    ann_outside_left = MOTAnnotation(frame_id=1, track_id=1, bb_left=-200.0, bb_top=100.0, bb_width=50.0, bb_height=50.0, conf=1.0)

    is_valid_1, _, status_1 = val.validate_annotation(ann_outside_right, 1920, 1080)
    assert is_valid_1 is False
    assert status_1 == "out_of_bounds"

    is_valid_2, _, status_2 = val.validate_annotation(ann_outside_left, 1920, 1080)
    assert is_valid_2 is False
    assert status_2 == "out_of_bounds"


def test_validate_annotation_partially_out_of_bounds_clipped() -> None:
    """Test clamping of partially out-of-bounds boxes when clip_boxes=True."""
    val = AnnotationValidator(clip_boxes=True)
    # Box starts at left -20 and width 50 (extends to +30)
    ann = MOTAnnotation(frame_id=1, track_id=1, bb_left=-20.0, bb_top=10.0, bb_width=50.0, bb_height=40.0, conf=1.0)

    is_valid, sanitized, status = val.validate_annotation(ann, 1920, 1080)
    assert is_valid is True
    assert status == "clipped"
    assert sanitized is not None
    assert sanitized[0] == 0.0  # Clamped to left border
    assert sanitized[1] == 10.0
    assert sanitized[2] == 30.0  # Width reduced to visible portion
    assert sanitized[3] == 40.0


def test_validate_annotation_partially_out_of_bounds_no_clip() -> None:
    """Test rejection of partially out-of-bounds boxes when clip_boxes=False."""
    val = AnnotationValidator(clip_boxes=False)
    ann = MOTAnnotation(frame_id=1, track_id=1, bb_left=-20.0, bb_top=10.0, bb_width=50.0, bb_height=40.0, conf=1.0)

    is_valid, sanitized, status = val.validate_annotation(ann, 1920, 1080)
    assert is_valid is False
    assert status == "partially_out_of_bounds"
    assert sanitized is None


def test_filter_and_validate_aggregation() -> None:
    """Test batch filtering and validation metrics compilation."""
    val = AnnotationValidator(clip_boxes=True)
    anns = [
        MOTAnnotation(1, 1, 10.0, 10.0, 50.0, 50.0, 1.0),     # Valid
        MOTAnnotation(1, 2, -10.0, 10.0, 50.0, 50.0, 1.0),    # Clipped
        MOTAnnotation(1, 3, 3000.0, 10.0, 50.0, 50.0, 1.0),   # Out of bounds
        MOTAnnotation(1, 4, 10.0, 10.0, -5.0, 50.0, 1.0),     # Non-positive dim
    ]

    valid_pairs, result = val.filter_and_validate(anns, image_width=1920, image_height=1080)

    assert len(valid_pairs) == 2
    assert isinstance(result, ValidationResult)
    assert result.total_inspected == 4
    assert result.valid_count == 2
    assert result.clipped_count == 1
    assert result.dropped_out_of_bounds == 1
    assert result.dropped_non_positive_dims == 1
    assert result.to_dict()["pass_rate_pct"] == 50.0
