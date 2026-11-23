"""Unit tests for YOLO format conversion, bbox math, and metadata sidecar generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sportiq.dataset.annotation_parser import load_sequence
from sportiq.dataset.class_mapping import ClassTaxonomyType
from sportiq.dataset.yolo_converter import YOLOConverter, YOLOInstance


def test_pixel_to_yolo_bbox_math() -> None:
    """Test math of pixel (x, y, w, h) to normalized (x_center, y_center, w, h)."""
    # Top-left (100, 200), width 50, height 100 on 1000x500 image:
    # x_center = (100 + 25) / 1000 = 0.125
    # y_center = (200 + 50) / 500 = 0.50
    # w_norm = 50 / 1000 = 0.05
    # h_norm = 100 / 500 = 0.20
    xc, yc, wn, hn = YOLOConverter.pixel_to_yolo_bbox(
        left=100.0,
        top=200.0,
        width=50.0,
        height=100.0,
        image_width=1000,
        image_height=500,
    )

    assert pytest.approx(xc, rel=1e-5) == 0.125
    assert pytest.approx(yc, rel=1e-5) == 0.50
    assert pytest.approx(wn, rel=1e-5) == 0.05
    assert pytest.approx(hn, rel=1e-5) == 0.20


def test_pixel_to_yolo_bbox_invalid_dimensions() -> None:
    """Test error raised when canvas dimension is non-positive."""
    with pytest.raises(ValueError, match="Invalid image dimensions"):
        YOLOConverter.pixel_to_yolo_bbox(10, 10, 20, 20, 0, 1080)


def test_yolo_instance_line() -> None:
    """Test YOLOInstance formatting to standard YOLO line."""
    instance = YOLOInstance(class_id=0, x_center=0.5, y_center=0.25, width=0.1, height=0.2)
    line = instance.to_yolo_line()
    assert line == "0 0.500000 0.250000 0.100000 0.200000"


def test_convert_sequence_four_class(synthetic_mot_sequence: Path, tmp_path: Path) -> None:
    """Test end-to-end YOLO sequence conversion in 4-class mode."""
    seq = load_sequence(synthetic_mot_sequence)
    out_dir = tmp_path / "yolo_export"

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    summary = converter.convert_sequence(seq, output_dir=out_dir)

    assert summary["sequence_name"] == "SNMOT-SYNTH-01"
    assert summary["taxonomy"] == "four_class"
    assert summary["total_annotations"] == 10
    # In 4-class: 10 annotations total, 2 staff annotations are skipped from YOLO txt
    # 8 exported to YOLO .txt
    assert summary["exported_yolo_boxes"] == 8
    assert summary["skipped_unmapped_boxes"] == 2
    assert summary["total_sidecar_records"] == 10

    labels_dir = Path(summary["labels_directory"])
    assert labels_dir.exists()
    assert (labels_dir / "000001.txt").exists()
    assert (labels_dir / "000002.txt").exists()

    # Verify frame 1 YOLO txt content
    frame1_lines = (labels_dir / "000001.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(frame1_lines) == 4  # player(0), goalkeeper(1), referee(2), ball(3)
    class_ids = [int(line.split()[0]) for line in frame1_lines]
    assert class_ids == [0, 1, 2, 3]

    # Verify sidecar JSONL manifest
    manifest_path = Path(summary["manifest_path"])
    assert manifest_path.exists()
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(records) == 10

    # Verify staff is preserved in sidecar with original track_id and raw class
    staff_records = [r for r in records if r["track_id"] == 5]
    assert len(staff_records) == 2
    for r in staff_records:
        assert r["raw_class"] == "staff"
        assert r["canonical_class"] == "staff"
        assert r["class_id"] is None  # Unmapped for 4-class detector, but preserved in sidecar


def test_convert_sequence_five_class(synthetic_mot_sequence: Path, tmp_path: Path) -> None:
    """Test YOLO sequence conversion in 5-class mode where staff is mapped to class 4."""
    seq = load_sequence(synthetic_mot_sequence)
    out_dir = tmp_path / "yolo_export_5class"

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FIVE_CLASS)
    summary = converter.convert_sequence(seq, output_dir=out_dir)

    assert summary["exported_yolo_boxes"] == 10
    assert summary["skipped_unmapped_boxes"] == 0

    labels_dir = Path(summary["labels_directory"])
    frame1_lines = (labels_dir / "000001.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(frame1_lines) == 5
    class_ids = [int(line.split()[0]) for line in frame1_lines]
    assert class_ids == [0, 1, 2, 3, 4]
