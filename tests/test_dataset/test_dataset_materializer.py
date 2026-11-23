"""Unit tests for YOLO dataset materializer, image-to-label alignment, and verification."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sportiq.dataset.annotation_parser import load_sequence
from sportiq.dataset.class_mapping import ClassTaxonomyType
from sportiq.dataset.dataset_materializer import (
    AlignmentReport,
    AlignmentVerificationError,
    DatasetMaterializer,
)
from sportiq.dataset.dataset_splitter import DatasetSplitter
from sportiq.dataset.yolo_converter import YOLOConverter


def test_materialize_manifest_mode(synthetic_mot_sequence: Path, tmp_path: Path) -> None:
    """Test materializing dataset in zero-duplication manifest mode."""
    seq = load_sequence(synthetic_mot_sequence)
    sequences = {seq.name: seq}

    # 1. Convert sequence to generate labels
    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    converter.convert_sequence(seq, output_dir=tmp_path / "yolo")
    labels_dir = tmp_path / "yolo" / "labels"

    # 2. Split
    splitter = DatasetSplitter(random_seed=42)
    split = splitter.split_sequences([seq.name], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    # 3. Materialize
    mat_dir = tmp_path / "materialized_manifest"
    summary = DatasetMaterializer.materialize(
        split=split,
        sequences=sequences,
        labels_base_dir=labels_dir,
        output_dir=mat_dir,
        mode="manifest",
    )

    assert summary["status"] == "materialized"
    assert summary["mode"] == "manifest"
    assert summary["total_images"] == 2
    assert summary["total_labeled_frames"] == 2

    # Verify manifest text files
    assert (mat_dir / "train.txt").exists()
    assert (mat_dir / "val.txt").exists()
    assert (mat_dir / "test.txt").exists()
    train_lines = (mat_dir / "train.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == 2

    # Verify dataset.yaml
    yaml_path = mat_dir / "dataset.yaml"
    assert yaml_path.exists()
    ydata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert ydata["train"] == "train.txt"

    # Verify alignment
    report = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert isinstance(report, AlignmentReport)
    assert report.is_aligned is True
    assert report.total_images == 2
    assert len(report.missing_images) == 0
    assert len(report.corrupted_labels) == 0


def test_materialize_copy_mode(synthetic_mot_sequence: Path, tmp_path: Path) -> None:
    """Test materializing dataset in physical directory layout (images/train, labels/train)."""
    seq = load_sequence(synthetic_mot_sequence)
    sequences = {seq.name: seq}

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    converter.convert_sequence(seq, output_dir=tmp_path / "yolo")
    labels_dir = tmp_path / "yolo" / "labels"

    splitter = DatasetSplitter(random_seed=42)
    split = splitter.split_sequences([seq.name], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    mat_dir = tmp_path / "materialized_copy"
    summary = DatasetMaterializer.materialize(
        split=split,
        sequences=sequences,
        labels_base_dir=labels_dir,
        output_dir=mat_dir,
        mode="copy",
    )

    assert summary["status"] == "materialized"
    assert (mat_dir / "images" / "train" / f"{seq.name}_000001.jpg").exists()
    assert (mat_dir / "labels" / "train" / f"{seq.name}_000001.txt").exists()

    yaml_path = mat_dir / "dataset.yaml"
    report = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert report.is_aligned is True
    assert report.total_images == 2


def test_verify_dataset_detects_missing_image(tmp_path: Path) -> None:
    """Test verification audit flags non-existent referenced image files."""
    dataset_dir = tmp_path / "broken_dataset"
    dataset_dir.mkdir()
    manifest_txt = dataset_dir / "train.txt"
    manifest_txt.write_text("C:/nonexistent/image_000001.jpg\n", encoding="utf-8")

    ydata = {
        "path": dataset_dir.as_posix(),
        "train": "train.txt",
        "val": "train.txt",
        "test": "train.txt",
        "names": {0: "player"},
    }
    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(yaml.dump(ydata), encoding="utf-8")

    report = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert report.is_aligned is False
    assert len(report.missing_images) > 0


def test_verify_dataset_detects_corrupted_boxes(synthetic_mot_sequence: Path, tmp_path: Path) -> None:
    """Test verification audit flags out-of-range coordinates in YOLO label files."""
    seq = load_sequence(synthetic_mot_sequence)
    sequences = {seq.name: seq}

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    converter.convert_sequence(seq, output_dir=tmp_path / "yolo")
    labels_dir = tmp_path / "yolo" / "labels"

    # Corrupt a label file: x_center = 2.5 (out of [0, 1])
    bad_lbl = labels_dir / seq.name / "000001.txt"
    bad_lbl.write_text("0 2.500000 0.500000 0.100000 0.200000\n", encoding="utf-8")

    splitter = DatasetSplitter(random_seed=42)
    split = splitter.split_sequences([seq.name], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    mat_dir = tmp_path / "mat_corrupt"
    with pytest.raises(AlignmentVerificationError, match="YOLO coordinate out of"):
        DatasetMaterializer.materialize(
            split=split,
            sequences=sequences,
            labels_base_dir=labels_dir,
            output_dir=mat_dir,
            mode="manifest",
        )
