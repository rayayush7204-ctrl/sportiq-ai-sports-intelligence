"""Unit tests for YOLO dataset materializer, image-to-label alignment, and verification."""

from __future__ import annotations

import math
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
    assert len(report.missing_labels) == 0
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
    assert len(report.missing_labels) == 0


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


def test_verify_dataset_detects_deleted_positive_label(synthetic_mot_sequence: Path, tmp_path: Path) -> None:
    """Test verification rejects datasets where a positive label file has been deleted."""
    seq = load_sequence(synthetic_mot_sequence)
    sequences = {seq.name: seq}

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    converter.convert_sequence(seq, output_dir=tmp_path / "yolo")
    labels_dir = tmp_path / "yolo" / "labels"

    splitter = DatasetSplitter(random_seed=42)
    split = splitter.split_sequences([seq.name], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    mat_dir = tmp_path / "mat_del_label"
    DatasetMaterializer.materialize(
        split=split,
        sequences=sequences,
        labels_base_dir=labels_dir,
        output_dir=mat_dir,
        mode="copy",
    )
    yaml_path = mat_dir / "dataset.yaml"

    # Ensure baseline is aligned
    rep_before = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert rep_before.is_aligned is True

    # Deliberately delete a positive label file
    lbl_to_delete = mat_dir / "labels" / "train" / f"{seq.name}_000001.txt"
    assert lbl_to_delete.exists()
    lbl_to_delete.unlink()

    # Re-verify: must be unaligned and missing_labels populated
    rep_after = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert rep_after.is_aligned is False
    assert len(rep_after.missing_labels) == 1
    assert str(lbl_to_delete) in rep_after.missing_labels


def test_verify_dataset_detects_deleted_image(synthetic_mot_sequence: Path, tmp_path: Path) -> None:
    """Test verification rejects datasets where an image file has been deleted."""
    seq = load_sequence(synthetic_mot_sequence)
    sequences = {seq.name: seq}

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    converter.convert_sequence(seq, output_dir=tmp_path / "yolo")
    labels_dir = tmp_path / "yolo" / "labels"

    splitter = DatasetSplitter(random_seed=42)
    split = splitter.split_sequences([seq.name], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    mat_dir = tmp_path / "mat_del_img"
    DatasetMaterializer.materialize(
        split=split,
        sequences=sequences,
        labels_base_dir=labels_dir,
        output_dir=mat_dir,
        mode="copy",
    )
    yaml_path = mat_dir / "dataset.yaml"

    # Deliberately delete an image
    img_to_delete = mat_dir / "images" / "train" / f"{seq.name}_000001.jpg"
    assert img_to_delete.exists()
    img_to_delete.unlink()

    rep = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert rep.is_aligned is False
    assert len(rep.missing_images) == 1
    assert str(img_to_delete) in rep.missing_images


@pytest.mark.parametrize(
    "corrupt_content,error_match",
    [
        ("99 0.500000 0.500000 0.100000 0.200000\n", "Invalid class ID 99"),
        ("0 nan 0.500000 0.100000 0.200000\n", "Non-finite YOLO coordinate"),
        ("0 inf 0.500000 0.100000 0.200000\n", "Non-finite YOLO coordinate"),
        ("0 1.500000 0.500000 0.100000 0.200000\n", "YOLO coordinate out of [0, 1] range"),
        ("0 0.500000 0.500000 0.100000\n", "expected 5 tokens"),
        ("abc 0.500000 0.500000 0.100000 0.200000\n", "Non-numeric token"),
    ],
)
def test_verify_dataset_detects_various_corrupted_labels(
    synthetic_mot_sequence: Path, tmp_path: Path, corrupt_content: str, error_match: str
) -> None:
    """Test verification detects nan, inf, out-of-bounds, invalid class ID, and malformed lines."""
    seq = load_sequence(synthetic_mot_sequence)
    sequences = {seq.name: seq}

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    converter.convert_sequence(seq, output_dir=tmp_path / "yolo")
    labels_dir = tmp_path / "yolo" / "labels"

    splitter = DatasetSplitter(random_seed=42)
    split = splitter.split_sequences([seq.name], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    mat_dir = tmp_path / f"mat_corrupt_{abs(hash(corrupt_content)) % 10000}"
    DatasetMaterializer.materialize(
        split=split,
        sequences=sequences,
        labels_base_dir=labels_dir,
        output_dir=mat_dir,
        mode="copy",
    )
    yaml_path = mat_dir / "dataset.yaml"

    # Inject corrupt content into one label file
    lbl = mat_dir / "labels" / "train" / f"{seq.name}_000001.txt"
    lbl.write_text(corrupt_content, encoding="utf-8")

    rep = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert rep.is_aligned is False
    assert len(rep.corrupted_labels) == 1
    assert error_match in rep.corrupted_labels[0]


def test_verify_dataset_intentional_negative_frame(synthetic_mot_sequence: Path, tmp_path: Path) -> None:
    """Test verification correctly accepts 0-byte empty label files as background frames."""
    seq = load_sequence(synthetic_mot_sequence)
    sequences = {seq.name: seq}

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    converter.convert_sequence(seq, output_dir=tmp_path / "yolo")
    labels_dir = tmp_path / "yolo" / "labels"

    splitter = DatasetSplitter(random_seed=42)
    split = splitter.split_sequences([seq.name], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    mat_dir = tmp_path / "mat_negative_frame"
    DatasetMaterializer.materialize(
        split=split,
        sequences=sequences,
        labels_base_dir=labels_dir,
        output_dir=mat_dir,
        mode="copy",
    )
    yaml_path = mat_dir / "dataset.yaml"

    # Turn frame 000001 into an intentional negative / background frame (0 bytes)
    lbl = mat_dir / "labels" / "train" / f"{seq.name}_000001.txt"
    lbl.write_text("", encoding="utf-8")

    rep = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert rep.is_aligned is True
    assert rep.total_images == 2
    assert rep.total_labels == 2
    assert rep.background_frames == 1
    assert rep.labeled_frames == 1
    assert len(rep.missing_labels) == 0
    assert len(rep.corrupted_labels) == 0


def test_verify_dataset_detects_broken_manifest_mapping(synthetic_mot_sequence: Path, tmp_path: Path) -> None:
    """Test verification audit rejects broken manifest mappings when label file is missing."""
    seq = load_sequence(synthetic_mot_sequence)
    sequences = {seq.name: seq}

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    converter.convert_sequence(seq, output_dir=tmp_path / "yolo")
    labels_dir = tmp_path / "yolo" / "labels"

    splitter = DatasetSplitter(random_seed=42)
    split = splitter.split_sequences([seq.name], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    mat_dir = tmp_path / "mat_broken_manifest"
    DatasetMaterializer.materialize(
        split=split,
        sequences=sequences,
        labels_base_dir=labels_dir,
        output_dir=mat_dir,
        mode="manifest",
    )
    yaml_path = mat_dir / "dataset.yaml"

    # Verify baseline is aligned
    rep_before = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert rep_before.is_aligned is True

    # Deliberately delete the source label for frame 000001
    source_lbl = labels_dir / seq.name / "000001.txt"
    assert source_lbl.exists()
    source_lbl.unlink()

    # Verify audit reports missing label
    rep_after = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert rep_after.is_aligned is False
    assert len(rep_after.missing_labels) >= 1


@pytest.mark.parametrize(
    "corrupt_content,error_match",
    [
        ("0 0.100000 0.500000 0.400000 0.200000\n", "left canvas boundary"),
        ("0 0.900000 0.500000 0.400000 0.200000\n", "right canvas boundary"),
        ("0 0.500000 0.050000 0.200000 0.200000\n", "top canvas boundary"),
        ("0 0.500000 0.950000 0.200000 0.200000\n", "bottom canvas boundary"),
    ],
)
def test_verify_dataset_geometric_bbox_boundaries_rejected(
    synthetic_mot_sequence: Path, tmp_path: Path, corrupt_content: str, error_match: str
) -> None:
    """Test verification rejects boxes extending beyond left, right, top, or bottom edges."""
    seq = load_sequence(synthetic_mot_sequence)
    sequences = {seq.name: seq}

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    converter.convert_sequence(seq, output_dir=tmp_path / "yolo")
    labels_dir = tmp_path / "yolo" / "labels"

    splitter = DatasetSplitter(random_seed=42)
    split = splitter.split_sequences([seq.name], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    mat_dir = tmp_path / f"mat_geom_{abs(hash(corrupt_content)) % 10000}"
    DatasetMaterializer.materialize(
        split=split,
        sequences=sequences,
        labels_base_dir=labels_dir,
        output_dir=mat_dir,
        mode="copy",
    )
    yaml_path = mat_dir / "dataset.yaml"

    # Inject geometric boundary violation
    lbl = mat_dir / "labels" / "train" / f"{seq.name}_000001.txt"
    lbl.write_text(corrupt_content, encoding="utf-8")

    rep = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert rep.is_aligned is False
    assert len(rep.corrupted_labels) == 1
    assert error_match in rep.corrupted_labels[0]


def test_verify_dataset_geometric_bbox_within_canvas_accepted(
    synthetic_mot_sequence: Path, tmp_path: Path
) -> None:
    """Test verification accepts boxes fully within canvas boundaries."""
    seq = load_sequence(synthetic_mot_sequence)
    sequences = {seq.name: seq}

    converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS)
    converter.convert_sequence(seq, output_dir=tmp_path / "yolo")
    labels_dir = tmp_path / "yolo" / "labels"

    splitter = DatasetSplitter(random_seed=42)
    split = splitter.split_sequences([seq.name], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    mat_dir = tmp_path / "mat_geom_valid"
    DatasetMaterializer.materialize(
        split=split,
        sequences=sequences,
        labels_base_dir=labels_dir,
        output_dir=mat_dir,
        mode="copy",
    )
    yaml_path = mat_dir / "dataset.yaml"

    # Box entirely within canvas: xc=0.5, yc=0.5, w=0.2, h=0.2 -> bounds [0.4, 0.4, 0.6, 0.6]
    lbl = mat_dir / "labels" / "train" / f"{seq.name}_000001.txt"
    lbl.write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")

    rep = DatasetMaterializer.verify_dataset_alignment(yaml_path)
    assert rep.is_aligned is True
    assert len(rep.corrupted_labels) == 0
