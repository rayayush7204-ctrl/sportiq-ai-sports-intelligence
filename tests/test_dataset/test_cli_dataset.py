"""Unit tests for dataset CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sportiq.cli import main


def test_cli_convert_dataset(synthetic_mot_sequence: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test sportiq convert-dataset CLI command."""
    out_dir = tmp_path / "cli_yolo_out"
    ret = main([
        "convert-dataset",
        str(synthetic_mot_sequence),
        "--output-dir", str(out_dir),
        "--taxonomy", "four_class",
        "--json",
    ])

    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["sequence_name"] == "SNMOT-SYNTH-01"
    assert data["exported_yolo_boxes"] == 8
    assert data["skipped_unmapped_boxes"] == 2


def test_cli_split_dataset(synthetic_multi_sequences: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test sportiq split-dataset CLI command."""
    yaml_out = tmp_path / "dataset.yaml"
    ret = main([
        "split-dataset",
        str(synthetic_multi_sequences),
        "--train-ratio", "0.6",
        "--val-ratio", "0.2",
        "--test-ratio", "0.2",
        "--seed", "42",
        "--output-yaml", str(yaml_out),
        "--json",
    ])

    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["total_count"] == 6
    assert yaml_out.exists()


def test_cli_dataset_stats(synthetic_mot_sequence: Path, capsys: pytest.CaptureFixture) -> None:
    """Test sportiq dataset-stats CLI command."""
    ret = main([
        "dataset-stats",
        str(synthetic_mot_sequence),
        "--taxonomy", "four_class",
        "--json",
    ])

    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["total_sequences"] == 1
    assert data["total_annotations"] == 10
    assert data["staff_annotation_count"] == 2


def test_cli_visualize_annotations(synthetic_mot_sequence: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test sportiq visualize-annotations CLI command."""
    out_img = tmp_path / "vis_output.jpg"
    ret = main([
        "visualize-annotations",
        str(synthetic_mot_sequence),
        "--frame-id", "1",
        "--output", str(out_img),
        "--json",
    ])

    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "success"
    assert out_img.exists()
