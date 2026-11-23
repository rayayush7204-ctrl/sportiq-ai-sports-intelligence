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


def test_cli_split_dataset_missing_game_id_strict(synthetic_multi_sequences: Path, capsys: pytest.CaptureFixture) -> None:
    """Test CLI split-dataset fails when match metadata is incomplete."""
    # Delete gameinfo.ini from one sequence to trigger incomplete match metadata
    bad_gameinfo = synthetic_multi_sequences / "SNMOT-001" / "gameinfo.ini"
    if bad_gameinfo.exists():
        bad_gameinfo.unlink()

    ret = main([
        "split-dataset",
        str(synthetic_multi_sequences),
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Incomplete match metadata" in captured.err


def test_cli_split_dataset_allow_missing_game_id(synthetic_multi_sequences: Path, capsys: pytest.CaptureFixture) -> None:
    """Test CLI split-dataset succeeds with warning when --allow-missing-game-id is passed."""
    # Delete gameinfo.ini from one sequence
    bad_gameinfo = synthetic_multi_sequences / "SNMOT-001" / "gameinfo.ini"
    if bad_gameinfo.exists():
        bad_gameinfo.unlink()

    ret = main([
        "split-dataset",
        str(synthetic_multi_sequences),
        "--allow-missing-game-id",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["total_count"] == 6


def test_cli_materialize_and_verify_dataset(synthetic_mot_sequence: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test sportiq materialize-dataset and verify-dataset CLI commands end-to-end."""
    yolo_labels = tmp_path / "yolo_labels"
    # Convert first and flush captured output
    main([
        "convert-dataset",
        str(synthetic_mot_sequence),
        "--output-dir", str(yolo_labels),
        "--json",
    ])
    capsys.readouterr()  # Clear capsys buffer

    mat_out = tmp_path / "materialized_yolo"
    ret_mat = main([
        "materialize-dataset",
        str(synthetic_mot_sequence.parent),
        "--labels-dir", str(yolo_labels / "labels"),
        "--output-dir", str(mat_out),
        "--mode", "manifest",
        "--train-ratio", "1.0",
        "--val-ratio", "0.0",
        "--test-ratio", "0.0",
        "--json",
    ])
    assert ret_mat == 0
    captured_mat = capsys.readouterr()
    mat_data = json.loads(captured_mat.out)
    assert mat_data["status"] == "materialized"

    # Now verify with CLI
    ret_ver = main([
        "verify-dataset",
        str(mat_out / "dataset.yaml"),
        "--json",
    ])
    assert ret_ver == 0
    captured_ver = capsys.readouterr()
    ver_data = json.loads(captured_ver.out)
    assert ver_data["is_aligned"] is True
    assert ver_data["total_images"] == 2


def test_cli_materialize_dataset_missing_game_id_strict(
    synthetic_multi_sequences: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test CLI materialize-dataset fails when match metadata is incomplete and strict mode is active."""
    # Convert all sequences first to provide labels
    yolo_labels = tmp_path / "multi_labels"
    for seq_dir in sorted(synthetic_multi_sequences.iterdir()):
        if seq_dir.is_dir() and (seq_dir / "seqinfo.ini").exists():
            main([
                "convert-dataset",
                str(seq_dir),
                "--output-dir", str(yolo_labels),
                "--json",
            ])
            capsys.readouterr()

    # Delete gameinfo.ini from SNMOT-001
    bad_gameinfo = synthetic_multi_sequences / "SNMOT-001" / "gameinfo.ini"
    if bad_gameinfo.exists():
        bad_gameinfo.unlink()

    mat_out = tmp_path / "mat_strict_fail"
    ret = main([
        "materialize-dataset",
        str(synthetic_multi_sequences),
        "--labels-dir", str(yolo_labels / "labels"),
        "--output-dir", str(mat_out),
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Incomplete match metadata" in captured.err
    assert "SNMOT-001" in captured.err


def test_cli_materialize_dataset_allow_missing_game_id(
    synthetic_multi_sequences: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test CLI materialize-dataset succeeds with warning when --allow-missing-game-id is passed."""
    yolo_labels = tmp_path / "multi_labels"
    for seq_dir in sorted(synthetic_multi_sequences.iterdir()):
        if seq_dir.is_dir() and (seq_dir / "seqinfo.ini").exists():
            main([
                "convert-dataset",
                str(seq_dir),
                "--output-dir", str(yolo_labels),
                "--json",
            ])
            capsys.readouterr()

    # Delete gameinfo.ini from SNMOT-001
    bad_gameinfo = synthetic_multi_sequences / "SNMOT-001" / "gameinfo.ini"
    if bad_gameinfo.exists():
        bad_gameinfo.unlink()

    mat_out = tmp_path / "mat_allow_ok"
    ret = main([
        "materialize-dataset",
        str(synthetic_multi_sequences),
        "--labels-dir", str(yolo_labels / "labels"),
        "--output-dir", str(mat_out),
        "--allow-missing-game-id",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Incomplete match metadata" in captured.err
    data = json.loads(captured.out)
    assert data["status"] == "materialized"
    assert (mat_out / "dataset.yaml").exists()


def test_cli_split_dataset_all_missing_game_id_fails(
    synthetic_multi_sequences: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test CLI split-dataset rejects when ALL sequences lack game_id."""
    for seq_dir in synthetic_multi_sequences.iterdir():
        if seq_dir.is_dir() and (seq_dir / "gameinfo.ini").exists():
            (seq_dir / "gameinfo.ini").unlink()

    ret = main(["split-dataset", str(synthetic_multi_sequences)])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Incomplete match metadata: 6 sequence(s) missing game_id" in captured.err


def test_cli_split_dataset_all_missing_game_id_allow_missing(
    synthetic_multi_sequences: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test CLI split-dataset succeeds when all sequences lack game_id if --allow-missing-game-id is passed."""
    for seq_dir in synthetic_multi_sequences.iterdir():
        if seq_dir.is_dir() and (seq_dir / "gameinfo.ini").exists():
            (seq_dir / "gameinfo.ini").unlink()

    ret = main([
        "split-dataset",
        str(synthetic_multi_sequences),
        "--allow-missing-game-id",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Incomplete match metadata: 6 sequence(s) missing game_id" in captured.err
    assert "Explicitly downgrading to sequence-level splitting" in captured.err
    data = json.loads(captured.out)
    assert data["total_count"] == 6


def test_cli_materialize_dataset_all_missing_game_id_fails(
    synthetic_multi_sequences: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test CLI materialize-dataset rejects when ALL sequences lack game_id."""
    yolo_labels = tmp_path / "multi_labels_all_miss"
    for seq_dir in sorted(synthetic_multi_sequences.iterdir()):
        if seq_dir.is_dir() and (seq_dir / "seqinfo.ini").exists():
            main([
                "convert-dataset",
                str(seq_dir),
                "--output-dir", str(yolo_labels),
                "--json",
            ])
            capsys.readouterr()

    for seq_dir in synthetic_multi_sequences.iterdir():
        if seq_dir.is_dir() and (seq_dir / "gameinfo.ini").exists():
            (seq_dir / "gameinfo.ini").unlink()

    mat_out = tmp_path / "mat_all_miss_fail"
    ret = main([
        "materialize-dataset",
        str(synthetic_multi_sequences),
        "--labels-dir", str(yolo_labels / "labels"),
        "--output-dir", str(mat_out),
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Incomplete match metadata: 6 sequence(s) missing game_id" in captured.err


def test_cli_materialize_dataset_all_missing_game_id_allow_missing(
    synthetic_multi_sequences: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test CLI materialize-dataset succeeds when all lack game_id if --allow-missing-game-id is passed."""
    yolo_labels = tmp_path / "multi_labels_all_allow"
    for seq_dir in sorted(synthetic_multi_sequences.iterdir()):
        if seq_dir.is_dir() and (seq_dir / "seqinfo.ini").exists():
            main([
                "convert-dataset",
                str(seq_dir),
                "--output-dir", str(yolo_labels),
                "--json",
            ])
            capsys.readouterr()

    for seq_dir in synthetic_multi_sequences.iterdir():
        if seq_dir.is_dir() and (seq_dir / "gameinfo.ini").exists():
            (seq_dir / "gameinfo.ini").unlink()

    mat_out = tmp_path / "mat_all_miss_ok"
    ret = main([
        "materialize-dataset",
        str(synthetic_multi_sequences),
        "--labels-dir", str(yolo_labels / "labels"),
        "--output-dir", str(mat_out),
        "--allow-missing-game-id",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Incomplete match metadata: 6 sequence(s) missing game_id" in captured.err
    data = json.loads(captured.out)
    assert data["status"] == "materialized"
    assert (mat_out / "dataset.yaml").exists()


def test_cli_split_dataset_single_sequence_no_gameinfo(
    synthetic_mot_sequence: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test CLI split-dataset rejects a single sequence missing gameinfo.ini unless permitted."""
    gameinfo = synthetic_mot_sequence / "gameinfo.ini"
    if gameinfo.exists():
        gameinfo.unlink()

    ret_fail = main(["split-dataset", str(synthetic_mot_sequence)])
    assert ret_fail == 1
    captured_fail = capsys.readouterr()
    assert "Incomplete match metadata" in captured_fail.err
    assert synthetic_mot_sequence.name in captured_fail.err

    ret_ok = main(["split-dataset", str(synthetic_mot_sequence), "--allow-missing-game-id", "--json"])
    assert ret_ok == 0
    captured_ok = capsys.readouterr()
    data = json.loads(captured_ok.out)
    assert data["total_count"] == 1


def test_cli_split_dataset_complete_match_metadata_grouped(
    synthetic_multi_sequences: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test CLI split-dataset keeps all sequences from the same match together without leakage."""
    ret = main([
        "split-dataset",
        str(synthetic_multi_sequences),
        "--train-ratio", "0.5",
        "--val-ratio", "0.5",
        "--test-ratio", "0.0",
        "--seed", "42",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    train_set = set(data["train_sequences"])
    val_set = set(data["val_sequences"])

    # MATCH-01 sequences must all be in the same split
    match_01_seqs = {"SNMOT-001", "SNMOT-002", "SNMOT-003"}
    assert match_01_seqs.issubset(train_set) or match_01_seqs.issubset(val_set)

    # MATCH-02 sequences must all be in the same split
    match_02_seqs = {"SNMOT-004", "SNMOT-005", "SNMOT-006"}
    assert match_02_seqs.issubset(train_set) or match_02_seqs.issubset(val_set)
