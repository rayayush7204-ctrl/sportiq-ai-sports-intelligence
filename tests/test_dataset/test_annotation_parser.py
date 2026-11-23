"""Unit tests for SoccerNet MOT annotation, seqinfo, and gameinfo parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from sportiq.dataset.annotation_parser import (
    GameInfo,
    MOTAnnotation,
    SequenceDataset,
    SequenceInfo,
    load_sequence,
    parse_gameinfo,
    parse_mot_gt,
    parse_seqinfo,
)


def test_parse_mot_gt_valid(synthetic_mot_sequence: Path) -> None:
    """Test parsing a valid gt.txt file."""
    gt_file = synthetic_mot_sequence / "gt" / "gt.txt"
    annotations = parse_mot_gt(gt_file)

    assert len(annotations) == 10
    first = annotations[0]
    assert isinstance(first, MOTAnnotation)
    assert first.frame_id == 1
    assert first.track_id == 1
    assert first.bb_left == 100.0
    assert first.bb_top == 200.0
    assert first.bb_width == 50.0
    assert first.bb_height == 120.0
    assert first.conf == 1.0
    assert first.world_x == -1.0


def test_parse_mot_gt_missing_file(tmp_path: Path) -> None:
    """Test error raised when gt.txt does not exist."""
    missing = tmp_path / "nonexistent_gt.txt"
    with pytest.raises(FileNotFoundError, match="not found"):
        parse_mot_gt(missing)


def test_parse_mot_gt_malformed(tmp_path: Path) -> None:
    """Test error raised on malformed MOT line with insufficient columns."""
    bad_gt = tmp_path / "bad_gt.txt"
    bad_gt.write_text("1, 2, 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected >= 7 columns"):
        parse_mot_gt(bad_gt)


def test_parse_seqinfo_valid(synthetic_mot_sequence: Path) -> None:
    """Test parsing seqinfo.ini metadata."""
    seqinfo_file = synthetic_mot_sequence / "seqinfo.ini"
    info = parse_seqinfo(seqinfo_file)

    assert isinstance(info, SequenceInfo)
    assert info.name == "SNMOT-SYNTH-01"
    assert info.im_dir == "img1"
    assert info.frame_rate == 25.0
    assert info.seq_length == 2
    assert info.im_width == 1920
    assert info.im_height == 1080
    assert info.im_ext == ".jpg"


def test_parse_seqinfo_missing(tmp_path: Path) -> None:
    """Test error raised when seqinfo.ini is missing."""
    with pytest.raises(FileNotFoundError, match="not found"):
        parse_seqinfo(tmp_path / "missing.ini")


def test_parse_gameinfo_valid(synthetic_mot_sequence: Path) -> None:
    """Test parsing gameinfo.ini tracks and game metadata."""
    gameinfo_file = synthetic_mot_sequence / "gameinfo.ini"
    info = parse_gameinfo(gameinfo_file)

    assert isinstance(info, GameInfo)
    assert info.game_id == "MATCH-2023-01"
    assert info.action == "Open Play"
    assert len(info.track_classes) == 5
    assert info.track_classes[1] == "player team left"
    assert info.track_classes[2] == "goalkeeper team right"
    assert info.track_classes[3] == "main referee"
    assert info.track_classes[4] == "ball"
    assert info.track_classes[5] == "staff"


def test_load_sequence_complete(synthetic_mot_sequence: Path) -> None:
    """Test loading entire sequence into SequenceDataset container."""
    seq_dataset = load_sequence(synthetic_mot_sequence)

    assert isinstance(seq_dataset, SequenceDataset)
    assert seq_dataset.name == "SNMOT-SYNTH-01"
    assert seq_dataset.num_frames == 2
    assert seq_dataset.total_annotations == 10
    assert seq_dataset.game_info is not None
    assert seq_dataset.game_info.game_id == "MATCH-2023-01"


def test_load_sequence_invalid_dir(tmp_path: Path) -> None:
    """Test error raised when directory does not exist."""
    with pytest.raises(NotADirectoryError, match="not found"):
        load_sequence(tmp_path / "nonexistent_dir")
