"""Unit tests for sequence-level and match-level deterministic dataset splitting."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sportiq.dataset.class_mapping import ClassTaxonomyType
from sportiq.dataset.dataset_splitter import DatasetSplit, DatasetSplitter


def test_split_sequences_disjoint() -> None:
    """Test sequence partitioning produces strictly disjoint sets (zero data leakage)."""
    splitter = DatasetSplitter(random_seed=42)
    seqs = [f"SNMOT-{i:03d}" for i in range(1, 21)]

    split = splitter.split_sequences(
        sequence_names=seqs,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    assert isinstance(split, DatasetSplit)
    s_train = set(split.train_sequences)
    s_val = set(split.val_sequences)
    s_test = set(split.test_sequences)

    # Disjoint checks
    assert not (s_train & s_val), "Data leakage between train and val!"
    assert not (s_train & s_test), "Data leakage between train and test!"
    assert not (s_val & s_test), "Data leakage between val and test!"

    # Total coverage
    assert s_train | s_val | s_test == set(seqs)
    assert len(split.train_sequences) == 14
    assert len(split.val_sequences) == 3
    assert len(split.test_sequences) == 3


def test_split_sequences_deterministic() -> None:
    """Test same seed produces identical splits, different seed produces different splits."""
    seqs = [f"SNMOT-{i:03d}" for i in range(1, 31)]

    s1 = DatasetSplitter(random_seed=42).split_sequences(seqs)
    s2 = DatasetSplitter(random_seed=42).split_sequences(seqs)
    s3 = DatasetSplitter(random_seed=999).split_sequences(seqs)

    assert s1.train_sequences == s2.train_sequences
    assert s1.val_sequences == s2.val_sequences
    assert s1.test_sequences == s2.test_sequences

    assert s1.train_sequences != s3.train_sequences


def test_split_sequences_with_match_grouping() -> None:
    """Test that all sequences belonging to the same match are grouped into the same partition."""
    splitter = DatasetSplitter(random_seed=42)
    seqs = ["SNMOT-01", "SNMOT-02", "SNMOT-03", "SNMOT-04", "SNMOT-05", "SNMOT-06"]
    match_grouping = {
        "SNMOT-01": "MATCH-A",
        "SNMOT-02": "MATCH-A",
        "SNMOT-03": "MATCH-A",
        "SNMOT-04": "MATCH-B",
        "SNMOT-05": "MATCH-B",
        "SNMOT-06": "MATCH-B",
    }

    split = splitter.split_sequences(
        sequence_names=seqs,
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        match_grouping=match_grouping,
    )

    # All of MATCH-A must be together, all of MATCH-B must be together
    match_a_seqs = {"SNMOT-01", "SNMOT-02", "SNMOT-03"}
    match_b_seqs = {"SNMOT-04", "SNMOT-05", "SNMOT-06"}

    train_set = set(split.train_sequences)
    val_set = set(split.val_sequences)

    assert train_set == match_a_seqs or train_set == match_b_seqs
    assert val_set == match_a_seqs or val_set == match_b_seqs


def test_split_partially_missing_game_id_strict_raises() -> None:
    """Test strict match-level splitting raises ValueError when any sequence lacks game_id."""
    splitter = DatasetSplitter(random_seed=42)
    seqs = ["SNMOT-01", "SNMOT-02", "SNMOT-03"]
    # SNMOT-03 missing from grouping
    incomplete_grouping = {"SNMOT-01": "MATCH-A", "SNMOT-02": "MATCH-A"}

    with pytest.raises(ValueError, match="Incomplete match metadata"):
        splitter.split_sequences(
            sequence_names=seqs,
            match_grouping=incomplete_grouping,
            strict_match_grouping=True,
        )


def test_split_partially_missing_game_id_fallback() -> None:
    """Test non-strict match-level splitting downgrades to sequence-level partition."""
    splitter = DatasetSplitter(random_seed=42)
    seqs = ["SNMOT-01", "SNMOT-02", "SNMOT-03"]
    incomplete_grouping = {"SNMOT-01": "MATCH-A", "SNMOT-02": "MATCH-A"}

    # With strict_match_grouping=False, it should warn and split at sequence level
    split = splitter.split_sequences(
        sequence_names=seqs,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        match_grouping=incomplete_grouping,
        strict_match_grouping=False,
    )

    assert len(split.train_sequences) == 3


def test_split_invalid_ratios() -> None:
    """Test error raised when split ratios do not sum to 1.0."""
    splitter = DatasetSplitter()
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        splitter.split_sequences(["SNMOT-01", "SNMOT-02"], train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)


def test_generate_yolo_yaml(tmp_path: Path) -> None:
    """Test generating a YOLO dataset.yaml configuration file."""
    yaml_path = tmp_path / "dataset.yaml"
    content = DatasetSplitter.generate_yolo_yaml(
        dataset_root=tmp_path,
        taxonomy=ClassTaxonomyType.FOUR_CLASS,
        output_path=yaml_path,
    )

    assert yaml_path.exists()
    parsed = yaml.safe_load(content)
    assert parsed["train"] == "images/train"
    assert parsed["val"] == "images/val"
    assert parsed["test"] == "images/test"
    assert parsed["names"] == {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
