"""Unit tests for SoccerNet multi-match dataset catalog and grouping logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sportiq.cli import build_parser, catalog_dataset_command
from sportiq.dataset.catalog import (
    MatchCatalogGroup,
    SequenceCatalogEntry,
    SoccerNetCatalog,
    build_soccernet_catalog,
    group_sequences_by_game,
)


def _make_dummy_entry(
    name: str,
    game_id: str = "4",
    split: str = "train",
    total_annotations: int = 100,
    has_local_frames: bool = False,
    local_frame_count: int = 0,
) -> SequenceCatalogEntry:
    """Helper to create a test catalog entry."""
    return SequenceCatalogEntry(
        sequence_name=name,
        game_id=game_id,
        split=split,
        frame_rate=25.0,
        seq_length=750,
        im_width=1920,
        im_height=1080,
        total_annotations=total_annotations,
        unique_track_ids=25,
        raw_class_distribution={"player team left": 80, "ball": 10, "goalkeeper team right": 5, "referee": 5},
        canonical_four_class_distribution={"player": 80, "goalkeeper": 5, "referee": 5, "ball": 10},
        staff_annotations=0,
        has_seqinfo=True,
        has_gameinfo=True,
        has_gt=True,
        has_local_frames=has_local_frames,
        local_frame_count=local_frame_count,
        action_class="Kick-off",
    )


def test_sequence_catalog_entry_serialization() -> None:
    """Test SequenceCatalogEntry fields and dictionary serialization."""
    entry = _make_dummy_entry("SNMOT-060", game_id="4")
    d = entry.to_dict()

    assert d["sequence_name"] == "SNMOT-060"
    assert d["game_id"] == "4"
    assert d["split"] == "train"
    assert d["total_annotations"] == 100
    assert d["canonical_four_class_distribution"]["player"] == 80
    assert d["canonical_four_class_distribution"]["ball"] == 10


def test_group_sequences_by_game() -> None:
    """Test group_sequences_by_game aggregates sequences into independent match groups."""
    entries = [
        _make_dummy_entry("SNMOT-060", game_id="4", split="train", total_annotations=100),
        _make_dummy_entry("SNMOT-061", game_id="4", split="train", total_annotations=150),
        _make_dummy_entry("SNMOT-116", game_id="7", split="test", total_annotations=200, has_local_frames=True, local_frame_count=3),
        _make_dummy_entry("SNMOT-999", game_id=None, split="train", total_annotations=50),
    ]

    groups = group_sequences_by_game(entries)

    assert "4" in groups
    assert "7" in groups
    assert "UNKNOWN" in groups

    # Match 4
    g4 = groups["4"]
    assert g4.game_id == "4"
    assert g4.total_sequences == 2
    assert g4.total_annotations == 250
    assert g4.sequence_names == ["SNMOT-060", "SNMOT-061"]
    assert g4.canonical_four_class_distribution["player"] == 160
    assert not g4.has_local_frames

    # Match 7
    g7 = groups["7"]
    assert g7.game_id == "7"
    assert g7.total_sequences == 1
    assert g7.total_annotations == 200
    assert g7.has_local_frames
    assert g7.total_local_frames == 3

    # Unknown
    gu = groups["UNKNOWN"]
    assert gu.total_sequences == 1


def test_soccernet_catalog_metrics_and_markdown() -> None:
    """Test SoccerNetCatalog container metrics and markdown generation."""
    e1 = _make_dummy_entry("SNMOT-060", game_id="4", split="train")
    e2 = _make_dummy_entry("SNMOT-116", game_id="7", split="test")

    groups = group_sequences_by_game([e1, e2])
    catalog = SoccerNetCatalog(
        entries=[e1, e2],
        match_groups=groups,
        total_sequences=2,
        unique_game_ids=["4", "7"],
        candidate_independent_matches=["4"],
    )

    assert catalog.total_annotations == 200
    assert len(catalog.unique_game_ids) == 2
    assert catalog.candidate_independent_matches == ["4"]

    # Markdown generation
    md = catalog.generate_markdown_report()
    assert "# SPORTIQ — SOCCERNET MULTI-MATCH DATASET CATALOG & AUDIT REPORT" in md
    assert "Game 4" in md
    assert "Game 7" in md
    assert "`SNMOT-060`" in md
    assert "`SNMOT-116`" in md

    # JSON export
    catalog_dict = json.loads(catalog.to_json())
    assert catalog_dict["total_sequences"] == 2
    assert catalog_dict["total_annotations"] == 200
    assert "4" in catalog_dict["match_groups"]


def test_build_soccernet_catalog_on_mock_dir(tmp_path: Path) -> None:
    """Test build_soccernet_catalog on a temporary mock directory structure."""
    split_dir = tmp_path / "train" / "SNMOT-001"
    split_dir.mkdir(parents=True)

    # Write seqinfo.ini
    (split_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\nimDir=img1\nframeRate=25\nseqLength=100\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n",
        encoding="utf-8",
    )

    # Write gameinfo.ini
    (split_dir / "gameinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\ngameID=12\nactionClass=Corner\nnum_tracklets=1\ntrackletID_1= player team left;10\n",
        encoding="utf-8",
    )

    # Write gt.txt
    (split_dir / "gt.txt").write_text(
        "1,1,100,200,50,80,1,-1,-1,-1\n2,1,102,202,50,80,1,-1,-1,-1\n",
        encoding="utf-8",
    )

    catalog = build_soccernet_catalog(tmp_path)
    assert catalog.total_sequences == 1
    assert catalog.unique_game_ids == ["12"]
    assert catalog.total_annotations == 2

    entry = catalog.entries[0]
    assert entry.sequence_name == "SNMOT-001"
    assert entry.game_id == "12"
    assert entry.split == "train"
    assert entry.action_class == "Corner"
    assert entry.canonical_four_class_distribution["player"] == 2


def test_cli_catalog_dataset_missing_dir(capsys: pytest.CaptureFixture[str]) -> None:
    """Test catalog-dataset CLI fails gracefully on missing directory."""
    parser = build_parser()
    args = parser.parse_args(["catalog-dataset", "--annotations-dir", "nonexistent/dir/path"])
    ret = catalog_dataset_command(args)
    assert ret == 1
    captured = capsys.readouterr()
    assert "Annotations directory not found" in captured.err


def test_cli_catalog_dataset_mocked_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test catalog-dataset CLI execution with mocked directory."""
    ann_dir = tmp_path / "ann"
    ann_dir.mkdir()
    seq_dir = ann_dir / "SNMOT-001"
    seq_dir.mkdir()

    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\nimDir=img1\nframeRate=25\nseqLength=100\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n",
        encoding="utf-8",
    )
    (seq_dir / "gameinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\ngameID=5\nactionClass=Goal\nnum_tracklets=1\ntrackletID_1= player team left;7\n",
        encoding="utf-8",
    )
    (seq_dir / "gt.txt").write_text("1,1,10,20,30,40,1,-1,-1,-1\n", encoding="utf-8")

    out_json = tmp_path / "out_catalog.json"
    out_md = tmp_path / "out_catalog.md"

    parser = build_parser()
    args = parser.parse_args([
        "catalog-dataset",
        "--annotations-dir", str(ann_dir),
        "--output-json", str(out_json),
        "--output-markdown", str(out_md),
    ])

    ret = catalog_dataset_command(args)
    assert ret == 0
    assert out_json.exists()
    assert out_md.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["total_sequences"] == 1
    assert data["unique_game_ids"] == ["5"]
