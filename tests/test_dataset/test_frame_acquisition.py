"""Unit tests for selective frame acquisition, path traversal safety, and match materialization."""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest
import cv2
import yaml

from sportiq.dataset.frame_acquisition import (
    INTERNAL_SPLIT_MATCHES,
    ImageIntegrityAuditor,
    MatchDatasetMaterializer,
    SequenceIntegrityReport,
    StorageBudgetChecker,
    ZipPathSafetyValidator,
)


def test_zip_path_safety_validator() -> None:
    """Test ZipPathSafetyValidator blocks path traversal, absolute paths, and unauthorized roots."""
    # Safe paths
    assert ZipPathSafetyValidator.is_safe_path("train/SNMOT-060/img1/000001.jpg")
    assert ZipPathSafetyValidator.is_safe_path("test/SNMOT-116/seqinfo.ini")
    assert ZipPathSafetyValidator.is_safe_path("SNMOT-060/img1/000001.jpg", allowed_root="SNMOT-060")

    # Path traversal
    assert not ZipPathSafetyValidator.is_safe_path("../evil.jpg")
    assert not ZipPathSafetyValidator.is_safe_path("train/SNMOT-060/../../evil.jpg")
    assert not ZipPathSafetyValidator.is_safe_path("SNMOT-060/../SNMOT-061/img1/000001.jpg")

    # Absolute paths & drive letters
    assert not ZipPathSafetyValidator.is_safe_path("/etc/passwd")
    assert not ZipPathSafetyValidator.is_safe_path("C:/Windows/System32/calc.exe")
    assert not ZipPathSafetyValidator.is_safe_path("D:\\data\\images.jpg")

    # Unmatched root
    assert not ZipPathSafetyValidator.is_safe_path("train/SNMOT-060/img1/000001.jpg", allowed_root="test")
    assert not ZipPathSafetyValidator.is_safe_path("")
    assert not ZipPathSafetyValidator.is_safe_path("   ")


def test_storage_budget_checker() -> None:
    """Test StorageBudgetChecker validates pre-approved budget and headroom boundaries."""
    # Under budget (e.g. 5 GiB)
    approved_under, stats_under = StorageBudgetChecker.check_disk_headroom(
        target_path="C:\\",
        required_bytes=5 * (1024**3),
    )
    assert stats_under["within_approved_budget"]
    assert stats_under["approved_budget_gib"] == 6.0

    # Over budget (e.g. 17.04 GiB for full 6 games)
    approved_over, stats_over = StorageBudgetChecker.check_disk_headroom(
        target_path="C:\\",
        required_bytes=int(17.04 * (1024**3)),
    )
    assert not stats_over["within_approved_budget"]
    assert not approved_over
    assert stats_over["required_gib"] == 17.04


def test_image_integrity_auditor_clean_sequence(tmp_path: Path) -> None:
    """Test ImageIntegrityAuditor on a complete valid sequence."""
    seq_dir = tmp_path / "SNMOT-001"
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True)

    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\nimDir=img1\nframeRate=25\nseqLength=3\nimWidth=100\nimHeight=100\nimExt=.jpg\n",
        encoding="utf-8",
    )

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(1, 4):
        cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), dummy_img)

    report = ImageIntegrityAuditor.audit_sequence(seq_dir)
    assert report.is_complete
    assert report.is_valid
    assert report.actual_frames_on_disk == 3
    assert len(report.missing_frame_indices) == 0


def test_image_integrity_auditor_missing_and_corrupt(tmp_path: Path) -> None:
    """Test ImageIntegrityAuditor flags missing, zero-byte, corrupt, and dimension-mismatch frames."""
    seq_dir = tmp_path / "SNMOT-002"
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True)

    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-002\nimDir=img1\nframeRate=25\nseqLength=4\nimWidth=100\nimHeight=100\nimExt=.jpg\n",
        encoding="utf-8",
    )

    # Frame 1: valid
    cv2.imwrite(str(img_dir / "000001.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))
    # Frame 2: zero-byte
    (img_dir / "000002.jpg").write_bytes(b"")
    # Frame 3: corrupt (not an image)
    (img_dir / "000003.jpg").write_bytes(b"corrupted_bytes_not_jpeg")
    # Frame 4: missing entirely

    report = ImageIntegrityAuditor.audit_sequence(seq_dir)
    assert not report.is_complete
    assert not report.is_valid
    assert report.actual_frames_on_disk == 3
    assert report.missing_frame_indices == [4]
    assert "000002.jpg" in report.zero_byte_frames
    assert "000003.jpg" in report.corrupted_frames


def test_assign_sequence_splits() -> None:
    """Test MatchDatasetMaterializer assigns sequences strictly based on game_id."""
    # Prepare dummy sequence paths with gameinfo.ini
    def _create_mock_seq(p: Path, name: str, game_id: str) -> Path:
        sdir = p / name
        sdir.mkdir(parents=True)
        (sdir / "gameinfo.ini").write_text(
            f"[Sequence]\nname={name}\ngameID={game_id}\n", encoding="utf-8"
        )
        return sdir

    tmp = Path("data")  # virtual path test
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        s1 = _create_mock_seq(tdp, "SNMOT-060", "4")
        s2 = _create_mock_seq(tdp, "SNMOT-097", "6")
        s3 = _create_mock_seq(tdp, "SNMOT-132", "8")
        s4 = _create_mock_seq(tdp, "SNMOT-116", "7")
        s5 = _create_mock_seq(tdp, "SNMOT-187", "11")

        splits = MatchDatasetMaterializer.assign_sequence_splits(
            [s1, s2, s3, s4, s5],
            split_match_map=INTERNAL_SPLIT_MATCHES,
        )

        assert len(splits["train"]) == 2  # 4 and 6
        assert len(splits["val"]) == 1    # 8
        assert len(splits["test"]) == 2   # 7 and 11


def test_assign_sequence_splits_disjoint_assertion() -> None:
    """Test MatchDatasetMaterializer raises error if a match is assigned to multiple splits."""
    invalid_map = {
        "train": ["4", "8"],
        "val": ["8"],  # duplicate gameID
    }
    with pytest.raises(ValueError, match="assigned to multiple splits"):
        MatchDatasetMaterializer.assign_sequence_splits([], split_match_map=invalid_map)


def test_materialize_yolo_dataset_mocked(tmp_path: Path) -> None:
    """Test end-to-end YOLO materialization with synthetic sequence."""
    seq_dir = tmp_path / "mock_seq"
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True)

    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=mock_seq\nimDir=img1\nframeRate=25\nseqLength=2\nimWidth=100\nimHeight=100\nimExt=.jpg\n",
        encoding="utf-8",
    )
    (seq_dir / "gameinfo.ini").write_text(
        "[Sequence]\nname=mock_seq\ngameID=4\nactionClass=Goal\nnum_tracklets=1\ntrackletID_1= player team left;10\n",
        encoding="utf-8",
    )
    (seq_dir / "gt.txt").write_text(
        "1,1,10,10,20,30,1,-1,-1,-1\n", encoding="utf-8"
    )

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "000001.jpg"), dummy_img)
    cv2.imwrite(str(img_dir / "000002.jpg"), dummy_img)

    out_dataset = tmp_path / "yolo_dataset"
    metrics = MatchDatasetMaterializer.materialize_yolo_dataset(
        split_sequences={"train": [seq_dir], "val": [], "test": []},
        output_dir=out_dataset,
        prefer_hardlink=False,  # use copy in test for safety
    )

    assert metrics["total_images"] == 2
    assert metrics["total_labels"] == 2

    # Check images/train and labels/train
    assert (out_dataset / "images" / "train" / "mock_seq_000001.jpg").is_file()
    assert (out_dataset / "images" / "train" / "mock_seq_000002.jpg").is_file()
    assert (out_dataset / "labels" / "train" / "mock_seq_000001.txt").is_file()
    assert (out_dataset / "labels" / "train" / "mock_seq_000002.txt").is_file()

    # Frame 2 had no annotations, so its label should be empty (negative frame)
    assert (out_dataset / "labels" / "train" / "mock_seq_000002.txt").read_text() == ""

    # Check dataset.yaml
    yaml_file = out_dataset / "dataset.yaml"
    assert yaml_file.is_file()
    ydata = yaml.safe_load(yaml_file.read_text())
    assert ydata["train"] == "images/train"
    assert ydata["names"][0] == "player"
