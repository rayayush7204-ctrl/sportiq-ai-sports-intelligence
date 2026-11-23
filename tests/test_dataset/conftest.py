"""Pytest fixtures for dataset engineering tests."""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np
import pytest

from sportiq.dataset.annotation_parser import SequenceDataset, load_sequence


@pytest.fixture
def synthetic_mot_sequence(tmp_path: Path) -> Path:
    """Create a minimal valid SoccerNet MOT sequence directory for unit testing.

    Directory structure:
      seq_dir/
        seqinfo.ini
        gameinfo.ini
        gt/
          gt.txt
        img1/
          000001.jpg
          000002.jpg
    """
    seq_dir = tmp_path / "SNMOT-SYNTH-01"
    seq_dir.mkdir(parents=True)

    # 1. seqinfo.ini
    seqinfo = configparser.ConfigParser()
    seqinfo["Sequence"] = {
        "name": "SNMOT-SYNTH-01",
        "imDir": "img1",
        "frameRate": "25.0",
        "seqLength": "2",
        "imWidth": "1920",
        "imHeight": "1080",
        "imExt": ".jpg",
    }
    with (seq_dir / "seqinfo.ini").open("w", encoding="utf-8") as f:
        seqinfo.write(f)

    # 2. gameinfo.ini
    gameinfo = configparser.ConfigParser()
    gameinfo["Sequence"] = {
        "game_id": "MATCH-2023-01",
        "action": "Open Play",
    }
    gameinfo["Track_1"] = {"class": "player team left", "jersey": "10"}
    gameinfo["Track_2"] = {"class": "goalkeeper team right", "jersey": "1"}
    gameinfo["Track_3"] = {"class": "main referee"}
    gameinfo["Track_4"] = {"class": "ball"}
    gameinfo["Track_5"] = {"class": "staff", "role": "coach"}

    with (seq_dir / "gameinfo.ini").open("w", encoding="utf-8") as f:
        gameinfo.write(f)

    # 3. gt/gt.txt (MOT format: frame, track, x, y, w, h, conf, -1, -1, -1)
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    gt_lines = [
        # Frame 1
        "1, 1, 100.0, 200.0, 50.0, 120.0, 1.0, -1, -1, -1",    # Track 1: Player
        "1, 2, 800.0, 400.0, 60.0, 140.0, 1.0, -1, -1, -1",    # Track 2: Goalkeeper
        "1, 3, 500.0, 300.0, 45.0, 110.0, 1.0, -1, -1, -1",    # Track 3: Referee
        "1, 4, 350.0, 450.0, 15.0, 15.0, 1.0, -1, -1, -1",     # Track 4: Ball
        "1, 5, 20.0, 80.0, 40.0, 100.0, 1.0, -1, -1, -1",      # Track 5: Staff
        # Frame 2
        "2, 1, 105.0, 202.0, 50.0, 120.0, 1.0, -1, -1, -1",    # Track 1
        "2, 2, 802.0, 401.0, 60.0, 140.0, 1.0, -1, -1, -1",    # Track 2
        "2, 3, 502.0, 301.0, 45.0, 110.0, 1.0, -1, -1, -1",    # Track 3
        "2, 4, 360.0, 455.0, 15.0, 15.0, 1.0, -1, -1, -1",     # Track 4
        "2, 5, 20.0, 80.0, 40.0, 100.0, 1.0, -1, -1, -1",      # Track 5
    ]
    (gt_dir / "gt.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")

    # 4. img1/ dummy images
    img1_dir = seq_dir / "img1"
    img1_dir.mkdir()
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dummy_img[:, :] = (34, 139, 34)  # Pitch green
    cv2.imwrite(str(img1_dir / "000001.jpg"), dummy_img)
    cv2.imwrite(str(img1_dir / "000002.jpg"), dummy_img)

    return seq_dir


@pytest.fixture
def synthetic_multi_sequences(tmp_path: Path) -> Path:
    """Create multiple sequences from two distinct matches for dataset splitting tests."""
    root = tmp_path / "multi_seqs"
    root.mkdir()

    seq_configs = [
        ("SNMOT-001", "MATCH-01"),
        ("SNMOT-002", "MATCH-01"),
        ("SNMOT-003", "MATCH-01"),
        ("SNMOT-004", "MATCH-02"),
        ("SNMOT-005", "MATCH-02"),
        ("SNMOT-006", "MATCH-02"),
    ]

    for name, match_id in seq_configs:
        sdir = root / name
        sdir.mkdir()

        # seqinfo
        seqinfo = configparser.ConfigParser()
        seqinfo["Sequence"] = {
            "name": name,
            "imDir": "img1",
            "frameRate": "25.0",
            "seqLength": "1",
            "imWidth": "1920",
            "imHeight": "1080",
            "imExt": ".jpg",
        }
        with (sdir / "seqinfo.ini").open("w", encoding="utf-8") as f:
            seqinfo.write(f)

        # gameinfo
        gameinfo = configparser.ConfigParser()
        gameinfo["Sequence"] = {"game_id": match_id}
        gameinfo["Track_1"] = {"class": "player"}
        with (sdir / "gameinfo.ini").open("w", encoding="utf-8") as f:
            gameinfo.write(f)

        # gt
        gt_dir = sdir / "gt"
        gt_dir.mkdir()
        (gt_dir / "gt.txt").write_text(
            "1, 1, 100.0, 200.0, 50.0, 100.0, 1.0, -1, -1, -1\n",
            encoding="utf-8",
        )

        # img1 dummy image
        img_dir = sdir / "img1"
        img_dir.mkdir()
        dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dummy_img[:, :] = (34, 139, 34)  # Pitch green
        cv2.imwrite(str(img_dir / "000001.jpg"), dummy_img)

    return root
