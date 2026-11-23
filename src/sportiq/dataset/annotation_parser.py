"""Parser for SoccerNet Multi-Object Tracking (MOT) dataset assets.

Parses:
  - gt.txt: MOT20-format 10-column CSV tracking ground-truth
  - seqinfo.ini: Video sequence metadata
  - gameinfo.ini: Contextual game metadata and track-to-class mappings
"""

from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MOTAnnotation:
    """Single-object detection instance in MOT format."""

    frame_id: int          # 1-based frame index
    track_id: int          # Unique integer track ID
    bb_left: float         # Top-left pixel X coordinate
    bb_top: float          # Top-left pixel Y coordinate
    bb_width: float        # Bounding box pixel width
    bb_height: float       # Bounding box pixel height
    conf: float            # Ground-truth confidence value (always preserved)
    world_x: float = -1.0  # MOT world coordinate X (unused, preserved)
    world_y: float = -1.0  # MOT world coordinate Y (unused, preserved)
    world_z: float = -1.0  # MOT world coordinate Z (unused, preserved)


@dataclass(frozen=True)
class SequenceInfo:
    """Metadata parsed from seqinfo.ini."""

    name: str
    im_dir: str
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str = ".jpg"


@dataclass(frozen=True)
class GameInfo:
    """Metadata and track mappings parsed from gameinfo.ini."""

    sequence_name: str
    game_id: Optional[str] = None
    action: Optional[str] = None
    track_classes: Dict[int, str] = field(default_factory=dict)
    track_metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SequenceDataset:
    """In-memory representation of an entire SoccerNet sequence."""

    sequence_path: Path
    seq_info: SequenceInfo
    game_info: Optional[GameInfo]
    annotations: List[MOTAnnotation]

    @property
    def name(self) -> str:
        return self.seq_info.name

    @property
    def num_frames(self) -> int:
        return self.seq_info.seq_length

    @property
    def total_annotations(self) -> int:
        return len(self.annotations)


def parse_mot_gt(gt_path: Union[str, Path]) -> List[MOTAnnotation]:
    """Parse a SoccerNet MOT gt.txt CSV file.

    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

    Args:
        gt_path: Path to gt.txt file.

    Returns:
        List[MOTAnnotation]: Parsed annotation records.

    Raises:
        FileNotFoundError: If gt_path does not exist.
        ValueError: If file formatting is invalid.
    """
    path = Path(gt_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"MOT ground truth file not found: {path}")

    annotations: List[MOTAnnotation] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            parts = [p.strip() for p in line_str.split(",")]
            if len(parts) < 7:
                # Try space-delimited fallback if comma not found
                parts = line_str.split()
                if len(parts) < 7:
                    raise ValueError(
                        f"Malformed MOT annotation at {path}:{line_num}: expected >= 7 columns, got {len(parts)}"
                    )

            try:
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                conf = float(parts[6])

                world_x = float(parts[7]) if len(parts) > 7 else -1.0
                world_y = float(parts[8]) if len(parts) > 8 else -1.0
                world_z = float(parts[9]) if len(parts) > 9 else -1.0

                annotations.append(
                    MOTAnnotation(
                        frame_id=frame_id,
                        track_id=track_id,
                        bb_left=bb_left,
                        bb_top=bb_top,
                        bb_width=bb_width,
                        bb_height=bb_height,
                        conf=conf,
                        world_x=world_x,
                        world_y=world_y,
                        world_z=world_z,
                    )
                )
            except Exception as e:
                raise ValueError(
                    f"Error parsing MOT record at {path}:{line_num} ('{line_str}'): {e}"
                ) from e

    return annotations


def parse_seqinfo(seqinfo_path: Union[str, Path]) -> SequenceInfo:
    """Parse a seqinfo.ini configuration file.

    Args:
        seqinfo_path: Path to seqinfo.ini file.

    Returns:
        SequenceInfo: Parsed sequence metadata.
    """
    path = Path(seqinfo_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"seqinfo.ini not found: {path}")

    config = configparser.ConfigParser()
    config.read(str(path), encoding="utf-8")

    section = "Sequence"
    if section not in config:
        # Fallback to first section if named differently
        if config.sections():
            section = config.sections()[0]
        else:
            raise ValueError(f"No sections found in {path}")

    sec = config[section]
    name = sec.get("name", path.parent.name)
    im_dir = sec.get("imDir", "img1")
    frame_rate = float(sec.get("frameRate", 25.0))
    seq_length = int(sec.get("seqLength", 0))
    im_width = int(sec.get("imWidth", 1920))
    im_height = int(sec.get("imHeight", 1080))
    im_ext = sec.get("imExt", ".jpg")

    return SequenceInfo(
        name=name,
        im_dir=im_dir,
        frame_rate=frame_rate,
        seq_length=seq_length,
        im_width=im_width,
        im_height=im_height,
        im_ext=im_ext,
    )


def parse_gameinfo(gameinfo_path: Union[str, Path]) -> GameInfo:
    """Parse gameinfo.ini configuration file.

    Maps [Track_<id>] sections to classes, jersey numbers, and team roles.

    Args:
        gameinfo_path: Path to gameinfo.ini file.

    Returns:
        GameInfo: Parsed game metadata and track-to-class mappings.
    """
    path = Path(gameinfo_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"gameinfo.ini not found: {path}")

    config = configparser.ConfigParser()
    config.read(str(path), encoding="utf-8")

    sequence_name = path.parent.name
    game_id: Optional[str] = None
    action: Optional[str] = None
    track_classes: Dict[int, str] = {}
    track_metadata: Dict[int, Dict[str, Any]] = {}

    if "Sequence" in config:
        seq_sec = config["Sequence"]
        game_id = seq_sec.get("game_id", seq_sec.get("gameId"))
        action = seq_sec.get("action")

    for sec_name in config.sections():
        # Match [Track_1], [Track_2], [track_1], etc.
        sec_lower = sec_name.lower()
        if sec_lower.startswith("track_"):
            try:
                track_id = int(sec_lower.split("track_")[1])
                sec_dict = dict(config[sec_name])
                class_val = sec_dict.get("class", sec_dict.get("role", "unclassified"))
                track_classes[track_id] = class_val
                track_metadata[track_id] = sec_dict
            except ValueError:
                continue

    return GameInfo(
        sequence_name=sequence_name,
        game_id=game_id,
        action=action,
        track_classes=track_classes,
        track_metadata=track_metadata,
    )


def load_sequence(sequence_dir: Union[str, Path]) -> SequenceDataset:
    """Load a complete SoccerNet sequence folder.

    Args:
        sequence_dir: Path to directory containing seqinfo.ini and gt/gt.txt.

    Returns:
        SequenceDataset: Fully loaded sequence.
    """
    seq_dir = Path(sequence_dir).resolve()
    if not seq_dir.is_dir():
        raise NotADirectoryError(f"Sequence directory not found: {seq_dir}")

    seqinfo_path = seq_dir / "seqinfo.ini"
    seq_info = parse_seqinfo(seqinfo_path)

    # Ground truth path can be gt/gt.txt or gt.txt
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        gt_path = seq_dir / "gt.txt"
    annotations = parse_mot_gt(gt_path)

    gameinfo_path = seq_dir / "gameinfo.ini"
    game_info = parse_gameinfo(gameinfo_path) if gameinfo_path.exists() else None

    return SequenceDataset(
        sequence_path=seq_dir,
        seq_info=seq_info,
        game_info=game_info,
        annotations=annotations,
    )
