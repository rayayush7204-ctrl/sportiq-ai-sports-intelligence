"""Selective frame acquisition, path traversal safety, and match-level YOLO dataset materializer."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import requests
import yaml

from sportiq.dataset.annotation_parser import load_sequence
from sportiq.dataset.annotation_validator import AnnotationValidator
from sportiq.dataset.class_mapping import ClassTaxonomy, ClassTaxonomyType, resolve_class
from sportiq.dataset.yolo_converter import YOLOConverter

logger = logging.getLogger(__name__)


# Exact custom internal partitions specified for SportIQ
INTERNAL_SPLIT_MATCHES = {
    "train": ["4", "6", "9"],
    "val": ["8"],
    "test": ["7", "11"],
}


class ZipPathSafetyValidator:
    """Validates ZIP member filenames against path traversal vulnerabilities."""

    @staticmethod
    def is_safe_path(member_path: str, allowed_root: Optional[str] = None) -> bool:
        """Validate whether a ZIP member path is strictly safe to extract.

        Args:
            member_path: Internal ZIP entry path string.
            allowed_root: Optional expected root folder prefix.

        Returns:
            bool: True if safe, False if traversal, absolute, or illegal.
        """
        if not member_path or not member_path.strip():
            return False

        normalized = member_path.replace("\\", "/")

        # Reject path traversal tokens
        parts = normalized.split("/")
        if ".." in parts or "." in parts[:-1]:
            return False

        # Reject absolute paths or Windows drive letters
        if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
            return False

        # Check allowed prefix if specified
        if allowed_root:
            clean_root = allowed_root.replace("\\", "/").strip("/")
            if not normalized.startswith(clean_root + "/") and normalized != clean_root:
                return False

        return True


class StorageBudgetChecker:
    """Verifies storage constraints and disk headroom before transferring data."""

    PREAPPROVED_MAX_BUDGET_BYTES: int = 6 * (1024**3)  # 6.00 GiB
    MIN_HEADROOM_RATIO: float = 0.20  # 20% minimum free-space headroom

    @classmethod
    def check_disk_headroom(
        cls,
        target_path: Union[str, Path] = "C:\\",
        required_bytes: int = 0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify available disk space and ensure required_bytes leaves >= 20% headroom.

        Args:
            target_path: Drive or path to inspect.
            required_bytes: Volume of bytes intended to be written.

        Returns:
            Tuple[bool, Dict[str, Any]]: (is_sufficient, stats_dict)
        """
        resolved = Path(target_path).resolve()
        drive_root = resolved.anchor or str(resolved)
        usage = shutil.disk_usage(drive_root)

        total_bytes = usage.total
        free_bytes = usage.free
        required_headroom_bytes = int(total_bytes * cls.MIN_HEADROOM_RATIO)

        remaining_free_after_transfer = free_bytes - required_bytes
        has_adequate_headroom = remaining_free_after_transfer >= required_headroom_bytes
        within_approved_budget = required_bytes <= cls.PREAPPROVED_MAX_BUDGET_BYTES

        stats = {
            "drive": drive_root,
            "total_bytes": total_bytes,
            "total_gib": round(total_bytes / (1024**3), 2),
            "free_bytes": free_bytes,
            "free_gib": round(free_bytes / (1024**3), 2),
            "required_bytes": required_bytes,
            "required_gib": round(required_bytes / (1024**3), 2),
            "required_headroom_bytes": required_headroom_bytes,
            "required_headroom_gib": round(required_headroom_bytes / (1024**3), 2),
            "remaining_free_after_gib": round(remaining_free_after_transfer / (1024**3), 2),
            "has_adequate_headroom": has_adequate_headroom,
            "within_approved_budget": within_approved_budget,
            "approved_budget_gib": round(cls.PREAPPROVED_MAX_BUDGET_BYTES / (1024**3), 2),
        }

        is_approved = has_adequate_headroom and within_approved_budget
        return is_approved, stats


@dataclass
class SequenceIntegrityReport:
    """Integrity audit result for a single acquired image sequence."""

    sequence_name: str
    expected_frames: int
    actual_frames_on_disk: int
    missing_frame_indices: List[int]
    corrupted_frames: List[str]
    zero_byte_frames: List[str]
    dimension_mismatches: List[str]
    is_complete: bool
    is_valid: bool


@dataclass
class DatasetIntegrityReport:
    """Aggregate integrity audit result across all sequences."""

    total_sequences: int
    complete_sequences: int
    total_expected_frames: int
    total_frames_on_disk: int
    total_corrupted: int
    total_zero_byte: int
    sequence_reports: Dict[str, SequenceIntegrityReport]

    @property
    def is_dataset_sound(self) -> bool:
        return (
            self.complete_sequences == self.total_sequences
            and self.total_corrupted == 0
            and self.total_zero_byte == 0
            and self.total_frames_on_disk == self.total_expected_frames
        )


class ImageIntegrityAuditor:
    """Audits image sequence files on disk against their seqinfo.ini metadata."""

    @staticmethod
    def audit_sequence(sequence_dir: Union[str, Path]) -> SequenceIntegrityReport:
        """Audit an individual sequence directory.

        Args:
            sequence_dir: Path to sequence folder containing seqinfo.ini and img1/

        Returns:
            SequenceIntegrityReport: Detailed audit findings.
        """
        seq_p = Path(sequence_dir).resolve()
        seqinfo_p = seq_p / "seqinfo.ini"
        if not seqinfo_p.is_file():
            raise FileNotFoundError(f"Missing seqinfo.ini in {seq_p}")

        from sportiq.dataset.annotation_parser import parse_seqinfo

        seq_info = parse_seqinfo(seqinfo_p)
        expected_len = seq_info.seq_length
        im_dir_name = seq_info.im_dir or "img1"
        im_ext = seq_info.im_ext or ".jpg"
        expected_w = seq_info.im_width
        expected_h = seq_info.im_height

        img_dir = seq_p / im_dir_name
        if not img_dir.is_dir():
            return SequenceIntegrityReport(
                sequence_name=seq_p.name,
                expected_frames=expected_len,
                actual_frames_on_disk=0,
                missing_frame_indices=list(range(1, expected_len + 1)),
                corrupted_frames=[],
                zero_byte_frames=[],
                dimension_mismatches=[],
                is_complete=False,
                is_valid=False,
            )

        # Collect existing image files
        existing_files = {f.name: f for f in img_dir.glob(f"*{im_ext}")}

        missing_indices: List[int] = []
        zero_byte_files: List[str] = []
        corrupt_files: List[str] = []
        dim_mismatches: List[str] = []

        # Audit expected frames 1 to expected_len (SoccerNet 1-based naming: 000001.jpg)
        for idx in range(1, expected_len + 1):
            fname = f"{idx:06d}{im_ext}"
            if fname not in existing_files:
                missing_indices.append(idx)
                continue

            file_p = existing_files[fname]
            size = file_p.stat().st_size
            if size == 0:
                zero_byte_files.append(fname)
                continue

            # Read image header / shape for verification
            img = cv2.imread(str(file_p))
            if img is None:
                corrupt_files.append(fname)
                continue

            h, w = img.shape[:2]
            if w != expected_w or h != expected_h:
                dim_mismatches.append(f"{fname} ({w}x{h} vs expected {expected_w}x{expected_h})")

        actual_count = len(existing_files)
        is_complete = (actual_count == expected_len) and (len(missing_indices) == 0)
        is_valid = is_complete and (len(zero_byte_files) == 0) and (len(corrupt_files) == 0) and (len(dim_mismatches) == 0)

        return SequenceIntegrityReport(
            sequence_name=seq_p.name,
            expected_frames=expected_len,
            actual_frames_on_disk=actual_count,
            missing_frame_indices=missing_indices,
            corrupted_frames=corrupt_files,
            zero_byte_frames=zero_byte_files,
            dimension_mismatches=dim_mismatches,
            is_complete=is_complete,
            is_valid=is_valid,
        )

    @classmethod
    def audit_dataset_root(cls, root_dir: Union[str, Path]) -> DatasetIntegrityReport:
        """Audit all sequence folders found recursively under root_dir."""
        p = Path(root_dir).resolve()
        seq_dirs: List[Path] = []
        for seqinfo_file in sorted(p.rglob("seqinfo.ini")):
            seq_dirs.append(seqinfo_file.parent)

        reports: Dict[str, SequenceIntegrityReport] = {}
        for sdir in seq_dirs:
            rep = cls.audit_sequence(sdir)
            reports[rep.sequence_name] = rep

        total_seqs = len(reports)
        complete_seqs = sum(1 for r in reports.values() if r.is_complete)
        total_exp = sum(r.expected_frames for r in reports.values())
        total_act = sum(r.actual_frames_on_disk for r in reports.values())
        total_corr = sum(len(r.corrupted_frames) for r in reports.values())
        total_zero = sum(len(r.zero_byte_frames) for r in reports.values())

        return DatasetIntegrityReport(
            total_sequences=total_seqs,
            complete_sequences=complete_seqs,
            total_expected_frames=total_exp,
            total_frames_on_disk=total_act,
            total_corrupted=total_corr,
            total_zero_byte=total_zero,
            sequence_reports=reports,
        )


class MatchDatasetMaterializer:
    """Materializes independent match partitions into Ultralytics YOLO training layout."""

    @staticmethod
    def assign_sequence_splits(
        sequence_dirs: List[Path],
        split_match_map: Dict[str, List[str]] = INTERNAL_SPLIT_MATCHES,
    ) -> Dict[str, List[Path]]:
        """Map sequence directories to splits strictly based on game_id.

        Asserts:
        - No sequence appears in multiple splits.
        - No game_id appears in multiple splits.
        - No sequence lacks a valid game_id.

        Args:
            sequence_dirs: List of candidate sequence directory paths.
            split_match_map: Mapping of split name to list of game_id strings.

        Returns:
            Dict[str, List[Path]]: Sequences mapped to 'train', 'val', 'test'.
        """
        # Validate that split_match_map has mutually disjoint match groups
        seen_games: Dict[str, str] = {}
        for split, games in split_match_map.items():
            for g in games:
                if g in seen_games:
                    raise ValueError(f"Match gameID={g} assigned to multiple splits: {seen_games[g]} and {split}")
                seen_games[g] = split

        assigned: Dict[str, List[Path]] = {s: [] for s in split_match_map.keys()}
        seen_seq_names: Set[str] = set()

        for sdir in sequence_dirs:
            if sdir.name in seen_seq_names:
                raise ValueError(f"Duplicate sequence name detected: {sdir.name}")
            seen_seq_names.add(sdir.name)

            gameinfo_p = sdir / "gameinfo.ini"
            if not gameinfo_p.is_file():
                raise ValueError(f"Sequence {sdir.name} is missing gameinfo.ini; cannot determine match assignment.")

            from sportiq.dataset.annotation_parser import parse_gameinfo
            ginfo = parse_gameinfo(gameinfo_p)
            gid = ginfo.game_id
            if not gid:
                raise ValueError(f"Sequence {sdir.name} has no valid game_id in {gameinfo_p}")

            target_split = seen_games.get(str(gid))
            if not target_split:
                raise ValueError(f"Sequence {sdir.name} belongs to gameID={gid} which is not in the target split map.")

            assigned[target_split].append(sdir)

        return assigned

    @classmethod
    def materialize_yolo_dataset(
        cls,
        split_sequences: Dict[str, List[Path]],
        output_dir: Union[str, Path],
        prefer_hardlink: bool = True,
        clip_boxes: bool = True,
    ) -> Dict[str, Any]:
        """Convert annotations and materialize images and labels into standard YOLO structure.

        Args:
            split_sequences: Mapping of split name ('train', 'val', 'test') to sequence paths.
            output_dir: Destination directory (e.g. data/processed/soccernet_yolo).
            prefer_hardlink: If True, uses hardlinks for images; falls back to copy.
            clip_boxes: Whether to clip boundary-exceeding annotations.

        Returns:
            Dict[str, Any]: Summary metrics of materialized dataset.
        """
        out_p = Path(output_dir).resolve()
        images_root = out_p / "images"
        labels_root = out_p / "labels"

        for split in ("train", "val", "test"):
            (images_root / split).mkdir(parents=True, exist_ok=True)
            (labels_root / split).mkdir(parents=True, exist_ok=True)

        validator = AnnotationValidator(clip_boxes=clip_boxes)
        converter = YOLOConverter(taxonomy=ClassTaxonomyType.FOUR_CLASS, validator=validator)
        split_metrics: Dict[str, Dict[str, Any]] = {}

        total_images_materialized = 0
        total_labels_materialized = 0

        for split, seq_list in split_sequences.items():
            split_img_dir = images_root / split
            split_lbl_dir = labels_root / split

            images_count = 0
            labels_count = 0

            for sdir in sorted(seq_list, key=lambda x: x.name):
                seq = load_sequence(sdir)

                # 1. Convert sequence to YOLO labels in a temp/intermediate folder
                seq_labels_dir = sdir / "_yolo_labels_tmp"
                seq_labels_dir.mkdir(parents=True, exist_ok=True)
                try:
                    res = converter.convert_sequence(seq, output_dir=seq_labels_dir)

                    # 2. Link or copy images and labels into split folder with sequence prefix
                    im_dir = sdir / (seq.seq_info.im_dir or "img1")
                    im_ext = seq.seq_info.im_ext or ".jpg"

                    # Collect images on disk
                    disk_images = sorted(im_dir.glob(f"*{im_ext}")) if im_dir.is_dir() else []

                    # Expected frame count from seqinfo
                    for frame_id in range(1, seq.seq_info.seq_length + 1):
                        frame_stem = f"{frame_id:06d}"
                        prefixed_stem = f"{seq.name}_{frame_stem}"

                        # Source image
                        src_img = im_dir / f"{frame_stem}{im_ext}"
                        dst_img = split_img_dir / f"{prefixed_stem}{im_ext}"

                        if src_img.is_file():
                            if not dst_img.exists():
                                if prefer_hardlink:
                                    try:
                                        os.link(src_img, dst_img)
                                    except (OSError, NotImplementedError):
                                        shutil.copy2(src_img, dst_img)
                                else:
                                    shutil.copy2(src_img, dst_img)
                            images_count += 1

                        # Source label (0-byte file created for background frames if none existed)
                        src_lbl = seq_labels_dir / f"{frame_stem}.txt"
                        dst_lbl = split_lbl_dir / f"{prefixed_stem}.txt"

                        if src_lbl.is_file():
                            shutil.copy2(src_lbl, dst_lbl)
                        else:
                            # Intentional empty negative label
                            dst_lbl.write_text("", encoding="utf-8")
                        labels_count += 1

                finally:
                    if seq_labels_dir.exists():
                        shutil.rmtree(seq_labels_dir, ignore_errors=True)

            split_metrics[split] = {
                "sequences_count": len(seq_list),
                "images_count": images_count,
                "labels_count": labels_count,
            }
            total_images_materialized += images_count
            total_labels_materialized += labels_count

        # Write dataset.yaml
        dataset_yaml_path = out_p / "dataset.yaml"
        yaml_content = {
            "path": out_p.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": ClassTaxonomy.FOUR_CLASS_NAMES,
        }
        with dataset_yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False)

        return {
            "output_dir": str(out_p),
            "dataset_yaml": str(dataset_yaml_path),
            "total_images": total_images_materialized,
            "total_labels": total_labels_materialized,
            "splits": split_metrics,
        }
