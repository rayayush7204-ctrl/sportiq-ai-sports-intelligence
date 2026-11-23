"""YOLO image-to-label alignment, indexing, and dataset materialization."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from sportiq.dataset.annotation_parser import SequenceDataset
from sportiq.dataset.class_mapping import ClassTaxonomy, ClassTaxonomyType
from sportiq.dataset.dataset_splitter import DatasetSplit

logger = logging.getLogger(__name__)


class AlignmentVerificationError(Exception):
    """Raised when YOLO image-to-label correspondence or validation fails."""


@dataclass
class AlignmentReport:
    """Report detailing image-to-label verification results."""

    dataset_yaml: str
    total_images: int = 0
    total_labels: int = 0
    labeled_frames: int = 0
    background_frames: int = 0
    missing_images: List[str] = field(default_factory=list)
    missing_labels: List[str] = field(default_factory=list)
    corrupted_labels: List[str] = field(default_factory=list)
    split_counts: Dict[str, int] = field(default_factory=dict)
    is_aligned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_yaml": self.dataset_yaml,
            "total_images": self.total_images,
            "total_labels": self.total_labels,
            "labeled_frames": self.labeled_frames,
            "background_frames": self.background_frames,
            "missing_images_count": len(self.missing_images),
            "missing_labels_count": len(self.missing_labels),
            "corrupted_labels_count": len(self.corrupted_labels),
            "split_counts": self.split_counts,
            "is_aligned": self.is_aligned,
        }


class DatasetMaterializer:
    """Materializes and verifies YOLO train/val/test dataset layouts."""

    SUPPORTED_MODES = ("manifest", "symlink", "hardlink", "copy")

    @classmethod
    def materialize(
        cls,
        split: DatasetSplit,
        sequences: Dict[str, SequenceDataset],
        labels_base_dir: Union[str, Path],
        output_dir: Union[str, Path],
        mode: str = "manifest",
        taxonomy: ClassTaxonomyType = ClassTaxonomyType.FOUR_CLASS,
    ) -> Dict[str, Any]:
        """Materialize a YOLO dataset partition preserving 1:1 image-label correspondence.

        Args:
            split: DatasetSplit with train, val, test sequences.
            sequences: Mapping of sequence_name -> loaded SequenceDataset.
            labels_base_dir: Directory containing exported YOLO labels per sequence (e.g. data/processed/yolo/labels).
            output_dir: Destination directory for dataset.yaml and index files/folders.
            mode: 'manifest' (writes train.txt/val.txt/test.txt without duplicating data),
                  'symlink', 'hardlink', or 'copy'.
            taxonomy: Detection taxonomy for class naming.

        Returns:
            Dict[str, Any]: Summary of materialized files and verification status.

        Raises:
            ValueError: If mode is unsupported.
            AlignmentVerificationError: If image-to-label pairing is broken.
        """
        if mode not in cls.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Choose from: {cls.SUPPORTED_MODES}")

        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        lbl_base = Path(labels_base_dir).resolve()

        split_map = {
            "train": split.train_sequences,
            "val": split.val_sequences,
            "test": split.test_sequences,
        }

        manifest_lines: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        total_paired_images = 0
        total_labeled_frames = 0
        total_background_frames = 0

        for split_name, seq_names in split_map.items():
            for seq_name in seq_names:
                if seq_name not in sequences:
                    raise AlignmentVerificationError(
                        f"Sequence '{seq_name}' specified in split '{split_name}' was not found in loaded sequences."
                    )

                seq = sequences[seq_name]
                seq_lbl_dir = lbl_base / seq_name
                im_dir = seq.sequence_path / seq.seq_info.im_dir
                im_ext = seq.seq_info.im_ext

                # Find all images for this sequence
                # SoccerNet MOT frame numbering: 1-based, padded to 6 digits (000001.jpg)
                frame_count = seq.seq_info.seq_length
                frame_indices = list(range(1, frame_count + 1)) if frame_count > 0 else []

                # If im_dir exists and contains extracted image files, prioritize physical files on disk
                if im_dir.exists():
                    found_imgs = sorted(list(im_dir.glob(f"*{im_ext}")) + list(im_dir.glob("*.jpg")) + list(im_dir.glob("*.png")))
                    found_fids = set()
                    for img_p in found_imgs:
                        try:
                            found_fids.add(int(img_p.stem))
                        except ValueError:
                            pass
                    if found_fids and (len(frame_indices) == 0 or len(found_fids) < len(frame_indices)):
                        frame_indices = sorted(list(found_fids))

                for fid in sorted(frame_indices):
                    img_path = im_dir / f"{fid:06d}{im_ext}"
                    if not img_path.exists():
                        # Try without padding or other extensions
                        cands = [
                            im_dir / f"{fid:06d}.jpg",
                            im_dir / f"{fid:06d}.png",
                            im_dir / f"{fid}{im_ext}",
                        ]
                        for c in cands:
                            if c.exists():
                                img_path = c
                                break

                    if not img_path.exists():
                        raise AlignmentVerificationError(
                            f"Missing source image frame {fid:06d} for sequence '{seq_name}' at {img_path}"
                        )

                    lbl_file = seq_lbl_dir / f"{fid:06d}.txt"
                    has_labels = lbl_file.exists() and lbl_file.stat().st_size > 0

                    if has_labels:
                        total_labeled_frames += 1
                        # Verify bounding box validity
                        cls._validate_yolo_label_file(lbl_file)
                    else:
                        total_background_frames += 1

                    if mode == "manifest":
                        # Record canonical POSIX image path in split manifest text file
                        manifest_lines[split_name].append(img_path.resolve().as_posix())
                    else:
                        # Materialize into images/<split> and labels/<split>
                        dst_img_dir = out_path / "images" / split_name
                        dst_lbl_dir = out_path / "labels" / split_name
                        dst_img_dir.mkdir(parents=True, exist_ok=True)
                        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

                        dst_img = dst_img_dir / f"{seq_name}_{fid:06d}{img_path.suffix}"
                        dst_lbl = dst_lbl_dir / f"{seq_name}_{fid:06d}.txt"

                        cls._link_or_copy(img_path, dst_img, mode=mode)
                        if has_labels:
                            cls._link_or_copy(lbl_file, dst_lbl, mode=mode)
                        else:
                            # Write empty file for negative / background frame
                            dst_lbl.touch(exist_ok=True)

                    split_counts[split_name] += 1
                    total_paired_images += 1

        # Write split manifests if in manifest mode
        yaml_split_refs = {}
        if mode == "manifest":
            for split_name in ("train", "val", "test"):
                txt_path = out_path / f"{split_name}.txt"
                content = "\n".join(manifest_lines[split_name]) + ("\n" if manifest_lines[split_name] else "")
                txt_path.write_text(content, encoding="utf-8")
                yaml_split_refs[split_name] = f"{split_name}.txt"
        else:
            for split_name in ("train", "val", "test"):
                yaml_split_refs[split_name] = f"images/{split_name}"

        # Generate dataset.yaml
        class_names = ClassTaxonomy.get_names(taxonomy)
        yolo_yaml_data = {
            "path": out_path.as_posix(),
            "train": yaml_split_refs["train"],
            "val": yaml_split_refs["val"],
            "test": yaml_split_refs["test"],
            "names": {int(k): v for k, v in class_names.items()},
        }
        yaml_path = out_path / "dataset.yaml"
        yaml_path.write_text(yaml.dump(yolo_yaml_data, sort_keys=False), encoding="utf-8")

        return {
            "output_directory": str(out_path),
            "dataset_yaml": str(yaml_path),
            "mode": mode,
            "taxonomy": taxonomy.value,
            "total_images": total_paired_images,
            "total_labeled_frames": total_labeled_frames,
            "total_background_frames": total_background_frames,
            "split_counts": split_counts,
            "status": "materialized",
        }

    @staticmethod
    def _validate_yolo_label_file(lbl_path: Path) -> None:
        """Verify that every line in a YOLO label file is syntactically and numerically valid."""
        with lbl_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 5:
                    raise AlignmentVerificationError(
                        f"Malformed YOLO label at {lbl_path}:{line_idx}: expected 5 tokens, got {len(parts)}"
                    )
                try:
                    cid = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError as e:
                    raise AlignmentVerificationError(
                        f"Non-numeric token in YOLO label {lbl_path}:{line_idx}: {e}"
                    ) from e

                for val, name in ((xc, "x_center"), (yc, "y_center"), (w, "width"), (h, "height")):
                    if not (0.0 <= val <= 1.0):
                        raise AlignmentVerificationError(
                            f"YOLO coordinate out of [0, 1] range ({name}={val}) at {lbl_path}:{line_idx}"
                        )

    @staticmethod
    def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
        """Safely create symlink, hardlink, or copy with Windows fallback."""
        if dst.exists():
            dst.unlink()

        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "hardlink":
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        elif mode == "symlink":
            try:
                os.symlink(src, dst)
            except OSError:
                # Windows without SeCreateSymbolicLinkPrivilege falls back to hardlink or copy
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

    @classmethod
    def verify_dataset_alignment(cls, dataset_yaml_path: Union[str, Path]) -> AlignmentReport:
        """Verify that all images and labels referenced by dataset.yaml exist and correspond 1:1.

        Args:
            dataset_yaml_path: Path to dataset.yaml.

        Returns:
            AlignmentReport: Complete audit metrics.
        """
        ypath = Path(dataset_yaml_path).resolve()
        if not ypath.exists():
            raise FileNotFoundError(f"dataset.yaml not found at {ypath}")

        data = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        root = Path(data.get("path", ypath.parent)).resolve()

        report = AlignmentReport(dataset_yaml=str(ypath))
        missing_imgs: List[str] = []
        missing_lbls: List[str] = []
        corrupt_lbls: List[str] = []
        split_counts: Dict[str, int] = {}

        for split in ("train", "val", "test"):
            split_ref = data.get(split)
            if not split_ref:
                continue

            ref_path = root / split_ref if not Path(split_ref).is_absolute() else Path(split_ref)
            img_list: List[Path] = []

            if ref_path.is_file() and ref_path.suffix == ".txt":
                # Manifest text file
                lines = ref_path.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    if line.strip():
                        img_list.append(Path(line.strip()))
            elif ref_path.is_dir():
                for ext in (".jpg", ".png", ".jpeg"):
                    img_list.extend(ref_path.glob(f"*{ext}"))

            split_counts[split] = len(img_list)
            report.total_images += len(img_list)

            for img_p in img_list:
                if not img_p.exists():
                    missing_imgs.append(str(img_p))
                    continue

                # Locate corresponding label
                # In standard layout: images/train/xxx.jpg -> labels/train/xxx.txt
                # In manifest mode: /.../SNMOT-01/img1/000001.jpg -> labels/SNMOT-01/000001.txt
                lbl_candidates = [
                    Path(str(img_p.parent).replace("images", "labels")) / f"{img_p.stem}.txt",
                    img_p.parent.parent / "yolo" / "labels" / img_p.parent.parent.name / f"{img_p.stem}.txt",
                    root / "labels" / split / f"{img_p.stem}.txt",
                ]

                lbl_found = None
                for cand in lbl_candidates:
                    if cand.exists():
                        lbl_found = cand
                        break

                if lbl_found is not None:
                    report.total_labels += 1
                    try:
                        cls._validate_yolo_label_file(lbl_found)
                        if lbl_found.stat().st_size > 0:
                            report.labeled_frames += 1
                        else:
                            report.background_frames += 1
                    except AlignmentVerificationError as e:
                        corrupt_lbls.append(f"{lbl_found}: {e}")
                else:
                    report.background_frames += 1

        report.missing_images = missing_imgs
        report.missing_labels = missing_lbls
        report.corrupted_labels = corrupt_lbls
        report.split_counts = split_counts
        report.is_aligned = (
            len(missing_imgs) == 0 and len(corrupt_lbls) == 0 and report.total_images > 0
        )
        return report
