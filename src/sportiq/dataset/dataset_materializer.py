"""YOLO image-to-label alignment, indexing, and dataset materialization.

Ultralytics YOLO Compatibility Notes:
  - Ultralytics YOLO resolves label paths from image paths by replacing the '/images/'
    directory component with '/labels/' and the image extension with '.txt'.
  - Directory modes ('copy', 'hardlink', 'symlink') produce standard, training-ready
    Ultralytics layouts:
        <output_dir>/
          dataset.yaml
          images/{train,val,test}/<seq>_<frame>.jpg
          labels/{train,val,test}/<seq>_<frame>.txt
    Every image has a corresponding label file: non-empty for positive detections,
    and 0-byte empty files for intentional negative / background frames.
  - Manifest mode ('manifest') generates train.txt, val.txt, test.txt containing
    resolved image paths without copying, alongside a labels_manifest.json sidecar
    mapping each image path to its exact ground-truth label file.
    If source paths do not have an '/images/' folder (e.g. SoccerNet's 'img1/'),
    Ultralytics training requires directory mode ('copy', 'symlink', 'hardlink')
    to map images to labels automatically.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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

        class_names = ClassTaxonomy.get_names(taxonomy)
        valid_class_ids = set(class_names.keys())

        split_map = {
            "train": split.train_sequences,
            "val": split.val_sequences,
            "test": split.test_sequences,
        }

        manifest_lines: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        manifest_image_to_label: Dict[str, str] = {}
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

                # Discover frames for this sequence
                frame_count = seq.seq_info.seq_length
                frame_indices = list(range(1, frame_count + 1)) if frame_count > 0 else []

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
                    # In standard pipeline, every frame must have an explicit label file (non-empty or 0-byte negative)
                    if not lbl_file.exists():
                        raise AlignmentVerificationError(
                            f"Missing expected label file for frame {fid:06d} of sequence '{seq_name}' at {lbl_file}"
                        )

                    is_positive = lbl_file.stat().st_size > 0
                    if is_positive:
                        total_labeled_frames += 1
                        cls._validate_yolo_label_file(lbl_file, valid_class_ids=valid_class_ids)
                    else:
                        total_background_frames += 1

                    resolved_img_str = img_path.resolve().as_posix()
                    resolved_lbl_str = lbl_file.resolve().as_posix()

                    if mode == "manifest":
                        manifest_lines[split_name].append(resolved_img_str)
                        manifest_image_to_label[resolved_img_str] = resolved_lbl_str
                    else:
                        dst_img_dir = out_path / "images" / split_name
                        dst_lbl_dir = out_path / "labels" / split_name
                        dst_img_dir.mkdir(parents=True, exist_ok=True)
                        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

                        dst_img = dst_img_dir / f"{seq_name}_{fid:06d}{img_path.suffix}"
                        dst_lbl = dst_lbl_dir / f"{seq_name}_{fid:06d}.txt"

                        cls._link_or_copy(img_path, dst_img, mode=mode)
                        if is_positive:
                            cls._link_or_copy(lbl_file, dst_lbl, mode=mode)
                        else:
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

            # Write exact image-to-label sidecar mapping for manifest mode
            labels_manifest_path = out_path / "labels_manifest.json"
            labels_manifest_path.write_text(json.dumps(manifest_image_to_label, indent=2), encoding="utf-8")
        else:
            for split_name in ("train", "val", "test"):
                yaml_split_refs[split_name] = f"images/{split_name}"

        # Generate dataset.yaml
        yolo_yaml_data = {
            "path": out_path.as_posix(),
            "train": yaml_split_refs["train"],
            "val": yaml_split_refs["val"],
            "test": yaml_split_refs["test"],
            "names": {int(k): v for k, v in class_names.items()},
        }
        if mode == "manifest":
            yolo_yaml_data["labels_manifest"] = "labels_manifest.json"
            yolo_yaml_data["labels_dir"] = lbl_base.as_posix()

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
    def _validate_yolo_label_file(
        lbl_path: Path,
        valid_class_ids: Optional[Set[int]] = None,
    ) -> None:
        """Validate syntax, finiteness, class ID and [0, 1] range bounds for YOLO label file."""
        if not lbl_path.exists():
            raise AlignmentVerificationError(f"Label file not found: {lbl_path}")

        content = lbl_path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()
        for line_idx, line in enumerate(lines, start=1):
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

            # Check finite values
            for val, name in ((xc, "x_center"), (yc, "y_center"), (w, "width"), (h, "height")):
                if not math.isfinite(val):
                    raise AlignmentVerificationError(
                        f"Non-finite YOLO coordinate ({name}={val}) at {lbl_path}:{line_idx}"
                    )

            # Check class ID against taxonomy
            if valid_class_ids is not None and cid not in valid_class_ids:
                raise AlignmentVerificationError(
                    f"Invalid class ID {cid} at {lbl_path}:{line_idx}: not defined in dataset names {valid_class_ids}"
                )

            # Check [0, 1] range bounds for centers and dimensions
            for val, name in ((xc, "x_center"), (yc, "y_center"), (w, "width"), (h, "height")):
                if not (0.0 <= val <= 1.0):
                    raise AlignmentVerificationError(
                        f"YOLO coordinate out of [0, 1] range ({name}={val}) at {lbl_path}:{line_idx}"
                    )

            # Check bounding-box geometric extent boundaries:
            # xc - width / 2 >= 0
            # xc + width / 2 <= 1
            # yc - height / 2 >= 0
            # yc + height / 2 <= 1
            eps = 1e-5
            x_min = xc - w / 2.0
            x_max = xc + w / 2.0
            y_min = yc - h / 2.0
            y_max = yc + h / 2.0

            if x_min < -eps:
                raise AlignmentVerificationError(
                    f"YOLO bounding box extends beyond left canvas boundary (x_min={x_min:.6f} < 0) at {lbl_path}:{line_idx}"
                )
            if x_max > 1.0 + eps:
                raise AlignmentVerificationError(
                    f"YOLO bounding box extends beyond right canvas boundary (x_max={x_max:.6f} > 1) at {lbl_path}:{line_idx}"
                )
            if y_min < -eps:
                raise AlignmentVerificationError(
                    f"YOLO bounding box extends beyond top canvas boundary (y_min={y_min:.6f} < 0) at {lbl_path}:{line_idx}"
                )
            if y_max > 1.0 + eps:
                raise AlignmentVerificationError(
                    f"YOLO bounding box extends beyond bottom canvas boundary (y_max={y_max:.6f} > 1) at {lbl_path}:{line_idx}"
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
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

    @classmethod
    def verify_dataset_alignment(cls, dataset_yaml_path: Union[str, Path]) -> AlignmentReport:
        """Verify that all images and labels referenced by dataset.yaml exist and correspond 1:1.

        Distinguishes legitimate empty negative labels (0 bytes) from missing labels.
        Validates class IDs against dataset.yaml names and verifies coordinate bounds.

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

        class_names_raw = data.get("names", {})
        valid_class_ids = set(int(k) for k in class_names_raw.keys()) if isinstance(class_names_raw, dict) else set()

        # Check for manifest-mode sidecar mapping
        manifest_label_map: Dict[str, str] = {}
        labels_manifest_key = data.get("labels_manifest")
        if labels_manifest_key:
            man_p = root / labels_manifest_key if not Path(labels_manifest_key).is_absolute() else Path(labels_manifest_key)
            if man_p.is_file():
                try:
                    manifest_label_map = json.loads(man_p.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning("Could not read %s: %s", man_p, e)
        elif (root / "labels_manifest.json").is_file():
            try:
                manifest_label_map = json.loads((root / "labels_manifest.json").read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Could not read labels_manifest.json: %s", e)

        labels_dir_key = data.get("labels_dir")
        configured_labels_dir = Path(labels_dir_key).resolve() if labels_dir_key else None

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
            is_manifest_mode = False

            if ref_path.is_file() and ref_path.suffix == ".txt":
                # Manifest text file
                is_manifest_mode = True
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

                lbl_found: Optional[Path] = None

                # Strategy A: Exact manifest lookup
                if manifest_label_map:
                    # Try resolved posix, absolute posix, and str
                    for k in (img_p.resolve().as_posix(), img_p.as_posix(), str(img_p.resolve()), str(img_p)):
                        if k in manifest_label_map:
                            exact_candidate = Path(manifest_label_map[k])
                            if exact_candidate.is_file():
                                lbl_found = exact_candidate
                            else:
                                missing_lbls.append(str(exact_candidate))
                            break

                    # If manifest explicitly mapped this image but candidate is missing, continue
                    if lbl_found is None and any(k in manifest_label_map for k in (img_p.resolve().as_posix(), img_p.as_posix())):
                        continue

                # Strategy B: Standard path discovery candidates
                if lbl_found is None:
                    lbl_candidates: List[Path] = []
                    # 1. Standard Ultralytics directory mapping: images/<split>/<stem>.<ext> -> labels/<split>/<stem>.txt
                    if "images" in img_p.parts:
                        lbl_candidates.append(
                            Path(str(img_p.parent).replace("images", "labels")) / f"{img_p.stem}.txt"
                        )
                    # 2. Split directory layout under root
                    lbl_candidates.append(root / "labels" / split / f"{img_p.stem}.txt")
                    # 3. Configured labels_dir from dataset.yaml
                    if configured_labels_dir:
                        lbl_candidates.append(configured_labels_dir / img_p.parent.parent.name / f"{img_p.stem}.txt")
                        lbl_candidates.append(configured_labels_dir / split / f"{img_p.stem}.txt")
                    # 4. Sequence directory layout under root
                    lbl_candidates.append(root / "labels" / img_p.parent.parent.name / f"{img_p.stem}.txt")
                    # 5. Same directory as image
                    lbl_candidates.append(img_p.parent / f"{img_p.stem}.txt")
                    # 6. yolo/labels layout in parent
                    lbl_candidates.append(
                        img_p.parent.parent / "yolo" / "labels" / img_p.parent.parent.name / f"{img_p.stem}.txt"
                    )

                    for cand in lbl_candidates:
                        if cand.is_file():
                            lbl_found = cand
                            break

                    if lbl_found is None:
                        # Missing label! Never count as background frame
                        primary_expected = lbl_candidates[0] if lbl_candidates else (root / "labels" / split / f"{img_p.stem}.txt")
                        missing_lbls.append(str(primary_expected))
                        continue

                report.total_labels += 1
                try:
                    # An empty label file (0 bytes or blank) is a verified intentional negative frame
                    is_empty = (lbl_found.stat().st_size == 0) or not lbl_found.read_text(encoding="utf-8").strip()
                    if is_empty:
                        report.background_frames += 1
                    else:
                        cls._validate_yolo_label_file(lbl_found, valid_class_ids=valid_class_ids)
                        report.labeled_frames += 1
                except AlignmentVerificationError as e:
                    corrupt_lbls.append(f"{lbl_found}: {e}")

            # Check reverse correspondence in directory mode to detect orphaned labels / deleted images
            if not is_manifest_mode and ref_path.is_dir():
                lbl_dir = root / "labels" / split
                if not lbl_dir.is_dir() and "images" in ref_path.parts:
                    lbl_dir = Path(str(ref_path).replace("images", "labels"))
                if lbl_dir.is_dir():
                    for lbl_file in lbl_dir.glob("*.txt"):
                        has_img = any(
                            (ref_path / f"{lbl_file.stem}{ext}").is_file()
                            for ext in (".jpg", ".png", ".jpeg")
                        )
                        if not has_img:
                            missing_imgs.append(str(ref_path / f"{lbl_file.stem}.jpg"))

        report.missing_images = sorted(list(set(missing_imgs)))
        report.missing_labels = sorted(list(set(missing_lbls)))
        report.corrupted_labels = corrupt_lbls
        report.split_counts = split_counts
        report.is_aligned = (
            len(report.missing_images) == 0
            and len(report.missing_labels) == 0
            and len(report.corrupted_labels) == 0
            and report.total_images > 0
        )
        return report
