"""Convert SoccerNet MOT annotations to normalized YOLO detection format.

Converts pixel bounding boxes (top-left x, y, width, height) to normalized
center coordinates (x_center, y_center, width, height) and generates sidecar
JSONL metadata preserving original tracking identities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sportiq.dataset.annotation_parser import SequenceDataset
from sportiq.dataset.annotation_validator import AnnotationValidator, ValidationResult
from sportiq.dataset.class_mapping import ClassTaxonomyType, resolve_class

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YOLOInstance:
    """Normalized YOLO detection instance."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_line(self) -> str:
        """Format as standard YOLO label line: <class_id> <x> <y> <w> <h>."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class YOLOConverter:
    """Converts SoccerNet tracking sequences into YOLO format with metadata sidecars."""

    def __init__(
        self,
        taxonomy: ClassTaxonomyType = ClassTaxonomyType.FOUR_CLASS,
        validator: Optional[AnnotationValidator] = None,
    ) -> None:
        """Initialize converter.

        Args:
            taxonomy: Detection taxonomy (FOUR_CLASS, FIVE_CLASS, TWO_CLASS).
            validator: Annotation validator instance.
        """
        self.taxonomy = taxonomy
        self.validator = validator or AnnotationValidator(clip_boxes=True)

    @staticmethod
    def pixel_to_yolo_bbox(
        left: float,
        top: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float, float, float]:
        """Convert pixel top-left (left, top, w, h) to normalized (x_center, y_center, w, h).

        Formulas:
          x_center = (left + width / 2.0) / image_width
          y_center = (top + height / 2.0) / image_height
          w_norm = width / image_width
          h_norm = height / image_height
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                f"Invalid image dimensions: {image_width}x{image_height}"
            )

        x_center = (left + (width / 2.0)) / image_width
        y_center = (top + (height / 2.0)) / image_height
        norm_width = width / image_width
        norm_height = height / image_height

        # Clamp to [0.0, 1.0] range
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_width = max(0.0, min(1.0, norm_width))
        norm_height = max(0.0, min(1.0, norm_height))

        return x_center, y_center, norm_width, norm_height

    def convert_sequence(
        self,
        sequence: SequenceDataset,
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """Convert a SequenceDataset into YOLO labels and sidecar metadata.

        Args:
            sequence: Loaded SequenceDataset.
            output_dir: Base directory for exported YOLO labels.

        Returns:
            Dict[str, Any]: Conversion statistics and generated file paths.
        """
        out_path = Path(output_dir).resolve()
        labels_dir = out_path / "labels" / sequence.name
        labels_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = labels_dir / "labels_metadata.jsonl"
        im_w = sequence.seq_info.im_width
        im_h = sequence.seq_info.im_height

        # Validate all annotations in sequence
        valid_pairs, val_result = self.validator.filter_and_validate(
            sequence.annotations, image_width=im_w, image_height=im_h
        )

        # Group valid annotations by frame_id
        frames_dict: Dict[int, List[Tuple[Any, Tuple[float, float, float, float]]]] = {}
        for ann, sanitized_xywh in valid_pairs:
            frames_dict.setdefault(ann.frame_id, []).append((ann, sanitized_xywh))

        exported_yolo_boxes = 0
        skipped_unmapped_boxes = 0
        total_sidecar_records = 0

        with manifest_path.open("w", encoding="utf-8") as manifest_f:
            for frame_id, ann_list in sorted(frames_dict.items()):
                # MOT is 1-based, YOLO frame naming is 0-based or 1-based.
                # Standard practice: match frame filenames (e.g. 000001.txt for frame 1)
                label_file = labels_dir / f"{frame_id:06d}.txt"
                yolo_lines: List[str] = []

                for ann, (x, y, w, h) in ann_list:
                    # Resolve class from gameinfo.ini mapping if available
                    raw_class = None
                    if sequence.game_info and ann.track_id in sequence.game_info.track_classes:
                        raw_class = sequence.game_info.track_classes[ann.track_id]

                    class_id, canonical_name, raw_norm = resolve_class(
                        raw_class, taxonomy=self.taxonomy
                    )

                    # Compute normalized YOLO coordinates
                    xc, yc, wn, hn = self.pixel_to_yolo_bbox(
                        x, y, w, h, image_width=im_w, image_height=im_h
                    )

                    # Write sidecar metadata record (preserves full lineage & track ID)
                    sidecar_record = {
                        "sequence_name": sequence.name,
                        "frame_id_1based": frame_id,
                        "frame_id_0based": frame_id - 1,
                        "track_id": ann.track_id,
                        "raw_class": raw_class or "unclassified",
                        "canonical_class": canonical_name,
                        "class_id": class_id,
                        "bbox_pixel_xywh": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                        "bbox_yolo_xywh": [round(xc, 6), round(yc, 6), round(wn, 6), round(hn, 6)],
                        "source_conf": ann.conf,
                    }
                    manifest_f.write(json.dumps(sidecar_record) + "\n")
                    total_sidecar_records += 1

                    if class_id is not None:
                        instance = YOLOInstance(
                            class_id=class_id,
                            x_center=xc,
                            y_center=yc,
                            width=wn,
                            height=hn,
                        )
                        yolo_lines.append(instance.to_yolo_line())
                        exported_yolo_boxes += 1
                    else:
                        skipped_unmapped_boxes += 1

                if yolo_lines:
                    label_file.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        summary = {
            "sequence_name": sequence.name,
            "taxonomy": self.taxonomy.value,
            "labels_directory": str(labels_dir),
            "manifest_path": str(manifest_path),
            "total_annotations": len(sequence.annotations),
            "exported_yolo_boxes": exported_yolo_boxes,
            "skipped_unmapped_boxes": skipped_unmapped_boxes,
            "total_sidecar_records": total_sidecar_records,
            "validation": val_result.to_dict(),
        }
        return summary
