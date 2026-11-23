"""Visualization utility for drawing ground-truth bounding boxes, track IDs, and classes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from sportiq.dataset.annotation_parser import GameInfo, MOTAnnotation, SequenceDataset, load_sequence
from sportiq.dataset.annotation_validator import AnnotationValidator
from sportiq.dataset.class_mapping import ClassTaxonomyType, resolve_class

logger = logging.getLogger(__name__)

# BGR color palette for visualization
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "player": (0, 200, 0),         # Green
    "goalkeeper": (0, 140, 255),    # Orange
    "referee": (0, 255, 255),       # Yellow
    "ball": (255, 255, 0),          # Cyan
    "staff": (255, 0, 255),         # Magenta
    "person": (0, 220, 100),        # Light Green
    "unclassified": (180, 180, 180), # Gray
}


def draw_annotations(
    image: np.ndarray,
    annotations: List[MOTAnnotation],
    game_info: Optional[GameInfo] = None,
    taxonomy: ClassTaxonomyType = ClassTaxonomyType.FOUR_CLASS,
    validator: Optional[AnnotationValidator] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes, track IDs, and class labels on an image.

    Args:
        image: BGR image array (H, W, C).
        annotations: List of MOTAnnotation instances for the current frame.
        game_info: Optional GameInfo containing track_id to class mapping.
        taxonomy: Selected ClassTaxonomyType.
        validator: Optional AnnotationValidator to validate/clip coordinates.
        line_thickness: Thickness of bounding box rectangles.
        font_scale: Scale for cv2 text rendering.

    Returns:
        np.ndarray: Annotated copy of the input image.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]

    val = validator or AnnotationValidator(clip_boxes=True)

    for ann in annotations:
        is_valid, sanitized_xywh, _ = val.validate_annotation(ann, image_width=w, image_height=h)
        if not is_valid or sanitized_xywh is None:
            continue

        x, y, bw, bh = sanitized_xywh
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + bw)), int(round(y + bh))

        # Resolve class
        raw_class = None
        if game_info and ann.track_id in game_info.track_classes:
            raw_class = game_info.track_classes[ann.track_id]

        class_id, canonical_name, _ = resolve_class(raw_class, taxonomy=taxonomy)
        color = CLASS_COLORS_BGR.get(canonical_name, (200, 200, 200))

        # Draw bounding rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)

        # Label text: e.g. "#1 player" or "#1 player (ID:0)"
        class_str = canonical_name if class_id is not None else f"{canonical_name}*"
        label = f"#{ann.track_id} {class_str}"

        # Draw label background box
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_y1 = max(0, y1 - label_h - 4)
        label_y2 = label_y1 + label_h + 4
        label_x2 = min(w, x1 + label_w + 4)

        cv2.rectangle(canvas, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, label_y2 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            1,
            cv2.LINE_AA,
        )

    return canvas


def visualize_sequence_frame(
    sequence: SequenceDataset,
    frame_id: int,
    output_path: Optional[Union[str, Path]] = None,
    image: Optional[np.ndarray] = None,
    taxonomy: ClassTaxonomyType = ClassTaxonomyType.FOUR_CLASS,
) -> np.ndarray:
    """Visualize annotations for a specific frame within a SequenceDataset.

    Args:
        sequence: SequenceDataset loaded from sequence directory.
        frame_id: 1-based frame index.
        output_path: Optional file path to save visualized image (JPEG/PNG).
        image: Optional existing image array. If None, checks sequence img1/ or creates a blank canvas.
        taxonomy: Detection taxonomy.

    Returns:
        np.ndarray: Visualized image array.
    """
    im_w = sequence.seq_info.im_width
    im_h = sequence.seq_info.im_height

    if image is None:
        # Check if matching image file exists in im_dir (e.g. img1/000001.jpg)
        img_candidates = [
            sequence.sequence_path / sequence.seq_info.im_dir / f"{frame_id:06d}{sequence.seq_info.im_ext}",
            sequence.sequence_path / sequence.seq_info.im_dir / f"{frame_id:06d}.jpg",
            sequence.sequence_path / sequence.seq_info.im_dir / f"{frame_id:06d}.png",
            sequence.sequence_path / "img1" / f"{frame_id:06d}.jpg",
        ]
        found_img_path = None
        for cand in img_candidates:
            if cand.exists():
                found_img_path = cand
                break

        if found_img_path is not None:
            image = cv2.imread(str(found_img_path))
            if image is None:
                logger.warning("Failed to decode image at %s, falling back to synthetic background.", found_img_path)
                image = np.full((im_h, im_w, 3), (34, 139, 34), dtype=np.uint8)  # Forest green pitch
        else:
            # Create synthetic football pitch green canvas
            image = np.full((im_h, im_w, 3), (34, 139, 34), dtype=np.uint8)

    # Filter annotations for frame_id
    frame_anns = [a for a in sequence.annotations if a.frame_id == frame_id]

    visualized = draw_annotations(
        image=image,
        annotations=frame_anns,
        game_info=sequence.game_info,
        taxonomy=taxonomy,
    )

    if output_path is not None:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), visualized)
        logger.info("Saved visualized frame %d to %s", frame_id, out)

    return visualized
