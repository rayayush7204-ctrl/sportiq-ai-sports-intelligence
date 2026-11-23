"""Bounding box validation and coordinate sanitization for MOT annotations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sportiq.dataset.annotation_parser import MOTAnnotation

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Summary of annotation validation metrics."""

    total_inspected: int = 0
    valid_count: int = 0
    clipped_count: int = 0
    dropped_out_of_bounds: int = 0
    dropped_non_positive_dims: int = 0

    def to_dict(self) -> dict:
        return {
            "total_inspected": self.total_inspected,
            "valid_count": self.valid_count,
            "clipped_count": self.clipped_count,
            "dropped_out_of_bounds": self.dropped_out_of_bounds,
            "dropped_non_positive_dims": self.dropped_non_positive_dims,
            "pass_rate_pct": (
                round((self.valid_count / self.total_inspected) * 100, 2)
                if self.total_inspected > 0
                else 100.0
            ),
        }


class AnnotationValidator:
    """Validate and sanitize MOT bounding boxes against image dimensions."""

    def __init__(self, clip_boxes: bool = True) -> None:
        """Initialize validator.

        Args:
            clip_boxes: If True, partially out-of-bounds boxes are clamped to image boundaries.
        """
        self.clip_boxes = clip_boxes

    def validate_annotation(
        self,
        ann: MOTAnnotation,
        image_width: int,
        image_height: int,
    ) -> Tuple[bool, Optional[Tuple[float, float, float, float]], str]:
        """Validate an individual MOT annotation.

        Args:
            ann: Input MOTAnnotation.
            image_width: Canvas width in pixels.
            image_height: Canvas height in pixels.

        Returns:
            Tuple[bool, Optional[Tuple[float, float, float, float]], str]:
              - is_valid: True if annotation is usable.
              - sanitized_xywh: (bb_left, bb_top, bb_width, bb_height) or None if dropped.
              - status: "valid", "clipped", or reason for rejection.
        """
        # 1. Non-positive dimensions check
        if ann.bb_width <= 0 or ann.bb_height <= 0:
            return False, None, "non_positive_dimensions"

        # 2. Check if box is entirely outside image canvas
        x1 = ann.bb_left
        y1 = ann.bb_top
        x2 = x1 + ann.bb_width
        y2 = y1 + ann.bb_height

        if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
            return False, None, "out_of_bounds"

        # 3. Check for partial boundary overlap
        needs_clipping = (x1 < 0) or (y1 < 0) or (x2 > image_width) or (y2 > image_height)

        if needs_clipping:
            if not self.clip_boxes:
                return False, None, "partially_out_of_bounds"

            cx1 = max(0.0, min(float(image_width), x1))
            cy1 = max(0.0, min(float(image_height), y1))
            cx2 = max(0.0, min(float(image_width), x2))
            cy2 = max(0.0, min(float(image_height), y2))

            cw = cx2 - cx1
            ch = cy2 - cy1

            if cw <= 1.0 or ch <= 1.0:
                return False, None, "clipped_to_zero"

            return True, (cx1, cy1, cw, ch), "clipped"

        return True, (x1, y1, ann.bb_width, ann.bb_height), "valid"

    def filter_and_validate(
        self,
        annotations: List[MOTAnnotation],
        image_width: int,
        image_height: int,
    ) -> Tuple[List[Tuple[MOTAnnotation, Tuple[float, float, float, float]]], ValidationResult]:
        """Validate a collection of annotations and return sanitized bounding boxes.

        Returns:
            Tuple: (List of (original_ann, sanitized_xywh), ValidationResult)
        """
        result = ValidationResult(total_inspected=len(annotations))
        valid_items: List[Tuple[MOTAnnotation, Tuple[float, float, float, float]]] = []

        for ann in annotations:
            is_valid, sanitized_xywh, status = self.validate_annotation(
                ann, image_width=image_width, image_height=image_height
            )

            if is_valid and sanitized_xywh is not None:
                valid_items.append((ann, sanitized_xywh))
                result.valid_count += 1
                if status == "clipped":
                    result.clipped_count += 1
            else:
                if status == "non_positive_dimensions":
                    result.dropped_non_positive_dims += 1
                elif "out_of_bounds" in status or status == "clipped_to_zero":
                    result.dropped_out_of_bounds += 1

        return valid_items, result
