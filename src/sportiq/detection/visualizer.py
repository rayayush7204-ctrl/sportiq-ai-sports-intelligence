"""Visualization utilities for rendering object detection bounding boxes onto images."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np

from sportiq.detection.prediction import FrameDetections


class DetectionVisualizer:
    """Renders detected bounding boxes, class labels, and confidence onto OpenCV frames."""

    # Default category color palette (BGR)
    COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
        "person": (0, 200, 0),        # Vibrant Green
        "sports ball": (0, 215, 255),   # Golden Yellow / Orange
        "player": (0, 200, 0),
        "goalkeeper": (0, 140, 255),
        "referee": (0, 255, 255),
        "ball": (0, 215, 255),
    }
    DEFAULT_COLOR = (255, 100, 0)  # Bright Blue fallback

    @classmethod
    def get_color(cls, class_name: str) -> Tuple[int, int, int]:
        """Lookup color by class name."""
        return cls.COLOR_PALETTE.get(class_name.lower(), cls.DEFAULT_COLOR)

    @classmethod
    def draw_detections(
        cls,
        image: np.ndarray,
        frame_detections: FrameDetections,
        box_thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """Draw bounding boxes and confidence labels on an image.

        Args:
            image: OpenCV BGR image (uint8).
            frame_detections: Detected bounding boxes and classes.
            box_thickness: Bounding box outline width in pixels.
            font_scale: Font scale for label text.

        Returns:
            np.ndarray: Annotated BGR image.
        """
        canvas = image.copy()
        for det in frame_detections.detections:
            box = det.box
            color = cls.get_color(det.class_name)

            pt1 = (int(round(box.x1)), int(round(box.y1)))
            pt2 = (int(round(box.x2)), int(round(box.y2)))

            # Draw bounding box
            cv2.rectangle(canvas, pt1, pt2, color, thickness=box_thickness)

            # Format label
            label = f"{det.class_name} {det.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            # Draw filled background for label text
            bg_pt1 = (pt1[0], max(0, pt1[1] - text_h - 6))
            bg_pt2 = (pt1[0] + text_w + 4, max(text_h + 6, pt1[1]))
            cv2.rectangle(canvas, bg_pt1, bg_pt2, color, -1)

            # Draw text in white or dark depending on color
            text_color = (0, 0, 0) if (color[0] + color[1] + color[2] > 400) else (255, 255, 255)
            cv2.putText(
                canvas,
                label,
                (pt1[0] + 2, bg_pt2[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # Draw summary overlay in top-left corner
        summary_text = (
            f"{frame_detections.model_name} | Detections: {frame_detections.total_detections} "
            f"| Time: {frame_detections.inference_time_ms:.1f}ms ({frame_detections.device})"
        )
        cv2.putText(
            canvas,
            summary_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            summary_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        return canvas

    @classmethod
    def visualize_and_save(
        cls,
        image_input: Union[str, Path, np.ndarray],
        frame_detections: FrameDetections,
        output_path: Union[str, Path],
    ) -> Path:
        """Draw detections and write image to disk.

        Args:
            image_input: Image path or loaded BGR numpy array.
            frame_detections: Detection predictions.
            output_path: Destination output file path.

        Returns:
            Path: Written image path.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"Failed to read image at {image_input}")
        else:
            img = image_input

        annotated = cls.draw_detections(img, frame_detections)

        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), annotated)
        return out_p
