"""YOLO object detector implementation with lazy loading and structured output."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import cv2
import numpy as np
import torch

from sportiq.detection.config import DetectorConfig
from sportiq.detection.prediction import BoundingBox, Detection, FrameDetections

logger = logging.getLogger(__name__)


class DetectionError(Exception):
    """Base exception for object detection errors."""


class ModelLoadError(DetectionError):
    """Raised when model loading or initialization fails."""


class InferenceError(DetectionError):
    """Raised when model inference fails on an input."""


class YOLODetector:
    """Modular YOLO detector wrapper supporting lazy loading and structured evaluation."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        """Initialize detector with configuration.

        Args:
            config: Optional DetectorConfig. Uses defaults if None.
        """
        self.config = config or DetectorConfig()
        self._model: Optional[Any] = None
        self._class_names: Dict[int, str] = {}

    @property
    def is_loaded(self) -> bool:
        """Check whether model weights are currently loaded in memory."""
        return self._model is not None

    @property
    def class_names(self) -> Dict[int, str]:
        """Return the model's actual class name mapping."""
        if not self.is_loaded:
            self.load_model()
        return self._class_names.copy()

    def load_model(self) -> None:
        """Lazily load model weights into memory with device validation.

        Raises:
            ModelLoadError: If model file is not found, device is invalid, or loading fails.
        """
        if self._model is not None:
            return

        model_path = Path(self.config.model_path)
        # Check local weights first, fallback to string if Ultralytics asset
        resolved_path = str(model_path.resolve()) if model_path.exists() else self.config.model_path

        # Validate CUDA device availability if requested
        if "cuda" in self.config.device:
            if not torch.cuda.is_available():
                raise ModelLoadError(
                    f"CUDA device '{self.config.device}' requested, but torch.cuda.is_available() is False."
                )

        try:
            from ultralytics import YOLO

            logger.info("Loading YOLO model from %s on device %s", resolved_path, self.config.device)
            model = YOLO(resolved_path)

            # Store class mapping from model
            if hasattr(model, "names") and isinstance(model.names, dict):
                self._class_names = {int(k): str(v) for k, v in model.names.items()}
            else:
                self._class_names = {}

            self._model = model
            logger.info(
                "YOLO model loaded successfully. Known classes: %d categories",
                len(self._class_names),
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model from '{resolved_path}': {e}") from e

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        frame_index: Optional[int] = None,
        timestamp_ms: Optional[float] = None,
        source_name: Optional[str] = None,
    ) -> FrameDetections:
        """Run object detection inference on a single image.

        Args:
            image_input: File path (str/Path) or pre-loaded OpenCV image (np.ndarray).
            frame_index: Optional source video frame index.
            timestamp_ms: Optional video timestamp in milliseconds.
            source_name: Optional source label; defaults to file path or 'ndarray'.

        Returns:
            FrameDetections: Structured detection results with bounding boxes and metadata.

        Raises:
            FileNotFoundError: If image file does not exist.
            InferenceError: If model inference fails.
        """
        if not self.is_loaded:
            self.load_model()

        source_str = source_name or (str(image_input) if isinstance(image_input, (str, Path)) else "ndarray")

        # Read dimensions
        if isinstance(image_input, (str, Path)):
            img_p = Path(image_input).resolve()
            if not img_p.exists():
                raise FileNotFoundError(f"Input image not found: {img_p}")
            img_bgr = cv2.imread(str(img_p))
            if img_bgr is None:
                raise InferenceError(f"Failed to decode image at {img_p}")
            h, w = img_bgr.shape[:2]
            inference_target: Union[str, np.ndarray] = str(img_p)
        elif isinstance(image_input, np.ndarray):
            h, w = image_input.shape[:2]
            inference_target = image_input
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")

        # Configure half-precision: enabled only for CUDA
        use_half = bool(self.config.half and ("cuda" in self.config.device.lower()))

        t0 = time.perf_counter()
        try:
            results = self._model(
                inference_target,
                device=self.config.device,
                imgsz=self.config.imgsz,
                conf=self.config.conf,
                iou=self.config.iou,
                half=use_half,
                classes=self.config.classes,
                max_det=self.config.max_det,
                verbose=False,
            )
        except Exception as e:
            raise InferenceError(f"Inference failed on {source_str}: {e}") from e
        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000.0

        detections: List[Detection] = []
        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None:
                xyxy_cpu = boxes.xyxy.cpu().numpy()
                conf_cpu = boxes.conf.cpu().numpy()
                cls_cpu = boxes.cls.cpu().numpy()

                for i in range(len(xyxy_cpu)):
                    x1, y1, x2, y2 = xyxy_cpu[i]
                    confidence = float(conf_cpu[i])
                    cid = int(cls_cpu[i])
                    cname = self._class_names.get(cid, f"class_{cid}")

                    bbox = BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    )
                    detections.append(
                        Detection(
                            box=bbox,
                            confidence=confidence,
                            class_id=cid,
                            class_name=cname,
                        )
                    )

        return FrameDetections(
            source=source_str,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            image_width=w,
            image_height=h,
            detections=detections,
            inference_time_ms=inference_time_ms,
            model_name=Path(self.config.model_path).name,
            device=self.config.device,
        )

    def predict_frames(
        self,
        frames_dir: Union[str, Path],
        manifest_path: Optional[Union[str, Path]] = None,
        output_jsonl: Optional[Union[str, Path]] = None,
    ) -> List[FrameDetections]:
        """Run object detection on an extracted-frames directory.

        Optionally aligns predictions with timestamps and frame indices from frames.jsonl.

        Args:
            frames_dir: Directory containing frame images.
            manifest_path: Optional path to frames.jsonl. If omitted, checks frames_dir / "frames.jsonl"
                           or parent directory.
            output_jsonl: Optional destination path to write prediction records.

        Returns:
            List[FrameDetections]: Ordered predictions per frame.
        """
        fdir = Path(frames_dir).resolve()
        if not fdir.is_dir():
            raise NotADirectoryError(f"Frames directory not found: {fdir}")

        # Try to resolve frames.jsonl manifest
        manifest_records: Dict[str, Dict[str, Any]] = {}
        candidate_manifests = [
            Path(manifest_path) if manifest_path else None,
            fdir / "frames.jsonl",
            fdir.parent / "frames.jsonl",
        ]
        resolved_manifest: Optional[Path] = None
        for cand in candidate_manifests:
            if cand and cand.is_file():
                resolved_manifest = cand
                break

        if resolved_manifest:
            try:
                for line in resolved_manifest.read_text(encoding="utf-8").strip().splitlines():
                    if line.strip():
                        rec = json.loads(line.strip())
                        # Recognize Phase 1 FrameMetadata schema: extracted_frame_filename
                        fname = (
                            rec.get("extracted_frame_filename")
                            or rec.get("frame_filename")
                            or rec.get("extracted_frame")
                        )
                        if fname:
                            # Normalize manifest filename to basename to match image filename on disk
                            manifest_records[Path(str(fname)).name] = rec
            except Exception as e:
                logger.warning("Could not parse manifest at %s: %s", resolved_manifest, e)

        # Collect image files sorted by name
        image_files: List[Path] = []
        for ext in (".jpg", ".jpeg", ".png"):
            image_files.extend(fdir.glob(f"*{ext}"))
        image_files = sorted(image_files)

        if not image_files:
            logger.warning("No image files found in %s", fdir)
            return []

        if output_jsonl:
            out_p = Path(output_jsonl).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.exists():
                out_p.unlink()

        results: List[FrameDetections] = []
        for img_p in image_files:
            rec = manifest_records.get(img_p.name, {})

            # Explicit None check to preserve 0-based frame index (including 0)
            frame_idx = rec.get("original_frame_index")
            if frame_idx is None:
                frame_idx = rec.get("frame_index")
            if frame_idx is not None:
                try:
                    frame_idx = int(frame_idx)
                except (ValueError, TypeError):
                    frame_idx = None

            # Preserve timestamp_ms (including legitimate 0.0)
            ts_ms = rec.get("timestamp_ms")
            if ts_ms is not None:
                try:
                    ts_ms = float(ts_ms)
                except (ValueError, TypeError):
                    ts_ms = None

            # Fallback frame index from numeric filename if missing from manifest
            if frame_idx is None:
                stem = img_p.stem.replace("frame_", "")
                if stem.isdigit():
                    frame_idx = int(stem)

            frame_det = self.predict_image(
                image_input=img_p,
                frame_index=frame_idx,
                timestamp_ms=ts_ms,
                source_name=img_p.name,
            )
            results.append(frame_det)

            if output_jsonl:
                frame_det.append_to_jsonl(output_jsonl)

        return results
