"""Frame extraction pipeline with timestamp preservation and manifest generation."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2

from sportiq.video.metadata import FrameMetadata, VideoMetadata
from sportiq.video.reader import VideoReader

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video at configurable intervals and generate manifests."""

    def __init__(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        every_seconds: Optional[float] = None,
        every_n_frames: Optional[int] = None,
        jpeg_quality: int = 95,
        overwrite: bool = False,
    ) -> None:
        """Initialize FrameExtractor.

        Args:
            video_path: Path to the target video file.
            output_dir: Destination directory. Defaults to data/processed/<video_stem>.
            every_seconds: Sample interval in seconds (e.g. 1.0 = 1 frame per second).
            every_n_frames: Sample interval in frame count (e.g. 30 = every 30th frame).
            jpeg_quality: JPEG compression quality from 1 to 100. Default 95.
            overwrite: If True, allows writing into an existing non-empty output directory.

        Raises:
            ValueError: If invalid parameters or mutually conflicting intervals are specified.
        """
        self.reader = VideoReader(video_path)
        self.video_path = self.reader.path

        if output_dir is None:
            self.output_dir = Path("data/processed") / self.video_path.stem
        else:
            self.output_dir = Path(output_dir).resolve()

        if every_seconds is not None and every_n_frames is not None:
            raise ValueError(
                "Specify either 'every_seconds' or 'every_n_frames', not both."
            )

        if every_seconds is not None:
            if every_seconds <= 0:
                raise ValueError(
                    f"every_seconds must be positive (> 0), got {every_seconds}"
                )
            self.every_seconds = float(every_seconds)
            self.every_n_frames = None
        elif every_n_frames is not None:
            if every_n_frames < 1:
                raise ValueError(
                    f"every_n_frames must be >= 1, got {every_n_frames}"
                )
            self.every_n_frames = int(every_n_frames)
            self.every_seconds = None
        else:
            # Default: sample every 1 second
            self.every_seconds = 1.0
            self.every_n_frames = None

        if not 1 <= jpeg_quality <= 100:
            raise ValueError(
                f"jpeg_quality must be between 1 and 100, got {jpeg_quality}"
            )
        self.jpeg_quality = int(jpeg_quality)
        self.overwrite = bool(overwrite)

    def _prepare_output_dir(self) -> Path:
        """Create and validate the output directory with overwrite protection."""
        frames_dir = self.output_dir / "frames"

        if self.output_dir.exists():
            existing_items = list(self.output_dir.iterdir())
            if existing_items:
                if not self.overwrite:
                    raise FileExistsError(
                        f"Output directory exists and is not empty: {self.output_dir}. "
                        "Use --overwrite to allow writing into this directory."
                    )
                logger.warning(
                    "Overwrite enabled: cleaning existing output directory: %s",
                    self.output_dir,
                )
                # Clean up existing frames directory if present
                if frames_dir.exists():
                    shutil.rmtree(frames_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        return frames_dir

    def calculate_sampling_interval_frames(self, fps: float) -> int:
        """Calculate the frame step interval from configured parameters."""
        if self.every_n_frames is not None:
            return self.every_n_frames
        assert self.every_seconds is not None
        interval = max(1, int(round(self.every_seconds * fps)))
        return interval

    def extract(self) -> Dict[str, Any]:
        """Execute the frame extraction pipeline.

        Returns:
            Dict[str, Any]: Execution summary containing extraction metrics and file paths.

        Raises:
            FileExistsError: If output directory is not empty and overwrite is False.
            InvalidVideoError: If video cannot be read.
        """
        metadata: VideoMetadata = self.reader.get_metadata()
        frames_dir = self._prepare_output_dir()
        step_frames = self.calculate_sampling_interval_frames(metadata.fps)

        manifest_path = self.output_dir / "frames.jsonl"
        metadata_path = self.output_dir / "video_metadata.json"

        # Write video_metadata.json
        metadata_path.write_text(metadata.to_json(indent=2), encoding="utf-8")

        extracted_manifest: List[FrameMetadata] = []
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            for original_idx, timestamp_ms, frame in self.reader.read_frames(
                sample_every_n_frames=step_frames
            ):
                frame_filename = f"frame_{original_idx:06d}.jpg"
                frame_rel_path = f"frames/{frame_filename}"
                frame_full_path = frames_dir / frame_filename

                # Save image
                write_ok = cv2.imwrite(
                    str(frame_full_path), frame, encode_params
                )
                if not write_ok:
                    raise IOError(
                        f"Failed to write image file: {frame_full_path}"
                    )

                height, width = frame.shape[:2]
                frame_meta = FrameMetadata(
                    source_filename=metadata.filename,
                    original_frame_index=original_idx,
                    timestamp_ms=round(timestamp_ms, 2),
                    extracted_frame_filename=frame_rel_path,
                    frame_width=width,
                    frame_height=height,
                )

                manifest_file.write(frame_meta.to_json() + "\n")
                extracted_manifest.append(frame_meta)

        summary = {
            "source_video": str(self.video_path),
            "video_filename": metadata.filename,
            "total_video_frames": metadata.total_frames,
            "fps": metadata.fps,
            "duration_seconds": metadata.duration_seconds,
            "sampling_step_frames": step_frames,
            "extracted_frames_count": len(extracted_manifest),
            "output_directory": str(self.output_dir),
            "frames_directory": str(frames_dir),
            "manifest_path": str(manifest_path),
            "video_metadata_path": str(metadata_path),
            "jpeg_quality": self.jpeg_quality,
        }

        logger.info(
            "Extracted %d frames from %s to %s",
            len(extracted_manifest),
            metadata.filename,
            self.output_dir,
        )
        return summary


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    every_seconds: Optional[float] = None,
    every_n_frames: Optional[int] = None,
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Convenience function to extract frames with configurable interval.

    Args:
        video_path: Target video file path.
        output_dir: Destination directory.
        every_seconds: Interval in seconds.
        every_n_frames: Interval in frame numbers.
        jpeg_quality: Quality of saved JPEG frames (1-100).
        overwrite: Overwrite existing output directory.

    Returns:
        Dict[str, Any]: Extraction summary.
    """
    extractor = FrameExtractor(
        video_path=video_path,
        output_dir=output_dir,
        every_seconds=every_seconds,
        every_n_frames=every_n_frames,
        jpeg_quality=jpeg_quality,
        overwrite=overwrite,
    )
    return extractor.extract()
