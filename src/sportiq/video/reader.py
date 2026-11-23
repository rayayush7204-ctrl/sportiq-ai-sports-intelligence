"""Video reader and metadata inspection using OpenCV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

from sportiq.video.metadata import VideoMetadata

logger = logging.getLogger(__name__)


class VideoReadError(Exception):
    """Base exception for video reading failures."""


class InvalidVideoError(VideoReadError):
    """Raised when a video file is corrupt, empty, or unreadable by OpenCV."""


class VideoReader:
    """Inspect and sequentially decode video streams with OpenCV.

    Ensures strict preservation of original frame indices and timestamps.
    """

    def __init__(self, video_path: Union[str, Path]) -> None:
        """Initialize VideoReader.

        Args:
            video_path: Path to the target video file.

        Raises:
            FileNotFoundError: If the specified video file does not exist.
            InvalidVideoError: If the file is not a valid video file or is empty.
        """
        self.path = Path(video_path).resolve()
        self._validate_file()
        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    def _validate_file(self) -> None:
        """Validate file existence and basic filesystem attributes."""
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        if not self.path.is_file():
            raise InvalidVideoError(f"Specified path is not a file: {self.path}")
        if self.path.stat().st_size == 0:
            raise InvalidVideoError(f"Video file is empty (0 bytes): {self.path}")

    def _open_capture(self) -> cv2.VideoCapture:
        """Open cv2.VideoCapture and verify the stream can be read."""
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise InvalidVideoError(
                f"Failed to open video file with OpenCV: {self.path}. "
                "The file may be corrupt or encoded in an unsupported codec."
            )
        return cap

    @staticmethod
    def _decode_fourcc(fourcc_int: float) -> str:
        """Convert OpenCV numeric fourcc into a 4-character string."""
        try:
            fourcc = int(fourcc_int)
            chars = [chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]
            # Check if all characters are printable ASCII
            if all(32 <= ord(c) <= 126 for c in chars):
                return "".join(chars)
            return f"0x{fourcc:08X}"
        except Exception:
            return "unknown"

    def _compute_safe_source_path(self) -> str:
        """Compute a safe relative POSIX path or filename without leaking absolute Windows host paths."""
        try:
            rel = self.path.relative_to(Path.cwd())
            return rel.as_posix()
        except (ValueError, RuntimeError):
            return self.path.name

    def get_metadata(self) -> VideoMetadata:
        """Inspect and return complete metadata for the video.

        Returns:
            VideoMetadata: Validated video metadata object.

        Raises:
            InvalidVideoError: If metadata cannot be extracted or dimensions are invalid.
        """
        if self._metadata is not None:
            return self._metadata

        cap = self._open_capture()
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc_raw = cap.get(cv2.CAP_PROP_FOURCC)
            codec = self._decode_fourcc(fourcc_raw)

            if width <= 0 or height <= 0 or fps <= 0 or total_frames <= 0:
                # Attempt to read the first frame to verify stream validity
                success, frame = cap.read()
                if not success or frame is None:
                    raise InvalidVideoError(
                        f"Video stream contains no readable frames: {self.path}"
                    )
                if width <= 0 or height <= 0:
                    height, width = frame.shape[:2]
                if fps <= 0:
                    fps = 25.0  # Fallback default if container lacks frame rate header
                if total_frames <= 0:
                    total_frames = 1

            file_size_bytes = self.path.stat().st_size
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
            duration_seconds = round(total_frames / fps, 3)

            safe_source_path = self._compute_safe_source_path()

            self._metadata = VideoMetadata(
                source_path=safe_source_path,
                filename=self.path.name,
                file_size_bytes=file_size_bytes,
                file_size_mb=file_size_mb,
                codec=codec,
                width=width,
                height=height,
                fps=round(fps, 3),
                total_frames=total_frames,
                duration_seconds=duration_seconds,
            )
            return self._metadata
        finally:
            cap.release()

    def read_frames(
        self,
        sample_every_n_frames: int = 1,
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Sequentially decode frames with original index and timestamp preservation.

        Args:
            sample_every_n_frames: Interval in frames between yielded frames (>= 1).

        Yields:
            Tuple[int, float, np.ndarray]:
                - original_frame_index: 0-based frame index in the source video
                - timestamp_ms: Timestamp in milliseconds from video start
                - frame_bgr: Decoded BGR frame as numpy uint8 array

        Raises:
            ValueError: If sample_every_n_frames < 1.
            InvalidVideoError: If video stream cannot be opened.
        """
        if sample_every_n_frames < 1:
            raise ValueError(
                f"sample_every_n_frames must be >= 1, got {sample_every_n_frames}"
            )

        metadata = self.get_metadata()
        fps = metadata.fps if metadata.fps > 0 else 25.0
        cap = self._open_capture()

        try:
            current_frame_idx = 0
            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    break

                if current_frame_idx % sample_every_n_frames == 0:
                    # Capture exact millisecond timestamp from video stream or derive from FPS
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if pos_msec is not None and pos_msec > 0:
                        timestamp_ms = round(float(pos_msec), 2)
                    else:
                        timestamp_ms = round((current_frame_idx / fps) * 1000.0, 2)

                    yield (current_frame_idx, timestamp_ms, frame)

                current_frame_idx += 1
        finally:
            cap.release()

    def __enter__(self) -> VideoReader:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._capture is not None and self._capture.isOpened():
            self._capture.release()
            self._capture = None
