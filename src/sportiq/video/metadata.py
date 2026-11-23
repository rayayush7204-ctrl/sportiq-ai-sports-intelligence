"""Data schemas and representations for video and frame metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata describing a video source."""

    source_path: str
    filename: str
    file_size_bytes: int
    file_size_mb: float
    codec: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert metadata to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class FrameMetadata:
    """Metadata describing an extracted video frame."""

    source_filename: str
    original_frame_index: int
    timestamp_ms: float
    extracted_frame_filename: str
    frame_width: int
    frame_height: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame metadata to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert frame metadata to a single-line JSON string (JSONL format)."""
        return json.dumps(self.to_dict())
