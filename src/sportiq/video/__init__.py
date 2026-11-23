"""Video ingestion, metadata extraction, and frame sampling package."""

from sportiq.video.metadata import FrameMetadata, VideoMetadata
from sportiq.video.reader import InvalidVideoError, VideoReadError, VideoReader
from sportiq.video.frame_extractor import FrameExtractor, extract_frames

__all__ = [
    "VideoMetadata",
    "FrameMetadata",
    "VideoReader",
    "InvalidVideoError",
    "VideoReadError",
    "FrameExtractor",
    "extract_frames",
]
