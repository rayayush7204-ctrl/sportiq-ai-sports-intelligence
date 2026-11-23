"""SportIQ — AI-Powered Sports Video Intelligence Platform."""

__version__ = "0.1.0"

from sportiq.video.metadata import FrameMetadata, VideoMetadata
from sportiq.video.reader import InvalidVideoError, VideoReadError, VideoReader
from sportiq.video.frame_extractor import FrameExtractor, extract_frames

__all__ = [
    "__version__",
    "VideoMetadata",
    "FrameMetadata",
    "VideoReader",
    "InvalidVideoError",
    "VideoReadError",
    "FrameExtractor",
    "extract_frames",
]
