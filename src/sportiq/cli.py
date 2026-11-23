"""SportIQ Command-Line Interface.

Provides CLI commands:
  - sportiq inspect-video <video_path>
  - sportiq extract-frames <video_path>
  - sportiq check-env
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sportiq import __version__
from sportiq.video.frame_extractor import FrameExtractor
from sportiq.video.reader import InvalidVideoError, VideoReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sportiq")


def check_env_command(args: argparse.Namespace) -> int:
    """Verify runtime environment, Python version, CUDA, and PyTorch availability."""
    env_info: Dict[str, Any] = {
        "sportiq_version": __version__,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }

    try:
        import cv2

        env_info["opencv_version"] = cv2.__version__
    except ImportError as e:
        env_info["opencv_version"] = f"Not available: {e}"

    try:
        import torch

        env_info["torch_version"] = torch.__version__
        cuda_avail = torch.cuda.is_available()
        env_info["cuda_available"] = cuda_avail

        if cuda_avail:
            env_info["cuda_version"] = torch.version.cuda
            env_info["device_count"] = torch.cuda.device_count()
            env_info["device_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            env_info["compute_capability"] = f"{cap[0]}.{cap[1]}"

            # Test CUDA tensor allocation and multiplication
            a = torch.randn(10, 10, device="cuda")
            b = torch.randn(10, 10, device="cuda")
            c = a @ b
            env_info["cuda_tensor_test"] = "PASSED" if c.is_cuda else "FAILED"
        else:
            env_info["cuda_version"] = None
            env_info["device_name"] = "CPU only"
            env_info["cuda_tensor_test"] = "SKIPPED (No CUDA device)"
    except ImportError as e:
        env_info["torch_version"] = f"Not available: {e}"
        env_info["cuda_available"] = False

    if getattr(args, "json", False):
        print(json.dumps(env_info, indent=2))
    else:
        print("=" * 60)
        print(f"SportIQ Video Intelligence — Environment Check (v{__version__})")
        print("=" * 60)
        print(f"Python:             {env_info['python_version']} ({env_info['python_executable']})")
        print(f"Platform:           {env_info['platform']}")
        print(f"OpenCV:             {env_info.get('opencv_version', 'N/A')}")
        print(f"PyTorch:            {env_info.get('torch_version', 'N/A')}")
        print(f"CUDA Available:     {env_info.get('cuda_available')}")
        if env_info.get("cuda_available"):
            print(f"CUDA Version:       {env_info.get('cuda_version')}")
            print(f"GPU Device:         {env_info.get('device_name')}")
            print(f"Compute Capability: {env_info.get('compute_capability')}")
            print(f"CUDA Tensor MatMul: {env_info.get('cuda_tensor_test')}")
        print("=" * 60)

    return 0


def inspect_video_command(args: argparse.Namespace) -> int:
    """Inspect and report video codec, dimensions, FPS, and duration."""
    video_path = Path(args.video_path)

    try:
        reader = VideoReader(video_path)
        metadata = reader.get_metadata()

        if getattr(args, "json", False):
            print(metadata.to_json(indent=2))
        else:
            print("=" * 60)
            print(f"SportIQ Video Inspection — {metadata.filename}")
            print("=" * 60)
            print(f"Source Path:      {metadata.source_path}")
            print(f"File Size:        {metadata.file_size_mb} MB ({metadata.file_size_bytes:,} bytes)")
            print(f"Video Codec:      {metadata.codec}")
            print(f"Resolution:       {metadata.width}x{metadata.height} pixels")
            print(f"Frame Rate:       {metadata.fps} FPS")
            print(f"Total Frames:     {metadata.total_frames:,}")
            print(f"Duration:         {metadata.duration_seconds:.2f} seconds")
            print("=" * 60)
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except InvalidVideoError as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during video inspection: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1


def extract_frames_command(args: argparse.Namespace) -> int:
    """Extract frames from video with timestamp preservation and manifest."""
    video_path = Path(args.video_path)

    try:
        extractor = FrameExtractor(
            video_path=video_path,
            output_dir=args.output_dir,
            every_seconds=args.every_seconds,
            every_n_frames=args.every_n_frames,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )

        summary = extractor.extract()

        if getattr(args, "json", False):
            print(json.dumps(summary, indent=2))
        else:
            print("=" * 60)
            print(f"SportIQ Frame Extraction Complete — {summary['video_filename']}")
            print("=" * 60)
            print(f"Extracted Frames:       {summary['extracted_frames_count']}")
            print(f"Sampling Step (frames): {summary['sampling_step_frames']}")
            print(f"Output Directory:       {summary['output_directory']}")
            print(f"Frames Folder:          {summary['frames_directory']}")
            print(f"Manifest JSONL:         {summary['manifest_path']}")
            print(f"Video Metadata JSON:    {summary['video_metadata_path']}")
            print("=" * 60)
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileExistsError as e:
        print(f"Overwrite Error: {e}", file=sys.stderr)
        print("Hint: Pass --overwrite to permit writing to an existing directory.", file=sys.stderr)
        return 1
    except (InvalidVideoError, ValueError) as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during frame extraction: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line parser for sportiq."""
    parser = argparse.ArgumentParser(
        prog="sportiq",
        description="SportIQ — AI-Powered Sports Video Intelligence CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="available commands",
        metavar="<command>",
    )

    # Command: check-env
    env_parser = subparsers.add_parser(
        "check-env",
        help="Verify Python, PyTorch, CUDA, and OpenCV runtime environment",
    )
    env_parser.add_argument(
        "--json",
        action="store_true",
        help="Output environment details in JSON format",
    )
    env_parser.set_defaults(func=check_env_command)

    # Command: inspect-video
    inspect_parser = subparsers.add_parser(
        "inspect-video",
        help="Inspect video codec, dimensions, FPS, frame count, and duration",
    )
    inspect_parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file (e.g. data/raw/match.mp4)",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Output video metadata as JSON",
    )
    inspect_parser.set_defaults(func=inspect_video_command)

    # Command: extract-frames
    extract_parser = subparsers.add_parser(
        "extract-frames",
        help="Sample and extract video frames with JSONL metadata manifest",
    )
    extract_parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file",
    )
    extract_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store extracted frames and manifest (default: data/processed/<video_stem>)",
    )
    extract_parser.add_argument(
        "--every-seconds",
        type=float,
        default=None,
        help="Extract one frame every N seconds (e.g. 1.0 or 0.5)",
    )
    extract_parser.add_argument(
        "--every-n-frames",
        type=int,
        default=None,
        help="Extract every N-th frame (e.g. 30)",
    )
    extract_parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG compression quality (1-100, default: 95)",
    )
    extract_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing non-empty output directory",
    )
    extract_parser.add_argument(
        "--json",
        action="store_true",
        help="Output extraction summary in JSON format",
    )
    extract_parser.set_defaults(func=extract_frames_command)

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """CLI application entrypoint."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not hasattr(parsed_args, "func"):
        parser.print_help()
        return 1

    return parsed_args.func(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
