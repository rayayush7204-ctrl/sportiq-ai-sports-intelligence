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
from sportiq.dataset.annotation_parser import load_sequence
from sportiq.dataset.annotation_validator import AnnotationValidator
from sportiq.dataset.class_mapping import ClassTaxonomyType
from sportiq.dataset.dataset_materializer import (
    AlignmentVerificationError,
    DatasetMaterializer,
)
from sportiq.dataset.dataset_splitter import DatasetSplitter
from sportiq.dataset.statistics import DatasetStatisticsReporter
from sportiq.dataset.visualizer import visualize_sequence_frame
from sportiq.dataset.yolo_converter import YOLOConverter
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


def convert_dataset_command(args: argparse.Namespace) -> int:
    """Convert SoccerNet MOT sequence annotations to YOLO format with metadata sidecar."""
    sequence_path = Path(args.sequence_dir)

    try:
        sequence = load_sequence(sequence_path)
        taxonomy = ClassTaxonomyType(args.taxonomy)
        validator = AnnotationValidator(clip_boxes=not args.no_clip)
        converter = YOLOConverter(taxonomy=taxonomy, validator=validator)

        output_dir = Path(args.output_dir) if args.output_dir else sequence_path / "yolo"
        summary = converter.convert_sequence(sequence=sequence, output_dir=output_dir)

        if getattr(args, "json", False):
            print(json.dumps(summary, indent=2))
        else:
            print("=" * 60)
            print(f"SportIQ YOLO Conversion Complete — {summary['sequence_name']}")
            print("=" * 60)
            print(f"Taxonomy:               {summary['taxonomy']}")
            print(f"Labels Directory:       {summary['labels_directory']}")
            print(f"Manifest JSONL:         {summary['manifest_path']}")
            print(f"Total Annotations:      {summary['total_annotations']}")
            print(f"Exported YOLO Boxes:    {summary['exported_yolo_boxes']}")
            print(f"Skipped Unmapped:       {summary['skipped_unmapped_boxes']}")
            val = summary["validation"]
            print(f"Validation Pass Rate:   {val['pass_rate_pct']}% (Clipped: {val['clipped_count']}, Dropped: {val['dropped_out_of_bounds'] + val['dropped_non_positive_dims']})")
            print("=" * 60)
        return 0

    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during YOLO conversion: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1


def split_dataset_command(args: argparse.Namespace) -> int:
    """Partition sequences deterministically across train, val, and test splits."""
    root_dir = Path(args.sequences_dir).resolve()

    try:
        # Collect sequences
        seq_names: List[str] = []
        match_grouping: Dict[str, str] = {}
        missing_match_metadata: List[str] = []

        if (root_dir / "seqinfo.ini").exists():
            seq_names = [root_dir.name]
            gameinfo_path = root_dir / "gameinfo.ini"
            if gameinfo_path.exists():
                try:
                    seq_obj = load_sequence(root_dir)
                    if seq_obj.game_info and seq_obj.game_info.game_id:
                        match_grouping[root_dir.name] = seq_obj.game_info.game_id
                    else:
                        missing_match_metadata.append(root_dir.name)
                except Exception as e:
                    logger.warning("Failed to parse gameinfo.ini for %s: %s", root_dir.name, e)
                    missing_match_metadata.append(root_dir.name)
            else:
                missing_match_metadata.append(root_dir.name)
        elif root_dir.is_dir():
            for child in sorted(root_dir.iterdir()):
                if child.is_dir() and (child / "seqinfo.ini").exists():
                    seq_names.append(child.name)
                    gameinfo_path = child / "gameinfo.ini"
                    if gameinfo_path.exists():
                        try:
                            seq_obj = load_sequence(child)
                            if seq_obj.game_info and seq_obj.game_info.game_id:
                                match_grouping[child.name] = seq_obj.game_info.game_id
                            else:
                                missing_match_metadata.append(child.name)
                        except Exception as e:
                            logger.warning("Failed to parse gameinfo.ini for %s: %s", child.name, e)
                            missing_match_metadata.append(child.name)
                    else:
                        missing_match_metadata.append(child.name)

        if not seq_names:
            print(f"Error: No valid sequences found in {root_dir}", file=sys.stderr)
            return 1

        allow_missing = getattr(args, "allow_missing_game_id", False)
        # If some sequences have match metadata but others are missing it:
        if match_grouping and missing_match_metadata:
            msg = (
                f"Incomplete match metadata: {len(missing_match_metadata)} sequence(s) missing game_id: {missing_match_metadata}."
            )
            if not allow_missing:
                print(
                    f"Error: {msg} Pass --allow-missing-game-id to explicitly downgrade to sequence-level splitting.",
                    file=sys.stderr,
                )
                return 1
            else:
                print(
                    f"Warning: {msg} Explicitly downgrading to sequence-level splitting.",
                    file=sys.stderr,
                )
                match_grouping = {}

        splitter = DatasetSplitter(random_seed=args.seed)
        split_result = splitter.split_sequences(
            sequence_names=seq_names,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            match_grouping=match_grouping if match_grouping else None,
            strict_match_grouping=not allow_missing,
        )

        yaml_content = None
        if args.output_yaml:
            taxonomy = ClassTaxonomyType(args.taxonomy)
            yaml_content = DatasetSplitter.generate_yolo_yaml(
                dataset_root=root_dir,
                taxonomy=taxonomy,
                output_path=args.output_yaml,
            )

        output_dict = split_result.to_dict()
        if args.output_yaml:
            output_dict["dataset_yaml"] = str(Path(args.output_yaml).resolve())

        if getattr(args, "json", False):
            print(json.dumps(output_dict, indent=2))
        else:
            print("=" * 60)
            print("SportIQ Deterministic Dataset Partition")
            print("=" * 60)
            print(f"Total Sequences:    {output_dict['total_count']}")
            print(f"Train Sequences:    {output_dict['train_count']} -> {', '.join(split_result.train_sequences) or 'None'}")
            print(f"Val Sequences:      {output_dict['val_count']} -> {', '.join(split_result.val_sequences) or 'None'}")
            print(f"Test Sequences:     {output_dict['test_count']} -> {', '.join(split_result.test_sequences) or 'None'}")
            if args.output_yaml:
                print(f"Saved dataset.yaml: {Path(args.output_yaml).resolve()}")
            print("=" * 60)
        return 0

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during dataset splitting: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1


def materialize_dataset_command(args: argparse.Namespace) -> int:
    """Materialize train/val/test splits preserving 1:1 image-to-label correspondence."""
    sequences_dir = Path(args.sequences_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    try:
        seq_dict = {}
        if (sequences_dir / "seqinfo.ini").exists():
            seq_dict[sequences_dir.name] = load_sequence(sequences_dir)
        elif sequences_dir.is_dir():
            for child in sorted(sequences_dir.iterdir()):
                if child.is_dir() and (child / "seqinfo.ini").exists():
                    try:
                        seq_dict[child.name] = load_sequence(child)
                    except Exception as e:
                        logger.warning("Could not load sequence %s: %s", child.name, e)

        if not seq_dict:
            print(f"Error: No valid sequences found in {sequences_dir}", file=sys.stderr)
            return 1

        seq_names = list(seq_dict.keys())
        match_grouping: Dict[str, str] = {}
        missing_match_metadata: List[str] = []

        for seq_name, seq_obj in seq_dict.items():
            if seq_obj.game_info and seq_obj.game_info.game_id:
                match_grouping[seq_name] = seq_obj.game_info.game_id
            else:
                missing_match_metadata.append(seq_name)

        allow_missing = getattr(args, "allow_missing_game_id", False)

        if match_grouping and missing_match_metadata:
            msg = (
                f"Incomplete match metadata: {len(missing_match_metadata)} sequence(s) missing game_id: {missing_match_metadata}."
            )
            if not allow_missing:
                print(
                    f"Error: {msg} Pass --allow-missing-game-id to explicitly permit downgrade to sequence-level splitting.",
                    file=sys.stderr,
                )
                return 1
            else:
                print(
                    f"Warning: {msg} Explicitly downgrading to sequence-level splitting.",
                    file=sys.stderr,
                )
                match_grouping = {}

        splitter = DatasetSplitter(random_seed=args.seed)
        split = splitter.split_sequences(
            sequence_names=seq_names,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            match_grouping=match_grouping if match_grouping else None,
            strict_match_grouping=not allow_missing,
        )

        taxonomy = ClassTaxonomyType(args.taxonomy)
        summary = DatasetMaterializer.materialize(
            split=split,
            sequences=seq_dict,
            labels_base_dir=labels_dir,
            output_dir=output_dir,
            mode=args.mode,
            taxonomy=taxonomy,
        )

        if getattr(args, "json", False):
            print(json.dumps(summary, indent=2))
        else:
            print("=" * 60)
            print("SportIQ YOLO Dataset Materialization Complete")
            print("=" * 60)
            print(f"Output Directory:       {summary['output_directory']}")
            print(f"Dataset YAML:           {summary['dataset_yaml']}")
            print(f"Materialization Mode:   {summary['mode']}")
            print(f"Total Paired Images:    {summary['total_images']}")
            print(f"Labeled Frames:         {summary['total_labeled_frames']}")
            print(f"Background Frames:      {summary['total_background_frames']}")
            print(f"Split Counts:           {summary['split_counts']}")
            print("=" * 60)
        return 0

    except (AlignmentVerificationError, ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during dataset materialization: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1


def verify_dataset_command(args: argparse.Namespace) -> int:
    """Audit and verify YOLO dataset image-to-label alignment from dataset.yaml."""
    yaml_path = Path(args.dataset_yaml).resolve()

    try:
        report = DatasetMaterializer.verify_dataset_alignment(yaml_path)
        report_dict = report.to_dict()

        if getattr(args, "json", False):
            print(json.dumps(report_dict, indent=2))
        else:
            print("=" * 60)
            print("SportIQ YOLO Dataset Alignment Audit")
            print("=" * 60)
            print(f"Dataset YAML:           {report.dataset_yaml}")
            print(f"Alignment Status:       {'ALIGNED' if report.is_aligned else 'BROKEN'}")
            print(f"Total Images:           {report.total_images}")
            print(f"Total Labels:           {report.total_labels}")
            print(f"Labeled Frames:         {report.labeled_frames}")
            print(f"Background Frames:      {report.background_frames}")
            print(f"Split Counts:           {report.split_counts}")
            if report.missing_images:
                print(f"Missing Images ({len(report.missing_images)}):")
                for m in report.missing_images[:5]:
                    print(f"  - {m}")
            if report.corrupted_labels:
                print(f"Corrupted Labels ({len(report.corrupted_labels)}):")
                for c in report.corrupted_labels[:5]:
                    print(f"  - {c}")
            print("=" * 60)
        return 0 if report.is_aligned else 1

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during dataset verification: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1


def dataset_stats_command(args: argparse.Namespace) -> int:
    """Analyze sequence ground-truth distributions, class counts, and track lengths."""
    root_dir = Path(args.sequences_dir).resolve()

    try:
        sequences: List[Any] = []
        if (root_dir / "seqinfo.ini").exists():
            sequences.append(load_sequence(root_dir))
        elif root_dir.is_dir():
            for child in sorted(root_dir.iterdir()):
                if child.is_dir() and (child / "seqinfo.ini").exists():
                    try:
                        sequences.append(load_sequence(child))
                    except Exception as e:
                        logger.warning("Could not load sequence %s: %e", child.name, e)

        if not sequences:
            print(f"Error: No valid sequences found in {root_dir}", file=sys.stderr)
            return 1

        taxonomy = ClassTaxonomyType(args.taxonomy)
        reporter = DatasetStatisticsReporter(taxonomy=taxonomy)
        stats = reporter.analyze(sequences)
        stats_dict = stats.to_dict()

        if getattr(args, "json", False):
            print(json.dumps(stats_dict, indent=2))
        else:
            print("=" * 60)
            print("SportIQ Ground-Truth Dataset Statistics")
            print("=" * 60)
            print(f"Sequences Analyzed:     {stats.total_sequences}")
            print(f"Total Frames:           {stats.total_frames:,}")
            print(f"Total Annotations:      {stats.total_annotations:,}")
            print(f"Unique Tracks:          {stats.unique_tracks:,}")
            print(f"Canonical Classes:      {stats.canonical_class_distribution}")
            print(f"Raw Classes:            {stats.raw_class_distribution}")
            print(f"Staff Annotations:      {stats.staff_annotation_count} ({stats.staff_percentage}%)")
            print(f"Track Length Mean:      {stats.track_length_metrics.get('mean_frames', 0)} frames (Median: {stats.track_length_metrics.get('median_frames', 0)})")
            print(f"BBox Mean Size:         {stats.bbox_dimension_metrics.get('mean_width_px', 0)}x{stats.bbox_dimension_metrics.get('mean_height_px', 0)} px")
            print("=" * 60)
        return 0

    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during dataset stats: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1


def visualize_annotations_command(args: argparse.Namespace) -> int:
    """Render ground-truth bounding boxes, track IDs, and classes onto frame image."""
    sequence_path = Path(args.sequence_dir)

    try:
        sequence = load_sequence(sequence_path)
        taxonomy = ClassTaxonomyType(args.taxonomy)
        output_path = Path(args.output).resolve() if args.output else sequence_path / f"vis_frame_{args.frame_id:06d}.jpg"

        visualize_sequence_frame(
            sequence=sequence,
            frame_id=args.frame_id,
            output_path=output_path,
            taxonomy=taxonomy,
        )

        if getattr(args, "json", False):
            print(json.dumps({
                "sequence_name": sequence.name,
                "frame_id": args.frame_id,
                "output_image": str(output_path),
                "status": "success",
            }, indent=2))
        else:
            print("=" * 60)
            print(f"SportIQ Frame Annotation Visualized — {sequence.name}")
            print("=" * 60)
            print(f"Frame ID:           {args.frame_id}")
            print(f"Output Image:       {output_path}")
            print("=" * 60)
        return 0

    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during visualization: %s", e)
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

    # Command: convert-dataset
    convert_parser = subparsers.add_parser(
        "convert-dataset",
        help="Convert SoccerNet MOT sequence annotations to normalized YOLO format",
    )
    convert_parser.add_argument(
        "sequence_dir",
        type=str,
        help="Path to SoccerNet sequence directory (containing seqinfo.ini and gt/gt.txt)",
    )
    convert_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for labels and metadata (default: <sequence_dir>/yolo)",
    )
    convert_parser.add_argument(
        "--taxonomy",
        type=str,
        choices=["four_class", "five_class", "two_class"],
        default="four_class",
        help="Detection taxonomy to apply (default: four_class)",
    )
    convert_parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable clamping partially out-of-bounds bounding boxes to image canvas",
    )
    convert_parser.add_argument(
        "--json",
        action="store_true",
        help="Output conversion summary in JSON format",
    )
    convert_parser.set_defaults(func=convert_dataset_command)

    # Command: split-dataset
    split_parser = subparsers.add_parser(
        "split-dataset",
        help="Deterministically split sequences/matches into train, val, and test sets",
    )
    split_parser.add_argument(
        "sequences_dir",
        type=str,
        help="Directory containing one or more SoccerNet sequence folders",
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction for training split (default: 0.70)",
    )
    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction for validation split (default: 0.15)",
    )
    split_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction for testing split (default: 0.15)",
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    split_parser.add_argument(
        "--allow-missing-game-id",
        action="store_true",
        help="Explicitly permit downgrade to sequence-level splitting if game_id is missing",
    )
    split_parser.add_argument(
        "--output-yaml",
        type=str,
        default=None,
        help="Optional path to export YOLO-format dataset.yaml",
    )
    split_parser.add_argument(
        "--taxonomy",
        type=str,
        choices=["four_class", "five_class", "two_class"],
        default="four_class",
        help="Detection taxonomy for dataset.yaml class names (default: four_class)",
    )
    split_parser.add_argument(
        "--json",
        action="store_true",
        help="Output split results in JSON format",
    )
    split_parser.set_defaults(func=split_dataset_command)

    # Command: materialize-dataset
    materialize_parser = subparsers.add_parser(
        "materialize-dataset",
        help="Materialize YOLO train/val/test splits preserving 1:1 image-to-label alignment",
    )
    materialize_parser.add_argument(
        "sequences_dir",
        type=str,
        help="Directory containing one or more sequence folders",
    )
    materialize_parser.add_argument(
        "--labels-dir",
        type=str,
        required=True,
        help="Base directory containing exported YOLO labels per sequence",
    )
    materialize_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Destination directory for materialized dataset and dataset.yaml",
    )
    materialize_parser.add_argument(
        "--mode",
        type=str,
        choices=["manifest", "symlink", "hardlink", "copy"],
        default="manifest",
        help="Materialization mode (default: manifest, zero disk duplication)",
    )
    materialize_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction for training split (default: 0.70)",
    )
    materialize_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction for validation split (default: 0.15)",
    )
    materialize_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction for testing split (default: 0.15)",
    )
    materialize_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    materialize_parser.add_argument(
        "--allow-missing-game-id",
        action="store_true",
        help="Permit downgrade to sequence-level partition if match metadata is incomplete",
    )
    materialize_parser.add_argument(
        "--taxonomy",
        type=str,
        choices=["four_class", "five_class", "two_class"],
        default="four_class",
        help="Taxonomy for dataset.yaml class names (default: four_class)",
    )
    materialize_parser.add_argument(
        "--json",
        action="store_true",
        help="Output summary in JSON format",
    )
    materialize_parser.set_defaults(func=materialize_dataset_command)

    # Command: verify-dataset
    verify_parser = subparsers.add_parser(
        "verify-dataset",
        help="Verify 1:1 image-label correspondence and bounds checking from dataset.yaml",
    )
    verify_parser.add_argument(
        "dataset_yaml",
        type=str,
        help="Path to YOLO dataset.yaml file to audit",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Output audit report in JSON format",
    )
    verify_parser.set_defaults(func=verify_dataset_command)

    # Command: dataset-stats
    stats_parser = subparsers.add_parser(
        "dataset-stats",
        help="Compute ground-truth distributions, class balance, and track length analytics",
    )
    stats_parser.add_argument(
        "sequences_dir",
        type=str,
        help="Directory containing one or more SoccerNet sequence folders",
    )
    stats_parser.add_argument(
        "--taxonomy",
        type=str,
        choices=["four_class", "five_class", "two_class"],
        default="four_class",
        help="Detection taxonomy to evaluate (default: four_class)",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Output dataset statistics in JSON format",
    )
    stats_parser.set_defaults(func=dataset_stats_command)

    # Command: visualize-annotations
    vis_parser = subparsers.add_parser(
        "visualize-annotations",
        help="Draw ground-truth bounding boxes, track IDs, and classes on a sequence frame",
    )
    vis_parser.add_argument(
        "sequence_dir",
        type=str,
        help="Path to SoccerNet sequence directory",
    )
    vis_parser.add_argument(
        "--frame-id",
        type=int,
        default=1,
        help="Frame ID to visualize (1-based, default: 1)",
    )
    vis_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: <sequence_dir>/vis_frame_<id>.jpg)",
    )
    vis_parser.add_argument(
        "--taxonomy",
        type=str,
        choices=["four_class", "five_class", "two_class"],
        default="four_class",
        help="Detection taxonomy (default: four_class)",
    )
    vis_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result in JSON format",
    )
    vis_parser.set_defaults(func=visualize_annotations_command)

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
