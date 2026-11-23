# Architecture & Engineering Specifications

## System Architecture

```text
Input Video File (.mp4, .mov, .avi)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Ingestion Layer (sportiq.video)                        │
│ ────────────────────────────────────────────────────── │
│ - VideoReader: OpenCV sequential decoder               │
│ - Metadata extraction: codec, resolution, fps, frames  │
│ - FrameExtractor: configurable temporal/frame sampling │
│ - Manifest generator: frames.jsonl, video_metadata.json│
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Detection Layer (sportiq.detection) — [Phase 2]        │
│ ────────────────────────────────────────────────────── │
│ - Player detector: YOLO person category baseline       │
│ - Ball detector: specialized football detector         │
│ - Output: BoundingBox2D(x1, y1, x2, y2, conf, class_id)│
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Tracking Layer (sportiq.tracking) — [Phase 3]          │
│ ────────────────────────────────────────────────────── │
│ - ByteTrack / BoT-SORT multi-object trackers           │
│ - Trajectory buffer & track ID assignment              │
│ - ID switch detection & track continuity metrics       │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Analytics Layer (sportiq.analytics) — [Phase 4]        │
│ ────────────────────────────────────────────────────── │
│ - Team classifier: Jersey colour clustering (k-means)  │
│ - Ball possession & player-ball proximity model        │
│ - Player speed, distance, and 2D spatial heatmaps      │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Visualization & Pitch Mapping (sportiq.visualization)  │
│ ────────────────────────────────────────────────────── │
│ - Video overlay annotator (boxes, IDs, trajectories)   │
│ - 2D pitch homography transformation                   │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Serving Layer (sportiq.api) — [Phase 5]                │
│ ────────────────────────────────────────────────────── │
│ - FastAPI endpoints for asynchronous job execution     │
│ - React dashboard for interactive match analytics      │
└────────────────────────────────────────────────────────┘
```

## Implemented Modules (Phase 1)

### 1. `sportiq.video.metadata`
- **`VideoMetadata`**: Immutable data container capturing source path, filename, byte size, size in MB, fourcc codec string, width, height, FPS, total frames, and duration in seconds. Provides `.to_dict()` and `.to_json()` helpers.
- **`FrameMetadata`**: Immutable data container capturing source filename, 0-based original frame index, timestamp in milliseconds, relative frame image path (`frames/frame_000000.jpg`), and frame width/height. Emits single-line JSON strings for `.jsonl` streaming.

### 2. `sportiq.video.reader`
- **`VideoReader`**: Encapsulates OpenCV's `cv2.VideoCapture` with robust validation:
  - File existence and non-empty check (> 0 bytes).
  - Fourcc integer decoding to printable ASCII characters.
  - Duration calculation from frame count and frame rate.
  - Safe stream release via context management (`with VideoReader(...) as reader:`).
  - Frame generator `read_frames(sample_every_n_frames)` guaranteeing sequential original frame index and timestamp preservation.

### 3. `sportiq.video.frame_extractor`
- **`FrameExtractor` / `extract_frames`**:
  - Configurable temporal interval (`--every-seconds`) or frame interval (`--every-n-frames`).
  - Strict overwrite protection: refuses to write into an existing non-empty output directory unless `overwrite=True` is explicitly passed.
  - High-quality JPEG frame serialization with configurable compression (`--jpeg-quality`).
  - Output manifest generation: produces `frames.jsonl` (line-by-line frame index and timestamp records) and `video_metadata.json`.

### 4. `sportiq.cli`
- Unified entry point `sportiq` installed via `pyproject.toml` or runnable via `python -m sportiq`:
  - `sportiq check-env`: inspects Python version, OpenCV, PyTorch, CUDA runtime, GPU device name, compute capability, and executes a live CUDA matrix multiplication test.
  - `sportiq inspect-video <path> [--json]`: validates and reports video properties.
  - `sportiq extract-frames <path> [--every-seconds N] [--every-n-frames N] [--overwrite] [--json]`: runs the end-to-end extraction pipeline.

## Engineering Rules

1. **Modular Independence**: Detection, tracking, analytics, and visualization reside in strictly decoupled modules.
2. **Timestamp Preservation**: Original frame indices and millisecond timestamps are immutably captured at frame ingestion; sampled indices are never confused with source frame numbers.
3. **Raw Output Persistence**: Ingestion manifests (`frames.jsonl`) and raw model predictions must be persisted before deriving higher-order statistics (possession, heatmaps).
4. **Explicit Uncertainty**: Uncertain ball detections and possession intervals must be flagged rather than silently interpolated.
5. **Hardware Constraints**: The target development machine features an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM. Model inference must use FP16 half precision (`half=True`) and compact architectures (`yolov8s` or `yolo11s`) to avoid Out of Memory (OOM) faults.
