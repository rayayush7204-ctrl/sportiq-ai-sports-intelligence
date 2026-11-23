# SportIQ — AI-Powered Sports Video Intelligence

> **Status: Phase 1, Phase 2 & Phase 3 Step 1 Baseline Implemented & Verified** · Computer Vision + Machine Learning Platform

SportIQ is a practical football-video analysis platform designed to turn ordinary match footage into structured, explainable match insights. The platform pipeline covers video ingestion, player and ball detection, multi-object tracking, team identification, pitch mapping, and movement/possession analytics.

The project is built as a rigorous computer-vision engineering system: reproducible experiments, explicit evaluation metrics, an API, and a lightweight dashboard—not just an isolated model demo.

## Project Pipeline

```text
Football Video (Local / Raw) & Tracking Datasets (SoccerNet MOT)
    │
    ▼
Video Ingestion & Frame Sampling (sportiq.video)       ◄ [PHASE 1 COMPLETE]
    │   ├── Codec, resolution, FPS, duration inspection
    │   ├── Timestamp & original frame index preservation
    │   └── JSONL metadata manifest + JPEG frame exports
    │
    ▼
Dataset Engineering & MOT/YOLO Converter (sportiq.dataset) ◄ [PHASE 2 COMPLETE]
    │   ├── SoccerNet MOT20 ground-truth parsing (gt.txt, seqinfo, gameinfo)
    │   ├── Bounding box validation, clamping, and coordinate sanitization
    │   ├── 4-class taxonomy (player, goalkeeper, referee, ball) + staff lineage
    │   ├── Deterministic match-level dataset splitting (zero frame leakage)
    │   ├── Normalized YOLO label conversion (.txt) + sidecar JSONL manifests
    │   └── Annotation visualization and empirical dataset analytics
    │
    ▼
Object Detection (YOLO / PyTorch)                     ◄ [PHASE 3 ACTIVE - STEP 1 COMPLETE]
    │   ├── Pretrained YOLO11n COCO baseline (GPU FP16)
    │   ├── Structured predictions & streaming JSONL export
    │   └── Detection visualizer with category overlays
    │
    ▼
Multi-Object Tracking (ByteTrack / BoT-SORT)
    │
    ├── Team Classification (jersey-colour clustering)
    ├── Ball Trajectory & Possession Estimation
    └── Pitch Mapping (homography calibration)
              │
              ▼
    Match Analytics & Visualizations
              │
              ▼
    FastAPI Service + React Dashboard
```

## Technology Stack

| Area | Tools & Packages | Status |
|---|---|---|
| **Language** | Python 3.11.9 (64-bit) | Active |
| **Deep Learning** | PyTorch 2.15.0+cu132, Torchvision 0.29.2+cu132 | Active (CUDA 13.2, RTX 3050 Laptop GPU) |
| **Dataset Engineering** | SoccerNet MOT parser, YOLO converter, PyYAML | Active (Phase 2) |
| **Object Detection** | Ultralytics YOLO (8.4.216) | Active (Phase 3 Baseline) |
| **Video Processing** | OpenCV (4.14.0) | Active |
| **Multi-Object Tracking** | ByteTrack, BoT-SORT | Planned (Phase 4) |
| **Data & Scientific** | NumPy (2.4.6), Pandas (2.3.3), Scipy (1.17.1) | Installed |
| **Machine Learning** | scikit-learn (1.9.1) | Installed |
| **Testing** | pytest (9.2.0) | Active (105 passing tests) |
| **API & Serving** | FastAPI, Uvicorn | Planned (Phase 6) |

## Repository Layout

```text
sportiq-ai-sports-intelligence/
├── data/
│   ├── raw/             # Local source videos; e.g. sample.mp4
│   ├── processed/       # Extracted frames and JSONL manifests
│   └── annotations/    # Labels and annotation exports
├── docs/                # Architecture, experiments, and project notes
│   ├── architecture.md
│   └── roadmap.md
├── models/weights/      # Local model weights (.pt, .onnx)
├── notebooks/           # Exploration and evaluation notebooks
├── scripts/             # CLI utilities and helpers
│   ├── generate_sample_video.py
│   ├── inspect_video.py
│   └── extract_frames.py
├── src/sportiq/         # Core Python package
│   ├── video/           # Ingestion, reader, metadata, extraction (Phase 1)
│   │   ├── metadata.py
│   │   ├── reader.py
│   │   └── frame_extractor.py
│   ├── dataset/         # Dataset engineering & MOT/YOLO pipeline (Phase 2)
│   │   ├── class_mapping.py        # 4-class, 5-class, 2-class taxonomies
│   │   ├── annotation_parser.py    # SoccerNet MOT20 gt.txt, seqinfo, gameinfo
│   │   ├── annotation_validator.py # Coordinate bounds checking & clipping
│   │   ├── yolo_converter.py       # Normalized YOLO labels & sidecar JSONL
│   │   ├── dataset_splitter.py     # Deterministic match-level splitting
│   │   ├── statistics.py           # Ground-truth class balance & track analytics
│   │   ├── visualizer.py           # Ground-truth bbox rendering on frames
│   │   └── downloader.py           # Remote HF inventory inspection helper
│   ├── detection/       # Object detection modules (Phase 3)
│   │   ├── config.py       # DetectorConfig (device, imgsz, conf, half)
│   │   ├── prediction.py   # BoundingBox, Detection, FrameDetections
│   │   ├── detector.py     # YOLODetector lazy loading & inference
│   │   └── visualizer.py   # DetectionVisualizer overlay renderer
│   ├── tracking/        # Multi-object tracking (Phase 4)
│   ├── analytics/       # Possession, team clustering, heatmaps
│   ├── visualization/   # Video and pitch renderers
│   └── cli.py           # Unified command-line interface
├── tests/               # Automated unit & integration tests
├── pyproject.toml       # Package configuration & entry points
├── requirements.txt     # Locked production dependencies
└── README.md
```

## Getting Started

### 1. Prerequisites
- **Python 3.11 (64-bit)**
- **NVIDIA GPU** with CUDA support (verified on NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM)
- **Git**

### 2. Environment Setup
Activate the virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Verify the environment and GPU acceleration:
```powershell
sportiq check-env
```
Expected output:
```text
============================================================
SportIQ Video Intelligence — Environment Check (v0.1.0)
============================================================
Python:             3.11.9
OpenCV:             4.14.0
PyTorch:            2.15.0+cu132
CUDA Available:     True
CUDA Version:       13.2
GPU Device:         NVIDIA GeForce RTX 3050 Laptop GPU
Compute Capability: 8.6
CUDA Tensor MatMul: PASSED
============================================================
```

### 3. Video Inspection & Frame Extraction CLI

#### Inspect a Video
Inspect resolution, frame rate, container codec, total frames, and duration:
```powershell
sportiq inspect-video data/raw/sample.mp4
```
For JSON output:
```powershell
sportiq inspect-video data/raw/sample.mp4 --json
```

#### Extract Frames with Timestamp Manifest
Extract frames sampled at fixed time intervals (e.g. 1 frame every second) or frame intervals:
```powershell
# Sample 1 frame every second
sportiq extract-frames data/raw/sample.mp4 --every-seconds 1.0

# Sample every 15 frames with overwrite allowed
sportiq extract-frames data/raw/sample.mp4 --every-n-frames 15 --overwrite
```

This generates:
- `data/processed/<video_stem>/frames/frame_000000.jpg` ...
- `data/processed/<video_stem>/frames.jsonl` (manifest recording `source_filename`, `original_frame_index`, `timestamp_ms`, `extracted_frame_filename`, `frame_width`, `frame_height`)
- `data/processed/<video_stem>/video_metadata.json` (complete video stream metadata)

### 4. Dataset Engineering CLI (Phase 2)

#### Convert SoccerNet MOT to YOLO Format
Convert MOT tracking sequences to normalized YOLO detection format (`.txt`) while preserving original tracking IDs and raw class lineage in `labels_metadata.jsonl`:
```powershell
# Convert sequence with primary 4-class taxonomy (player, goalkeeper, referee, ball)
sportiq convert-dataset data/raw/soccernet/SNMOT-060 --output-dir data/processed/yolo --taxonomy four_class

# Output conversion summary as JSON
sportiq convert-dataset data/raw/soccernet/SNMOT-060 --json
```

#### Deterministic Match-Level Dataset Splitting
Split sequences into disjoint train/val/test partitions guaranteeing zero frame leakage (grouped by match/game ID when available) and generate YOLO `dataset.yaml`:
```powershell
sportiq split-dataset data/raw/soccernet/ --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --output-yaml data/processed/yolo/dataset.yaml
```

#### Ground-Truth Empirical Statistics & Class Distribution
Inspect class balance, staff frequency, track duration statistics, and bounding-box dimensions:
```powershell
sportiq dataset-stats data/raw/soccernet/ --taxonomy four_class
```

#### Annotation Visualizer
Draw ground-truth bounding boxes, track IDs, and color-coded classes onto sequence frames:
```powershell
sportiq visualize-annotations data/raw/soccernet/SNMOT-060 --frame-id 1 --output data/processed/vis_frame_1.jpg
```

#### Materialize YOLO Dataset Splits (Zero-Duplication Manifests or Directory Layout)
Materialize verified train, validation, and test partitions with 1:1 image-to-label alignment:
```powershell
# Manifest mode: creates train.txt, val.txt, test.txt and dataset.yaml without duplicating large image files
sportiq materialize-dataset data/raw/soccernet/ --labels-dir data/processed/yolo/labels --output-dir data/processed/yolo_dataset --mode manifest

# Copy or symlink mode: creates images/train, labels/train, etc.
sportiq materialize-dataset data/raw/soccernet/ --labels-dir data/processed/yolo/labels --output-dir data/processed/yolo_dataset --mode copy
```

#### Audit and Verify Dataset Alignment
Verify every referenced image exists, corresponds to a valid label, and has valid normalized coordinates:
```powershell
sportiq verify-dataset data/processed/yolo_dataset/dataset.yaml
```

#### Run Pretrained YOLO11n Object Detection
Run inference on single frames or extracted frame directories:
```powershell
# Run inference on frame directory and export predictions
sportiq detect-frames data/processed/sample/frames --output-jsonl data/processed/predictions.jsonl --visualize data/processed/vis_frames

# Run inference on a single image with JSON summary
sportiq detect-frames data/processed/sample/frames/frame_000000.jpg --json
```

The detector automatically discovers and parses the Phase 1 `frames.jsonl` manifest (`extracted_frame_filename`, `original_frame_index`, `timestamp_ms`), preserving exact source frame numbers (including 0-based frame 0) and timestamps (including 0.0 ms) across streaming JSONL exports.

### 5. Running Automated Tests
Run the complete automated test suite (105 unit and integration tests):
```powershell
pytest -v
```

## Development Roadmap

- [x] **Day 1: Environment & GPU verification** (Python 3.11, PyTorch 2.15+cu132, RTX 3050)
- [x] **Day 2: Video inspection and frame extraction** (`sportiq.video`, CLI, JSONL manifest, tests)
- [x] **Day 3: Dataset engineering & MOT/YOLO pipeline** (`sportiq.dataset`, 4-class taxonomy, MOT20 parser, validator, deterministic splitter, visualizer, materializer, 90 tests)
- [x] **Day 4: Pretrained detector baseline** (Phase 3 Step 1: `sportiq.detection`, YOLO11n, GPU FP16, JSONL prediction export, 105 tests)
- [ ] **Day 5: Multi-object tracking (ByteTrack)** (track association across frames, ID retention)
- [ ] **Day 6: BoT-SORT comparison and team clustering** (jersey appearance features)
- [ ] **Day 7: Ball association, possession estimation, and heatmaps**
- [ ] **Day 8: Homography prototype and pitch visualization**
- [ ] **Day 9: Evaluation report, API skeleton, and dashboard prototype**
- [ ] **Day 10: Cleanup, full integration tests, Docker containerization**

## Author
Ayush Kumar Ray  
GitHub: [@rayayush7204-ctrl](https://github.com/rayayush7204-ctrl)
