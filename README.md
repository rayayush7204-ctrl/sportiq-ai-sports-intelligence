# SportIQ — AI-Powered Sports Video Intelligence

> **Status: Phase 1 Implemented & Verified** · Computer Vision + Machine Learning Platform

SportIQ is a practical football-video analysis platform designed to turn ordinary match footage into structured, explainable match insights. The platform pipeline covers video ingestion, player and ball detection, multi-object tracking, team identification, pitch mapping, and movement/possession analytics.

The project is built as a rigorous computer-vision engineering system: reproducible experiments, explicit evaluation metrics, an API, and a lightweight dashboard—not just an isolated model demo.

## Project Pipeline

```text
Football Video (Local / Raw)
    │
    ▼
Video Ingestion & Frame Sampling (sportiq.video)   ◄ [PHASE 1 COMPLETE]
    │   ├── Codec, resolution, FPS, duration inspection
    │   ├── Timestamp & original frame index preservation
    │   └── JSONL metadata manifest + JPEG frame exports
    │
    ▼
Object Detection (YOLO / PyTorch)                 ◄ [PHASE 2 - NEXT]
    │   ├── Player / person detection baseline
    │   └── Football ball detection
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
| **Object Detection** | Ultralytics YOLO (8.4.216) | Installed |
| **Video Processing** | OpenCV (4.14.0) | Active |
| **Multi-Object Tracking** | ByteTrack, BoT-SORT | Planned (Phase 3) |
| **Data & Scientific** | NumPy (2.4.6), Pandas (2.3.3), Scipy (1.17.1) | Installed |
| **Machine Learning** | scikit-learn (1.9.1) | Installed |
| **Testing** | pytest (9.2.0) | Active (17 passing tests) |
| **API & Serving** | FastAPI, Uvicorn | Planned (Phase 5) |

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
│   ├── video/           # Ingestion, reader, metadata, extraction
│   │   ├── metadata.py
│   │   ├── reader.py
│   │   └── frame_extractor.py
│   ├── detection/       # Object detection modules (Phase 2)
│   ├── tracking/        # Multi-object tracking (Phase 3)
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

### 4. Running Automated Tests
Run the test suite with synthetic video fixtures:
```powershell
pytest -v
```

## Development Roadmap

- [x] **Day 1: Environment & GPU verification** (Python 3.11, PyTorch 2.15+cu132, RTX 3050)
- [x] **Day 2: Video inspection and frame extraction** (`sportiq.video`, CLI, JSONL manifest, tests)
- [ ] **Day 3: Pretrained player detector baseline** (YOLOv8 / YOLO11 person detection, mAP metrics)
- [ ] **Day 4: Football-ball detection baseline / fine-tuning**
- [ ] **Day 5: ByteTrack integration** (track identities across frames)
- [ ] **Day 6: BoT-SORT comparison and team clustering** (jersey appearance features)
- [ ] **Day 7: Ball association, possession estimation, and heatmaps**
- [ ] **Day 8: Homography prototype and pitch visualization**
- [ ] **Day 9: Evaluation report, API skeleton, and dashboard prototype**
- [ ] **Day 10: Cleanup, full integration tests, Docker containerization**

## Author
Ayush Kumar Ray  
GitHub: [@rayayush7204-ctrl](https://github.com/rayayush7204-ctrl)
