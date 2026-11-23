# 10-Day MVP Roadmap

| Day | Milestone / Deliverable | Acceptance Check | Status |
|---|---|---|---|
| **1** | **Environment, GPU verification, data/licence inventory** | Reproducible environment (Python 3.11, PyTorch 2.15+cu132, CUDA 13.2, RTX 3050 verified) | **COMPLETED** |
| **2** | **Video inspection and frame extraction pipeline** | CLI commands (`inspect-video`, `extract-frames`, `check-env`), JSONL manifest, timestamp preservation, 20 passing unit/integration tests | **COMPLETED** |
| **3** | **Dataset engineering & annotation pipeline** | SoccerNet MOT parser, 4-class taxonomy, coordinate validator, YOLO converter, match-level splitter, visualizer, 57 passing tests | **COMPLETED** |
| **4** | **Pretrained player & ball detector baseline** | YOLOv8/11 person/ball detection baseline, annotated sample clip, and precision/recall evaluation | Pending |
| **5** | **Multi-object tracking (ByteTrack)** | Persistent track IDs across frames, bounding box association, and ID retention metrics | Pending |
| **6** | **BoT-SORT comparison and team clustering** | Comparison of ByteTrack vs BoT-SORT; jersey colour feature extraction and k-means clustering | Pending |
| **7** | **Ball association, possession, heatmaps** | Player-ball proximity modeling, continuous possession estimation, and 2D spatial occupancy heatmaps | Pending |
| **8** | **Homography prototype and pitch visualization** | 2D pitch keypoint calibration, homography transformation matrix, and tactical top-down view | Pending |
| **9** | **Evaluation report, API skeleton, dashboard prototype** | Standardized evaluation script, FastAPI backend skeleton, and prototype UI | Pending |
| **10** | **Cleanup, tests, Docker, README, demonstration** | End-to-end reproducible pipeline, Docker containerization, full documentation | Pending |

---

### Phase 1 Acceptance Summary (Days 1 & 2)
- **Environment**: Python 3.11.9 (64-bit), PyTorch 2.15.0+cu132, CUDA 13.2, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, Compute Capability 8.6).
- **Core Package**: `src/sportiq/` modular structure with `video` subpackage (`metadata.py`, `reader.py`, `frame_extractor.py`) and unified CLI (`cli.py`).
- **Pipeline Validation**: Synthetic football match clip generated (`data/raw/sample.mp4`), inspected, and extracted to `data/processed/sample/` with `frames.jsonl` and `video_metadata.json`.
- **Test Suite**: Passing automated unit and integration tests under `pytest` with synthetic video fixtures.

### Phase 2 Acceptance Summary (Day 3)
- **SoccerNet MOT Tracking Parsing**: Full support for MOT20 10-column ground truth format (`gt.txt`), sequence configurations (`seqinfo.ini`), and contextual game metadata (`gameinfo.ini`).
- **Target Taxonomy Implementation**:
  - Primary 4-class model: `0: player`, `1: goalkeeper`, `2: referee`, `3: ball`.
  - Experimental 5-class model: adds `4: staff`.
  - Baseline 2-class model: `0: person`, `1: ball`.
  - Strict preservation of raw class labels, canonical class names, and original track IDs in `labels_metadata.jsonl`.
- **Annotation Validation & Sanitization**: Complete bounds checking ($0 \le x < W$, $0 \le y < H$), non-positive dimension rejection, and bounding-box edge clamping.
- **YOLO Conversion & Sidecar Metadata**: Normalized YOLO detection labels (`<class_id> <x_center> <y_center> <w> <h>`) exported to frame `.txt` files with full tracking metadata sidecars.
- **Deterministic Partitioning**: Zero frame-leakage splitting via Largest Remainder quota distribution at sequence and match levels, with YOLO `dataset.yaml` generation.
- **Annotation Visualizer & CLI Tools**: Integrated CLI commands (`convert-dataset`, `split-dataset`, `dataset-stats`, `visualize-annotations`) and OpenCV color-coded bounding-box renderer.
- **Automated Test Coverage**: 57 automated tests passing under `pytest` across video and dataset packages.
