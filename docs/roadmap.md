# 10-Day MVP Roadmap

| Day | Milestone / Deliverable | Acceptance Check | Status |
|---|---|---|---|
| **1** | **Environment, GPU verification, data/licence inventory** | Reproducible environment (Python 3.11, PyTorch 2.15+cu132, CUDA 13.2, RTX 3050 verified) | **COMPLETED** |
| **2** | **Video inspection and frame extraction pipeline** | CLI commands (`inspect-video`, `extract-frames`, `check-env`), JSONL manifest, timestamp preservation, 20 passing unit/integration tests | **COMPLETED** |
| **3** | **Dataset engineering & annotation pipeline** | SoccerNet MOT parser, 4-class taxonomy, coordinate validator, YOLO converter, match-level splitter, visualizer, 90 passing tests | **COMPLETED** |
| **4** | **Pretrained detector baseline & readiness audit** | YOLO11n person/ball detection baseline, readiness audit report, FP16 CUDA inference, 104 passing tests | **COMPLETED (Step 1)** |
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
- **Annotation Validation & Sanitization**: Complete bounds checking ($0 \le x < W$, $0 \le y < H$), finite-number validation (`math.isfinite()` rejecting NaN and +/-Inf), non-positive dimension rejection, and bounding-box edge clamping.
- **YOLO Conversion & Sidecar Metadata**: Normalized YOLO detection labels (`<class_id> <x_center> <y_center> <w> <h>`) exported to frame `.txt` files with full tracking metadata sidecars.
- **Deterministic Partitioning & Strict Match Grouping**: Zero frame-leakage splitting via Largest Remainder quota distribution at sequence and match levels, with strict game_id completeness checks and YOLO `dataset.yaml` generation.
- **Dataset Materialization & Alignment Verification**: `DatasetMaterializer` supporting zero-duplication manifest and physical directory layouts (`copy`, `hardlink`, `symlink`), explicit 1:1 image-to-label alignment auditing, distinguishing intentional 0-byte negative frames from missing labels, class ID validation, and exact manifest-mode mapping verification.
- **Annotation Visualizer & CLI Tools**: Integrated CLI commands (`convert-dataset`, `split-dataset`, `dataset-stats`, `visualize-annotations`, `materialize-dataset`, `verify-dataset`) and OpenCV color-coded bounding-box renderer.
- **Automated Test Coverage**: 90 automated tests passing under `pytest` across video, dataset, materialization, label integrity, geometric boundary verification, and validation modules.

### Phase 3 Acceptance Summary (Step 1: Baseline & Readiness Audit)
- **Dataset Readiness Audit**: Formally documented in `docs/detection_readiness_report.md`. Audited local SoccerNet assets (`SNMOT-116`, `gameID=7`). Formally marked custom model training and evaluation as **BLOCKED** pending acquisition of $\ge 3$ independent matches to guarantee zero match-level data leakage.
- **Pretrained Baseline Model**: Implemented modular `sportiq.detection` subpackage (`config.py`, `prediction.py`, `detector.py`, `visualizer.py`) wrapping Ultralytics `YOLO11n` COCO model.
- **GPU Acceleration**: Verified FP16 half-precision CUDA GPU 0 inference on NVIDIA GeForce RTX 3050 Laptop GPU (20-40 ms / frame latency).
- **Metadata & Manifest Preservation**: Original frame indices and millisecond timestamps are seamlessly preserved from ingestion manifests (`frames.jsonl`) into output detection records.
- **Machine-Readable & Visual Outputs**: Streamlined streaming prediction export to JSONL (`FrameDetections.append_to_jsonl()`) and visual bounding box overlays (`DetectionVisualizer`).
- **Unified CLI Integration**: `sportiq detect-frames` supporting single images and extracted-frame directories.
- **Automated Test Suite**: 104 passing automated tests under `pytest` across video, dataset, and detection modules.
