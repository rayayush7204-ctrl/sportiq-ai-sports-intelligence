# 10-Day MVP Roadmap

| Day | Milestone / Deliverable | Acceptance Check | Status |
|---|---|---|---|
| **1** | **Environment, GPU verification, data/licence inventory** | Reproducible environment (Python 3.11, PyTorch 2.15+cu132, CUDA 13.2, RTX 3050 verified) | **COMPLETED** |
| **2** | **Video inspection and frame extraction pipeline** | CLI commands (`inspect-video`, `extract-frames`, `check-env`), JSONL manifest, timestamp preservation, 17 passing unit/integration tests | **COMPLETED** |
| **3** | **Pretrained player detector baseline** | Person detection baseline with YOLOv8/11, annotated sample clip, and precision/recall evaluation | Pending |
| **4** | **Football ball detection baseline / fine-tuning** | Ball candidate detection, visual validation, false-positive analysis, and dataset decision | Pending |
| **5** | **ByteTrack integration** | Persistent track IDs across frames, bounding box association, and ID retention metrics | Pending |
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
- **Test Suite**: 17 passing automated unit and integration tests under `pytest` with synthetic video fixtures.
