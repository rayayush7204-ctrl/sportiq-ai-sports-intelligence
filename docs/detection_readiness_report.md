# SportIQ — Detection Readiness Audit Report

> **Document Status**: Approved / Baseline Audit  
> **Date**: 2026-11-23  
> **Phase**: Phase 3, Step 1 (Object Detection Readiness Audit)  
> **Overall Verdict**: **BLOCKED** for Custom Model Training & Match-Level Evaluation  

---

## 1. Executive Summary & Verdict

Before initiating object detection model training, an empirical readiness audit was performed on the local ground-truth dataset assets using SportIQ's dataset auditing tools (`sportiq dataset-stats` and `sportiq verify-dataset`).

| Audit Dimension | Result | Verdict |
|---|---|---|
| **Available Sequences** | 1 (`SNMOT-116`) | Inadequate for Multi-Sequence Partitioning |
| **Unique Match Groups (`game_id`)** | 1 (`gameID=7`) | **BLOCKED**: Cannot create disjoint train/val/test matches |
| **Physical Image Frames** | 3 frames (`000001.jpg`, `000002.jpg`, `000003.jpg`) | Inadequate for deep learning training |
| **Ground-Truth Sequence Length** | 750 frames (25 FPS, 30.0s) | Metadata and annotations present (11,460 labels) |
| **Image-to-Label Alignment** | 100% Aligned (`is_aligned: True`, 0 missing, 0 corrupt) | **PASSED** (Pipeline Integrity Verified) |
| **Training Feasibility** | Single Match Group | **BLOCKED** pending acquisition of $\ge 3$ distinct matches |

### Formal Readiness Verdict: BLOCKED FOR CUSTOM TRAINING
Per rigorous computer-vision engineering guidelines and strict match-level splitting policies:
1. **Zero Data Leakage Enforcement**: Machine learning models for sports tracking cannot be evaluated by splitting frames or sequences from the same match into train and test sets. High temporal and environmental correlation (same stadium, lighting, pitch quality, team kits, camera vantage) creates severe data leakage, resulting in inflated evaluation metrics that fail to generalize.
2. **Current Constraint**: Only one match group (`gameID=7`) is available locally. Creating independent train, validation, and test partitions requires a minimum of 3 independent match groups.
3. **Strict Policy Compliance**: We refuse to synthesize artificial match IDs or split frames within the same match to simulate independent evaluation. Custom four-class model training and evaluation are therefore formally **BLOCKED** until additional authorized match sequences are ingested.
4. **Immediate Path Forward**: Proceed with Step 1 implementation: build the modular inference pipeline using the pretrained `YOLO11n` COCO baseline (`person` and `sports ball` detection).

---

## 2. Sequence Inventory & Metadata

The local SoccerNet ground-truth repository was scanned across `data/raw/` and `data/processed/`:

```text
Sequence Name:      SNMOT-116
Sequence Path:      data/raw/soccernet/SNMOT-116
Game ID:            7 (Match Group 7)
Action Class:       Corner
Action Position:    975,293 ms
Clip Window:        969,000 ms to 999,000 ms (Game Time: 1 - 16:09 to 16:39)
Sequence Length:    750 frames (at 25 FPS = 30.0 seconds)
Native Resolution:  1920 x 1080 px
Local Image Frames: 3 frames present on disk (000001.jpg, 000002.jpg, 000003.jpg)
```

---

## 3. Ground-Truth Class Distribution & Statistics

Using `sportiq dataset-stats data/raw/soccernet/SNMOT-116`, all 11,460 sequence annotations across 27 ground-truth tracklets were analyzed under the 4-class taxonomy:

### Canonical Class Distribution (Full Sequence gt.txt)
- **Player** (`0`): 9,550 annotations (83.33%)
  - *Raw breakdown*: `player team right`: 5,235; `player team left`: 4,315
- **Ball** (`3`): 724 annotations (6.32%)
- **Goalkeeper** (`1`): 601 annotations (5.24%)
  - *Raw breakdown*: `goalkeepers team left`: 601
- **Referee** (`2`): 585 annotations (5.10%)
  - *Raw breakdown*: `main referee`: 300; `side referee`: 285
- **Staff**: 0 annotations (0.0%)

### Track & Spatial Dimension Metrics
- **Unique Tracklets**: 27 tracks
- **Track Duration**: Mean: 424.4 frames (17.0s); Median: 465.0 frames; Range: [120, 724] frames
- **Bounding Box Dimensions (1080p)**:
  - Mean Width: 53.18 px (Min: 3 px, Max: 172 px)
  - Mean Height: 117.56 px (Min: 11 px, Max: 243 px)
  - Mean Aspect Ratio ($W/H$): 0.486

---

## 4. Image-to-Label Alignment Audit

Using `sportiq verify-dataset data/processed/soccernet_val/materialized/dataset.yaml`, all materialized local frames were audited for structural and numerical integrity:

```json
{
  "dataset_yaml": "data/processed/soccernet_val/materialized/dataset.yaml",
  "total_images": 3,
  "total_labels": 3,
  "labeled_frames": 3,
  "background_frames": 0,
  "missing_images_count": 0,
  "missing_labels_count": 0,
  "corrupted_labels_count": 0,
  "split_counts": {
    "train": 3,
    "val": 0,
    "test": 0
  },
  "is_aligned": true
}
```

### Integrity Checklist
- [x] **1:1 Image-to-Label Correspondence**: Verified for all 3 materialized frames.
- [x] **Finite Coordinates**: All coordinates pass `math.isfinite()` bounds checking.
- [x] **Geometric Canvas Bounds**: All bounding box extents ($x_c \pm w/2, y_c \pm h/2$) strictly reside within $[0.0, 1.0]$.
- [x] **Zero Disk Duplication**: Manifest mode links directly to source images without duplicating multi-megabyte frame assets.

---

## 5. Match-Level Partitioning & Data Leakage Analysis

Attempting a match-level partition with `sportiq split-dataset` or `sportiq materialize-dataset` in strict mode confirms the fundamental barrier:

```powershell
sportiq split-dataset data/raw/soccernet
# Output:
# Error: Incomplete match metadata: only 1 unique match found.
# Cannot partition a single match group across disjoint train, val, and test splits without leakage.
```

When all sequences share `gameID=7`, any allocation of sequences or frames into train, validation, or test partitions would introduce fatal frame leakage. The model would be tested on the exact same players, kits, pitch, lighting, and camera conditions as seen in training.

---

## 6. Unblocking Requirements & Action Plan

### Prerequisites to Unblock Custom Four-Class Fine-Tuning:
1. **Acquire Additional Match Sequences**:
   - Ingest at least 2 additional authorized SoccerNet tracking sequences representing distinct `gameID` matches (e.g. `SNMOT-117` and `SNMOT-118` from separate matches, or synthetic match footage with distinct match IDs).
   - Minimum requirement: 3 distinct match IDs ($\ge 1$ Train match, $\ge 1$ Validation match, $\ge 1$ Test match).
2. **Download Corresponding Image Frames**:
   - For training to be viable, download the complete frame image sets for the selected sequences.

### Phase 3, Step 1 Execution Strategy (Pretrained Baseline):
1. Implement a clean, modular detection pipeline in `src/sportiq/detection/`.
2. Wrap the official Ultralytics `YOLO11n` pretrained COCO model.
3. Verify GPU-accelerated FP16 inference on NVIDIA RTX 3050 Laptop GPU (4 GB VRAM).
4. Run inference smoke test on local permitted frames (`data/raw/sample.mp4` and `SNMOT-116/img1/000001.jpg`).
5. Export structured JSONL predictions and visually annotated inspection images.
6. Rigorously report detection metrics only where ground-truth evaluation sets are scientifically valid.
