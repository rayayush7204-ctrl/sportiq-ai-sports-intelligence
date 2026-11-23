# Architecture & Engineering Specifications

## System Architecture

```text
Input Video File (.mp4, .mov, .avi)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Ingestion Layer (sportiq.video) — [Phase 1]            │
│ ────────────────────────────────────────────────────── │
│ - VideoReader: OpenCV sequential decoder               │
│ - Metadata extraction: codec, resolution, fps, frames  │
│ - FrameExtractor: configurable temporal/frame sampling │
│ - Manifest generator: frames.jsonl, video_metadata.json│
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Dataset Engineering (sportiq.dataset) — [Phase 2]      │
│ ────────────────────────────────────────────────────── │
│ - SoccerNet MOT20 ground-truth parser                  │
│ - Normalized YOLO converter & 4-class taxonomy         │
│ - Largest Remainder match-level dataset splitter       │
│ - DatasetMaterializer & 1:1 alignment verifier         │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Detection Layer (sportiq.detection) — [Phase 3]        │
│ ────────────────────────────────────────────────────── │
│ - Pretrained YOLO11n baseline (COCO person, ball)      │
│ - Lazy model loading, FP16 CUDA GPU 0 acceleration     │
│ - BoundingBox, Detection, FrameDetections containers   │
│ - Structured JSONL predictions & visual overlay        │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Tracking Layer (sportiq.tracking) — [Phase 4]          │
│ ────────────────────────────────────────────────────── │
│ - ByteTrack / BoT-SORT multi-object trackers           │
│ - Trajectory buffer & track ID assignment              │
│ - ID switch detection & track continuity metrics       │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Analytics Layer (sportiq.analytics) — [Phase 5]        │
│ ────────────────────────────────────────────────────── │
│ - Team classifier: Jersey colour clustering (k-means)  │
│ - Ball possession & player-ball proximity model        │
│ - 2D pitch homography transformation & heatmaps        │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Serving Layer (sportiq.api) — [Phase 6]                │
│ ────────────────────────────────────────────────────── │
│ - FastAPI endpoints for asynchronous job execution     │
│ - React dashboard for interactive match analytics      │
└────────────────────────────────────────────────────────┘
```

## Implemented Modules (Phase 1 & Phase 2)

### Phase 1: Video Ingestion & Inspection (`sportiq.video`)

#### 1. `sportiq.video.metadata`
- **`VideoMetadata`**: Immutable data container capturing source path, filename, byte size, size in MB, fourcc codec string, width, height, FPS, total frames, and duration in seconds. Provides `.to_dict()` and `.to_json()` helpers. Path sanitization guarantees no Windows host or user directory paths leak.
- **`FrameMetadata`**: Immutable data container capturing source filename, 0-based original frame index, timestamp in milliseconds, relative frame image path (`frames/frame_000000.jpg`), and frame width/height. Emits single-line JSON strings for `.jsonl` streaming.

#### 2. `sportiq.video.reader`
- **`VideoReader`**: Encapsulates OpenCV's `cv2.VideoCapture` with robust validation:
  - File existence and non-empty check (> 0 bytes).
  - Fourcc integer decoding to printable ASCII characters.
  - Duration calculation from frame count and frame rate.
  - Safe stream release via context management (`with VideoReader(...) as reader:`).
  - Frame generator `read_frames(sample_every_n_frames)` guaranteeing sequential original frame index and timestamp preservation.

#### 3. `sportiq.video.frame_extractor`
- **`FrameExtractor` / `extract_frames`**:
  - Configurable temporal interval (`--every-seconds`) or frame interval (`--every-n-frames`).
  - Strict overwrite protection: refuses to write into an existing non-empty output directory unless `overwrite=True` is explicitly passed.
  - High-quality JPEG frame serialization with configurable compression (`--jpeg-quality`).
  - Output manifest generation: produces `frames.jsonl` (line-by-line frame index and timestamp records) and `video_metadata.json`.

---

### Phase 2: Dataset Engineering & Annotation Pipeline (`sportiq.dataset`)

#### 1. `sportiq.dataset.class_mapping`
- **Target Taxonomies**:
  - **`FOUR_CLASS` (Primary Detection Model)**:
    - `0: player` (mapped from `player team left`, `player team right`, `player`)
    - `1: goalkeeper` (mapped from `goalkeeper team left`, `goalkeeper team right`, `goalkeeper`)
    - `2: referee` (mapped from `main referee`, `side referee`, `referee`)
    - `3: ball` (mapped from `ball`)
    - `staff`: Kept with canonical name `"staff"` and original track ID in `labels_metadata.jsonl`; `class_id` is `None` (unmapped for 4-class detector, not conflated with referee).
  - **`FIVE_CLASS` (Experimental Model)**:
    - `0: player`, `1: goalkeeper`, `2: referee`, `3: ball`, `4: staff`
  - **`TWO_CLASS` (Person/Ball Baseline Model)**:
    - `0: person` (`player`, `goalkeeper`, `referee`, `staff`), `1: ball` (`ball`)
- Preserves full lineage: `(class_id, canonical_name, raw_normalized)`.

#### 2. `sportiq.dataset.annotation_parser`
- **`parse_mot_gt`**: Parses SoccerNet MOT20 10-column CSV format (`frame_id`, `track_id`, `bb_left`, `bb_top`, `bb_width`, `bb_height`, `conf`, `world_x`, `world_y`, `world_z`).
  - MOT ground-truth confidence is always `1.0` and world coordinates are `-1.0` (preserved immutably).
- **`parse_seqinfo`**: Parses `seqinfo.ini` for frame rate, sequence length, width, height, and image directory.
- **`parse_gameinfo`**: Parses `gameinfo.ini` extracting `game_id`, `action`, and individual track role definitions from `[Track_<id>]` sections.
- **`load_sequence`**: Assembles a `SequenceDataset` container with complete sequence context.

#### 3. `sportiq.dataset.annotation_validator`
- **`AnnotationValidator`**:
  - Checks for non-positive widths or heights ($w \le 0$ or $h \le 0$) and drops invalid records.
  - Checks if box is entirely out of bounds ($x_2 \le 0$ or $y_2 \le 0$ or $x_1 \ge W$ or $y_1 \ge H$).
  - Clips partially out-of-bounds boxes to $[0, W] \times [0, H]$ when `clip_boxes=True`.
  - Produces detailed `ValidationResult` metrics: total inspected, valid count, clipped count, dropped count, pass rate %.

#### 4. `sportiq.dataset.yolo_converter`
- **`YOLOConverter`**:
  - Normalizes pixel bounding box coordinates to YOLO center coordinates:
    $$x_{\text{center}} = \frac{x_{\text{left}} + \frac{w}{2}}{W}, \quad y_{\text{center}} = \frac{y_{\text{top}} + \frac{h}{2}}{H}$$
    $$w_{\text{norm}} = \frac{w}{W}, \quad h_{\text{norm}} = \frac{h}{H}$$
  - Generates YOLO label files (`000001.txt`, `000002.txt`, ...) corresponding to 1-based frame numbers.
  - Generates sidecar manifest `labels_metadata.jsonl` recording sequence name, 1-based and 0-based frame IDs, original integer `track_id`, raw class string, canonical class name, YOLO class ID, pixel coordinates, and confidence.

#### 5. `sportiq.dataset.dataset_splitter`
- **`DatasetSplitter`**:
  - Partitions sequences into `train`, `val`, and `test` splits deterministically via seeded RNG (`random_seed=42`).
  - Supports sequence-level and match-level grouping (grouping all sequences sharing `game_id` to guarantee zero data leakage).
  - Largest Remainder quota algorithm ensures exact integer partition sizes summing to sequence total.
  - Generates YOLO `dataset.yaml` specifying paths and class names dictionary.

#### 6. `sportiq.dataset.statistics`
- **`DatasetStatisticsReporter`**:
  - Analyzes class distributions (raw vs canonical), staff count and percentage, unique tracks, track duration percentiles (min, max, mean, median), and bounding box spatial metrics (mean width, height, aspect ratio).

#### 7. `sportiq.dataset.visualizer`
- **`draw_annotations` / `visualize_sequence_frame`**:
  - Color-codes bounding boxes by canonical class (green for player, orange for goalkeeper, yellow for referee, cyan for ball, magenta for staff).
  - Renders labels with track IDs and class names onto OpenCV frames.

#### 8. `sportiq.dataset.dataset_materializer`
- **`DatasetMaterializer` / `verify_dataset_alignment`**:
  - Materializes train/val/test splits preserving 1:1 image-to-label correspondence.
  - Zero-duplication manifest mode generates `train.txt`, `val.txt`, `test.txt` containing resolved POSIX image paths, a `labels_manifest.json` sidecar for explicit verification, and standard `dataset.yaml`.
  - Directory mode (`copy`, `symlink`, `hardlink`) materializes standard Ultralytics-compatible `images/<split>` and `labels/<split>` directories.
  - Alignment audit validates existence of every image, 1:1 correspondence with label files (distinguishing intentional 0-byte negative frames from missing labels), validates class IDs against `dataset.yaml` names, checks finite coordinates (`math.isfinite()`), and verifies that bounding box geometric extents ($x_c \pm w/2, y_c \pm h/2$) strictly reside within $[0.0, 1.0]$.
  - Strict match grouping policy guarantees zero match-level leakage: rejects incomplete or entirely missing match metadata unless `--allow-missing-game-id` is explicitly passed.
  - Reverse directory check detects deleted/orphaned images when labels exist on disk.
  - Compatibility note: Ultralytics YOLO requires `/images/` $\to$ `/labels/` path substitution when training from text manifests. For datasets without an `/images/` path token (such as SoccerNet's `img1/`), directory mode (`copy`, `symlink`, `hardlink`) provides standard native compatibility.

#### 9. `sportiq.dataset.catalog`
- **`build_soccernet_catalog` / `group_sequences_by_game`**:
  - `SequenceCatalogEntry`: Captures sequence dimensions, frame rate, duration, annotation count, track count, 4-class canonical distribution, and local frame presence.
  - `MatchCatalogGroup`: Groups sequences strictly by verified `game_id` from `gameinfo.ini`, guaranteeing no synthetic match generation and reporting aggregate class balance per match.
  - `SoccerNetCatalog`: Serializes machine-readable JSON catalogs and comprehensive markdown reports auditing remote repository inventories, match distribution, and image availability gaps.

---

### Phase 3: Object Detection Baseline (`sportiq.detection`)

#### 1. `sportiq.detection.config`
- **`DetectorConfig`**: Dataclass controlling model weights path (`yolo11n.pt`), target device (`cuda:0` / `cpu`), inference image size (`imgsz=640`), confidence threshold (`conf=0.25`), IoU threshold (`iou=0.45`), FP16 half precision (`half=True`), and batch size (`1`). Validates parameter boundaries.

#### 2. `sportiq.detection.prediction`
- **`BoundingBox`**: Pixel-coordinate bounding box container ($x_1, y_1, x_2, y_2$) with derived dimensions ($w, h$, centers) and normalized YOLO coordinate export.
- **`Detection`**: Individual object prediction instance with bounding box, confidence score, integer class ID, and class name.
- **`FrameDetections`**: Complete frame detection record capturing source image path, frame index, millisecond timestamp, image dimensions, inference latency (ms), model identifier, device, and class frequency distribution. Supports single-line JSON serialization and streaming `.append_to_jsonl()`.

#### 3. `sportiq.detection.detector`
- **`YOLODetector`**:
  - Lazy model initialization: weights are loaded only on first inference or explicit `load_model()`.
  - Explicit device validation: verifies CUDA availability when `cuda` is requested, preventing silent fallback to CPU.
  - FP16 half-precision execution enabled on CUDA GPU 0.
  - Exposes `class_names` mapping directly from model.
  - `predict_image()`: Processes single image or numpy BGR array.
  - `predict_frames()`: Processes frame directories, automatically resolving `frames.jsonl` manifests according to the Phase 1 `FrameMetadata` schema (`extracted_frame_filename`, 0-based `original_frame_index`, and `timestamp_ms`) to preserve exact source timestamps and frame indices in structured JSONL predictions.

#### 4. `sportiq.detection.visualizer`
- **`DetectionVisualizer`**: Renders detected bounding boxes, confidence tags, and class badges onto frames with category color coding and summary overlay. Exports annotated frames to disk.

---

### Phase 1, 2 & 3 CLI (`sportiq.cli`)
- Unified entry point `sportiq`:
  - `check-env`: inspects environment and GPU acceleration.
  - `inspect-video`: extracts video stream metadata.
  - `extract-frames`: samples video frames with manifest.
  - `convert-dataset`: converts SoccerNet MOT sequences to YOLO `.txt` + `labels_metadata.jsonl`.
  - `split-dataset`: partitions datasets deterministically and writes `dataset.yaml` with strict match-level grouping.
  - `dataset-stats`: computes ground-truth statistics across sequences.
  - `visualize-annotations`: renders annotated frames with ground-truth boxes and track IDs.
  - `materialize-dataset`: builds zero-duplication YOLO datasets with split manifests or physical layouts.
  - `verify-dataset`: audits image-to-label alignment, missing frames, and coordinate validity.
  - `catalog-dataset`: catalogs multi-match tracking sequences, groups by game ID, and audits image/annotation availability.
  - `detect-frames`: runs object detection inference on images or extracted-frame directories, exporting JSONL predictions and visual overlays.

## Engineering Rules

1. **Modular Independence**: Detection, tracking, analytics, and visualization reside in strictly decoupled modules.
2. **Timestamp Preservation**: Original frame indices and millisecond timestamps are immutably captured at frame ingestion; sampled indices are never confused with source frame numbers. Frame manifest metadata (`extracted_frame_filename`, `original_frame_index`, `timestamp_ms`) is tested and preserved through detection predictions.
3. **Raw Output Persistence**: Ingestion manifests (`frames.jsonl`) and raw model predictions must be persisted before deriving higher-order statistics (possession, heatmaps).
4. **Explicit Uncertainty**: Uncertain ball detections and possession intervals must be flagged rather than silently interpolated.
5. **Hardware Constraints**: The target development machine features an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM. Model inference must use FP16 half precision (`half=True`) and lightweight compact architectures (`yolo11n` baseline, ~2.6M parameters) to strictly avoid Out of Memory (OOM) faults within the 4 GB VRAM budget.
