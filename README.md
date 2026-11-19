# SportIQ — AI-Powered Sports Video Intelligence

> **Status: In development** · Individual Computer Vision + Machine Learning project

SportIQ is a football-video analysis project that aims to turn ordinary match footage into structured, explainable match insights. The planned pipeline covers player and ball detection, multi-object tracking, team identification, pitch mapping, and movement/possession analytics.

The project is being built as a practical computer-vision engineering system: reproducible experiments, explicit evaluation, an API, and a lightweight dashboard are part of the plan—not just a model demo.

## Project goals

- Detect players and the ball in football footage.
- Track players across frames and maintain identities where possible.
- Group players by team using visual appearance cues.
- Estimate ball possession from tracked player and ball positions.
- Visualize player movement and heatmaps.
- Map image coordinates to a 2D pitch when suitable calibration points are available.
- Evaluate detection and tracking quality on held-out match footage.

**Important:** These are project goals, not completed capabilities. Features, metrics, and screenshots will be updated only after they are implemented and tested.

## Planned pipeline

```text
Football video
    │
    ▼
Frame sampling and preprocessing (OpenCV)
    │
    ▼
Object detection (YOLO / PyTorch)
    │
    ▼
Multi-object tracking (ByteTrack baseline; BoT-SORT comparison)
    │
    ├── Team classification (jersey-colour features + clustering)
    ├── Ball trajectory and possession estimation
    └── Pitch mapping (homography, where calibration is possible)
              │
              ▼
      Match analytics and visualizations
              │
              ▼
      FastAPI service + React dashboard (planned)
```

## Planned technology stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning | PyTorch, Ultralytics YOLO |
| Video processing | OpenCV |
| Multi-object tracking | ByteTrack, BoT-SORT |
| Data processing | NumPy, Pandas |
| Clustering / utilities | scikit-learn |
| API (planned) | FastAPI |
| Dashboard (planned) | React |
| Experiment tracking (planned) | MLflow |
| Packaging / deployment (planned) | Docker |

The detector, tracker, and dependencies will be selected and pinned after baseline experiments. Check the licences and terms for model code, pretrained weights, and datasets before reuse or redistribution.

## Repository layout

```text
sportiq-ai-sports-intelligence/
├── data/
│   ├── raw/             # Local source videos; do not commit large/restricted files
│   ├── processed/       # Generated frames and intermediate outputs
│   └── annotations/    # Labels and annotation exports
├── docs/                # Architecture, experiments, and project notes
├── models/weights/      # Local model weights; normally excluded from Git
├── notebooks/           # Exploration and evaluation notebooks
├── scripts/             # Reproducible command-line utilities
├── src/sportiq/
│   ├── detection/        # Player / ball detection
│   ├── tracking/         # Multi-object tracking
│   ├── analytics/        # Team, possession, and movement analytics
│   ├── visualization/    # Annotated video and pitch visualizations
│   └── api/              # API application (planned)
├── tests/               # Unit and integration tests
├── requirements.txt
└── README.md
```

## Getting started

### 1. Requirements

- Python 3.10 or 3.11 (the project environment will be fixed during setup)
- Git
- An NVIDIA GPU is useful for model training and accelerated inference; CPU-only development can still be used for data preparation and tests.

### 2. Clone and create an environment

```bash
git clone https://github.com/rayayush7204-ctrl/sportiq-ai-sports-intelligence.git
cd sportiq-ai-sports-intelligence
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

Install a PyTorch build compatible with your installed NVIDIA driver and CUDA setup using the official PyTorch selector: https://pytorch.org/get-started/locally/ . Then install the project packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the GPU

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Evaluation plan

| Component | Metrics to report |
|---|---|
| Player / ball detection | Precision, recall, mAP@50, mAP@50:95 |
| Multi-object tracking | HOTA, IDF1, ID switches |
| Team classification | Accuracy, macro-F1 |
| Possession estimation | Agreement with manually labelled reference intervals |
| Runtime | Throughput (FPS), per-frame latency, GPU memory |

Evaluation will use held-out videos or matches—not random neighbouring frames from the same footage—to reduce data leakage. Results will include the dataset split, hardware, image size, confidence threshold, and model version.

## Development roadmap

- [ ] Establish environment and inspect GPU/CUDA compatibility
- [ ] Select a permitted football dataset and document its licence
- [ ] Build frame extraction and dataset inspection utilities
- [ ] Establish pretrained player-detection baseline
- [ ] Fine-tune / evaluate football-specific ball detection
- [ ] Implement and compare ByteTrack and BoT-SORT
- [ ] Add team-colour clustering
- [ ] Estimate possession and generate movement heatmaps
- [ ] Prototype pitch mapping with homography
- [ ] Add evaluation scripts and reproducible reports
- [ ] Add FastAPI endpoints and a React dashboard
- [ ] Containerize and publish a demonstration

## Data, privacy, and licensing

Do not commit match videos, extracted frames, annotations, or model weights unless their licences explicitly allow it. Keep large assets outside Git and document dataset source, attribution, permitted uses, and any redistribution restrictions. This repository will contain code and documentation; data should be downloaded separately by each user where required.

## Current state

This repository is an initial project scaffold. Detection, tracking, analytics, API, and dashboard functionality are not yet implemented. Implementation and measured results will be added incrementally.

## Author

Ayush Kumar Ray  
GitHub: [@rayayush7204-ctrl](https://github.com/rayayush7204-ctrl)
