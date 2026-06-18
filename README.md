# CV / Anomaly Detection Workspace

This repository contains several related pipelines for object detection and
anomaly detection in images and video:

- **YOLO26n / YOLO-World local inference** (`src/inference.py`, `src/inference_world.py`)
- **EdgeAnomalyCCTV** (`EdgeAnomalyCCTV/`) — a 5-layer edge pipeline with YOLOv8,
  gating filters, and a VLM outlier classifier.
- **Paper cascade** (`Paper/cascade.py`) — a reproducible implementation of the
  cascading multi-agent anomaly-detection framework from arXiv:2601.06204v3.
- **Presentation** (`edgeanomaly-presentation/`) — a Vite + React frontend for
  demoing / visualizing results.

---

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Model weights are downloaded automatically on first use.

---

## YOLO26n / YOLO-World Local Inference

### YOLO26n

```bash
# Default sample image
python src/inference.py

# Your own image/video
python src/inference.py --source path/to/image.jpg
python src/inference.py --source path/to/video.mp4

# Webcam
python src/inference.py --source 0 --show

# Save annotated results
python src/inference.py --source path/to/image.jpg --save
```

Saved results are written to `results/runs/detect/predict/`.

### YOLO-World (Open-Vocabulary)

YOLO-World can detect **custom classes** defined by text, not just COCO's 80 classes.

```bash
# Default classes on the sample image
python src/inference_world.py

# Custom classes
python src/inference_world.py \
  --source path/to/image.jpg \
  --classes "glasses,laptop,coffee cup"

# Video + save
python src/inference_world.py \
  --source path/to/video.mp4 \
  --classes "person,car,bicycle" \
  --save

# Webcam
python src/inference_world.py --source 0 --classes "person,phone" --show
```

### Common inference options

| Flag        | Default                                    | Description                 |
|-------------|--------------------------------------------|-----------------------------|
| `--source`  | `https://ultralytics.com/images/bus.jpg`   | Input source                |
| `--model`   | `weights/yolo/yolo26n.pt` / `weights/yolo/yolov8m-world.pt` | Model weights |
| `--classes` | `person,bus,car` (YOLO-World only)         | Comma-separated class names |
| `--imgsz`   | `640`                                      | Input image size            |
| `--conf`    | `0.25`                                     | Confidence threshold        |
| `--iou`     | `0.45`                                     | NMS IoU threshold           |
| `--save`    | `False`                                    | Save annotated outputs      |
| `--show`    | `False`                                    | Display results in a window |

### Export

```python
from ultralytics import YOLO

model = YOLO("weights/yolo/yolo26n.pt")
model.export(format="onnx")
```

See the [Ultralytics export docs](https://docs.ultralytics.com/modes/export/) for supported formats.

> **Note:** YOLO-World models use a CLIP text encoder and are typically exported differently. Refer to the Ultralytics YOLO-World documentation for export details.

---

## EdgeAnomalyCCTV

A unified framework for outlier anomaly detection supporting both RTSP video
streams and single images.

### Architecture

There are 5 layers:

1. **Ingestion:** Frame buffer from RTSP streams or image inputs.
2. **Detection & Tracking:** YOLOv8 + ByteTrack (for video), YOLOv8 (synthetic tracking for image).
3. **Outlier Filter:** Gates for deduplication, auto-pass, and uncertainty check.
4. **LLM Outlier Classifier:** Qwen3-VL-2B-Instruct outlier-detection async queue.
5. **Render & Alert:** Output overlays, MQTT alerts, or API response.

### Run the pipeline

```bash
# Video / webcam mode
python -u EdgeAnomalyCCTV/src/main.py --mode video

# Single image (default bundled benchmark image)
python EdgeAnomalyCCTV/src/main.py --mode graph

# Single image or video file of your choice
python EdgeAnomalyCCTV/src/main.py --mode graph --input path/to/image.jpg
python EdgeAnomalyCCTV/src/main.py --mode video --input path/to/video.mp4
```

### OOD-Class Benchmark

The framework treats `COCO_CLASSES` as the known closed set.  Anything else
should ideally be flagged as an **OUTLIER** by the LLM classifier.

Helper scripts live in `EdgeAnomalyCCTV/benchmarks/`:

| File | Purpose |
|------|---------|
| `ood_classes.py` | Curated list of OOD classes and helpers. |
| `prepare_openimages_ood.py` | Download / prepare an OOD image benchmark. |
| `run_ood_benchmark.py` | Run the full pipeline on the benchmark and report statistics. |

#### Quick start (no extra installs)

```bash
# Create a small 18-image benchmark from Caltech101 OOD classes
python EdgeAnomalyCCTV/benchmarks/prepare_openimages_ood.py \
    --backend torchvision \
    --torchvision-dataset Caltech101 \
    --max-per-class 3 \
    --output-dir benchmark_data/ood_openimages_small \
    --classes octopus,lobster,scorpion,helicopter,crab,starfish

# Run the EdgeAnomalyCCTV pipeline on it
python EdgeAnomalyCCTV/benchmarks/run_ood_benchmark.py \
    --benchmark-dir benchmark_data/ood_openimages_small \
    --output-dir benchmark_data/ood_results_small
```

#### Larger / more realistic benchmark (OpenImages)

```bash
pip install fiftyone

python EdgeAnomalyCCTV/benchmarks/prepare_openimages_ood.py \
    --backend openimages \
    --max-per-class 20 \
    --output-dir benchmark_data/ood_openimages

python EdgeAnomalyCCTV/benchmarks/run_ood_benchmark.py \
    --benchmark-dir benchmark_data/ood_openimages \
    --output-dir benchmark_data/ood_results
```

#### Use your own images

```bash
python EdgeAnomalyCCTV/benchmarks/prepare_openimages_ood.py \
    --backend local \
    --local-dir /path/to/your/ood_images \
    --output-dir benchmark_data/ood_local
```

#### Interpreting results

`run_ood_benchmark.py` reports:

- **Total images evaluated**
- **Images with detections**
- **Images flagged with outlier**
- **Total object detections**
- **Total objects flagged outlier**
- **Object-level outlier recall** = outlier detections / total detections
- **Image-level outlier recall** = images with at least one outlier / images with detections

For a perfect OOD detector, both recall values should be high (close to 100%).

---

## Paper — Cascading Multi-Agent Anomaly Detection

This folder contains a reproducible implementation of the cascading multi-agent
anomaly detection framework introduced in the paper *"Cascading Multi-Agent
Anomaly Detection in Surveillance Systems via Vision-Language Models and
Embedding-Based Classification"* (arXiv:2601.06204v3).

### Main architecture

1. **Multi-Agent Orchestration**
   - `MessageBroker`: In-process publish-subscribe broker.
   - `EventDrivenAgent`: Triggered by asynchronous silent alarms.
   - `CyclicalMonitoringAgent`: Systematic health checks including Shannon-entropy computation.

2. **Cascading Detection Pipeline** (`CascadingMultiAgentPipeline`)
   - **Stage I:** `CascadableYOLO` (YOLOv8n) object-level detection.
   - **Stage II:** `ReconstructionScorer` convolutional autoencoder.
   - **Stage III:** `VLMReasoner` semantic reasoning with Moondream2, BLIP, LLaVA-7B, or a dummy backend.

3. **System-Level Response**
   - `joint_severity_score(...)` fuses visual and contextual confidence.

### Run the paper pipeline

```bash
# Install dependencies
pip install ultralytics transformers sentence-transformers opencv-python torch moondream

# Run on the bundled benchmark images
python Paper/cascade.py

# Treat every COCO class as an anomaly cue
python Paper/cascade.py --all-classes-anomaly
```

Annotated outputs are saved to `benchmark_results/visualized/`.

### Verification smoke test

```bash
python - <<'PY'
import cv2
from pathlib import Path
from Paper.cascade import (
    CascadableYOLO, ReconstructionScorer, VLMReasoner,
    shannon_entropy, joint_severity_score, CascadingMultiAgentPipeline
)
frame = cv2.imread('benchmark_data/legacy/kitchen_person.jpg')
print(CascadableYOLO(conf_threshold=0.45).infer(frame))
print(ReconstructionScorer().infer(frame))
print(VLMReasoner(mode='moondream', device='cpu').infer(frame))
print('entropy gray:', shannon_entropy(frame * 0 + 128))
print('severity:', joint_severity_score(0.92, 0.84))

pipe = CascadingMultiAgentPipeline(vlm_mode='moondream', device='cpu')
for p in sorted(Path('benchmark_data/legacy').glob('*.jpg')):
    print(f"\n{p.name}:")
    print(pipe.process_frame(cv2.imread(str(p))))
PY
```

---

## Project Layout

```text
.
├── src/                              # YOLO26n / YOLO-World inference scripts
├── EdgeAnomalyCCTV/
│   ├── src/                          # 5-layer edge pipeline
│   └── benchmarks/                   # OOD-class benchmark helpers
├── Paper/
│   ├── cascade.py                    # Cascading multi-agent pipeline
│   └── requirements.txt
├── benchmark_data/                   # Sample images, videos, and generated OOD benchmarks
│   └── legacy/                       # Original bundled sample media (untracked)
├── weights/                          # Model weights
│   └── yolo/                         # YOLO .pt files
├── edgeanomaly-presentation/         # Vite + React presentation frontend
├── requirements.txt                  # Root Python dependencies
└── README.md                         # This file
```
