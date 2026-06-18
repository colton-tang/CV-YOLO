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

#### How to run it

All benchmark scripts should be run from the project root with the virtual
environment activated:

```bash
cd /Users/t/CV
source venv/bin/activate
```

##### Quick start (no extra installs)

Uses **Caltech101** via `torchvision` and finishes in a few minutes.

```bash
# 1. Prepare a small 18-image benchmark (6 OOD classes, 3 images each)
python EdgeAnomalyCCTV/benchmarks/prepare_openimages_ood.py \
    --backend torchvision \
    --torchvision-dataset Caltech101 \
    --max-per-class 3 \
    --output-dir benchmark_data/ood_openimages_small \
    --classes octopus,lobster,scorpion,helicopter,crab,starfish

# 2. Run the EdgeAnomalyCCTV pipeline on it
python EdgeAnomalyCCTV/benchmarks/run_ood_benchmark.py \
    --benchmark-dir benchmark_data/ood_openimages_small \
    --output-dir benchmark_data/ood_results_small
```

Results are saved to `benchmark_data/ood_results_small/ood_benchmark_summary.json`.

##### Larger / more realistic benchmark (OpenImages)

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

##### Use your own images

Organize your images in folders named by class:

```text
/path/to/your/ood_images/
├── crab/
├── helicopter/
└── scorpion/
```

```bash
python EdgeAnomalyCCTV/benchmarks/prepare_openimages_ood.py \
    --backend local \
    --local-dir /path/to/your/ood_images \
    --output-dir benchmark_data/ood_local

python EdgeAnomalyCCTV/benchmarks/run_ood_benchmark.py \
    --benchmark-dir benchmark_data/ood_local \
    --output-dir benchmark_data/ood_results_local
```

##### Single-image sanity check

Run one OOD image through the main pipeline:

```bash
python EdgeAnomalyCCTV/src/main.py --mode graph \
    --input benchmark_data/ood_openimages_small/helicopter/helicopter_5528.jpg
```

#### Interpreting results

`run_ood_benchmark.py` prints a summary like this:

```text
Total images evaluated       : 18
Images with detections       : 13
Images flagged with outlier  : 13
Total object detections      : 17
Total objects flagged outlier: 17
Object-level outlier recall  : 100.00%
Image-level outlier recall   : 100.00%
```

- **Object-level outlier recall** — what fraction of detected objects were flagged as OUTLIER.
- **Image-level outlier recall** — what fraction of images with detections had at least one OUTLIER.

For a good OOD detector, both values should be high, because every detected OOD
object should be classified as an outlier.

#### Verifying correctness

The benchmark runner now compares the VLM's `KNOWN`/`OUTLIER` decision against
the ground-truth class (taken from the image's parent folder).  It reports a
confusion matrix and standard classification metrics:

```text
Ground-truth metrics (folder labels):
  Confusion matrix: TP=17 TN=0 FP=0 FN=0
  Accuracy         : 100.00%
  Precision        : 100.00%
  Recall           : 100.00%
  F1-score         : 100.00%
```

- **TP** — OOD object flagged as OUTLIER
- **TN** — COCO object flagged as KNOWN
- **FP** — COCO object wrongly flagged as OUTLIER
- **FN** — OOD object wrongly flagged as KNOWN

#### Optional: judge VLM decisions

`judge_vlm_correctness.py` evaluates each saved crop against the ground-truth
folder name.  It always reports two deterministic metrics:

* **Class-match accuracy** — does the VLM's `final_class` exactly match the
  benchmark folder name (ground-truth class)?
* **Decision accuracy / precision / recall / F1** — is the `KNOWN`/`OUTLIER`
  decision correct relative to the ground-truth OOD status?

Run the fast deterministic comparison without loading any model:

```bash
python EdgeAnomalyCCTV/benchmarks/judge_vlm_correctness.py \
    --summary benchmark_data/ood_results_small/ood_benchmark_summary.json \
    --skip-llm-judge
```

You can also ask a second VLM to double-check the first VLM's decisions.
Two backends are supported: a local Qwen3-VL model or the Kimi API.

First make sure the benchmark was run with `--save-crops`:

```bash
python EdgeAnomalyCCTV/benchmarks/run_ood_benchmark.py \
    --benchmark-dir benchmark_data/ood_openimages_small \
    --output-dir benchmark_data/ood_results_small \
    --save-crops
```

Then run the judge with a local model:

```bash
python EdgeAnomalyCCTV/benchmarks/judge_vlm_correctness.py \
    --summary benchmark_data/ood_results_small/ood_benchmark_summary.json \
    --judge-backend local \
    --judge-model Qwen/Qwen3-VL-2B-Instruct
```

Or run the judge with the Kimi API:

```bash
python EdgeAnomalyCCTV/benchmarks/judge_vlm_correctness.py \
    --summary benchmark_data/ood_results_small/ood_benchmark_summary.json \
    --judge-backend kimi
```

Credentials are read from a project-root `.env` file by default (e.g.
`KIMI_API_KEY`, `KIMI_API_BASE`, `KIMI_MODEL_NAME`).  You can still override
them with environment variables or with `--kimi-api-key`, `--kimi-api-base`,
and `--kimi-model-name`.

The report is written to `vlm_judgement_report.json` next to the summary file.

> **Note:** a small local judge (e.g. Qwen3-VL-2B) can be unreliable.  For
> trustworthy human-like review, use a larger model or the Kimi API.
> `kimi-code` is a reasoning model, so the script sets a large token budget
> to accommodate its reasoning before the final JSON answer.

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
