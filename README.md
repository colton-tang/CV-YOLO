# CV / Anomaly Detection Workspace

This repository contains several related pipelines for object detection and
anomaly detection in images and video:

- **YOLO26n / YOLO-World local inference** (`src/inference.py`, `src/inference_world.py`)
- **EdgeAnomalyCCTV** (`EdgeAnomalyCCTV/`) — a 5-layer edge pipeline with YOLOv8,
  gating filters, and a VLM outlier classifier.
- **Benchmarking suite** (`EdgeAnomalyCCTV/benchmarks/`) — comprehensive OOD-class
  evaluation, matrix comparisons, and VLM-as-a-judge verification.
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

## Benchmarking

The benchmarking suite is the fastest way to evaluate how well the pipeline
detects **out-of-distribution (OOD)** objects — anything that is not in the
COCO closed set.

### Quick start: one-shot benchmark

```bash
# Full pipeline on the small Caltech101 benchmark, with Kimi judge
python EdgeAnomalyCCTV/benchmarks/main.py --judge-backend kimi

# Use a local judge model instead
python EdgeAnomalyCCTV/benchmarks/main.py --judge-backend local \
    --judge-model Qwen/Qwen3-VL-2B-Instruct

# Skip preparation if the benchmark already exists
python EdgeAnomalyCCTV/benchmarks/main.py \
    --skip-prepare \
    --judge-backend kimi
```

### Benchmark scripts

| File | Purpose |
|------|---------|
| `00_ood_classes.py` | Curated list of OOD classes and helpers. |
| `01_prepare_openimages_ood.py` | Download / prepare an OOD image benchmark. |
| `02_run_ood_benchmark.py` | Run the full pipeline on the benchmark and report statistics. |
| `03_judge_vlm_correctness.py` | Independently judge VLM decisions (deterministic or LLM-as-a-judge). |
| `main.py` | One-shot orchestrator that runs all of the above. |
| `run_benchmark_matrix.py` | Compare multiple detector/framework variants side-by-side. |

### Detailed benchmark guide

#### 1. Prepare a benchmark

**Small / fast — Caltech101 via torchvision:**

```bash
python EdgeAnomalyCCTV/benchmarks/01_prepare_openimages_ood.py \
    --backend torchvision \
    --torchvision-dataset Caltech101 \
    --max-per-class 3 \
    --output-dir benchmark_data/ood_openimages_small \
    --classes octopus,lobster,scorpion,helicopter,crab,starfish
```

**Larger / more realistic — OpenImages:**

```bash
pip install fiftyone

python EdgeAnomalyCCTV/benchmarks/01_prepare_openimages_ood.py \
    --backend openimages \
    --max-per-class 20 \
    --output-dir benchmark_data/ood_openimages
```

**Use your own images:**

Organize your images in folders named by class:

```text
/path/to/your/ood_images/
├── crab/
├── helicopter/
└── scorpion/
```

```bash
python EdgeAnomalyCCTV/benchmarks/01_prepare_openimages_ood.py \
    --backend local \
    --local-dir /path/to/your/ood_images \
    --output-dir benchmark_data/ood_local
```

#### 2. Run the benchmark

```bash
python EdgeAnomalyCCTV/benchmarks/02_run_ood_benchmark.py \
    --benchmark-dir benchmark_data/ood_openimages_small \
    --output-dir benchmark_data/ood_results_small
```

Results are saved to `benchmark_data/ood_results_small/ood_benchmark_summary.json`.

#### 3. Judge VLM decisions (optional)

```bash
# Fast deterministic comparison (no model load)
python EdgeAnomalyCCTV/benchmarks/03_judge_vlm_correctness.py \
    --summary benchmark_data/ood_results_small/ood_benchmark_summary.json \
    --skip-llm-judge

# Local judge model
python EdgeAnomalyCCTV/benchmarks/03_judge_vlm_correctness.py \
    --summary benchmark_data/ood_results_small/ood_benchmark_summary.json \
    --judge-backend local \
    --judge-model Qwen/Qwen3-VL-2B-Instruct

# Kimi API judge
python EdgeAnomalyCCTV/benchmarks/03_judge_vlm_correctness.py \
    --summary benchmark_data/ood_results_small/ood_benchmark_summary.json \
    --judge-backend kimi
```

### Benchmark matrix: compare variants

`run_benchmark_matrix.py` runs the same benchmark across multiple
 detector / framework combinations and produces a single combined summary:

```bash
python EdgeAnomalyCCTV/benchmarks/run_benchmark_matrix.py \
    --judge-backend kimi
```

Available variants:

| Variant | Model | Mode |
|---------|-------|------|
| `yolov8n_framework` | `yolov8n.pt` | Full EdgeAnomalyCCTV framework |
| `yolo_world_only` | `yolov8m-world.pt` | Detector-only evaluation |
| `yolo_world_framework` | `yolov8m-world.pt` | Full framework with YOLO-World |

Customize the matrix:

```bash
# Run only selected variants
python EdgeAnomalyCCTV/benchmarks/run_benchmark_matrix.py \
    --variants yolov8n_framework,yolo_world_framework \
    --judge-backend kimi

# Larger OpenImages benchmark
python EdgeAnomalyCCTV/benchmarks/run_benchmark_matrix.py \
    --backend openimages \
    --max-per-class 20 \
    --benchmark-dir benchmark_data/ood_openimages \
    --output-dir benchmark_data/benchmark_matrix_results \
    --judge-backend kimi

# Skip steps when re-running
python EdgeAnomalyCCTV/benchmarks/run_benchmark_matrix.py \
    --skip-prepare \
    --skip-judge
```

The combined summary is written to
`benchmark_data/benchmark_matrix_results/benchmark_matrix_summary.json`.

### Interpreting results

`02_run_ood_benchmark.py` prints a summary like this:

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

Ground-truth metrics are also reported (folder labels treated as truth):

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

### Single-image sanity check

Run one OOD image through the main pipeline:

```bash
python EdgeAnomalyCCTV/src/main.py --mode graph \
    --input benchmark_data/ood_openimages_small/helicopter/helicopter_5528.jpg
```

---

## Project Layout

```text
.
├── src/                              # YOLO26n / YOLO-World inference scripts
├── EdgeAnomalyCCTV/
│   ├── src/                          # 5-layer edge pipeline
│   └── benchmarks/                   # OOD-class benchmark helpers + matrix runner
├── benchmark_data/                   # Sample images, videos, and generated OOD benchmarks
│   ├── benchmark_matrix_results/     # Combined matrix benchmark outputs
│   ├── legacy/                       # Original bundled sample media
│   ├── ood_openimages/               # Full OpenImages OOD benchmark
│   ├── ood_openimages_small/         # Small Caltech101 OOD benchmark
│   └── ood_results_small/            # Results for the small benchmark
├── weights/                          # Model weights
│   ├── clip/                         # CLIP weights
│   └── yolo/                         # YOLO .pt files
├── edgeanomaly-presentation/         # Vite + React presentation frontend
├── requirements.txt                  # Root Python dependencies
└── README.md                         # This file
```
