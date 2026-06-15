# YOLO26n / YOLO-World Local Deployment

This folder contains a minimal local deployment setup for:
- [Ultralytics YOLO26n](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/) (`yolov8m-world.pt`)

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

## YOLO26n

### Run inference on the default sample image

```bash
python src/inference.py
```

### Run inference on your own image/video

```bash
python src/inference.py --source path/to/image.jpg
python src/inference.py --source path/to/video.mp4
```

### Use the webcam

```bash
python src/inference.py --source 0 --show
```

### Save annotated results

```bash
python src/inference.py --source path/to/image.jpg --save
```

Saved results are written to `results/runs/detect/predict/`.

---

## YOLO-World (Open-Vocabulary)

YOLO-World can detect **custom classes** defined by text, not just COCO's 80 classes.

### Run with default classes on the sample image

```bash
python src/inference_world.py
```

### Detect custom classes

```bash
python src/inference_world.py \
  --source path/to/image.jpg \
  --classes "glasses,laptop,coffee cup"
```

### Run on video and save output

```bash
python src/inference_world.py \
  --source path/to/video.mp4 \
  --classes "person,car,bicycle" \
  --save
```

### Use the webcam

```bash
python src/inference_world.py --source 0 --classes "person,phone" --show
```

---

## Outlier Detection Benchmark

Compare how often YOLO26n and YOLOv8m-World produce false-positive detections for objects outside a target vocabulary.

### Run the benchmark

```bash
python src/compare_outlier_detection.py
```

This will:
1. Download 5 benchmark images into `benchmark_data/`
2. Run both models on each image
3. Count inlier vs outlier detections
4. Save results to `results/benchmark_results/outlier_summary.csv`
5. Generate `results/benchmark_results/outlier_rate_comparison.png`

### How it works

For each image, a small set of "inlier" classes is defined (e.g., `person`, `bus`).
- **YOLO26n** runs with its full COCO vocabulary. Detections whose predicted class is not in the inlier set are counted as outlier false positives.
- **YOLOv8m-World** runs with only the inlier classes set via `set_classes()`. By construction, it should not produce outlier detections.

### Interpreting results

| Metric | Meaning |
|--------|---------|
| `total` | Total number of detections |
| `inlier_count` | Detections matching the target vocabulary |
| `outlier_count` | Detections outside the target vocabulary |
| `outlier_rate` | `outlier_count / total` |
| `avg_inlier_conf` | Average confidence of inlier detections |
| `avg_outlier_conf` | Average confidence of outlier detections |

### Add your own images

Edit `src/compare_outlier_detection.py` and add entries to `BENCHMARK_IMAGES`:

```python
{
    "name": "my_scene",
    "url": "https://example.com/my_image.jpg",
    "inlier_classes": ["person", "car"],
    "description": "Street scene with pedestrians and vehicles",
}
```

---

## Common Options

| Flag        | Default                                    | Description                 |
|-------------|--------------------------------------------|-----------------------------|
| `--source`  | `https://ultralytics.com/images/bus.jpg`   | Input source                |
| `--model`   | `yolo26n.pt` / `yolov8m-world.pt`          | Model weights               |
| `--classes` | `person,bus,car` (YOLO-World only)         | Comma-separated class names |
| `--imgsz`   | `640`                                      | Input image size            |
| `--conf`    | `0.25`                                     | Confidence threshold        |
| `--iou`     | `0.45`                                     | NMS IoU threshold           |
| `--save`    | `False`                                    | Save annotated outputs      |
| `--show`    | `False`                                    | Display results in a window |

---

## Export for other formats

To export YOLO26n to ONNX, TensorRT, etc.:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx")
```

See the [Ultralytics export docs](https://docs.ultralytics.com/modes/export/) for supported formats.

> **Note:** YOLO-World models use a CLIP text encoder and are typically exported differently. Refer to the Ultralytics YOLO-World documentation for export details.
