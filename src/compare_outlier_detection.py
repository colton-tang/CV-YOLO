#!/usr/bin/env python3
"""
Out-of-distribution / outlier detection benchmark.

Compares YOLO26n (closed-vocabulary COCO) against YOLOv8m-World
(open-vocabulary with a restricted class list) on their tendency to
produce false-positive detections for objects outside the target
vocabulary.

For each benchmark image we define a small set of "inlier" classes.
- YOLO26n runs with its full COCO vocabulary.
- YOLO-World runs with only the inlier classes set via set_classes().

Any detection whose predicted label is NOT in the inlier set is counted
as an outlier false positive.
"""

import csv
import json
from pathlib import Path
import requests

import matplotlib.pyplot as plt
from ultralytics import YOLO


BENCHMARK_DATA_DIR = Path("benchmark_data")
RESULTS_DIR = Path("results") / "benchmark_results"

BENCHMARK_IMAGES = [
    {
        "name": "bus",
        "url": "https://ultralytics.com/images/bus.jpg",
        "inlier_classes": ["person", "bus"],
        "description": "People boarding a bus",
    },
    {
        "name": "zidane",
        "url": "https://ultralytics.com/images/zidane.jpg",
        "inlier_classes": ["person", "tie"],
        "description": "Two soccer players",
    },
    {
        "name": "kitchen_person",
        "url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "inlier_classes": ["person"],
        "description": "Kitchen with one person; pots/pans/food are outliers",
    },
    {
        "name": "kitchen_fruit",
        "url": "http://images.cocodataset.org/val2017/000000037777.jpg",
        "inlier_classes": ["refrigerator", "bowl"],
        "description": "Kitchen with fridge and fruit bowl; chairs/oven/table are outliers",
    },
    {
        "name": "street_signs",
        "url": "http://images.cocodataset.org/val2017/000000058636.jpg",
        "inlier_classes": ["stop sign"],
        "description": "Street signs on a pole; stop sign is not actually present",
    },
]


def download_images() -> None:
    BENCHMARK_DATA_DIR.mkdir(exist_ok=True)
    for item in BENCHMARK_IMAGES:
        target = BENCHMARK_DATA_DIR / item["name"]
        target = target.with_suffix(Path(item["url"]).suffix or ".jpg")
        if not target.exists():
            print(f"Downloading {item['name']} ...")
            response = requests.get(item["url"], allow_redirects=True, timeout=60)
            response.raise_for_status()
            target.write_bytes(response.content)
        item["path"] = target


def evaluate_yolo26n(model: YOLO, image_path: Path, inlier_classes: list[str]) -> dict:
    results = model.predict(str(image_path), verbose=False)
    result = results[0]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detections.append({"class": cls_name, "conf": conf, "inlier": cls_name in inlier_classes})

    total = len(detections)
    inlier_dets = [d for d in detections if d["inlier"]]
    outlier_dets = [d for d in detections if not d["inlier"]]

    return {
        "model": "YOLO26n",
        "total": total,
        "inlier_count": len(inlier_dets),
        "outlier_count": len(outlier_dets),
        "outlier_rate": len(outlier_dets) / total if total else 0.0,
        "avg_inlier_conf": sum(d["conf"] for d in inlier_dets) / len(inlier_dets) if inlier_dets else 0.0,
        "avg_outlier_conf": sum(d["conf"] for d in outlier_dets) / len(outlier_dets) if outlier_dets else 0.0,
        "outlier_classes": sorted(set(d["class"] for d in outlier_dets)),
    }


def evaluate_yolo_world(model: YOLO, image_path: Path, inlier_classes: list[str]) -> dict:
    model.set_classes(inlier_classes)
    results = model.predict(str(image_path), verbose=False)
    result = results[0]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            # With set_classes(), every prediction is restricted to the vocabulary.
            detections.append({"class": cls_name, "conf": conf, "inlier": True})

    total = len(detections)

    return {
        "model": "YOLOv8m-World",
        "total": total,
        "inlier_count": total,
        "outlier_count": 0,
        "outlier_rate": 0.0,
        "avg_inlier_conf": sum(d["conf"] for d in detections) / total if total else 0.0,
        "avg_outlier_conf": 0.0,
        "outlier_classes": [],
    }


def plot_summary(summary_rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    images = [row["image"] for row in summary_rows if row["model"] == "YOLO26n"]
    yolo26n_rates = [row["outlier_rate"] for row in summary_rows if row["model"] == "YOLO26n"]
    world_rates = [row["outlier_rate"] for row in summary_rows if row["model"] == "YOLOv8m-World"]

    x = range(len(images))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], yolo26n_rates, width, label="YOLO26n", color="#FF6B6B")
    ax.bar([i + width / 2 for i in x], world_rates, width, label="YOLOv8m-World", color="#4ECDC4")

    ax.set_ylabel("Outlier False-Positive Rate")
    ax.set_title("Outlier Detection Comparison: YOLO26n vs YOLOv8m-World")
    ax.set_xticks(x)
    ax.set_xticklabels(images, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plot_path = RESULTS_DIR / "outlier_rate_comparison.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


def main() -> None:
    download_images()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading models ...")
    yolo26n = YOLO("yolo26n.pt")
    yolo_world = YOLO("yolov8m-world.pt")

    summary_rows = []
    detailed_rows = []

    for item in BENCHMARK_IMAGES:
        print(f"\n--- {item['name']} ({item['description']}) ---")
        print(f"Inlier vocabulary: {item['inlier_classes']}")

        r26 = evaluate_yolo26n(yolo26n, item["path"], item["inlier_classes"])
        rw = evaluate_yolo_world(yolo_world, item["path"], item["inlier_classes"])

        for r in (r26, rw):
            r["image"] = item["name"]
            r["inlier_classes"] = ",".join(item["inlier_classes"])
            summary_rows.append(r)

        print(f"YOLO26n       -> total: {r26['total']}, inliers: {r26['inlier_count']}, "
              f"outliers: {r26['outlier_count']}, outlier_rate: {r26['outlier_rate']:.2%}")
        if r26["outlier_classes"]:
            print(f"  Outlier classes detected: {', '.join(r26['outlier_classes'])}")

        print(f"YOLOv8m-World -> total: {rw['total']}, inliers: {rw['inlier_count']}, "
              f"outliers: {rw['outlier_count']}, outlier_rate: {rw['outlier_rate']:.2%}")

    # Save summary CSV
    csv_path = RESULTS_DIR / "outlier_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "model",
                "inlier_classes",
                "total",
                "inlier_count",
                "outlier_count",
                "outlier_rate",
                "avg_inlier_conf",
                "avg_outlier_conf",
                "outlier_classes",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\nSaved summary CSV to {csv_path}")

    # Save config for reproducibility
    config_path = RESULTS_DIR / "benchmark_config.json"
    config_path.write_text(
        json.dumps(
            [{k: v for k, v in item.items() if k != "path"} for item in BENCHMARK_IMAGES],
            indent=2,
        )
    )

    # Plot
    plot_summary(summary_rows)

    # Final aggregate summary
    print("\n=== Aggregate Summary ===")
    for model_name in ("YOLO26n", "YOLOv8m-World"):
        rows = [r for r in summary_rows if r["model"] == model_name]
        total_dets = sum(r["total"] for r in rows)
        total_outliers = sum(r["outlier_count"] for r in rows)
        overall_rate = total_outliers / total_dets if total_dets else 0.0
        print(f"{model_name}: {total_outliers} outlier detections out of {total_dets} total "
              f"(overall outlier rate: {overall_rate:.2%})")


if __name__ == "__main__":
    main()
