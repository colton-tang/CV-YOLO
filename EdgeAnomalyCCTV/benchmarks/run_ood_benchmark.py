#!/usr/bin/env python3
"""
Run EdgeAnomalyCCTV on an OOD-class benchmark and report outlier-detection
statistics.

Usage:
    # Run on the default benchmark directory
    python run_ood_benchmark.py

    # Run on a custom directory of images
    python run_ood_benchmark.py --benchmark-dir ./my_ood_images

    # Save annotated images to a results folder
    python run_ood_benchmark.py --save-visualizations --output-dir ./ood_results
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add repository root / EdgeAnomalyCCTV/src to path
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "EdgeAnomalyCCTV" / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT / "EdgeAnomalyCCTV" / "benchmarks"))

from constants import COCO_CLASSES  # noqa: E402
from layer1_ingestion import IngestionLayer  # noqa: E402
from layer2_detection import DetectionTrackingLayer  # noqa: E402
from layer3_filtering import GateOutlierFilterLayer  # noqa: E402
from layer4_llm_classifier import LLMClassifierLayer  # noqa: E402
from layer5_render import RenderAlertLayer  # noqa: E402
from ood_classes import is_ood  # noqa: E402

DEFAULT_BENCHMARK_DIR = ROOT / "benchmark_data" / "ood_openimages"
DEFAULT_OUTPUT_DIR = ROOT / "EdgeAnomalyCCTV" / "ood_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EdgeAnomalyCCTV on OOD benchmark")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=str(DEFAULT_BENCHMARK_DIR),
        help="Directory containing OOD benchmark images (default: benchmark_data/ood_openimages)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save result JSON and optional visualizations",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save annotated output images for each benchmark image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/yolo/yolov8n.pt",
        help="YOLOv8 model path/name (default: weights/yolo/yolov8n.pt)",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save cropped object images for later review or judging",
    )
    return parser.parse_args()


def _collect_image_paths(benchmark_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for p in benchmark_dir.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        # Skip hidden directories (e.g. caches) and any nested cache folder.
        if any(part.startswith(".") for part in p.relative_to(benchmark_dir).parts[:-1]):
            continue
        paths.append(p)
    return sorted(paths)


def _crop_bbox(frame: np.ndarray, bbox: list) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


async def run_benchmark(args: argparse.Namespace) -> dict:
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_image_paths(benchmark_dir)
    if not image_paths:
        print(f"ERROR: no images found in {benchmark_dir}")
        print("Prepare a benchmark first with prepare_openimages_ood.py")
        sys.exit(1)

    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Images to evaluate:  {len(image_paths)}")

    # Initialize pipeline once so the VLM is loaded only one time.
    detection = DetectionTrackingLayer(model_path=args.model, known_classes=COCO_CLASSES)
    filtering = GateOutlierFilterLayer(known_classes=COCO_CLASSES)
    classifier = LLMClassifierLayer(
        queue=filtering.llm_queue,
        known_classes=COCO_CLASSES,
        track_state_db=filtering.track_state_db,
    )
    render = RenderAlertLayer()

    llm_task = asyncio.create_task(classifier.run())

    results = []
    try:
        for img_path in image_paths:
            print(f"\n{'='*60}")
            print(f"Processing: {img_path.relative_to(benchmark_dir)}")

            ingestion = IngestionLayer(mode="IMAGE", source=str(img_path))
            frame_data = ingestion.get_frame()
            if frame_data is None:
                print("  warning: could not read frame, skipping")
                continue

            tracks = detection.process(frame_data)
            await filtering.process(tracks)
            await filtering.llm_queue.join()

            # Ground-truth class from the parent folder name.
            gt_class = img_path.parent.name.lower().strip()

            # Summarize per-image results
            img_result = {
                "image": str(img_path),
                "relative": str(img_path.relative_to(benchmark_dir)),
                "ground_truth_class": gt_class,
                "num_detections": len(tracks),
                "tracks": [],
            }
            outlier_count = 0
            for track in tracks:
                tid = track["track_id"]
                state = filtering.track_state_db.get(tid, {})
                status = state.get("status", "UNKNOWN")
                cls = state.get("display_class") or state.get("class") or track.get("display_class") or track.get("class") or "unknown"
                conf = state.get("confidence")
                if conf is None:
                    conf = track.get("conf", 0.0)

                if status == "OUTLIER":
                    outlier_count += 1

                img_result["tracks"].append({
                    "track_id": tid,
                    "ground_truth_class": gt_class,
                    "yolo_class": track.get("class"),
                    "yolo_display_class": track.get("display_class"),
                    "yolo_conf": track.get("conf"),
                    "status": status,
                    "final_class": cls,
                    "final_confidence": conf,
                    "reason": state.get("reason", ""),
                    "is_ood_class": is_ood(track.get("class", "")),
                })

            img_result["outlier_count"] = outlier_count
            img_result["any_outlier"] = outlier_count > 0
            results.append(img_result)

            if args.save_visualizations:
                vis_dir = output_dir / "visualizations" / img_path.parent.relative_to(benchmark_dir)
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / f"{img_path.stem}_annotated.jpg"
                render.process(
                    filtering.track_state_db,
                    source_type="IMAGE",
                    raw_frame=frame_data["raw_frame"],
                    tracks=tracks,
                )
                # Render layer writes to EdgeAnomalyCCTV/output.jpg for IMAGE mode.
                # Copy that file to the requested visualization path.
                default_out = ROOT / "EdgeAnomalyCCTV" / "output.jpg"
                if default_out.exists():
                    cv2.imwrite(str(vis_path), cv2.imread(str(default_out)))
                    print(f"  visualization saved to: {vis_path}")

            if args.save_crops:
                crop_dir = output_dir / "crops" / img_path.parent.relative_to(benchmark_dir)
                crop_dir.mkdir(parents=True, exist_ok=True)
                for track in tracks:
                    tid = track["track_id"]
                    crop = _crop_bbox(frame_data["raw_frame"], track["bbox"])
                    crop_path = crop_dir / f"{img_path.stem}_{tid}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    # Link crop path back to the track record.
                    for rec in img_result["tracks"]:
                        if rec["track_id"] == tid:
                            rec["crop_path"] = str(crop_path)
                            break
                print(f"  {len(tracks)} crop(s) saved to: {crop_dir}")

            # Reset per-image track state so the next image starts fresh.
            filtering.track_state_db.clear()

    finally:
        classifier.shutdown()
        llm_task.cancel()
        try:
            await asyncio.wait_for(llm_task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    # Aggregate statistics
    total_images = len(results)
    images_with_detections = sum(1 for r in results if r["num_detections"] > 0)
    images_with_outliers = sum(1 for r in results if r["any_outlier"])
    total_detections = sum(r["num_detections"] for r in results)
    total_outliers = sum(r["outlier_count"] for r in results)

    # Ground-truth confusion matrix (treat OUTLIER as the positive class).
    # The folder name of each image is the ground-truth class.
    tp = tn = fp = fn = 0
    for r in results:
        gt_is_ood = is_ood(r.get("ground_truth_class", ""))
        for track in r.get("tracks", []):
            status = track.get("status", "UNKNOWN")
            predicted_outlier = status == "OUTLIER"
            if predicted_outlier and gt_is_ood:
                tp += 1
            elif predicted_outlier and not gt_is_ood:
                fp += 1
            elif not predicted_outlier and gt_is_ood:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "model": args.model,
        "known_classes": COCO_CLASSES,
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "images_with_outliers": images_with_outliers,
        "total_detections": total_detections,
        "total_outliers": total_outliers,
        "outlier_recall": total_outliers / total_detections if total_detections else 0.0,
        "image_level_outlier_recall": images_with_outliers / images_with_detections if images_with_detections else 0.0,
        "ground_truth_metrics": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "results": results,
    }

    summary_path = output_dir / "ood_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    gt = summary["ground_truth_metrics"]
    print(f"\n{'='*60}")
    print("OOD BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total images evaluated       : {total_images}")
    print(f"Images with detections       : {images_with_detections}")
    print(f"Images flagged with outlier  : {images_with_outliers}")
    print(f"Total object detections      : {total_detections}")
    print(f"Total objects flagged outlier: {total_outliers}")
    print(f"Object-level outlier recall  : {summary['outlier_recall']:.2%}")
    print(f"Image-level outlier recall   : {summary['image_level_outlier_recall']:.2%}")
    print(f"\nGround-truth metrics (folder labels):")
    print(f"  Confusion matrix: TP={gt['tp']} TN={gt['tn']} FP={gt['fp']} FN={gt['fn']}")
    print(f"  Accuracy         : {gt['accuracy']:.2%}")
    print(f"  Precision        : {gt['precision']:.2%}")
    print(f"  Recall           : {gt['recall']:.2%}")
    print(f"  F1-score         : {gt['f1_score']:.2%}")
    print(f"\nSummary saved to: {summary_path}")

    return summary


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
