#!/usr/bin/env python3
"""
Run EdgeAnomalyCCTV on an OOD-class benchmark and report outlier-detection
statistics.

Usage:
    # Run on the default benchmark directory
    python 02_run_ood_benchmark.py

    # Run on a custom directory of images
    python 02_run_ood_benchmark.py --benchmark-dir ./my_ood_images

    # Save annotated images to a results folder
    python 02_run_ood_benchmark.py --save-visualizations --output-dir ./ood_results
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add repository root / EdgeAnomalyCCTV/src to path
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "EdgeAnomalyCCTV" / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT / "EdgeAnomalyCCTV" / "benchmarks"))

import importlib  # noqa: E402

from constants import COCO_CLASSES  # noqa: E402
from layer1_ingestion import IngestionLayer  # noqa: E402
from layer2_detection import DetectionTrackingLayer  # noqa: E402
from layer3_filtering import GateOutlierFilterLayer  # noqa: E402
from layer4_llm_classifier import LLMClassifierLayer  # noqa: E402
from layer5_render import RenderAlertLayer  # noqa: E402

_ood_classes = importlib.import_module("00_ood_classes")
is_ood = _ood_classes.is_ood

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
        "--evaluation-mode",
        type=str,
        choices=["framework", "detector_only"],
        default="framework",
        help="Benchmark evaluation mode (default: framework)",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save cropped object images for later review or judging",
    )
    parser.add_argument(
        "--top-k-per-image",
        type=int,
        default=1,
        help="Keep only the top-K highest-confidence detections per image (0 = keep all, default: 1)",
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


def _apply_top_k_per_image(tracks: list[dict], k: int) -> list[dict]:
    """Keep only the top-K detections by confidence for a single image."""
    if k <= 0 or len(tracks) <= k:
        return tracks
    return sorted(tracks, key=lambda t: t.get("conf", 0.0), reverse=True)[:k]


async def run_benchmark(args: argparse.Namespace) -> dict:
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_image_paths(benchmark_dir)
    if not image_paths:
        print(f"ERROR: no images found in {benchmark_dir}")
        print("Prepare a benchmark first with 01_prepare_openimages_ood.py")
        sys.exit(1)

    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Images to evaluate:  {len(image_paths)}")
    print(f"Evaluation mode:     {args.evaluation_mode}")

    detection = DetectionTrackingLayer(model_path=args.model, known_classes=COCO_CLASSES)
    filtering = None
    classifier = None
    render = None
    llm_task = None

    if args.evaluation_mode == "framework":
        # Initialize the full pipeline once so the VLM is loaded only one time.
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
            tracks = _apply_top_k_per_image(tracks, args.top_k_per_image)

            # Ground-truth class from the parent folder name.
            gt_class = img_path.parent.name.lower().strip()

            img_result = {
                "image": str(img_path),
                "relative": str(img_path.relative_to(benchmark_dir)),
                "ground_truth_class": gt_class,
                "num_detections": len(tracks),
                "tracks": [],
            }

            if args.evaluation_mode == "framework":
                await filtering.process(tracks)
                await filtering.llm_queue.join()

                outlier_count = 0
                for track in tracks:
                    tid = track["track_id"]
                    state = filtering.track_state_db.get(tid, {})
                    status = state.get("status", "UNKNOWN")
                    cls = (
                        state.get("display_class")
                        or state.get("class")
                        or track.get("display_class")
                        or track.get("class")
                        or "unknown"
                    )
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
            else:
                non_coco_count = 0
                for track in tracks:
                    detected_label = track.get("display_class") or track.get("class") or "unknown"
                    predicted_is_non_coco = is_ood(detected_label)
                    if predicted_is_non_coco:
                        non_coco_count += 1

                    img_result["tracks"].append({
                        "track_id": track["track_id"],
                        "ground_truth_class": gt_class,
                        "yolo_class": track.get("class"),
                        "yolo_display_class": track.get("display_class"),
                        "yolo_conf": track.get("conf"),
                        "bbox": track.get("bbox"),
                        "predicted_is_non_coco": predicted_is_non_coco,
                    })

                img_result["non_coco_detection_count"] = non_coco_count
                img_result["any_non_coco_detection"] = non_coco_count > 0

            results.append(img_result)

            if args.save_visualizations:
                vis_dir = output_dir / "visualizations" / img_path.parent.relative_to(benchmark_dir)
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / f"{img_path.stem}_annotated.jpg"
                if args.evaluation_mode == "framework":
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
                else:
                    annotated = frame_data["raw_frame"].copy()
                    for track in tracks:
                        x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
                        label = track.get("display_class") or track.get("class") or "unknown"
                        color = (0, 200, 255) if is_ood(label) else (0, 255, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                    cv2.imwrite(str(vis_path), annotated)
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
            if filtering is not None:
                filtering.track_state_db.clear()

    finally:
        if classifier is not None and llm_task is not None:
            classifier.shutdown()
            llm_task.cancel()
            try:
                await asyncio.wait_for(llm_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    total_images = len(results)
    images_with_detections = sum(1 for r in results if r["num_detections"] > 0)
    total_detections = sum(r["num_detections"] for r in results)

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "model": args.model,
        "evaluation_mode": args.evaluation_mode,
        "known_classes": COCO_CLASSES,
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "total_detections": total_detections,
        "results": results,
    }

    if args.evaluation_mode == "framework":
        images_with_outliers = sum(1 for r in results if r["any_outlier"])
        total_outliers = sum(r["outlier_count"] for r in results)

        # Ground-truth confusion matrix (treat OUTLIER as the positive class).
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

        summary.update({
            "images_with_outliers": images_with_outliers,
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
        })
    else:
        images_with_non_coco = sum(1 for r in results if r["any_non_coco_detection"])
        total_non_coco_detections = sum(r["non_coco_detection_count"] for r in results)
        summary.update({
            "images_with_non_coco_detections": images_with_non_coco,
            "total_non_coco_detections": total_non_coco_detections,
            "detection_level_non_coco_rate": (
                total_non_coco_detections / total_detections if total_detections else 0.0
            ),
            "image_level_non_coco_rate": (
                images_with_non_coco / images_with_detections if images_with_detections else 0.0
            ),
        })

    summary_path = output_dir / "ood_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("OOD BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total images evaluated       : {total_images}")
    print(f"Images with detections       : {images_with_detections}")
    print(f"Total object detections      : {total_detections}")
    if args.evaluation_mode == "framework":
        gt = summary["ground_truth_metrics"]
        print(f"Images flagged with outlier  : {summary['images_with_outliers']}")
        print(f"Total objects flagged outlier: {summary['total_outliers']}")
        print(f"Object-level outlier recall  : {summary['outlier_recall']:.2%}")
        print(f"Image-level outlier recall   : {summary['image_level_outlier_recall']:.2%}")
        print(f"\nGround-truth metrics (folder labels):")
        print(f"  Confusion matrix: TP={gt['tp']} TN={gt['tn']} FP={gt['fp']} FN={gt['fn']}")
        print(f"  Accuracy         : {gt['accuracy']:.2%}")
        print(f"  Precision        : {gt['precision']:.2%}")
        print(f"  Recall           : {gt['recall']:.2%}")
        print(f"  F1-score         : {gt['f1_score']:.2%}")
    else:
        print(f"Images with non-COCO label   : {summary['images_with_non_coco_detections']}")
        print(f"Non-COCO detections          : {summary['total_non_coco_detections']}")
        print(f"Detection-level non-COCO rate: {summary['detection_level_non_coco_rate']:.2%}")
        print(f"Image-level non-COCO rate    : {summary['image_level_non_coco_rate']:.2%}")
    print(f"\nSummary saved to: {summary_path}")

    return summary


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
