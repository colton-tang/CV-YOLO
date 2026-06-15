#!/usr/bin/env python3
"""
YOLO-World (yolov8m-world) local inference script.

YOLO-World is an open-vocabulary detector: you can define custom classes
with text instead of being limited to COCO's 80 classes.

Usage:
    python inference_world.py --source path/to/image.jpg --classes "person,car,dog"
    python inference_world.py --source path/to/video.mp4 --classes "person,car" --save
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO-World inference")
    parser.add_argument(
        "--source",
        type=str,
        default="https://ultralytics.com/images/bus.jpg",
        help="Image, video, directory, URL, or webcam index",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m-world.pt",
        help="Path or name of the YOLO-World model (default: yolov8m-world.pt)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="person,bus,car",
        help="Comma-separated list of custom class names (default: person,bus,car)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated results to results/runs/detect/predict",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in a window (not available in headless environments)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"Setting custom classes: {class_names}")
    model.set_classes(class_names)

    print(f"Running inference on: {args.source}")
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        show=args.show,
        project="results/runs",
        name="predict_world",
        exist_ok=True,
        verbose=True,
    )

    # Print a concise summary for each result
    for i, result in enumerate(results):
        boxes = result.boxes
        num_detections = len(boxes) if boxes is not None else 0
        print(f"Result {i + 1}: {num_detections} detection(s)")
        if num_detections > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names.get(cls_id, "unknown")
                xyxy = box.xyxy[0].tolist()
                print(f"  - {name}: {conf:.2f} at {[round(v, 1) for v in xyxy]}")


if __name__ == "__main__":
    main()
