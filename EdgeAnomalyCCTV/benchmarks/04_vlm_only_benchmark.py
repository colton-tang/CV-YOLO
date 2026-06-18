#!/usr/bin/env python3
"""
VLM-only OOD benchmark using the Kimi API.

This baseline bypasses the object detector and asks a vision-language model
(Kimi) directly whether each benchmark image belongs to the closed COCO set
or is an outlier/anomaly.  The resulting summary is produced in the same
schema as 02_run_ood_benchmark.py so the matrix orchestrator can consume it.

Usage:
    python 04_vlm_only_benchmark.py --benchmark-dir benchmark_data/ood_openimages_small \
        --output-dir benchmark_data/benchmark_matrix_results/vlm_only

Credentials are read from the project-root .env file or environment variables:
    KIMI_API_KEY, KIMI_API_BASE, KIMI_MODEL_NAME, KIMI_USER_AGENT
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import requests

# Add repository root / EdgeAnomalyCCTV paths so we can import shared code.
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "EdgeAnomalyCCTV" / "src"
BENCHMARK_DIR = ROOT / "EdgeAnomalyCCTV" / "benchmarks"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

import importlib  # noqa: E402

from constants import COCO_CLASSES  # noqa: E402

_ood = importlib.import_module("00_ood_classes")
is_ood = _ood.is_ood

KIMI_DEFAULT_BASE = "https://api.kimi.com/coding"
KIMI_DEFAULT_MODEL = "kimi-code"
KIMI_DEFAULT_USER_AGENT = "claude-code/0.1.0"


def _load_dotenv(dotenv_path: Path | None = None) -> None:
    """Load environment variables from a .env file (no external deps)."""
    if dotenv_path is None:
        dotenv_path = ROOT / ".env"
    if not dotenv_path.exists():
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if (
                    (value.startswith('"') and value.endswith('"'))
                    or (value.startswith("'") and value.endswith("'"))
                ):
                    value = value[1:-1]
                if key and os.environ.get(key) is None:
                    os.environ[key] = value
    except Exception as exc:
        print(f"[VLM-ONLY] warning: could not load {dotenv_path}: {exc}")


# Load project-root .env by default so credentials are available without exporting.
_load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a VLM-only OOD benchmark with the Kimi API"
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=str(ROOT / "benchmark_data" / "ood_openimages_small"),
        help="Directory containing OOD benchmark images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the VLM-only summary JSON",
    )
    parser.add_argument(
        "--kimi-api-key",
        type=str,
        default=os.getenv("KIMI_API_KEY"),
        help="Kimi API key (default: KIMI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--kimi-api-base",
        type=str,
        default=os.getenv("KIMI_API_BASE", KIMI_DEFAULT_BASE),
        help=f"Kimi API base URL (default: {KIMI_DEFAULT_BASE})",
    )
    parser.add_argument(
        "--kimi-model-name",
        type=str,
        default=os.getenv("KIMI_MODEL_NAME", KIMI_DEFAULT_MODEL),
        help=f"Kimi model name (default: {KIMI_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--kimi-user-agent",
        type=str,
        default=os.getenv("KIMI_USER_AGENT", KIMI_DEFAULT_USER_AGENT),
        help=f"User-Agent header for Kimi API (default: {KIMI_DEFAULT_USER_AGENT})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent Kimi API calls (default: 5)",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=512,
        help="Resize the longest image edge to this value before sending (default: 512)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per Kimi API call in seconds (default: 60)",
    )
    return parser.parse_args()


def _collect_image_paths(benchmark_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for p in benchmark_dir.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        if any(part.startswith(".") for part in p.relative_to(benchmark_dir).parts[:-1]):
            continue
        paths.append(p)
    return sorted(paths)


def _resize_frame(frame: np.ndarray, max_size: int) -> np.ndarray:
    """Resize frame so its longest edge is at most max_size."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_image_to_base64(image_path: Path, max_size: int) -> str | None:
    """Read an image, resize it, and return a JPEG base64 data URL."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None
    frame = _resize_frame(frame, max_size)
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_messages(known_classes_text: str, image_data_url: str) -> list[dict]:
    system_prompt = (
        "You are a strict visual classifier for an open-vocabulary anomaly-detection benchmark. "
        "Given an image and a closed set of known classes, decide whether the main object "
        "in the image belongs to the known set or is an outlier/anomaly."
    )
    user_prompt = (
        f"Known classes (closed set): {known_classes_text}.\n\n"
        "Look at the image and decide whether the main object belongs to one of the known classes "
        "or is an outlier/anomaly. "
        'Respond with ONLY a JSON object in this exact format:\n'
        '{"type": "KNOWN" or "OUTLIER", "class": "short class name", '
        '"confidence": "high/medium/low", "reason": "one sentence"}'
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _parse_vlm_response(response: str) -> dict:
    response = response.strip()
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            obj_type = str(parsed.get("type", "OUTLIER")).upper()
            if obj_type not in {"KNOWN", "OUTLIER"}:
                upper = response.upper()
                if "KNOWN" in upper and "OUTLIER" not in upper:
                    obj_type = "KNOWN"
                else:
                    obj_type = "OUTLIER"
            return {
                "type": obj_type,
                "class": str(parsed.get("class", "unknown")),
                "confidence": str(parsed.get("confidence", "low")).lower(),
                "reason": str(parsed.get("reason", "")),
            }
        except json.JSONDecodeError:
            pass

    upper = response.upper()
    if "KNOWN" in upper and "OUTLIER" not in upper:
        obj_type = "KNOWN"
    elif "OUTLIER" in upper or "ANOMALY" in upper or "UNKNOWN" in upper:
        obj_type = "OUTLIER"
    else:
        obj_type = "OUTLIER"

    return {
        "type": obj_type,
        "class": "unknown",
        "confidence": "low",
        "reason": response,
    }


def _call_kimi(
    messages: list[dict],
    api_key: str,
    base_url: str,
    model_name: str,
    user_agent: str,
    timeout: int,
) -> dict:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": user_agent,
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 256,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return _parse_vlm_response(content)


def _create_classifier(
    args: argparse.Namespace,
) -> Callable[[Path], dict]:
    if not args.kimi_api_key:
        print(
            "ERROR: --kimi-api-key or KIMI_API_KEY environment variable is required "
            "for the VLM-only Kimi backend"
        )
        sys.exit(1)

    print(
        f"[VLM-ONLY] Using Kimi API at {args.kimi_api_base} "
        f"(model={args.kimi_model_name}, concurrency={args.concurrency})"
    )

    known_classes_text = ", ".join(COCO_CLASSES)

    def classify(image_path: Path) -> dict:
        image_data_url = _encode_image_to_base64(image_path, args.max_image_size)
        if image_data_url is None:
            return {
                "type": "OUTLIER",
                "class": "unknown",
                "confidence": "low",
                "reason": "Failed to read or encode image",
            }
        messages = _build_messages(known_classes_text, image_data_url)
        return _call_kimi(
            messages,
            args.kimi_api_key,
            args.kimi_api_base,
            args.kimi_model_name,
            args.kimi_user_agent,
            args.timeout,
        )

    return classify


async def _classify_one(
    image_path: Path,
    gt_class: str,
    classify_fn: Callable[[Path], dict],
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        return await asyncio.to_thread(classify_fn, image_path)


async def run_benchmark(args: argparse.Namespace) -> dict:
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_image_paths(benchmark_dir)
    if not image_paths:
        print(f"ERROR: no images found in {benchmark_dir}")
        sys.exit(1)

    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Images to evaluate:  {len(image_paths)}")
    print(f"VLM backend:         Kimi ({args.kimi_model_name})")

    classify_fn = _create_classifier(args)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    tasks = [
        _classify_one(
            img_path,
            img_path.parent.name.lower().strip(),
            classify_fn,
            semaphore,
        )
        for img_path in image_paths
    ]
    vlm_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    tp = tn = fp = fn = 0
    class_match_count = 0
    total_decisions = 0
    images_with_outliers = 0
    total_outliers = 0

    for img_path, vlm_result in zip(image_paths, vlm_results):
        gt_class = img_path.parent.name.lower().strip()
        gt_is_ood = is_ood(gt_class)

        if isinstance(vlm_result, Exception):
            print(f"[VLM-ONLY] error on {img_path}: {vlm_result}")
            traceback.print_exc()
            vlm_result = {
                "type": "OUTLIER",
                "class": "unknown",
                "confidence": "low",
                "reason": f"API error: {vlm_result}",
            }

        status = vlm_result["type"]
        final_class = vlm_result["class"]
        predicted_outlier = status == "OUTLIER"
        normalized_final = final_class.lower().strip()
        normalized_gt = gt_class.lower().strip()

        class_match = (
            not predicted_outlier
            and normalized_final == normalized_gt
            and normalized_final not in {"", "unknown"}
        )

        total_decisions += 1
        if predicted_outlier:
            images_with_outliers += 1
            total_outliers += 1

        if predicted_outlier and gt_is_ood:
            tp += 1
        elif predicted_outlier and not gt_is_ood:
            fp += 1
        elif not predicted_outlier and gt_is_ood:
            fn += 1
        else:
            tn += 1

        if class_match:
            class_match_count += 1

        img_result = {
            "image": str(img_path),
            "relative": str(img_path.relative_to(benchmark_dir)),
            "ground_truth_class": gt_class,
            "num_detections": 1,
            "tracks": [
                {
                    "track_id": 1,
                    "ground_truth_class": gt_class,
                    "yolo_class": "vlm_only",
                    "yolo_display_class": "vlm_only",
                    "yolo_conf": 1.0,
                    "status": status,
                    "final_class": final_class,
                    "final_confidence": vlm_result.get("confidence", "low"),
                    "reason": vlm_result.get("reason", ""),
                    "is_ood_class": predicted_outlier,
                }
            ],
            "outlier_count": 1 if predicted_outlier else 0,
            "any_outlier": predicted_outlier,
        }
        results.append(img_result)

    total_images = len(results)
    images_with_detections = total_decisions

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    class_match_accuracy = class_match_count / total_decisions if total_decisions else 0.0

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "model": args.kimi_model_name,
        "evaluation_mode": "vlm_only",
        "known_classes": COCO_CLASSES,
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "total_detections": total_decisions,
        "detector_non_coco_count": total_outliers,
        "detector_non_coco_rate": total_outliers / total_decisions if total_decisions else 0.0,
        "images_with_outliers": images_with_outliers,
        "total_outliers": total_outliers,
        "outlier_recall": total_outliers / total_decisions if total_decisions else 0.0,
        "image_level_outlier_recall": images_with_outliers / total_decisions if total_decisions else 0.0,
        "ground_truth_metrics": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "class_match_count": class_match_count,
            "class_match_accuracy": class_match_accuracy,
        },
        "results": results,
    }

    summary_path = output_dir / "ood_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("VLM-ONLY BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total images evaluated       : {total_images}")
    print(f"Images with VLM decisions    : {images_with_detections}")
    print(f"Total VLM decisions          : {total_decisions}")
    print(f"Objects flagged outlier      : {total_outliers}")
    print(f"Object-level outlier recall  : {summary['outlier_recall']:.2%}")
    print(f"Image-level outlier recall   : {summary['image_level_outlier_recall']:.2%}")
    print(f"\nGround-truth metrics (folder labels):")
    print(f"  Confusion matrix: TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  Accuracy         : {accuracy:.2%}")
    print(f"  Precision        : {precision:.2%}")
    print(f"  Recall           : {recall:.2%}")
    print(f"  F1-score         : {f1:.2%}")
    print(f"  Class-match acc. : {class_match_accuracy:.2%}")
    print(f"\nSummary saved to: {summary_path}")

    return summary


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
