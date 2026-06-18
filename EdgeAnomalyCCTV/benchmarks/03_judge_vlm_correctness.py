#!/usr/bin/env python3
"""
Judge the correctness of EdgeAnomalyCCTV's VLM decisions.

Two judgement modes are supported:

1. Deterministic comparison (no LLM call):
   - Compare the benchmark folder name (ground-truth class) with the
     VLM-generated `final_class` and report class-match accuracy.
   - Compare the KNOWN/OUTLIER decision with the ground-truth OOD status
     and report decision-level accuracy / precision / recall / F1.

2. LLM-as-a-judge:
   - A second VLM (local Qwen3-VL or Kimi API) looks at the crop and the
     first VLM's decision, then says whether that decision is correct.

Usage:
    # Deterministic comparison only (fast, no API/model load)
    python 03_judge_vlm_correctness.py \
        --summary benchmark_data/ood_results_small/ood_benchmark_summary.json \
        --skip-llm-judge

    # Judge with a local VLM
    python 03_judge_vlm_correctness.py \
        --summary benchmark_data/ood_results_small/ood_benchmark_summary.json \
        --judge-backend local \
        --judge-model Qwen/Qwen3-VL-2B-Instruct

    # Judge with the Kimi API (credentials are read from .env by default)
    python 03_judge_vlm_correctness.py \
        --summary benchmark_data/ood_results_small/ood_benchmark_summary.json \
        --judge-backend kimi

Output:
    {output_dir}/vlm_judgement_report.json
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
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Add repository root / EdgeAnomalyCCTV/src to path
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "EdgeAnomalyCCTV" / "src"
sys.path.insert(0, str(SRC_DIR))

from constants import COCO_CLASSES  # noqa: E402


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
        print(f"[JUDGE] warning: could not load {dotenv_path}: {exc}")


# Load project-root .env by default so credentials are available without exporting.
_load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge VLM outlier decisions")
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to ood_benchmark_summary.json produced by 02_run_ood_benchmark.py",
    )
    parser.add_argument(
        "--judge-backend",
        type=str,
        choices=["local", "kimi"],
        default="local",
        help="Backend for the judge VLM (default: local)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Local VLM model to use as judge (default: Qwen/Qwen3-VL-2B-Instruct)",
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
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save judgement report (default: same as summary directory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for local judge model (cpu/cuda/mps). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip the LLM-as-a-judge step and only compute deterministic metrics",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=None,
        help="Max concurrent LLM judge calls. Default: 5 for kimi, 1 for local.",
    )
    return parser.parse_args()


def _select_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_local_judge(model_id: str, device: str):
    print(f"[JUDGE] Loading local judge model {model_id} on {device}...")
    dtype = torch.float16 if device != "cpu" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print("[JUDGE] Local judge model loaded.")
    return processor, model


def _build_judge_messages_local(crop_bgr, gt_class: str, first_vlm_decision: dict, known_classes: list):
    image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")

    status = first_vlm_decision.get("status", "UNKNOWN")
    final_class = first_vlm_decision.get("final_class", "unknown")
    reason = first_vlm_decision.get("reason", "")
    yolo_class = first_vlm_decision.get("yolo_class", "unknown")
    yolo_conf = first_vlm_decision.get("yolo_conf", 0.0)

    known_text = ", ".join(known_classes)
    gt_in_known = gt_class.lower().strip() in {c.lower() for c in known_classes}
    expected_status = "KNOWN" if gt_in_known else "OUTLIER"

    system_prompt = (
        "You are a strict classification evaluator. You must decide whether a "
        "KNOWN/OUTLIER decision is correct, using ONLY the provided known-classes "
        "list. Do not rely on your own prior knowledge of COCO or any other dataset."
    )

    user_prompt = (
        f"Known classes (closed set): {known_text}.\n\n"
        f"Ground-truth class of the object in the image: '{gt_class}'.\n"
        f"Because '{gt_class}' is {'IN' if gt_in_known else 'NOT in'} the known classes list, "
        f"the CORRECT decision is: {expected_status}.\n\n"
        f"Another model made this decision:\n"
        f"  - status: {status}\n"
        f"  - class: {final_class}\n"
        f"  - YOLO initial guess: {yolo_class} (conf={yolo_conf:.2f})\n"
        f"  - reason: {reason}\n\n"
        f"Question: Is '{status}' the correct decision?\n"
        "Answer with ONLY a JSON object:\n"
        '{"correct": true or false, "explanation": "one sentence"}'
    )

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _parse_judge_response(response: str) -> dict:
    response = response.strip()
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return {
                "correct": bool(parsed.get("correct", False)),
                "explanation": parsed.get("explanation", ""),
            }
        except json.JSONDecodeError:
            pass

    # Fallback heuristic
    upper = response.upper()
    return {
        "correct": "TRUE" in upper and "FALSE" not in upper,
        "explanation": response,
    }


def _run_local_judge(processor, model, device, messages) -> dict:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return _parse_judge_response(response)


def _encode_image_to_base64(crop_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", crop_bgr)
    if not success:
        raise RuntimeError("Failed to encode crop to JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def _build_judge_messages_kimi(gt_class: str, first_vlm_decision: dict) -> tuple[str, str]:
    status = first_vlm_decision.get("status", "UNKNOWN")
    final_class = first_vlm_decision.get("final_class", "unknown")
    reason = first_vlm_decision.get("reason", "")

    system_prompt = (
        "You are a strict text-based classification evaluator. "
        "Decide whether a predicted class label matches a ground-truth class label. "
        "Respond with ONLY a JSON object, no reasoning, no markdown."
    )

    user_prompt = (
        f"Ground-truth class: '{gt_class}'.\n"
        f"Model classification: '{final_class}' (status={status}).\n"
        f"Model reasoning: {reason}\n\n"
        "Does the model's class label match the ground-truth class? "
        "Be lenient with synonyms and descriptors (e.g., 'blue crab' matches 'crab').\n"
        'Answer with ONLY this exact JSON format: '
        '{"correct": true or false, "class_match": true or false, "explanation": "one sentence"}'
    )

    return system_prompt, user_prompt


def _call_kimi_judge(
    api_key: str,
    base_url: str,
    model_name: str,
    user_agent: str,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": user_agent,
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2048,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return _parse_judge_response(content)


def _create_judge(args: argparse.Namespace):
    """Return a callable judge_fn(crop_bgr, gt_class, track) -> dict."""
    if args.skip_llm_judge:
        return None

    if args.judge_backend == "local":
        device = _select_device(args.device)
        processor, model = _load_local_judge(args.judge_model, device)

        def judge_fn(crop_bgr, gt_class: str, track: dict) -> dict:
            messages = _build_judge_messages_local(
                crop_bgr, gt_class, track, COCO_CLASSES
            )
            return _run_local_judge(processor, model, device, messages)

        return judge_fn

    if args.judge_backend == "kimi":
        if not args.kimi_api_key:
            print(
                "ERROR: --kimi-api-key or KIMI_API_KEY environment variable is required "
                "for Kimi backend"
            )
            sys.exit(1)

        print(
            f"[JUDGE] Using Kimi API judge at {args.kimi_api_base} "
            f"(model={args.kimi_model_name})"
        )

        def judge_fn(crop_bgr, gt_class: str, track: dict) -> dict:
            system_prompt, user_prompt = _build_judge_messages_kimi(gt_class, track)
            return _call_kimi_judge(
                api_key=args.kimi_api_key,
                base_url=args.kimi_api_base,
                model_name=args.kimi_model_name,
                user_agent=args.kimi_user_agent,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        return judge_fn

    raise ValueError(f"Unknown judge backend: {args.judge_backend}")


def _normalize(label: str) -> str:
    return label.lower().strip()


def _is_known_class(label: str, known_classes: set) -> bool:
    return _normalize(label) in known_classes


def _normalize_track_for_judge(track: dict, img_result: dict) -> dict:
    """Normalize framework and detector_only tracks into a common schema."""
    normalized = dict(track)

    if "predicted_is_non_coco" in track:
        # detector_only schema
        normalized["status"] = "OUTLIER" if track.get("predicted_is_non_coco") else "KNOWN"
        normalized["final_class"] = track.get("yolo_display_class") or track.get("yolo_class") or "unknown"
        normalized["reason"] = ""
        normalized["yolo_class"] = track.get("yolo_class", "unknown")
        normalized["yolo_conf"] = track.get("yolo_conf", 0.0)
    else:
        # framework schema
        normalized["status"] = track.get("status", "UNKNOWN")
        normalized["final_class"] = track.get("final_class", "unknown")
        normalized["reason"] = track.get("reason", "")
        normalized["yolo_class"] = track.get("yolo_class", "unknown")
        normalized["yolo_conf"] = track.get("yolo_conf", 0.0)

    # Ensure a crop path exists. If the benchmark didn't save crops, try to
    # generate one on the fly from the bbox and original image.
    if not normalized.get("crop_path") or not Path(normalized["crop_path"]).exists():
        bbox = track.get("bbox")
        image_path = img_result.get("image")
        if bbox and image_path and Path(image_path).exists():
            try:
                frame = cv2.imread(str(image_path))
                if frame is not None:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
                    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        out_dir = Path(image_path).parent.parent / "judge_crops"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = out_dir / f"{Path(image_path).stem}_{track.get('track_id', 'unknown')}.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        normalized["crop_path"] = str(crop_path)
            except Exception:
                pass

    return normalized


async def _judge_one(
    crop,
    gt_class: str,
    norm_track: dict,
    judge_fn: Callable,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run a single LLM judge call under a concurrency semaphore."""
    async with semaphore:
        return await asyncio.to_thread(judge_fn, crop, gt_class, norm_track)


async def _run_judgements(
    judge_items: list[tuple],
    judge_fn: Callable,
    concurrency: int,
) -> list:
    """Run all judge calls concurrently with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _judge_one(crop, gt_class, norm_track, judge_fn, semaphore)
        for (_, crop, gt_class, norm_track) in judge_items
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: summary file not found: {summary_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    known_classes = set(c.lower() for c in summary.get("known_classes", COCO_CLASSES))
    results = summary.get("results", [])

    judge_fn = _create_judge(args)

    judgements = []
    judge_items = []  # (entry, crop, gt_class, norm_track) for concurrent LLM judging
    total = 0

    # Deterministic class-match counts
    class_match_count = 0

    # Deterministic KNOWN/OUTLIER decision counts
    decision_correct = 0
    tp = tn = fp = fn = 0

    # LLM judge agreement counts
    judge_total = 0
    judge_correct_count = 0
    judge_agrees_with_class_match = 0
    judge_agrees_with_decision = 0

    for img_result in results:
        gt_class = img_result.get("ground_truth_class", "unknown")
        gt_is_known = _is_known_class(gt_class, known_classes)

        for track in img_result.get("tracks", []):
            norm_track = _normalize_track_for_judge(track, img_result)

            crop_path = norm_track.get("crop_path")
            if not crop_path or not Path(crop_path).exists():
                print(f"[JUDGE] skipping track {track.get('track_id', 'unknown')} (no crop saved)")
                continue

            crop = cv2.imread(str(crop_path))
            if crop is None:
                print(f"[JUDGE] skipping track {track.get('track_id', 'unknown')} (crop unreadable)")
                continue

            status = norm_track.get("status", "UNKNOWN")
            final_class = norm_track.get("final_class", "unknown")
            first_is_known = status == "KNOWN"

            # Deterministic class-match metric
            normalized_final = _normalize(final_class)
            normalized_gt = _normalize(gt_class)
            class_match = normalized_final == normalized_gt and normalized_final not in {"", "unknown"}

            # Deterministic KNOWN/OUTLIER decision correctness
            decision_is_correct = (
                (status == "OUTLIER" and not gt_is_known)
                or (status == "KNOWN" and gt_is_known)
            )

            total += 1
            if class_match:
                class_match_count += 1
            if decision_is_correct:
                decision_correct += 1

            # Confusion matrix relative to OUTLIER as the positive class.
            if first_is_known and gt_is_known:
                tn += 1
            elif first_is_known and not gt_is_known:
                fn += 1
            elif not first_is_known and gt_is_known:
                fp += 1
            else:
                tp += 1

            entry = {
                "image": img_result.get("relative"),
                "track_id": track.get("track_id", "unknown"),
                "ground_truth_class": gt_class,
                "ground_truth_is_known": gt_is_known,
                "first_vlm_status": status,
                "first_vlm_class": final_class,
                "class_match": class_match,
                "decision_correct": decision_is_correct,
                "crop_path": crop_path,
                "judge_correct": None,
                "judge_class_match": None,
                "judge_explanation": None,
            }

            judgements.append(entry)
            if judge_fn is not None:
                judge_items.append((entry, crop, gt_class, norm_track))

    # Run LLM judge calls concurrently with bounded concurrency.
    if judge_fn is not None and judge_items:
        concurrency = args.judge_concurrency
        if concurrency is None:
            concurrency = 5 if args.judge_backend == "kimi" else 1
        print(
            f"[JUDGE] Running {len(judge_items)} LLM judge calls "
            f"concurrently (max {concurrency} at a time)..."
        )
        judge_results = asyncio.run(_run_judgements(judge_items, judge_fn, concurrency))

        for (entry, _, _, _), judgement in zip(judge_items, judge_results):
            if isinstance(judgement, Exception):
                print(f"[JUDGE] error on track {entry['track_id']}: {judgement}")
                traceback.print_exc()
                judgement = {"correct": False, "explanation": f"ERROR: {judgement}"}

            judge_total += 1
            judge_correct = bool(judgement.get("correct", False))
            judge_class_match = bool(judgement.get("class_match", judge_correct))
            judge_explanation = judgement.get("explanation", "")

            if judge_correct:
                judge_correct_count += 1
            if judge_class_match == entry["class_match"]:
                judge_agrees_with_class_match += 1
            if judge_correct == entry["decision_correct"]:
                judge_agrees_with_decision += 1

            entry["judge_correct"] = judge_correct
            entry["judge_class_match"] = judge_class_match
            entry["judge_explanation"] = judge_explanation

    # Print all judgement results.
    for entry in judgements:
        print(
            f"[JUDGE] {entry['image']} | "
            f"gt={entry['ground_truth_class']} | "
            f"first={entry['first_vlm_status']}/{entry['first_vlm_class']} | "
            f"class_match={entry['class_match']} | "
            f"decision_correct={entry['decision_correct']}"
            + (
                f" | judge_correct={entry['judge_correct']}"
                if entry["judge_correct"] is not None
                else ""
            )
        )

    # Deterministic metrics
    class_match_accuracy = class_match_count / total if total else 0.0
    decision_accuracy = decision_correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # LLM judge metrics
    judge_accuracy = judge_correct_count / judge_total if judge_total else None
    judge_class_agreement = judge_agrees_with_class_match / judge_total if judge_total else None
    judge_decision_agreement = judge_agrees_with_decision / judge_total if judge_total else None

    report = {
        "summary_file": str(summary_path),
        "judge_backend": "none" if args.skip_llm_judge else args.judge_backend,
        "judge_model": args.judge_model if args.judge_backend == "local" else args.kimi_model_name,
        "kimi_api_base": args.kimi_api_base if args.judge_backend == "kimi" else None,
        "total_judged": total,
        "deterministic_metrics": {
            "class_match_count": class_match_count,
            "class_match_accuracy": class_match_accuracy,
            "decision_correct_count": decision_correct,
            "decision_accuracy": decision_accuracy,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "llm_judge_metrics": {
            "judge_total": judge_total,
            "judge_correct_count": judge_correct_count,
            "judge_accuracy": judge_accuracy,
            "judge_class_agreement": judge_class_agreement,
            "judge_decision_agreement": judge_decision_agreement,
        } if judge_fn is not None else None,
        "judgements": judgements,
    }

    report_path = output_dir / "vlm_judgement_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("VLM JUDGEMENT REPORT")
    print(f"{'='*60}")
    print(f"Judge backend     : {report['judge_backend']}")
    if report['judge_backend'] != "none":
        print(f"Judge model       : {report['judge_model']}")
    print(f"Total judged      : {total}")
    print(f"Class-match acc.  : {class_match_accuracy:.2%} ({class_match_count}/{total})")
    print(f"Decision accuracy : {decision_accuracy:.2%} ({decision_correct}/{total})")
    print(f"Precision (OUTLIER): {precision:.2%}")
    print(f"Recall (OUTLIER)  : {recall:.2%}")
    print(f"F1-score          : {f1:.2%}")
    print(f"Confusion matrix  : TP={tp} TN={tn} FP={fp} FN={fn}")
    if report["llm_judge_metrics"] is not None:
        print(f"\nLLM judge accuracy       : {judge_accuracy:.2%}" if judge_accuracy is not None else "")
        print(f"LLM agrees w/ class match: {judge_class_agreement:.2%}" if judge_class_agreement is not None else "")
        print(f"LLM agrees w/ decision   : {judge_decision_agreement:.2%}" if judge_decision_agreement is not None else "")
    print(f"Report saved to   : {report_path}")

    # Release model memory if local backend was used
    if args.judge_backend == "local" and not args.skip_llm_judge:
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
