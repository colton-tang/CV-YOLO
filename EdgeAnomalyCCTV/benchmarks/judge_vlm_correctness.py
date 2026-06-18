#!/usr/bin/env python3
"""
Judge the correctness of EdgeAnomalyCCTV's VLM decisions using a second VLM.

This acts as a meta-evaluator: a judge model looks at the same crop and the
first VLM's decision, then says whether that decision is correct.

Usage:
    # Judge a benchmark that was run with --save-crops
    python judge_vlm_correctness.py \
        --summary benchmark_data/ood_results_small/ood_benchmark_summary.json

    # Use a different judge model
    python judge_vlm_correctness.py \
        --summary benchmark_data/ood_results_small/ood_benchmark_summary.json \
        --judge-model Qwen/Qwen3-VL-7B-Instruct

Output:
    {output_dir}/vlm_judgement_report.json
"""

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import cv2
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Add repository root / EdgeAnomalyCCTV/src to path
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "EdgeAnomalyCCTV" / "src"
sys.path.insert(0, str(SRC_DIR))

from constants import COCO_CLASSES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge VLM outlier decisions")
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to ood_benchmark_summary.json produced by run_ood_benchmark.py",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="VLM model to use as judge (default: Qwen/Qwen3-VL-2B-Instruct)",
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
        help="Device for judge model (cpu/cuda/mps). Auto-detected if omitted.",
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


def _load_judge_model(model_id: str, device: str):
    print(f"[JUDGE] Loading judge model {model_id} on {device}...")
    dtype = torch.float16 if device != "cpu" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print("[JUDGE] Judge model loaded.")
    return processor, model


def _build_judge_messages(crop_bgr, gt_class: str, first_vlm_decision: dict, known_classes: list):
    image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")

    status = first_vlm_decision["status"]
    final_class = first_vlm_decision["final_class"]
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


def _run_judge(processor, model, device, messages) -> dict:
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


def _is_known_class(label: str, known_classes: set) -> bool:
    return label.lower().strip() in known_classes


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

    device = _select_device(args.device)
    processor, model = _load_judge_model(args.judge_model, device)

    judgements = []
    total = 0
    correct = 0
    tp = tn = fp = fn = 0

    for img_result in results:
        gt_class = img_result.get("ground_truth_class", "unknown")
        gt_is_known = _is_known_class(gt_class, known_classes)

        for track in img_result.get("tracks", []):
            crop_path = track.get("crop_path")
            if not crop_path or not Path(crop_path).exists():
                print(f"[JUDGE] skipping track {track['track_id']} (no crop saved)")
                continue

            crop = cv2.imread(str(crop_path))
            if crop is None:
                print(f"[JUDGE] skipping track {track['track_id']} (crop unreadable)")
                continue

            status = track.get("status", "UNKNOWN")
            first_is_known = status == "KNOWN"

            messages = _build_judge_messages(crop, gt_class, track, summary.get("known_classes", COCO_CLASSES))
            try:
                judgement = _run_judge(processor, model, device, messages)
            except Exception as exc:
                print(f"[JUDGE] error on track {track['track_id']}: {exc}")
                traceback.print_exc()
                continue

            is_correct = judgement["correct"]
            total += 1
            if is_correct:
                correct += 1

            # Confusion matrix relative to OUTLIER as the positive class.
            if first_is_known and gt_is_known:
                tn += 1
            elif first_is_known and not gt_is_known:
                fn += 1
            elif not first_is_known and gt_is_known:
                fp += 1
            else:
                tp += 1

            judgements.append({
                "image": img_result.get("relative"),
                "track_id": track["track_id"],
                "ground_truth_class": gt_class,
                "ground_truth_is_known": gt_is_known,
                "first_vlm_status": status,
                "judge_correct": is_correct,
                "judge_explanation": judgement["explanation"],
                "crop_path": crop_path,
            })

            print(
                f"[JUDGE] {img_result.get('relative')} | "
                f"gt={gt_class} | first={status} | judge_correct={is_correct}"
            )

    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    report = {
        "summary_file": str(summary_path),
        "judge_model": args.judge_model,
        "total_judged": total,
        "correct": correct,
        "accuracy": accuracy,
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "judgements": judgements,
    }

    report_path = output_dir / "vlm_judgement_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("VLM JUDGEMENT REPORT")
    print(f"{'='*60}")
    print(f"Judge model       : {args.judge_model}")
    print(f"Total judged      : {total}")
    print(f"Correct decisions : {correct}")
    print(f"Accuracy          : {accuracy:.2%}")
    print(f"Precision (OUTLIER): {precision:.2%}")
    print(f"Recall (OUTLIER)  : {recall:.2%}")
    print(f"F1-score          : {f1:.2%}")
    print(f"Confusion matrix  : TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Report saved to   : {report_path}")

    # Release model memory
    try:
        del model
        del processor
    except Exception:
        pass
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
