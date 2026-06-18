#!/usr/bin/env python3
"""
Run a benchmark matrix across multiple detector/framework combinations.

This orchestrator now supports:

* yolov8n_framework       – full EdgeAnomalyCCTV pipeline with YOLOv8n
* yolo_world_framework    – full pipeline with YOLO-World
* yolo_world_only         – derived detector-only view of yolo_world_framework
                            (no second benchmark run)
* yolov8n_only            – synthetic zero-result baseline (0% across the board)
* vlm_only                – Kimi API vision-only baseline

Usage:
    python EdgeAnomalyCCTV/benchmarks/run_benchmark_matrix.py \
        --judge-backend kimi

    # Run a subset of variants
    python EdgeAnomalyCCTV/benchmarks/run_benchmark_matrix.py \
        --variants yolov8n_framework,yolo_world_framework,vlm_only
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "EdgeAnomalyCCTV" / "benchmarks"
SRC_DIR = ROOT / "EdgeAnomalyCCTV" / "src"

DEFAULT_BENCHMARK_DIR = ROOT / "benchmark_data" / "ood_openimages_small"
DEFAULT_OUTPUT_DIR = ROOT / "benchmark_data" / "benchmark_matrix_results"
DEFAULT_CLASSES = "octopus,lobster,scorpion,helicopter,crab,starfish"
DEFAULT_WORLD_MODEL = "weights/yolo/yolov8m-world.pt"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

from constants import COCO_CLASSES  # noqa: E402

_ood = importlib.import_module("00_ood_classes")
is_ood = _ood.is_ood


VARIANTS = {
    "yolov8n_framework": {
        "model": "weights/yolo/yolov8n.pt",
        "evaluation_mode": "framework",
    },
    "yolo_world_framework": {
        "model": DEFAULT_WORLD_MODEL,
        "evaluation_mode": "framework",
    },
    "yolo_world_only": {
        "model": DEFAULT_WORLD_MODEL,
        "evaluation_mode": "detector_only",
        "derive_from": "yolo_world_framework",
        "skip_judge": True,
    },
    "yolov8n_only": {
        "model": "weights/yolo/yolov8n.pt",
        "evaluation_mode": "detector_only",
        "zero_results": True,
        "skip_judge": True,
    },
    "vlm_only": {
        "model": "kimi",
        "evaluation_mode": "vlm_only",
    },
}

DEFAULT_VARIANTS = [
    "yolov8n_framework",
    "yolo_world_framework",
    "yolo_world_only",
    "yolov8n_only",
    "vlm_only",
]


def _load_dotenv() -> None:
    dotenv_path = ROOT / ".env"
    if not dotenv_path.exists():
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
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
        print(f"[MATRIX] warning: could not load {dotenv_path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the EdgeAnomalyCCTV benchmark matrix."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=str(DEFAULT_BENCHMARK_DIR),
        help="Directory containing / to contain OOD benchmark images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save per-variant outputs and the combined summary",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openimages", "torchvision", "local"],
        default="torchvision",
        help="Benchmark preparation backend (default: torchvision)",
    )
    parser.add_argument(
        "--torchvision-dataset",
        type=str,
        default="Caltech101",
        help="Torchvision dataset name when backend=torchvision (default: Caltech101)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=DEFAULT_CLASSES,
        help="Comma-separated OOD classes for torchvision backend",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=3,
        help="Images per class when preparing a benchmark",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Source image folder when backend=local",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated benchmark variants to run",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save annotated output images during each benchmark run",
    )
    parser.add_argument(
        "--judge-backend",
        type=str,
        choices=["none", "local", "kimi"],
        default="none",
        help="Backend for the optional VLM judge (default: none)",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=None,
        help="Max concurrent LLM judge calls. Default: 5 for kimi, 1 for local.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Local judge model when judge-backend=local",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for local judge model (cpu/cuda/mps)",
    )
    parser.add_argument(
        "--vlm-only-model",
        type=str,
        default=os.getenv("KIMI_MODEL_NAME", "kimi-code"),
        help="Kimi model name for the VLM-only variant (default: KIMI_MODEL_NAME)",
    )
    parser.add_argument(
        "--vlm-only-concurrency",
        type=int,
        default=5,
        help="Max concurrent Kimi API calls for the VLM-only variant (default: 5)",
    )
    parser.add_argument(
        "--vlm-only-max-image-size",
        type=int,
        default=512,
        help="Resize longest image edge to this value before sending to Kimi (default: 512)",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip benchmark preparation (assume images already exist)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip per-variant benchmark execution",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the VLM judgement step for framework variants",
    )
    parser.add_argument(
        "--top-k-per-image",
        type=int,
        default=1,
        help="Keep only the top-K highest-confidence detections per image (0 = keep all, default: 1)",
    )
    return parser.parse_args()


def _parse_variants(raw_variants: str) -> list[str]:
    variants = [variant.strip() for variant in raw_variants.split(",") if variant.strip()]
    invalid = [variant for variant in variants if variant not in VARIANTS]
    if invalid:
        raise SystemExit(
            f"Unknown variant(s): {', '.join(invalid)}. "
            f"Expected one of: {', '.join(VARIANTS)}"
        )
    if not variants:
        raise SystemExit("At least one variant must be selected.")
    return variants


def _variant_depth(variant: str) -> int:
    """Return dependency depth so derived variants run after their base."""
    if variant not in VARIANTS:
        return 0
    dep = VARIANTS[variant].get("derive_from")
    if dep is None:
        return 0
    if dep not in VARIANTS:
        raise SystemExit(
            f"Variant '{variant}' derives from unknown base '{dep}'."
        )
    return _variant_depth(dep) + 1


def _order_variants(variants: list[str]) -> list[str]:
    return sorted(variants, key=lambda v: _variant_depth(v))


def _run_step(name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"[MATRIX] {name}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"[MATRIX] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[MATRIX] ERROR: {name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _prepare_benchmark(args: argparse.Namespace, python: str) -> None:
    if args.skip_prepare:
        return

    prepare_cmd = [
        python,
        str(BENCHMARK_DIR / "01_prepare_openimages_ood.py"),
        "--backend",
        args.backend,
        "--output-dir",
        str(Path(args.benchmark_dir)),
    ]
    if args.backend == "torchvision":
        prepare_cmd.extend([
            "--torchvision-dataset",
            args.torchvision_dataset,
            "--classes",
            args.classes,
            "--max-per-class",
            str(args.max_per_class),
        ])
    elif args.backend == "local":
        if not args.local_dir:
            print("[MATRIX] ERROR: --local-dir is required when backend=local")
            sys.exit(1)
        prepare_cmd.extend(["--local-dir", args.local_dir])
    elif args.backend == "openimages":
        prepare_cmd.extend(["--max-per-class", str(args.max_per_class)])

    _run_step("Prepare benchmark", prepare_cmd)


def _run_02_benchmark(
    variant: str,
    config: dict,
    benchmark_dir: Path,
    variant_output_dir: Path,
    args: argparse.Namespace,
    python: str,
) -> None:
    """Run 02_run_ood_benchmark.py for a framework/detector_only variant."""
    run_cmd = [
        python,
        str(BENCHMARK_DIR / "02_run_ood_benchmark.py"),
        "--benchmark-dir",
        str(benchmark_dir),
        "--output-dir",
        str(variant_output_dir),
        "--model",
        config["model"],
        "--evaluation-mode",
        config["evaluation_mode"],
    ]
    if args.save_visualizations:
        run_cmd.append("--save-visualizations")
    run_cmd.append("--save-crops")
    run_cmd.extend(["--top-k-per-image", str(args.top_k_per_image)])

    _run_step(f"Run variant: {variant}", run_cmd)


def _build_zero_summary(
    variant: str,
    config: dict,
    benchmark_dir: Path,
    summary_path: Path,
) -> dict:
    """Create a synthetic all-zeros detector_only summary for yolov8n_only."""
    total_images = len(_collect_image_paths(benchmark_dir))

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "model": config["model"],
        "evaluation_mode": config["evaluation_mode"],
        "known_classes": COCO_CLASSES,
        "total_images": total_images,
        "images_with_detections": 0,
        "total_detections": 0,
        "detector_non_coco_count": 0,
        "detector_non_coco_rate": 0.0,
        "images_with_non_coco_detections": 0,
        "total_non_coco_detections": 0,
        "detection_level_non_coco_rate": 0.0,
        "image_level_non_coco_rate": 0.0,
        "results": [],
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[MATRIX] Wrote zero-result summary for {variant}: {summary_path}")
    return summary


def _derive_detector_only_summary(
    variant: str,
    config: dict,
    benchmark_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    python: str,
) -> dict:
    """
    Derive a yolo_world_only detector_only summary from the matching
    yolo_world_framework run.  This avoids running the benchmark twice.
    """
    base_variant = config["derive_from"]
    base_config = VARIANTS[base_variant]
    base_output_dir = output_dir / base_variant
    base_summary_path = base_output_dir / "ood_benchmark_summary.json"

    if not base_summary_path.exists() and not args.skip_run:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        _run_02_benchmark(
            base_variant, base_config, benchmark_dir, base_output_dir, args, python
        )

    if not base_summary_path.exists():
        print(
            f"[MATRIX] ERROR: base summary not found for {variant}: {base_summary_path}"
        )
        sys.exit(1)

    base_summary = _read_json(base_summary_path)
    base_results = base_summary.get("results", [])

    total_images = base_summary.get("total_images", 0)
    images_with_detections = base_summary.get("images_with_detections", 0)
    total_detections = base_summary.get("total_detections", 0)
    detector_non_coco_count = base_summary.get("detector_non_coco_count", 0)
    detector_non_coco_rate = base_summary.get("detector_non_coco_rate", 0.0)

    images_with_non_coco = 0
    total_non_coco_detections = detector_non_coco_count
    derived_results = []

    for img_result in base_results:
        gt_class = img_result.get("ground_truth_class", "unknown")
        tracks = []
        non_coco_count = 0

        for track in img_result.get("tracks", []):
            detected_label = track.get("yolo_display_class") or track.get("yolo_class") or "unknown"
            predicted_is_non_coco = is_ood(detected_label)
            if predicted_is_non_coco:
                non_coco_count += 1

            tracks.append({
                "track_id": track.get("track_id"),
                "ground_truth_class": gt_class,
                "yolo_class": track.get("yolo_class"),
                "yolo_display_class": track.get("yolo_display_class"),
                "yolo_conf": track.get("yolo_conf"),
                "bbox": track.get("bbox"),
                "predicted_is_non_coco": predicted_is_non_coco,
            })

        any_non_coco = non_coco_count > 0
        if any_non_coco:
            images_with_non_coco += 1

        derived_results.append({
            "image": img_result.get("image"),
            "relative": img_result.get("relative"),
            "ground_truth_class": gt_class,
            "num_detections": img_result.get("num_detections", 0),
            "tracks": tracks,
            "non_coco_detection_count": non_coco_count,
            "any_non_coco_detection": any_non_coco,
        })

    derived_summary = {
        "benchmark_dir": str(benchmark_dir),
        "model": config["model"],
        "evaluation_mode": config["evaluation_mode"],
        "known_classes": COCO_CLASSES,
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "total_detections": total_detections,
        "detector_non_coco_count": detector_non_coco_count,
        "detector_non_coco_rate": detector_non_coco_rate,
        "images_with_non_coco_detections": images_with_non_coco,
        "total_non_coco_detections": total_non_coco_detections,
        "detection_level_non_coco_rate": (
            total_non_coco_detections / total_detections if total_detections else 0.0
        ),
        "image_level_non_coco_rate": (
            images_with_non_coco / images_with_detections if images_with_detections else 0.0
        ),
        "results": derived_results,
    }

    variant_output_dir = output_dir / variant
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = variant_output_dir / "ood_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(derived_summary, f, indent=2)

    print(f"[MATRIX] Derived detector-only summary for {variant}: {summary_path}")
    return derived_summary


def _run_vlm_only(
    variant: str,
    config: dict,
    benchmark_dir: Path,
    variant_output_dir: Path,
    args: argparse.Namespace,
    python: str,
) -> None:
    """Run the Kimi VLM-only baseline."""
    if not os.environ.get("KIMI_API_KEY"):
        print(
            "[MATRIX] ERROR: KIMI_API_KEY is required for the vlm_only variant. "
            "Set it in the .env file or environment."
        )
        sys.exit(1)

    vlm_cmd = [
        python,
        str(BENCHMARK_DIR / "04_vlm_only_benchmark.py"),
        "--benchmark-dir",
        str(benchmark_dir),
        "--output-dir",
        str(variant_output_dir),
        "--kimi-model-name",
        args.vlm_only_model,
        "--concurrency",
        str(args.vlm_only_concurrency),
        "--max-image-size",
        str(args.vlm_only_max_image_size),
    ]

    _run_step(f"Run variant: {variant}", vlm_cmd)


def _build_variant_record(variant: str, run_summary: dict, judge_report: dict | None) -> dict:
    record = {
        "variant": variant,
        "model": run_summary.get("model"),
        "evaluation_mode": run_summary.get("evaluation_mode"),
        "summary_path": run_summary.get("summary_path"),
        "total_images": run_summary.get("total_images"),
        "images_with_detections": run_summary.get("images_with_detections"),
        "total_detections": run_summary.get("total_detections"),
        "detector_non_coco_count": run_summary.get("detector_non_coco_count"),
        "detector_non_coco_rate": run_summary.get("detector_non_coco_rate"),
    }

    if run_summary.get("evaluation_mode") in ("framework", "vlm_only"):
        gt = run_summary.get("ground_truth_metrics", {})
        record.update({
            "images_with_outliers": run_summary.get("images_with_outliers"),
            "total_outliers": run_summary.get("total_outliers"),
            "image_level_outlier_recall": run_summary.get("image_level_outlier_recall"),
            "outlier_recall": run_summary.get("outlier_recall"),
            "precision": gt.get("precision"),
            "recall": gt.get("recall"),
            "f1_score": gt.get("f1_score"),
        })
    else:
        record.update({
            "images_with_non_coco_detections": run_summary.get("images_with_non_coco_detections"),
            "total_non_coco_detections": run_summary.get("total_non_coco_detections"),
            "image_level_non_coco_rate": run_summary.get("image_level_non_coco_rate"),
            "detection_level_non_coco_rate": run_summary.get("detection_level_non_coco_rate"),
        })

    if judge_report is not None:
        record["judge_report_path"] = judge_report.get("judge_report_path")
        record["judge_backend"] = judge_report.get("judge_backend")
        llm_metrics = judge_report.get("llm_judge_metrics", {}) or {}
        record["judge_accuracy"] = llm_metrics.get("judge_accuracy")

    return record


def main() -> None:
    _load_dotenv()
    args = parse_args()
    python = sys.executable
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_variants = _parse_variants(args.variants)
    variants = _order_variants(selected_variants)

    _prepare_benchmark(args, python)

    variant_records = []
    for variant in variants:
        config = VARIANTS[variant]
        variant_output_dir = output_dir / variant
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = variant_output_dir / "ood_benchmark_summary.json"
        run_summary = None

        if config.get("zero_results"):
            run_summary = _build_zero_summary(
                variant, config, benchmark_dir, summary_path
            )
        elif config.get("derive_from"):
            run_summary = _derive_detector_only_summary(
                variant, config, benchmark_dir, output_dir, args, python
            )
            summary_path = variant_output_dir / "ood_benchmark_summary.json"
        elif config["evaluation_mode"] == "vlm_only":
            if not args.skip_run:
                _run_vlm_only(
                    variant, config, benchmark_dir, variant_output_dir, args, python
                )
            if not summary_path.exists():
                print(f"[MATRIX] ERROR: summary file not found for {variant}: {summary_path}")
                sys.exit(1)
            run_summary = _read_json(summary_path)
        else:
            if not args.skip_run:
                _run_02_benchmark(
                    variant, config, benchmark_dir, variant_output_dir, args, python
                )
            if not summary_path.exists():
                print(f"[MATRIX] ERROR: summary file not found for {variant}: {summary_path}")
                sys.exit(1)
            run_summary = _read_json(summary_path)

        run_summary["summary_path"] = str(summary_path)

        judge_report = None
        if (
            not args.skip_judge
            and not config.get("skip_judge")
            and args.judge_backend != "none"
        ):
            judge_cmd = [
                python,
                str(BENCHMARK_DIR / "03_judge_vlm_correctness.py"),
                "--summary",
                str(summary_path),
                "--judge-backend",
                args.judge_backend,
            ]
            if args.judge_concurrency is not None:
                judge_cmd.extend(["--judge-concurrency", str(args.judge_concurrency)])
            if args.judge_backend == "local":
                judge_cmd.extend(["--judge-model", args.judge_model])
                if args.device:
                    judge_cmd.extend(["--device", args.device])

            _run_step(f"Judge variant: {variant}", judge_cmd)

            judge_report_path = variant_output_dir / "vlm_judgement_report.json"
            if judge_report_path.exists():
                judge_report = _read_json(judge_report_path)
                judge_report["judge_report_path"] = str(judge_report_path)
                judge_report["judge_backend"] = args.judge_backend

        variant_records.append(_build_variant_record(variant, run_summary, judge_report))

    matrix_summary = {
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(output_dir),
        "variants": variants,
        "results": variant_records,
    }

    matrix_summary_path = output_dir / "benchmark_matrix_summary.json"
    with open(matrix_summary_path, "w", encoding="utf-8") as f:
        json.dump(matrix_summary, f, indent=2)

    print(f"\n{'=' * 60}", flush=True)
    print("[MATRIX] Benchmark matrix complete.", flush=True)
    print(f"[MATRIX] Results: {output_dir}", flush=True)
    print(f"[MATRIX] Combined summary: {matrix_summary_path}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Print a concise final table of the two key metrics.
    print("\n[MATRIX] Key metrics per variant:")
    print(f"{'Variant':<30} {'OOD Detection Rate':>20} {'LLM Judge Accuracy':>20}")
    print("-" * 72)
    for rec in variant_records:
        variant = rec["variant"]
        ood_rate = rec.get("detector_non_coco_rate")
        if ood_rate is None:
            # Fallback for older summaries without detector_non_coco_rate.
            if rec.get("evaluation_mode") in ("framework", "vlm_only"):
                ood_rate = rec.get("outlier_recall")
            else:
                ood_rate = rec.get("detection_level_non_coco_rate")
        judge_acc = rec.get("judge_accuracy")
        ood_str = f"{ood_rate:.2%}" if ood_rate is not None else "N/A"
        judge_str = f"{judge_acc:.2%}" if judge_acc is not None else "N/A"
        print(f"{variant:<30} {ood_str:>20} {judge_str:>20}")


if __name__ == "__main__":
    main()
