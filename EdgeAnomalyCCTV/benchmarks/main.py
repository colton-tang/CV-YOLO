#!/usr/bin/env python3
"""
End-to-end OOD benchmark orchestrator.

Runs the full evaluation pipeline in order:

1. 01_prepare_openimages_ood.py  — prepare / validate the OOD benchmark images
2. 02_run_ood_benchmark.py       — run EdgeAnomalyCCTV on the benchmark
3. 03_judge_vlm_correctness.py   — judge VLM decisions (optional)

Usage:
    # Full pipeline on the small Caltech101 benchmark with Kimi judge
    cd /Users/t/CV
    source venv/bin/activate
    python EdgeAnomalyCCTV/benchmarks/main.py \
        --judge-backend kimi

    # Skip preparation if images already exist
    python EdgeAnomalyCCTV/benchmarks/main.py \
        --skip-prepare \
        --judge-backend local \
        --judge-model Qwen/Qwen3-VL-2B-Instruct

    # Larger OpenImages benchmark (requires fiftyone)
    python EdgeAnomalyCCTV/benchmarks/main.py \
        --backend openimages \
        --max-per-class 20 \
        --benchmark-dir benchmark_data/ood_openimages \
        --output-dir benchmark_data/ood_results \
        --judge-backend kimi
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add repository root to path so this file can be imported or run directly.
ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "EdgeAnomalyCCTV" / "benchmarks"

DEFAULT_BENCHMARK_DIR = ROOT / "benchmark_data" / "ood_openimages_small"
DEFAULT_OUTPUT_DIR = ROOT / "benchmark_data" / "ood_results_small"
DEFAULT_CLASSES = "octopus,lobster,scorpion,helicopter,crab,starfish"


def _load_dotenv() -> None:
    """Load environment variables from a project-root .env file."""
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
        print(f"[MAIN] warning: could not load {dotenv_path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full OOD benchmark pipeline end-to-end."
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
        help="Directory to save benchmark results and optional crops",
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
        "--model",
        type=str,
        default="weights/yolo/yolov8n.pt",
        help="YOLOv8 model path/name for the detection layer",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save annotated output images during the benchmark run",
    )
    parser.add_argument(
        "--judge-backend",
        type=str,
        choices=["none", "local", "kimi"],
        default="none",
        help="Backend for the optional VLM judge (default: none)",
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
        "--skip-prepare",
        action="store_true",
        help="Skip benchmark preparation (assume images already exist)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip the EdgeAnomalyCCTV benchmark run",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the VLM judgement step",
    )
    return parser.parse_args()


def _run_step(name: str, cmd: list[str]) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"[MAIN] {name}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"[MAIN] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[MAIN] ERROR: {name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    _load_dotenv()
    args = parse_args()

    python = sys.executable
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    summary_file = output_dir / "ood_benchmark_summary.json"

    # 1. Prepare benchmark
    if not args.skip_prepare:
        prepare_cmd = [
            python,
            str(BENCHMARK_DIR / "01_prepare_openimages_ood.py"),
            "--backend", args.backend,
            "--output-dir", str(benchmark_dir),
        ]
        if args.backend == "torchvision":
            prepare_cmd.extend([
                "--torchvision-dataset", args.torchvision_dataset,
                "--classes", args.classes,
                "--max-per-class", str(args.max_per_class),
            ])
        elif args.backend == "local":
            if not args.local_dir:
                print("[MAIN] ERROR: --local-dir is required when backend=local")
                sys.exit(1)
            prepare_cmd.extend(["--local-dir", args.local_dir])
        elif args.backend == "openimages":
            prepare_cmd.extend(["--max-per-class", str(args.max_per_class)])

        _run_step("Prepare benchmark", prepare_cmd)

    # 2. Run benchmark
    if not args.skip_run:
        run_cmd = [
            python,
            str(BENCHMARK_DIR / "02_run_ood_benchmark.py"),
            "--benchmark-dir", str(benchmark_dir),
            "--output-dir", str(output_dir),
            "--model", args.model,
            "--save-crops",  # crops are required by the judge
        ]
        if args.save_visualizations:
            run_cmd.append("--save-visualizations")

        _run_step("Run benchmark", run_cmd)

    # 3. Judge results
    if not args.skip_judge and args.judge_backend != "none":
        if not summary_file.exists():
            print(f"\n[MAIN] ERROR: summary file not found: {summary_file}")
            print("[MAIN] Run the benchmark first or remove --skip-run.")
            sys.exit(1)

        judge_cmd = [
            python,
            str(BENCHMARK_DIR / "03_judge_vlm_correctness.py"),
            "--summary", str(summary_file),
            "--judge-backend", args.judge_backend,
        ]
        if args.judge_backend == "local":
            judge_cmd.extend(["--judge-model", args.judge_model])
            if args.device:
                judge_cmd.extend(["--device", args.device])

        _run_step("Judge VLM decisions", judge_cmd)

    print(f"\n{'='*60}", flush=True)
    print("[MAIN] Benchmark pipeline complete.", flush=True)
    print(f"[MAIN] Results: {output_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
