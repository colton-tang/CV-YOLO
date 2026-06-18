#!/usr/bin/env python3
"""
Prepare a small OOD-class benchmark from OpenImages (or a fallback URL set).

The benchmark is designed for EdgeAnomalyCCTV: COCO_CLASSES are treated as the
known closed set, and any object class NOT in COCO is semantically OOD.

Usage:
    # Install optional dependency for full OpenImages download
    pip install fiftyone

    # Download up to 20 images per OOD class from OpenImages V7
    python 01_prepare_openimages_ood.py --backend openimages --max-per-class 20

    # Use the tiny fallback URL set (no extra dependencies)
    python 01_prepare_openimages_ood.py --backend fallback

    # Use your own local image folder (just validate it)
    python 01_prepare_openimages_ood.py --backend local --local-dir ./my_ood_images

Output:
    benchmark_data/ood_openimages/
        <class_name>/<image_id>.jpg
    benchmark_data/ood_openimages_manifest.json
"""

import argparse
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

_ood_classes = importlib.import_module("00_ood_classes")
OOD_CLASSES = _ood_classes.OOD_CLASSES

DEFAULT_OUTPUT_DIR = ROOT / "benchmark_data" / "ood_openimages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OOD class benchmark")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openimages", "torchvision", "local"],
        default="torchvision",
        help=(
            "Download source: openimages (requires fiftyone, largest/standard), "
            "torchvision (uses Caltech101, quick start), or validate a local directory"
        ),
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=20,
        help="Maximum images to download per OOD class (openimages backend only)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of OOD classes to download (default: curated OOD_CLASSES)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save benchmark images",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local image directory to validate (local backend only)",
    )
    parser.add_argument(
        "--torchvision-dataset",
        type=str,
        default="Caltech101",
        help="torchvision dataset name to use as quick OOD source (Caltech101, Flowers102, Food101, etc.)",
    )
    return parser.parse_args()


def _load_image_from_url(url: str) -> np.ndarray | None:
    """Download an image and return it as a BGR numpy array."""
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' is required for fallback URL download.")
        return None

    headers = {
        "User-Agent": (
            "EdgeAnomalyCCTV-ODDBenchmark/1.0 "
            "(https://github.com/; research benchmark downloader)"
        ),
    }

    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        print(f"  failed to fetch {url}: {exc}")
        return None


def _save_image(img: np.ndarray, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(dest), img)


def _load_torchvision_dataset(dataset_name: str, root: Path):
    """Load a torchvision image-classification dataset by name."""
    import torchvision.datasets as tv_datasets

    root.mkdir(parents=True, exist_ok=True)
    loader = getattr(tv_datasets, dataset_name, None)
    if loader is None:
        raise ValueError(f"unknown torchvision dataset: {dataset_name}")

    # Most torchvision classification datasets accept a 'download' kwarg.
    try:
        return loader(root=str(root), download=True)
    except TypeError:
        return loader(root=str(root))


def _download_torchvision(classes, explicit_classes, dataset_name: str, output_dir: Path, max_per_class: int = 0) -> dict:
    """Use a torchvision classification dataset as a quick OOD source.

    Images are grouped by their dataset class label.  Classes that overlap
    with COCO are skipped so every retained image is semantically OOD.
    """
    manifest = {"backend": "torchvision", "dataset": dataset_name, "classes": {}, "images": []}

    # Keep the raw torchvision download cache outside the benchmark folder so
    # the runner does not accidentally recurse into it.
    cache_root = Path.home() / ".cache" / "edgeanomaly_ood" / "torchvision"
    ds = _load_torchvision_dataset(dataset_name, cache_root)
    class_counts = {}

    # Build a map from class index -> class name.
    class_names = getattr(ds, "classes", None) or getattr(ds, "categories", None)
    if class_names:
        idx_to_name = {i: name.lower().strip().replace("_", " ") for i, name in enumerate(class_names)}
    else:
        idx_to_name = {}

    for sample_idx, (img, label_idx) in enumerate(ds):
        cls = idx_to_name.get(label_idx, f"class_{label_idx}")

        # Skip classes that are in COCO (e.g. Caltech101 has airplane, car, ...).
        if cls in {c.lower() for c in COCO_CLASSES}:
            continue

        # If the user supplied an explicit class list, only keep those.
        # Otherwise keep every non-COCO class from the torchvision dataset.
        if explicit_classes and cls not in explicit_classes:
            continue

        if max_per_class and class_counts.get(cls, 0) >= max_per_class:
            continue

        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        dest = cls_dir / f"{cls}_{sample_idx:04d}.jpg"

        try:
            # PIL Image -> BGR numpy array.
            import numpy as np
            rgb = np.array(img)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if _save_image(bgr, dest):
                class_counts[cls] = class_counts.get(cls, 0) + 1
                manifest["classes"][cls] = manifest["classes"].get(cls, 0) + 1
                manifest["images"].append({
                    "class": cls,
                    "path": str(dest),
                    "source": f"{dataset_name}:{sample_idx}",
                })
        except Exception as exc:
            print(f"  failed to save {cls} sample {sample_idx}: {exc}")

    return manifest


def _download_openimages(classes, max_per_class: int, output_dir: Path) -> dict:
    """Download a subset of OpenImages V7 for the requested OOD classes."""
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("ERROR: openimages backend requires 'fiftyone'.")
        print("Install it with:  pip install fiftyone")
        sys.exit(1)

    manifest = {"backend": "openimages", "classes": {}, "images": []}

    # Load OpenImages V7 validation split.  This is a once-off download that
    # fiftyone will cache in ~/fiftyone.
    print("[openimages] loading OpenImages V7 validation split via fiftyone ...")
    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            label_types=["detections"],
            classes=classes,
            max_samples=max_per_class * len(classes),
        )
    except Exception as exc:
        print(f"ERROR: failed to load OpenImages: {exc}")
        sys.exit(1)

    # Group samples by the OOD class we asked for.  A single image may contain
    # multiple OOD objects; we assign it to the first requested class found.
    for sample in dataset.iter_samples(progress=True):
        detections = sample.ground_truth.detections if sample.ground_truth else []
        matched_class = None
        for det in detections:
            label = det.label.lower().strip()
            if label in classes:
                matched_class = label
                break
        if matched_class is None:
            continue

        cls_dir = output_dir / matched_class
        cls_dir.mkdir(parents=True, exist_ok=True)
        dest = cls_dir / f"{matched_class}_{sample.id}.jpg"
        src_path = Path(sample.filepath)
        try:
            img = cv2.imread(str(src_path))
            if img is None:
                continue
            if _save_image(img, dest):
                manifest["classes"][matched_class] = manifest["classes"].get(matched_class, 0) + 1
                manifest["images"].append({
                    "class": matched_class,
                    "path": str(dest),
                    "source": str(src_path),
                })
        except Exception as exc:
            print(f"  failed to copy {src_path}: {exc}")

    return manifest


def _validate_local(local_dir: Path, classes, output_dir: Path) -> dict:
    """Validate that images in a local directory contain OOD classes."""
    manifest = {"backend": "local", "classes": {}, "images": []}
    if not local_dir.exists():
        print(f"ERROR: local directory does not exist: {local_dir}")
        sys.exit(1)

    for img_path in sorted(local_dir.rglob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        # Try to infer class from parent folder name
        cls = img_path.parent.name.lower().strip()
        if cls not in classes:
            cls = "unknown"
        manifest["classes"][cls] = manifest["classes"].get(cls, 0) + 1
        manifest["images"].append({
            "class": cls,
            "path": str(img_path),
            "source": "local",
        })
    return manifest


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit_classes = [c.strip().lower() for c in args.classes.split(",")] if args.classes else []
    classes = explicit_classes if explicit_classes else OOD_CLASSES
    print(f"OOD classes for this benchmark: {classes}")

    if args.backend == "torchvision":
        manifest = _download_torchvision(classes, explicit_classes, args.torchvision_dataset, output_dir, args.max_per_class)
    elif args.backend == "openimages":
        manifest = _download_openimages(classes, args.max_per_class, output_dir)
    elif args.backend == "local":
        if args.local_dir is None:
            print("ERROR: --local-dir is required for local backend")
            sys.exit(1)
        manifest = _validate_local(Path(args.local_dir), classes, output_dir)
    else:
        raise ValueError(f"unknown backend: {args.backend}")

    manifest_path = output_dir / "ood_openimages_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nBenchmark preparation complete.")
    print(f"  images saved to: {output_dir}")
    print(f"  manifest saved to: {manifest_path}")
    print(f"  images per class: {manifest['classes']}")
    print(f"  total images: {len(manifest['images'])}")


if __name__ == "__main__":
    main()
