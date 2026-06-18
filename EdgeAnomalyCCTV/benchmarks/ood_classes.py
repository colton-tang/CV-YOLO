"""
OOD (Out-of-Distribution) class definitions for benchmarking EdgeAnomalyCCTV.

The framework treats COCO_CLASSES as the known closed-set.  Any object class
that is NOT in COCO_CLASSES is semantically OOD for this detector and should
ideally be flagged by the LLM outlier classifier as OUTLIER.
"""

import sys
from pathlib import Path

# Make the src package importable from benchmarks/
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import COCO_CLASSES

KNOWN_CLASSES = COCO_CLASSES
KNOWN_SET = set(KNOWN_CLASSES)

# A curated list of object classes that are NOT in COCO but are common enough
# to appear in OpenImages / real-world images.  These are good candidates for
# an OOD-class benchmark.
OOD_CLASSES = [
    "snake",
    "lizard",
    "frog",
    "crab",
    "lobster",
    "shrimp",
    "octopus",
    "squid",
    "jellyfish",
    "starfish",
    "seahorse",
    "mushroom",
    "cactus",
    "pine tree",
    "helicopter",
    "drone",
    "robot",
    "gun",
    "rifle",
    "bow and arrow",
    "fire",
    "smoke",
    "helmet",
    "mask",
    "microscope",
    "telescope",
    "piano",
    "guitar",
    "drum",
    "volcano",
    "tornado",
    "waterfall",
    "canyon",
    "glacier",
    "pyramid",
    "windmill",
    "satellite dish",
    "solar panel",
    "traffic cone",
    "manhole cover",
]

# Verify none of these accidentally overlap with COCO.
OVERLAPS = [c for c in OOD_CLASSES if c in KNOWN_SET]
if OVERLAPS:
    raise ValueError(f"OOD_CLASSES overlap with COCO: {OVERLAPS}")

# A tiny fallback benchmark: a handful of public-domain/CC0 Wikimedia Commons
# URLs that can be used immediately if OpenImages is not downloaded.  These are
# meant for quick smoke-testing only; for a rigorous benchmark use OpenImages.
# NOTE: URLs must be real, reachable image URLs.  If you add entries here, test
# them first.
FALLBACK_OOD_URLS = {}


def is_ood(label: str) -> bool:
    """Return True if a label is outside the known COCO set."""
    return label.lower().strip() not in KNOWN_SET


def list_ood_labels():
    """Return the curated OOD class list."""
    return OOD_CLASSES.copy()
