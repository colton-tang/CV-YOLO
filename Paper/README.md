# Cascading Multi-Agent Anomaly Detection Framework

This folder contains a reproducible implementation of the cascading multi-agent anomaly detection framework introduced in the paper *"Cascading Multi-Agent Anomaly Detection in Surveillance Systems via Vision-Language Models and Embedding-Based Classification"* (arXiv:2601.06204v3).

## Main Architecture Structure

The codebase maps to the architectural components designed in the paper:

1. **Multi-Agent Orchestration**:
   - `MessageBroker`: In-process publish-subscribe broker that mimics the Redis Pub/Sub broker mentioned in the paper (Sec. 3.1).
   - `EventDrivenAgent`: Triggered by asynchronous silent alarms (`α ∈ {0,1}`); captures streams and routes them for assessment.
   - `CyclicalMonitoringAgent`: Performs systematic cyclical health checks, including Shannon-entropy computation for obstruction/frozen-stream detection (Sec. 4.1).

2. **Cascading Detection Pipeline** (`CascadingMultiAgentPipeline`, Eq. (1)):
   - **Stage I: Object-Level Detection (`CascadableYOLO`)**: YOLOv8n rapid object-level assessment. Configurable threshold `τ_y` (default `0.45`, matching Sec. 5.4). Only configured anomaly classes (e.g., `person`) cause an early-exit anomaly; other detections are allowed to continue to Stage II.
   - **Stage II: Reconstruction-Based Scoring (`ReconstructionScorer`)**: Convolutional autoencoder (`AnomalyAutoencoder`) with the paper's channel sizes (`3→16→32→64`) and a 7×7 bottleneck convolution, mirrored by transposed convolutions and a Sigmoid output. Uses Eq. (2) MSE reconstruction error at `128×128` with threshold `τ_r = 1.5×10⁻³`.
   - **Stage III: Semantic Reasoning (`VLMReasoner`)**: Fallback for ambiguous cases. Supports **Moondream2 (~1.6B)** as the recommended lightweight default, BLIP (smallest fallback), LLaVA-7B (paper backend), or a deterministic dummy for smoke testing. Sentence embeddings (`all-mpnet-base-v2`) and class centroids computed from a few-shot prototype bank are used for cosine-similarity classification with abstention threshold `τ_c = 0.54`.

3. **System-Level Response**:
   - `joint_severity_score(...)` implements the fused score `S = λ₁·conf_visual + λ₂·conf_contextual` from Sec. 4.1 (`λ₁=0.4`, `λ₂=0.6`, `τ_S=0.75`).

## How to Run

The `cascade.py` file is ready out-of-the-box and does not require a cluster. It uses multithreading to simulate the multi-agent asynchronous publish/subscribe behaviours, but it now runs on the real sample images in `benchmark_data/` instead of synthetic dummy frames.

Install dependencies:

```bash
pip install ultralytics transformers sentence-transformers opencv-python torch moondream
```

For the paper-faithful LLaVA-7B backend you also need `bitsandbytes` (Linux/WSL) or a GPU with enough VRAM.  The default **Moondream2** backend downloads its weights automatically on first use and runs locally on CPU, MPS, or CUDA.

Execute the pipeline:

```bash
python Paper/cascade.py
```

By default only the `person` class is treated as a YOLO anomaly cue. To flag **every** detected COCO class as an anomaly, run:

```bash
python Paper/cascade.py --all-classes-anomaly
```

Annotated output images are saved to `benchmark_results/visualized/`. Each image shows:
- **Red boxes + labels** for Stage I YOLO anomaly detections.
- **Green boxes + labels** for Stage I non-anomaly YOLO detections.
- **Top-banner label** for Stage II (autoencoder score) and Stage III (VLM classification) decisions.

The pipeline result dict now also includes a `detections` list with every object YOLO found, e.g.:

```python
{
  'stage': 1,
  'label': 'bus',
  'confidence': 0.87,
  'anomaly': True,
  'detections': [
    {'label': 'bus',     'confidence': 0.87, 'box': [...], 'is_anomaly': True},
    {'label': 'person',  'confidence': 0.86, 'box': [...], 'is_anomaly': True},
    {'label': 'person',  'confidence': 0.85, 'box': [...], 'is_anomaly': True},
  ]
}
```

## Reproducibility Notes / Known Gaps

The script reproduces the **architecture, control flow, thresholds, and metrics** described in the paper.  The following items require additional artifacts that cannot be bundled in a standalone demo file:

| Paper element | Status in `cascade.py` | What is needed to fully reproduce |
|---|---|---|
| YOLOv8n object detector | ✅ Uses pre-trained `yolov8n.pt` | Fine-tuned weights for the paper's custom classes (e.g., `obstructed view`) used in Sec. 4 |
| Autoencoder architecture | ✅ Implemented as described | **Pre-trained weights** trained on ~1.1M normal UCF-Crime frames; random weights give meaningless reconstruction errors |
| VLM backend | ✅ Moondream2 (~1.6B) is the default lightweight option; LLaVA-7B and BLIP also supported | Moondream2 auto-downloads from Hugging Face; LLaVA-7B needs ~13 GB GPU/VRAM |
| Few-shot centroid bank | ✅ Prototype bank with centroid averaging | Paper's exact 20 curated examples per class are not publicly provided |
| UCF-Crime evaluation | ❌ Not included | The full dataset, training loop, and evaluation dashboard (Fig. 5) |

To run with a specific backend:

```python
pipeline = CascadingMultiAgentPipeline(
    yolo_conf=0.45,
    ae_threshold=1.5e-3,
    vlm_mode='moondream',  # 'blip', 'llava', or 'dummy'
    ae_weights='path/to/ae_ucf_crime.pth',
    device='cpu'           # or 'cuda', 'mps'
)
```

Moondream2 will be downloaded automatically from Hugging Face the first time you use it.

To force a frame through to the VLM (Stage III) for demonstration when the autoencoder is not trained, temporarily raise the autoencoder threshold:

```python
import cv2
from Paper.cascade import CascadingMultiAgentPipeline, Visualizer

pipeline = CascadingMultiAgentPipeline(vlm_mode='moondream', ae_threshold=0.5, device='cpu')
frame = cv2.imread('benchmark_data/kitchen_fruit.jpg')
result = pipeline.process_frame(frame)
print(result)

Visualizer().draw(frame, result, detector=pipeline.stage1, source_file='vlm_demo.jpg')
```

## Verification

A quick smoke test that exercises all three stages plus the entropy/severity helpers:

```python
python - <<'PY'
import cv2
from pathlib import Path
from Paper.cascade import (
    CascadableYOLO, ReconstructionScorer, VLMReasoner,
    shannon_entropy, joint_severity_score, CascadingMultiAgentPipeline
)
frame = cv2.imread('benchmark_data/kitchen_person.jpg')
print(CascadableYOLO(conf_threshold=0.45).infer(frame))
print(ReconstructionScorer().infer(frame))
print(VLMReasoner(mode='moondream', device='cpu').infer(frame))
print('entropy gray:', shannon_entropy(frame * 0 + 128))
print('severity:', joint_severity_score(0.92, 0.84))

# Full cascade on a real sample image
pipe = CascadingMultiAgentPipeline(vlm_mode='moondream', device='cpu')
for p in sorted(Path('benchmark_data').glob('*.jpg')):
    print(f"\n{p.name}:")
    print(pipe.process_frame(cv2.imread(str(p))))
PY
```
