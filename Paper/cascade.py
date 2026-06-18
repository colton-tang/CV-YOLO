import cv2
import torch
import torch.nn as nn
import numpy as np
import threading
import time
import queue
from pathlib import Path
from ultralytics import YOLO


class CascadableYOLO:
    """Stage I: Object-level detection with YOLOv8.

    Matches the paper's description in Sec. 3.2 and Sec. 5.4:
    - YOLOv8n (~7M params) is the lightweight detector.
    - A frame exits early if max_k P_1(y=k|x_t) >= tau_y.
    - Sec. 5.4 specifically issues a "person" cue when any detection exceeds
      tau_y = 0.45 confidence.  The case study (Sec. 4) uses tau_1 = 0.85 for
      the custom "obstructed view" class; both thresholds are configurable.
    """

    def __init__(self,
                 model_path='weights/yolo/yolov8n.pt',
                 conf_threshold=0.45,
                 anomaly_classes=None,
                 person_cue=True,
                 all_classes_anomaly=False):
        self.detector = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.person_cue = person_cue
        self.all_classes_anomaly = all_classes_anomaly
        self._last_results = None

        # Default anomaly-related YOLO classes.  Pre-trained COCO does not
        # contain "obstructed view"; that class requires a fine-tuned model as
        # used in the paper's case study.  We keep the mapping explicit so the
        # intended behaviour is clear.
        self.anomaly_classes = set(anomaly_classes or [])
        if person_cue:
            self.anomaly_classes.add('person')

    def infer(self, frame):
        results = self.detector.predict(frame, verbose=False, conf=self.conf_threshold)
        self._last_results = results
        if len(results) > 0 and len(results[0].boxes) > 0:
            confs = results[0].boxes.conf
            boxes = results[0].boxes
            detections = []
            for i in range(len(boxes)):
                conf = float(confs[i])
                label_idx = int(boxes.cls[i].item())
                label = self.detector.names[label_idx]
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": boxes.xyxy[i].tolist(),
                    "is_anomaly": label in self.anomaly_classes or self.all_classes_anomaly,
                })

            max_conf = confs.max().item()
            best_idx = int(confs.argmax().item())
            best_label = self.detector.names[int(boxes.cls[best_idx].item())]

            if max_conf >= self.conf_threshold:
                # The paper states that Stage I can classify frames as nominal
                # or anomalous depending on the detected category.  We only
                # exit as an anomaly for configured anomaly classes (or all
                # classes if all_classes_anomaly=True); otherwise the frame is
                # allowed to continue to Stage II.
                is_anomaly = (
                    self.all_classes_anomaly
                    or best_label in self.anomaly_classes
                )
                return {
                    "stage": 1,
                    "anomaly": is_anomaly,
                    "confidence": max_conf,
                    "label": best_label,
                    "exit": is_anomaly,
                    "reason": f"YOLO detection '{best_label}' (conf={max_conf:.3f})",
                    "detections": detections,
                }
        return {"stage": 1, "anomaly": False, "confidence": 0.0, "exit": False, "detections": []}

    def get_boxes(self):
        """Return the raw YOLO boxes from the most recent inference."""
        if self._last_results is None or len(self._last_results) == 0:
            return []
        return self._last_results[0].boxes


class AnomalyAutoencoder(nn.Module):
    """Convolutional autoencoder used for Stage II reconstruction scoring.

    Architecture from Sec. 3.2 / Sec. 5.4:
    - Encoder: three stride-2 convolutions (3->16->32->64) followed by a 7x7
      bottleneck convolution.  The 7x7 *kernel* reduces the 16x16 feature map
      to a 10x10 latent representation.
    - Decoder: mirrored transposed convolutions that reconstruct the original
      128x128 input, with a final Sigmoid activation.

    Input is normalised to [0, 1] and resized to 128x128.
    """

    def __init__(self):
        super(AnomalyAutoencoder, self).__init__()
        # Encoder: 3->16, 16->32, 32->64, bottleneck 7x7 kernel
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=0),  # 16 -> 10 (7x7 kernel)
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=1, padding=0),   # 10 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # 64 -> 128
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ReconstructionScorer:
    """Stage II: reconstruction-based anomaly scoring.

    Implements Eq. (2) from the paper:
        e(x) = (1 / (3*H*W)) * ||x - x'||_2^2,   H = W = 128
    and flags an anomaly when e(x) >= tau_r = 1.5e-3.

    NOTE: The autoencoder must be trained on normal surveillance frames before
    deployment; the current script initialises weights randomly for
    demonstration.  Pre-trained weights are required for meaningful scores.
    """

    def __init__(self, threshold=1.5e-3, device='cpu', model_path=None):
        self.model = AnomalyAutoencoder().to(device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.threshold = threshold
        self.device = device

    def infer(self, frame):
        resized = cv2.resize(frame, (128, 128))
        img_tensor = (torch.from_numpy(resized)
                          .permute(2, 0, 1)
                          .float()
                          .unsqueeze(0) / 255.0)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            reconstructed = self.model(img_tensor)

        # Equivalent to Eq. (2): mean over the 3*H*W elements.
        mse = torch.mean((img_tensor - reconstructed) ** 2).item()

        if mse >= self.threshold:
            return {
                "stage": 2,
                "anomaly": True,
                "score": mse,
                "reason": "High reconstruction error",
                "exit": True
            }
        return {"stage": 2, "anomaly": False, "score": mse, "exit": False}


class VLMReasoner:
    """Stage III: semantic reasoning with a VLM and embedding-based classifier.

    Paper specification (Sec. 3.2 / Sec. 5.4):
    - Vision-language backbone: LLaVA-7B (or LLaVA-Next).
    - Sentence embeddings: all-mpnet-base-v2.
    - Class centroids mu_k computed from a few-shot bank (20 curated examples
      per class).
    - Cosine similarity s_k = cos(E(t), mu_k); accept if max_k s_k >= tau_c,
      otherwise abstain (Benign).  Default tau_c = 0.54.

    Because LLaVA-7B is too heavy for a lightweight demo, this class supports
    several backends, from lightweight edge models to the paper-faithful one:
        1. moondream - Moondream2 (~1.6B).  Recommended lightweight default.
        2. blip      - BLIP captioning fallback (smallest, lower fidelity).
        3. llava     - LLaVA-7B via transformers (paper-faithful, heaviest).
        4. dummy     - deterministic stub for smoke testing.
    """

    # Few-shot prototype bank.  Each key maps to a list of descriptive phrases.
    # In the paper these are aggregated into centroids mu_k by averaging their
    # sentence embeddings.
    DEFAULT_FEW_SHOT_BANK = {
        "camera_blocked": [
            "lens obstruction camera blocked",
            "obscured lens covered camera",
            "camera view blocked by object",
            "blurred or blocked surveillance lens",
        ],
        "person_detected": [
            "unauthorized person detected",
            "hand covering lens",
            "individual in restricted area",
            "person standing where they should not be",
        ],
        "suspicious_behavior": [
            "suspicious behavior loitering",
            "individual loitering near restricted gate",
            "person acting suspiciously",
            "unusual activity in monitored zone",
        ],
        "person_with_weapon": [
            "person holding a weapon",
            "armed individual visible",
            "someone brandishing a knife or gun",
        ],
        "collapsed_individual": [
            "person collapsed on the ground",
            "individual lying down unconscious",
            "fallen person needing assistance",
        ],
    }

    def __init__(self,
                 mode='moondream',
                 tau_c=0.54,
                 few_shot_bank=None,
                 device='cpu'):
        """
        Args:
            mode: 'moondream' | 'blip' | 'llava' | 'dummy'.
            tau_c: classifier acceptance threshold (default 0.54).
            few_shot_bank: dict mapping label -> list of text descriptions.
            device: torch device string.
        """
        self.mode = mode.lower()
        self.tau_c = tau_c
        self.device = device
        self.few_shot_bank = few_shot_bank or self.DEFAULT_FEW_SHOT_BANK
        self.prototypes = {}

        if self.mode == 'llava':
            from transformers import (
                LlavaForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig,
            )
            model_id = "llava-hf/llava-1.5-7b-hf"
            processor = AutoProcessor.from_pretrained(model_id)
            # Load in 4-bit to make local execution feasible on modest hardware.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self.vlm = (model, processor)
            self._build_centroids()

        elif self.mode == 'moondream':
            # Use the official moondream pip package; it downloads the weights
            # once and runs locally without the transformers compatibility
            # issues of the remote-code Hub wrapper.
            import moondream as md
            self.vlm = md.vl(model="moondream2", local=True)
            self._build_centroids()

        elif self.mode == 'blip':
            from transformers import pipeline
            # Newer transformers uses "image-text-to-text" instead of
            # the legacy "image-to-text" task name.
            self.vlm = pipeline(
                "image-text-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=0 if device == 'cuda' else -1,
            )
            self._build_centroids()

        elif self.mode == 'dummy':
            self.vlm = None
            self._build_centroids()

        else:
            raise ValueError(f"Unsupported VLM mode: {mode}")

    def _build_centroids(self):
        """Compute centroid mu_k for each class from the few-shot bank."""
        if self.mode == 'dummy':
            # Deterministic hash-based embeddings so the dummy is reproducible
            # without sentence-transformers installed.
            np.random.seed(42)
            dim = 768
            for label, phrases in self.few_shot_bank.items():
                embs = []
                for phrase in phrases:
                    embs.append(self._dummy_embedding(phrase, dim))
                self.prototypes[label] = np.mean(embs, axis=0)
            return

        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-mpnet-base-v2')
        for label, phrases in self.few_shot_bank.items():
            embs = embedder.encode(phrases, convert_to_numpy=True)
            self.prototypes[label] = np.mean(embs, axis=0)

    @staticmethod
    def _dummy_embedding(text, dim=768):
        # Stable hash so dummy centroids are reproducible across runs.
        import hashlib
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)
        emb = rng.normal(size=dim).astype(np.float32)
        emb /= np.linalg.norm(emb)
        return emb

    def _caption_llava(self, pil_img):
        model, processor = self.vlm
        prompt = (
            "USER: <image>\nDescribe what is happening in this surveillance "
            "frame, focusing on any unusual or suspicious activity.\nASSISTANT:"
        )
        inputs = processor(text=prompt, images=pil_img, return_tensors="pt").to(self.device)
        output = model.generate(**inputs, max_new_tokens=100)
        return processor.batch_decode(output, skip_special_tokens=True)[0]

    def _caption_blip(self, pil_img):
        # Newer transformers image-text-to-text pipelines require both image
        # and text.  An empty prompt triggers BLIP's captioning behaviour.
        out = self.vlm({"images": pil_img, "text": ""})
        # Output format: [{'text': '...'}] or [{'generated_text': '...'}]
        if isinstance(out, list) and len(out) > 0:
            out = out[0]
        return out.get("text") or out.get("generated_text", "")

    def _caption_moondream(self, pil_img):
        model = self.vlm
        encoded_image = model.encode_image(pil_img)
        prompt = (
            "Describe what is happening in this surveillance frame, "
            "focusing on any unusual or suspicious activity."
        )
        return model.query(encoded_image, prompt)["answer"]

    def infer(self, frame):
        if self.mode == 'dummy':
            # Deterministic dummy: use a hash of the frame to pick a class so
            # repeated runs are stable and the abstention path can be exercised.
            rng = np.random.default_rng(int(frame[:8, :8].sum()) % (2**31))
            labels = list(self.prototypes.keys()) + ["Benign"]
            label = str(rng.choice(labels))
            confidence = 0.55 + rng.random() * 0.15
            if label == "Benign":
                confidence = 0.30 + rng.random() * 0.20
            return {
                "stage": 3,
                "anomaly": label != "Benign",
                "label": label,
                "confidence": float(confidence),
                "text": f"dummy caption -> {label}",
                "exit": True
            }

        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if self.mode == 'llava':
            text_out = self._caption_llava(pil_img)
        elif self.mode == 'moondream':
            text_out = self._caption_moondream(pil_img)
        else:  # blip
            text_out = self._caption_blip(pil_img)

        emb = self._embed(text_out)

        best_label = "Benign"
        max_sim = -1.0
        for label, centroid in self.prototypes.items():
            sim = float(
                np.dot(emb, centroid)
                / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            )
            if sim > max_sim:
                max_sim = sim
                best_label = label

        if max_sim >= self.tau_c:
            return {
                "stage": 3,
                "anomaly": True,
                "label": best_label,
                "confidence": max_sim,
                "text": text_out,
                "exit": True
            }
        else:
            return {
                "stage": 3,
                "anomaly": False,
                "label": "Benign",
                "confidence": max_sim,
                "text": text_out,
                "exit": True
            }

    def _embed(self, text):
        if self.mode == 'dummy':
            return self._dummy_embedding(text)
        if not hasattr(self, '_embedder'):
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-mpnet-base-v2')
        return self._embedder.encode(text, convert_to_numpy=True)


class CascadingMultiAgentPipeline:
    """Three-stage early-exit cascade described in Eq. (1) of the paper."""

    def __init__(self,
                 yolo_conf=0.45,
                 ae_threshold=1.5e-3,
                 vlm_mode='moondream',
                 ae_weights=None,
                 device='cpu',
                 yolo_all_classes_anomaly=False):
        self.stage1 = CascadableYOLO(
            conf_threshold=yolo_conf,
            all_classes_anomaly=yolo_all_classes_anomaly,
        )
        self.stage2 = ReconstructionScorer(threshold=ae_threshold,
                                           device=device,
                                           model_path=ae_weights)
        self.stage3 = VLMReasoner(mode=vlm_mode, device=device)

    def process_frame(self, frame):
        res1 = self.stage1.infer(frame)
        if res1["exit"]:
            return res1

        res2 = self.stage2.infer(frame)
        if res2["exit"]:
            return res2

        res3 = self.stage3.infer(frame)
        return res3


class MessageBroker:
    """Publish-subscribe broker.

    The paper mentions Redis Pub/Sub as the production broker.  This local
    in-process Queue implementation preserves the same pub/sub semantics for
    reproducible single-machine demos.
    """

    def __init__(self):
        self.task_queue = queue.Queue()
        self._lock = threading.Lock()
        self._subscribers = []

    def publish(self, task):
        with self._lock:
            self.task_queue.put(task)
            for subscriber in self._subscribers:
                subscriber(task)

    def subscribe(self):
        return self.task_queue.get()

    def register_callback(self, callback):
        with self._lock:
            self._subscribers.append(callback)


def shannon_entropy(frame, bins=256):
    """Compute average Shannon entropy of frame intensities (Sec. 4.1).

    H(x_t) = -sum_i p_i log(p_i),  where p_i are histogram bin probabilities.
    The paper uses tau_H = 2.3 as a nominal threshold for obstruction detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel()
    hist = hist.astype(np.float64) + 1e-10
    hist /= hist.sum()
    return -np.sum(hist * np.log2(hist))


class EventDrivenAgent(threading.Thread):
    """Triggered by asynchronous silent alarms (Sec. 3.1)."""

    def __init__(self, broker, camera_id="camera_1", alert_signal=True, frame=None,
                 source_file=None):
        super().__init__()
        self.broker = broker
        self.camera_id = camera_id
        self.alert_signal = alert_signal
        self.frame = frame
        self._source_file = source_file or 'event_frame'
        self.daemon = True

    def run(self):
        time.sleep(1)  # Simulate waiting for an alarm event
        # In a real deployment this frame comes from the alerted camera stream.
        frame = self.frame if self.frame is not None else np.random.randint(
            0, 255, (720, 1280, 3), dtype=np.uint8
        )
        self.broker.publish({
            "type": "event",
            "camera_id": self.camera_id,
            "frame": frame,
            "alert_signal": int(self.alert_signal),
            "source_file": getattr(self, '_source_file', 'event_frame'),
        })


class CyclicalMonitoringAgent(threading.Thread):
    """Executes continuous health verification (Sec. 3.1 / Sec. 4.1).

    Verifies camera operational status by computing the Shannon entropy of
    incoming frames and flagging possible obstruction/illumination faults when
    H(x_t) < tau_H.
    """

    def __init__(self, broker, camera_id="camera_1", interval=2, tau_h=2.3,
                 frames=None, source_files=None):
        super().__init__()
        self.broker = broker
        self.camera_id = camera_id
        self.interval = interval
        self.tau_h = tau_h
        self.frames = frames or []
        self.source_files = source_files or [f"monitoring_frame_{i}" for i in range(len(self.frames))]
        self.daemon = True

    def run(self):
        idx = 0
        while True:
            time.sleep(self.interval)
            if self.frames:
                frame = self.frames[idx % len(self.frames)]
                source_file = self.source_files[idx % len(self.source_files)]
                idx += 1
            else:
                # Constant gray frame: low entropy, simulates an obstructed/frozen
                # camera as described in the case study.
                frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
                source_file = "gray_frame"
            entropy = shannon_entropy(frame)
            healthy = entropy >= self.tau_h
            self.broker.publish({
                "type": "monitoring",
                "camera_id": self.camera_id,
                "frame": frame,
                "entropy": float(entropy),
                "healthy": bool(healthy),
                "source_file": source_file,
            })


def joint_severity_score(visual_conf, contextual_conf, lambdas=(0.4, 0.6)):
    """Compute system-level severity score from Sec. 4.1:

    S = lambda_1 * conf_visual + lambda_2 * conf_contextual
    High-priority threshold tau_S = 0.75.
    """
    return lambdas[0] * visual_conf + lambdas[1] * contextual_conf


class Visualizer:
    """Draw detection boxes and stage labels on frames for inspection."""

    def __init__(self, output_dir='benchmark_results/visualized'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _put_text(img, text, y_offset, color=(0, 255, 0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = img.shape[:2]
        # Adapt font scale to image width so long labels fit on small frames.
        scale = min(0.7, max(0.35, (w / 1280) * 0.7))
        thickness = max(1, int(scale * 2.5))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = 10, y_offset
        cv2.rectangle(img, (x, y - th - 5), (x + tw + 10, y + 5), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 5, y), font, scale, color, thickness, cv2.LINE_AA)

    def draw(self, frame, result, detector=None, source_file='frame.jpg'):
        """Annotate a frame with the cascade decision and save it.

        Args:
            frame: BGR image (numpy array).
            result: dict returned by the pipeline.
            detector: CascadableYOLO instance, used to draw YOLO boxes.
            source_file: filename used for the saved output image.

        Returns:
            Path to the saved annotated image.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        stage = result.get('stage')

        # Draw YOLO boxes for Stage I results.
        if stage == 1 and detector is not None:
            # Prefer the detections list from the result dict; fall back to raw boxes.
            detections = result.get("detections", [])
            if not detections and detector.get_boxes() is not None:
                boxes = detector.get_boxes()
                for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = detector.detector.names[cls_id]
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "box": box.xyxy[0].tolist(),
                        "is_anomaly": label in detector.anomaly_classes,
                    })

            for det in detections:
                x1, y1, x2, y2 = map(int, det["box"])
                label = det["label"]
                conf = det["confidence"]
                color = (0, 0, 255) if det["is_anomaly"] else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                box_text = f"{label} {conf:.2f}"
                self._put_text(annotated, box_text, y1 - 10 if y1 > 30 else y2 + 20, color)

        # Overlay the stage-level decision at the top of the image.
        if stage == 1:
            status = f"STAGE I - YOLO: {result['label']} ({result['confidence']:.2f})"
            color = (0, 0, 255) if result['anomaly'] else (0, 255, 0)
        elif stage == 2:
            status = f"STAGE II - AE: {result['reason']} ({result['score']:.4f})"
            color = (0, 0, 255) if result['anomaly'] else (0, 255, 0)
        elif stage == 3:
            label = result['label']
            status = f"STAGE III - VLM: {label} ({result['confidence']:.2f})"
            color = (0, 0, 255) if result['anomaly'] and label != "Benign" else (0, 255, 0)
        else:
            status = "UNKNOWN STAGE"
            color = (128, 128, 128)

        self._put_text(annotated, status, 30, color)

        out_path = self.output_dir / source_file
        cv2.imwrite(str(out_path), annotated)
        return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Cascading multi-agent anomaly detection demo"
    )
    parser.add_argument(
        "--all-classes-anomaly",
        action="store_true",
        help="Treat every YOLO detection above the confidence threshold as an anomaly "
             "(default: only the configured anomaly classes, e.g., 'person')."
    )
    parser.add_argument(
        "--vlm-mode",
        default='moondream',
        choices=['moondream', 'blip', 'llava', 'dummy'],
        help="VLM backend to use for Stage III (default: moondream)."
    )
    args = parser.parse_args()

    broker = MessageBroker()

    # Try the recommended lightweight VLM (Moondream2).  If the model is not
    # downloaded or the runtime lacks dependencies, fall back to dummy mode so
    # the script still demonstrates the cascade structure.
    vlm_mode = args.vlm_mode
    try:
        pipeline = CascadingMultiAgentPipeline(
            yolo_conf=0.45,
            ae_threshold=1.5e-3,
            vlm_mode=vlm_mode,
            device='cpu',
            yolo_all_classes_anomaly=args.all_classes_anomaly,
        )
        print(f"Loaded VLM backend: {vlm_mode}")
    except Exception as e:
        print(f"WARNING: could not load VLM backend '{vlm_mode}': {e}")
        print("Falling back to dummy VLM for demonstration.")
        pipeline = CascadingMultiAgentPipeline(
            yolo_conf=0.45,
            ae_threshold=1.5e-3,
            vlm_mode='dummy',
            device='cpu',
            yolo_all_classes_anomaly=args.all_classes_anomaly,
        )

    # Load real sample images from the repository instead of using synthetic
    # dummy frames.  We use one person-containing image for the event-driven
    # alarm and rotate through the remaining images for cyclical monitoring.
    sample_dir = Path(__file__).resolve().parent.parent / "benchmark_data"
    sample_images = sorted(sample_dir.glob("*.jpg"))
    if not sample_images:
        # Fallback to the project root if benchmark_data is missing.
        sample_dir = Path(__file__).resolve().parent.parent
        sample_images = sorted(sample_dir.glob("*.jpg"))

    if not sample_images:
        raise RuntimeError("No sample .jpg images found in benchmark_data/ or project root.")

    print(f"Using sample images from: {sample_dir}")
    for p in sample_images:
        print(f"  - {p.name}")

    # Pick a likely person/anomaly frame for the silent-alarm event.
    event_path = None
    for p in sample_images:
        if 'person' in p.name.lower() or 'zidane' in p.name.lower():
            event_path = p
            break
    if event_path is None:
        event_path = sample_images[0]
    monitoring_paths = [p for p in sample_images if p != event_path]

    event_frame = cv2.imread(str(event_path))
    monitoring_frames = [cv2.imread(str(p)) for p in monitoring_paths]

    event_agent = EventDrivenAgent(broker, frame=event_frame,
                                   source_file=event_path.name)
    monitor_agent = CyclicalMonitoringAgent(broker, interval=2,
                                            frames=monitoring_frames,
                                            source_files=[p.name for p in monitoring_paths])
    event_agent.start()
    monitor_agent.start()

    visualizer = Visualizer(output_dir='benchmark_results/visualized')
    print(f"\nAnnotated outputs will be saved to: {visualizer.output_dir}")
    print("Multi-agent orchestration started. Waiting for tasks...")

    # Worker process subscribing to the broker
    num_tasks = 1 + len(monitoring_frames)  # 1 event + all monitoring frames
    for _ in range(num_tasks):
        task = broker.subscribe()
        source = task['type']
        source_file = task.get('source_file', source)
        print(f"\nReceived task from {source} agent ({source_file})...")
        result = pipeline.process_frame(task['frame'])
        print(f"Pipeline result for {source} task: {result}")

        # Draw boxes/labels and save the annotated image.
        out_path = visualizer.draw(
            task['frame'], result,
            detector=pipeline.stage1,
            source_file=source_file
        )
        print(f"Saved annotated image: {out_path}")

        # Demonstrate the system-level severity score when both agents fire.
        if source == "monitoring" and not task.get('healthy', True):
            visual_conf = 0.92  # e.g. corroborated Stage I/II confidence
            contextual_conf = result.get('confidence', 0.0)
            score = joint_severity_score(visual_conf, contextual_conf)
            print(f"Joint severity score S = {score:.3f}")
