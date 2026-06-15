import asyncio
import json
import re

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class LLMClassifierLayer:
    def __init__(self, queue, known_classes, track_state_db=None, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        self.queue = queue
        self.known_classes = known_classes
        self.track_state_db = track_state_db if track_state_db is not None else {}
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = self._select_device()
        self.dtype = self._select_dtype(self.device)
        self.max_new_tokens = 96

    async def run(self):
        while True:
            item = await self.queue.get()
            source_type = item["source_type"]
            track_id = item.get("track_id")

            try:
                result = await self._run_llm(item)
            except Exception as exc:
                result = self._fallback_result(item, error=str(exc))

            status = "UNKNOWN"
            if result["type"] == "KNOWN":
                status = "RESOLVED"
            elif result["type"] == "OUTLIER":
                status = "OUTLIER"

            if self.track_state_db is not None:
                dedup_key = track_id
                self.track_state_db[dedup_key] = {
                    "status": status,
                    "class": result["class"],
                    "display_class": result.get("display_class", result["class"]),
                    "confidence": result["confidence"],
                    "bbox": item.get("bbox"),
                    "llm_response": result.get("raw_response"),
                    "llm_error": result.get("error"),
                }

            if source_type == "IMAGE":
                pass

            self.queue.task_done()

    async def _run_llm(self, item):
        await self._ensure_model_loaded()
        return await asyncio.to_thread(self._infer_sync, item)

    async def _ensure_model_loaded(self):
        if self.model is not None and self.processor is not None:
            return
        await asyncio.to_thread(self._load_model)

    def _load_model(self):
        if self.model is not None and self.processor is not None:
            return

        model_kwargs = {"torch_dtype": self.dtype}
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _infer_sync(self, item):
        image = self._to_pil_image(item["crop"])
        prompt = self._build_prompt(item)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat_text], images=[image], padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = self._parse_response(output_text, item)
        parsed["raw_response"] = output_text
        return parsed

    def _build_prompt(self, item):
        yolo_class = item.get("yolo_class", "unknown")
        known_classes = ", ".join(self.known_classes)
        return (
            "You are classifying one cropped CCTV object.\n"
            f"YOLO hint: class='{yolo_class}', confidence={item.get('yolo_conf', 0.0):.3f}.\n"
            f"Known classes: {known_classes}.\n"
            "Choose the best label.\n"
            "If the object clearly belongs to one known class, set type to KNOWN and class to that known class.\n"
            "If it does not belong to any known class, set type to OUTLIER and class to a short specific label.\n"
            "Confidence must be one of: low, medium, high.\n"
            "Reply with JSON only using this schema:\n"
            '{"type":"KNOWN|OUTLIER","class":"label","confidence":"low|medium|high","reason":"short reason"}'
        )

    def _parse_response(self, output_text, item):
        yolo_class = item.get("yolo_class", "unknown")
        display_class = item.get("display_class", yolo_class)

        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not match:
            return self._fallback_result(item, raw_response=output_text)

        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return self._fallback_result(item, raw_response=output_text)

        result_type = str(payload.get("type", "")).upper()
        raw_class = str(payload.get("class", "")).strip().lower()
        confidence = str(payload.get("confidence", "medium")).strip().lower()

        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"

        if result_type not in {"KNOWN", "OUTLIER"}:
            result_type = "KNOWN" if raw_class in self.known_classes else "OUTLIER"

        if result_type == "KNOWN":
            canonical_class = self._match_known_class(raw_class, yolo_class)
            return {
                "type": "KNOWN",
                "class": canonical_class,
                "display_class": display_class if canonical_class == yolo_class else canonical_class,
                "confidence": confidence,
            }

        outlier_class = raw_class or yolo_class
        return {
            "type": "OUTLIER",
            "class": outlier_class,
            "display_class": outlier_class,
            "confidence": confidence,
        }

    def _match_known_class(self, raw_class, fallback_class):
        if raw_class in self.known_classes:
            return raw_class

        aliases = {
            "car": "vehicle",
            "bus": "vehicle",
            "truck": "vehicle",
            "van": "vehicle",
            "motorcycle": "vehicle",
            "bicycle": "vehicle",
            "bike": "vehicle",
            "bagpack": "backpack",
            "handbag": "bag",
            "trolley": "cart",
        }

        if raw_class in aliases and aliases[raw_class] in self.known_classes:
            return aliases[raw_class]
        if fallback_class in self.known_classes:
            return fallback_class
        if fallback_class in aliases and aliases[fallback_class] in self.known_classes:
            return aliases[fallback_class]
        return "unknown"

    def _fallback_result(self, item, raw_response=None, error=None):
        yolo_class = item.get("yolo_class", "unknown")
        display_class = item.get("display_class", yolo_class)

        if yolo_class in self.known_classes:
            return {
                "type": "KNOWN",
                "class": yolo_class,
                "display_class": display_class,
                "confidence": "medium",
                "raw_response": raw_response,
                "error": error,
            }

        return {
            "type": "OUTLIER",
            "class": yolo_class,
            "display_class": display_class,
            "confidence": "medium",
            "raw_response": raw_response,
            "error": error,
        }

    def _to_pil_image(self, crop):
        if isinstance(crop, Image.Image):
            return crop
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _select_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _select_dtype(self, device):
        if device in {"cuda", "mps"}:
            return torch.float16
        return torch.float32
