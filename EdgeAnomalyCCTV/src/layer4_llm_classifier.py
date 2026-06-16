import asyncio
import json
import re
import traceback

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class LLMClassifierLayer:
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

    def __init__(self, queue, known_classes, track_state_db=None):
        self.queue = queue
        self.known_classes = known_classes or []
        self.track_state_db = track_state_db if track_state_db is not None else {}

        self.device = self._select_device()
        print(f"[LLM] Loading {self.MODEL_ID} on {self.device}...")
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32

        try:
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=self.dtype,
                device_map=None,  # we move manually to keep control
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
            print(f"[LLM] {self.MODEL_ID} loaded successfully.")
        except Exception as exc:
            print(f"[LLM] Failed to load {self.MODEL_ID}: {exc}")
            traceback.print_exc()
            self.processor = None
            self.model = None

    @staticmethod
    def _select_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    async def run(self):
        while True:
            item = await self.queue.get()
            track_id = item.get("track_id")
            yolo_class = item.get("yolo_class", "unknown")
            yolo_conf = item.get("yolo_conf", 0.0)
            trigger_reason = item.get("trigger_reason", "")

            print(
                f"[VLM] Processing track {track_id}: yolo='{yolo_class}' "
                f"conf={yolo_conf:.2f} reason='{trigger_reason}'"
            )

            try:
                result = await self._run_llm(item)

                status = "UNKNOWN"
                if result["type"] == "KNOWN":
                    status = "RESOLVED"
                elif result["type"] == "OUTLIER":
                    status = "OUTLIER"

                print(
                    f"[VLM] Result for track {track_id}: status={status} "
                    f"class='{result.get('class', 'unknown')}' "
                    f"confidence='{result.get('confidence', 'low')}'"
                )

                if self.track_state_db is not None:
                    self.track_state_db[track_id] = {
                        "status": status,
                        "class": result["class"],
                        "display_class": result["class"],
                        "confidence": result.get("confidence", "low"),
                        "reason": result.get("reason", ""),
                        "bbox": item.get("bbox"),
                        "yolo_class": yolo_class,
                        "yolo_conf": yolo_conf,
                        "vlm_processed": True,
                    }
            except Exception as exc:
                print(f"[LLM] Error processing item {track_id}: {exc}")
                traceback.print_exc()
            finally:
                self.queue.task_done()

    async def _run_llm(self, item):
        if self.model is None or self.processor is None:
            # Model failed to load; return a fallback so the pipeline doesn't hang.
            return {
                "type": "OUTLIER",
                "class": item.get("display_class", "unknown"),
                "confidence": "low",
                "reason": "VLM model failed to load",
            }

        crop = item["crop"]
        yolo_class = item.get("yolo_class", "unknown")
        yolo_conf = item.get("yolo_conf", 0.0)
        trigger_reason = item.get("trigger_reason", "Uncertain detection")
        known_classes_text = ", ".join(self.known_classes)

        # Convert OpenCV BGR ndarray to PIL RGB image.
        image = Image.fromarray(crop).convert("RGB")

        system_prompt = (
            "You are a visual classifier for a security camera anomaly-detection pipeline. "
            "Given a cropped image of a single object, decide if it belongs to a known set of "
            "everyday classes or if it is an outlier/anomaly/foreign object."
        )

        user_prompt = (
            f"The detector initially thought this object was '{yolo_class}' with confidence {yolo_conf:.2f}. "
            f"Trigger reason: {trigger_reason}.\n\n"
            f"Known classes: {known_classes_text}.\n\n"
            "Analyze the image. Respond with ONLY a JSON object in this exact format:\n"
            '{"type": "KNOWN" or "OUTLIER", "class": "short class name", "confidence": "high/medium/low", "reason": "one-sentence explanation"}'
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # Run the heavy inference in a thread pool so the asyncio event loop stays responsive.
        return await asyncio.to_thread(self._infer, messages)

    def _infer(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response):
        response = response.strip()
        # Try to extract JSON from the response.
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                return {
                    "type": parsed.get("type", "OUTLIER").upper(),
                    "class": parsed.get("class", "unknown"),
                    "confidence": parsed.get("confidence", "low").lower(),
                    "reason": parsed.get("reason", ""),
                }
            except json.JSONDecodeError:
                pass

        # Fallback heuristic: look for KNOWN/OUTLIER in the text.
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
