import asyncio
from constants import COCO_CLASSES, HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD


class GateOutlierFilterLayer:
    def __init__(self, known_classes=None):
        self.known_classes = known_classes or COCO_CLASSES
        self.track_state_db = {}
        self.llm_queue = asyncio.Queue(maxsize=100)

    async def process(self, tracking_results):
        for obj in tracking_results:
            track_id = obj["track_id"]
            source_type = obj["source_type"]
            source_id = obj["source_id"]
            conf = obj["conf"]
            cls = obj["class"]
            display_class = obj.get("display_class", cls)

            dedup_key = track_id
            existing_state = self.track_state_db.get(dedup_key)
            current_status = existing_state.get("status") if existing_state else None

            # Gate 1: already finalized -> drop
            if current_status in ("RESOLVED", "OUTLIER", "UNKNOWN"):
                continue

            # Gate 2: high-confidence known class -> auto-pass
            if conf > HIGH_CONFIDENCE_THRESHOLD and cls in self.known_classes:
                self.track_state_db[dedup_key] = {
                    "status": "RESOLVED",
                    "class": cls,
                    "display_class": display_class,
                    "confidence": conf,
                    "bbox": obj["bbox"],
                }
                continue

            # Gate 3: uncertain or unknown class -> send to LLM (once)
            if (LOW_CONFIDENCE_THRESHOLD < conf <= HIGH_CONFIDENCE_THRESHOLD) or (cls not in self.known_classes):
                if current_status != "VERIFYING":
                    self.track_state_db[dedup_key] = {
                        "status": "VERIFYING",
                        "class": cls,
                        "display_class": display_class,
                        "confidence": conf,
                        "bbox": obj["bbox"],
                    }
                    try:
                        await self.llm_queue.put({
                            "track_id": track_id,
                            "crop": self._crop_bbox(obj["raw_frame"], obj["bbox"]),
                            "yolo_class": cls,
                            "display_class": display_class,
                            "yolo_conf": conf,
                            "trigger_reason": "Uncertain class" if cls in self.known_classes else "Unknown class",
                            "source_type": source_type,
                            "source_id": source_id,
                            "bbox": obj["bbox"],
                        })
                    except asyncio.QueueFull:
                        pass
                else:
                    # Keep VERIFYING state but refresh bbox/ confidence for rendering.
                    self.track_state_db[dedup_key]["bbox"] = obj["bbox"]
                    self.track_state_db[dedup_key]["confidence"] = conf

            # Below low threshold: drop (do not store)
            elif conf <= LOW_CONFIDENCE_THRESHOLD:
                continue

    def _crop_bbox(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]
