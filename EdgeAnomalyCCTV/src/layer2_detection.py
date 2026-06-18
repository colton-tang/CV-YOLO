import hashlib
import time
from constants import LOW_CONFIDENCE_THRESHOLD
from ultralytics import YOLO


class TrackLifeManager:
    """TrackLife Manager: birth_time, last_seen, trajectory, age, velocity."""

    def __init__(self):
        self.tracks = {}

    def update(self, track_id, bbox, timestamp):
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "birth_time": timestamp,
                "last_seen": timestamp,
                "trajectory": [bbox],
                "age": 1,
            }
        else:
            record = self.tracks[track_id]
            record["last_seen"] = timestamp
            record["trajectory"].append(bbox)
            record["age"] += 1

    def get_state(self, track_id):
        record = self.tracks.get(track_id, {})
        trajectory = record.get("trajectory", [])

        velocity = 0.0
        if len(trajectory) >= 2:
            last = trajectory[-1]
            prev = trajectory[-2]
            cx_last = (last[0] + last[2]) / 2.0
            cy_last = (last[1] + last[3]) / 2.0
            cx_prev = (prev[0] + prev[2]) / 2.0
            cy_prev = (prev[1] + prev[3]) / 2.0
            velocity = ((cx_last - cx_prev) ** 2 + (cy_last - cy_prev) ** 2) ** 0.5

        return {
            "birth_time": record.get("birth_time"),
            "last_seen": record.get("last_seen"),
            "trajectory": trajectory,
            "age": record.get("age", 0),
            "velocity": velocity,
        }


class DetectionTrackingLayer:
    CLASS_ALIASES = {
        "car": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
        "train": "vehicle",
        "motorcycle": "vehicle",
        "bicycle": "vehicle",
        "boat": "vehicle",
        "handbag": "bag",
    }

    def __init__(self, model_path="weights/yolo/yolov8n.pt", conf_threshold=LOW_CONFIDENCE_THRESHOLD, known_classes=None):
        self.conf_threshold = conf_threshold
        self.known_classes = set(known_classes or [])
        self.model = YOLO(model_path)
        self.track_life = TrackLifeManager()

        if "world" in model_path.lower():
            import yaml
            from ultralytics.utils.checks import check_yaml
            try:
                with open(check_yaml("lvis.yaml"), "r", encoding="utf-8") as f:
                    lvis_data = yaml.safe_load(f)
                yolo_classes = list(lvis_data["names"].values())
            except Exception:
                yolo_classes = list(self.known_classes) + ["unknown object", "anomaly", "foreign object"]
            self.model.set_classes(yolo_classes)

    def process(self, input_data):
        raw_frame = input_data["raw_frame"]
        source_type = input_data["source_type"]
        source_id = input_data["source_id"]
        timestamp = input_data.get("timestamp", time.time())

        if source_type == "VIDEO":
            return self._process_video(raw_frame, source_id, timestamp)
        elif source_type == "IMAGE":
            return self._process_image(raw_frame, source_id)
        return []

    def _process_video(self, raw_frame, source_id, timestamp):
        results = []
        yolo_results = self.model.track(
            raw_frame,
            conf=0.1,  # Lower confidence to let tracker receive low-confidence detections (0.1-0.3)
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
        )

        if not yolo_results:
            return results

        result = yolo_results[0]
        if result.boxes is None or result.boxes.id is None:
            return results

        names = result.names
        boxes = result.boxes
        for box, track_id_tensor in zip(boxes, boxes.id):
            cls_id = int(box.cls[0].item())
            detected_label = names[cls_id]
            canonical_label = self._canonicalize_label(detected_label)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            track_id = int(track_id_tensor.item())

            self.track_life.update(track_id, [x1, y1, x2, y2], timestamp)

            # Filter out low-confidence detections (below self.conf_threshold) from being propagated
            # to downstream application layers, while still allowing the tracker to update them.
            if conf < self.conf_threshold:
                continue

            life = self.track_life.get_state(track_id)

            results.append({
                "track_id": track_id,
                "class": canonical_label,
                "display_class": detected_label,
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
                "age": life["age"],
                "velocity": life["velocity"],
                "source_type": "VIDEO",
                "source_id": source_id,
                "raw_frame": raw_frame,
                "birth_time": life["birth_time"],
                "last_seen": life["last_seen"],
                "trajectory": life["trajectory"],
            })

        return results

    def _process_image(self, raw_frame, source_id):
        results = []
        boxes, classes, display_classes, confs = self._run_yolo(raw_frame)
        base_track_id = hashlib.md5(str(source_id).encode()).hexdigest()

        for idx, (bbox, cls, display_cls, conf) in enumerate(zip(boxes, classes, display_classes, confs)):
            # Give each detection in an image its own track id so the filter
            # can evaluate every object independently (IMAGE mode has no
            # temporal tracking anyway).
            synthetic_track_id = f"{base_track_id}_{idx}"
            results.append({
                "track_id": synthetic_track_id,
                "class": cls,
                "display_class": display_cls,
                "conf": conf,
                "bbox": bbox,
                "age": 1,
                "velocity": 0,
                "source_type": "IMAGE",
                "source_id": source_id,
                "raw_frame": raw_frame,
            })

        return results

    def _run_yolo(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        boxes = []
        classes = []
        display_classes = []
        confs = []

        if not results:
            return boxes, classes, display_classes, confs

        result = results[0]
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            detected_label = names[cls_id]
            canonical_label = self._canonicalize_label(detected_label)

            boxes.append([x1, y1, x2, y2])
            classes.append(canonical_label)
            display_classes.append(detected_label)
            confs.append(float(box.conf[0].item()))

        return boxes, classes, display_classes, confs

    def _canonicalize_label(self, detected_label):
        alias_label = self.CLASS_ALIASES.get(detected_label)
        if alias_label and alias_label in self.known_classes:
            return alias_label
        return detected_label
