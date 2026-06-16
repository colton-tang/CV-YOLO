from constants import LOW_CONFIDENCE_THRESHOLD
from ultralytics import YOLO


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

    def __init__(self, model_path="yolov8n.pt", conf_threshold=LOW_CONFIDENCE_THRESHOLD, known_classes=None):
        self.conf_threshold = conf_threshold
        self.known_classes = set(known_classes or [])
        self.model = YOLO(model_path)

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

        boxes, classes, display_classes, confs = self._run_yolo(raw_frame)

        results = []
        if source_type == "VIDEO":
            tracks = self._run_tracking(boxes, classes, display_classes, confs)
            for t in tracks:
                results.append({
                    "track_id": t["id"],
                    "class": t["class"],
                    "display_class": t.get("display_class", t["class"]),
                    "conf": t["conf"],
                    "bbox": t["bbox"],
                    "age": t["age"],
                    "velocity": t["velocity"],
                    "source_type": source_type,
                    "source_id": source_id,
                    "raw_frame": raw_frame,
                })
        elif source_type == "IMAGE":
            for i, (bbox, cls, display_cls, conf) in enumerate(zip(boxes, classes, display_classes, confs)):
                synthetic_track_id = f"{source_id}_{i}"
                results.append({
                    "track_id": synthetic_track_id,
                    "class": cls,
                    "display_class": display_cls,
                    "conf": conf,
                    "bbox": bbox,
                    "age": 1,
                    "velocity": 0,
                    "source_type": source_type,
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

    def _run_tracking(self, boxes, classes, display_classes, confs):
        return []

    def _canonicalize_label(self, detected_label):
        alias_label = self.CLASS_ALIASES.get(detected_label)
        if alias_label and alias_label in self.known_classes:
            return alias_label
        return detected_label
