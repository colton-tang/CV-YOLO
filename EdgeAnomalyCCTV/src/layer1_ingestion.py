import cv2
import hashlib
from pathlib import Path

class IngestionLayer:
    def __init__(self, mode="VIDEO", source=None):
        self.mode = mode
        self.source = source
        self.cap = None
        if self.mode == "VIDEO" and source:
            self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        if self.mode == "VIDEO":
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    return {
                        "raw_frame": frame,
                        "source_type": "VIDEO",
                        "source_id": self.source,
                        "timestamp": cv2.getTickCount()
                    }
            return None
        elif self.mode == "IMAGE":
            image_path = self._resolve_source_path(self.source)
            frame = cv2.imread(str(image_path))
            if frame is None:
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                print(f"Warning: Could not read image '{self.source}'. Using a blank image for testing.")
            return {
                "raw_frame": frame,
                "source_type": "IMAGE",
                "source_id": hashlib.md5(str(image_path).encode()).hexdigest(),
                "timestamp": cv2.getTickCount()
            }

    def _resolve_source_path(self, source):
        if source is None:
            return Path("")

        source_path = Path(source)
        if source_path.is_absolute():
            return source_path

        cwd_candidate = Path.cwd() / source_path
        if cwd_candidate.exists():
            return cwd_candidate

        project_root_candidate = Path(__file__).resolve().parents[2] / source_path
        if project_root_candidate.exists():
            return project_root_candidate

        workspace_candidate = Path(__file__).resolve().parents[3] / source_path
        if workspace_candidate.exists():
            return workspace_candidate

        return cwd_candidate
