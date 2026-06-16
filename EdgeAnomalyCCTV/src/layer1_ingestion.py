import cv2
import hashlib

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
            frame = cv2.imread(self.source)
            if frame is None:
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                print(f"Warning: Could not read {self.source}. Using a blank image for testing.")
            return {
                "raw_frame": frame,
                "source_type": "IMAGE",
                "source_id": hashlib.md5(self.source.encode()).hexdigest(),
                "timestamp": cv2.getTickCount()
            }
