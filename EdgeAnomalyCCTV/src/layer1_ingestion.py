import cv2
import time
from collections import deque


class IngestionLayer:
    """Layer 1: Ingestion (Camera / Image → Frame Buffer)."""

    def __init__(self, mode="VIDEO", source=None, buffer_seconds=2.0, target_fps=25.0):
        self.mode = mode
        self.source = self._normalize_source(source)
        self.source_id = str(self.source)
        self.cap = None
        self._opened = False
        self._is_live_stream = self._infer_is_live_stream(self.source)

        # Ring buffer configuration (last N seconds at target_fps)
        self.buffer_seconds = buffer_seconds
        self.target_fps = target_fps
        self.buffer_size = int(buffer_seconds * target_fps)
        self.frame_buffer = deque(maxlen=self.buffer_size)

        if self.mode == "VIDEO" and source is not None:
            self._open_capture()

    @staticmethod
    def _normalize_source(source):
        """Convert string digits to integers so OpenCV treats them as camera indices."""
        if isinstance(source, str) and source.strip().lstrip("-").isdigit():
            return int(source)
        return source

    @staticmethod
    def _infer_is_live_stream(source):
        """Heuristic: integer indices and network URLs are treated as live streams."""
        if source is None:
            return False
        if isinstance(source, int):
            return True
        s = str(source).lower()
        return s.startswith(("rtsp://", "http://", "https://", "rtp://", "udp://"))

    def _open_capture(self):
        """Open (or reopen) the video capture stream."""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        self._opened = self.cap.isOpened()
        return self._opened

    def get_frame(self):
        """Return the latest frame in the unified format.

        Unified Output: {raw_frame, source_type, source_id, timestamp}
        """
        if self.mode == "VIDEO":
            if self.cap is None or not self._opened:
                if not self._open_capture():
                    return None

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # For live streams, attempt one reconnection before giving up.
                if not self._is_live_stream or not self._open_capture():
                    return None
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return None

            frame_data = {
                "raw_frame": frame,
                "source_type": "VIDEO",
                "source_id": self.source_id,
                "timestamp": time.time(),
            }
            self.frame_buffer.append(frame_data)
            return frame_data

        elif self.mode == "IMAGE":
            frame = cv2.imread(self.source)
            if frame is None:
                import numpy as np

                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                print(f"Warning: Could not read {self.source}. Using a blank image for testing.")

            return {
                "raw_frame": frame,
                "source_type": "IMAGE",
                "source_id": self.source_id,
                "timestamp": time.time(),
            }

        return None
