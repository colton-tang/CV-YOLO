import cv2
import json


class RenderAlertLayer:
    """Layer 5: Render + Alert (Display / API Response)."""

    # BGR color palette matching the spec
    COLORS = {
        "RESOLVED": (0, 255, 0),    # Green
        "VERIFYING": (0, 255, 255), # Yellow
        "OUTLIER": (0, 0, 255),     # Red
        "UNKNOWN": (255, 0, 0),     # Blue
        "NEW": (128, 128, 128),     # Gray
    }
    WINDOW_NAME = "EdgeAnomalyCCTV"

    def __init__(self):
        self._window_created = False
        self._last_video_summary = None

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            self._window_created = True

    def _get_color(self, status):
        return self.COLORS.get(status, self.COLORS["NEW"])

    def _draw_overlay(self, frame, tracks, tracking_state):
        for track in tracks:
            track_id = track["track_id"]
            bbox = track.get("bbox")
            if bbox is None:
                continue

            state = tracking_state.get(track_id, {})
            status = state.get("status")
            cls = state.get("display_class") or state.get("class") or track.get("display_class") or track.get("class") or "unknown"
            conf = state.get("confidence")
            if conf is None:
                conf = track.get("conf", 0.0)

            color = self._get_color(status)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # VLM confidence is a string (high/medium/low), YOLO confidence is a float.
            if isinstance(conf, str):
                conf_str = conf
            else:
                conf_str = f"{conf:.2f}"

            label = f"{cls} | {status or 'NEW'} | {conf_str}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def process(self, tracking_state, source_type, raw_frame=None, mqtt_client=None, tracks=None):
        tracks = tracks or []

        if source_type == "VIDEO":
            self._ensure_window()
            if raw_frame is not None:
                overlay = raw_frame.copy()
                overlay = self._draw_overlay(overlay, tracks, tracking_state)
                cv2.imshow(self.WINDOW_NAME, overlay)

            outlier_count = sum(1 for s in tracking_state.values() if s.get("status") == "OUTLIER")
            summary = (len(tracks), outlier_count)
            if summary != self._last_video_summary:
                print(f"\n[VIDEO] Rendered {len(tracks)} tracks | outliers: {outlier_count}")
                self._last_video_summary = summary

        elif source_type == "IMAGE":
            print(f"\n--- Output Results (Source: {source_type}) ---")
            if not tracking_state:
                print("Image Render: No detections survived the pipeline for this image.")

            if raw_frame is not None:
                output_image = raw_frame.copy()
                output_image = self._draw_overlay(output_image, tracks, tracking_state)
                cv2.imwrite("EdgeAnomalyCCTV/output.jpg", output_image)
                print("Image Render: Static image annotated and saved to 'EdgeAnomalyCCTV/output.jpg'.")

            print("JSON Response:")
            print(json.dumps(tracking_state, indent=2))
            print("-------------------------------------------\n")
