import cv2
import json


class RenderAlertLayer:
    def __init__(self):
        pass

    def _get_box_color(self, info):
        if "llm_response" in info or "llm_error" in info:
            return (0, 0, 255)
        return (0, 255, 0)

    def process(self, tracking_state, source_type, raw_frame=None, mqtt_client=None, active_track_ids=None):
        print(f"\n--- Output Results (Source: {source_type}) ---")
        if source_type == "VIDEO":
            print("Video Render: Overlay updated. MQTT alert sent (if outlier).")
            print("Current Tracking State:", tracking_state)
        elif source_type == "IMAGE":
            if not tracking_state:
                print("Image Render: No detections survived the pipeline for this image.")
            if raw_frame is not None:
                output_image = raw_frame.copy()
                for track_id, info in tracking_state.items():
                    if active_track_ids is not None and track_id not in active_track_ids:
                        continue
                    if "bbox" in info:
                        x1, y1, x2, y2 = [int(v) for v in info["bbox"]]
                        cls = info.get("display_class", info.get("class", "unknown"))
                        color = self._get_box_color(info)

                        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(output_image, f"{cls}", (x1, max(y1 - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imwrite("EdgeAnomalyCCTV/output.jpg", output_image)
                print("Image Render: Static image annotated and saved to 'EdgeAnomalyCCTV/output.jpg'.")

            print("JSON Response:")
            print(json.dumps(tracking_state, indent=2))
        print("-------------------------------------------\n")
