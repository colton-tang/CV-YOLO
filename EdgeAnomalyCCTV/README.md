# EdgeAnomalyCCTV

A unified framework for outlier anomaly detection supporting both RTSP Video streams and single Images.

## Architecture Let
There are 5 layers:
1. **Ingestion:** Frame buffer from RTSP streams or image inputs.
2. **Detection & Tracking:** YOLOv8 + ByteTrack (for video), YOLOv8 (synthetic tracking for image).
3. **Outlier Filter:** Gates for Deduplication, Auto-Pass, Uncertainty Check.
4. **LLM Outlier Classifier:** Qwen2-VL-2B-Instruct outlier detection async queue.
5. **Render & Alert:** Output overlays, MQTT alerts, or API response.
