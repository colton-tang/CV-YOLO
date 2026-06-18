import asyncio
import cv2
import gc
import sys
import argparse
from pathlib import Path
import torch
from constants import COCO_CLASSES
from layer1_ingestion import IngestionLayer
from layer2_detection import DetectionTrackingLayer
from layer3_filtering import GateOutlierFilterLayer
from layer5_render import RenderAlertLayer

KNOWN_CLASSES = COCO_CLASSES

async def main():
    parser = argparse.ArgumentParser(description="EdgeAnomalyCCTV Main Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["graph", "video"],
        help="Mode to run: 'graph' (static image/directory benchmark) or 'video' (real-time camera/video)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Input source. For graph mode: path to a single image. "
            "For video mode: path to a video file, RTSP/HTTP URL, or camera index (e.g., 0). "
            "If omitted, graph mode defaults to the bundled benchmark image and video mode defaults to camera 0."
        ),
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default="weights/yolo/yolov8n.pt",
        help="Path to the YOLO detector weights to use.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the LLM verification stage and render detector/gate results only.",
    )
    args, unknown = parser.parse_known_args()

    mode = args.mode
    if not mode:
        print("Select run mode:")
        print("1. Graph (Outlier Detection Benchmark & Plot)")
        print("2. Video (Real-time Camera Detection)")
        try:
            choice = input("Enter choice (1/2 or graph/video): ").strip().lower()
            if choice in ("1", "graph"):
                mode = "graph"
            elif choice in ("2", "video"):
                mode = "video"
            else:
                print("Invalid choice. Defaulting to 'video'.")
                mode = "video"
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    # Resolve input source
    if args.input is None:
        resolved_input = str(Path(__file__).resolve().parents[2] / "benchmark_data" / "kitchen_person.jpg") if mode == "graph" else "0"
    else:
        resolved_input = args.input

    # Initialization based on mode
    if mode == "graph":
        print("\nStarting Static Image Detection (Graph mode)...")
        ingestion = IngestionLayer(mode="IMAGE", source=resolved_input)
    else:
        print("\nStarting Real-time Camera Detection (Video mode)...")
        ingestion = IngestionLayer(mode="VIDEO", source=resolved_input, width=640, height=480)
        if ingestion.mode == "VIDEO" and (not ingestion.cap or not ingestion.cap.isOpened()):
            print(f"\nError: Video source '{resolved_input}' could not be opened.")
            print("Please check if:")
            print("1. Another application is currently using your webcam.")
            print("2. The camera index / path / URL is correct.")
            print("3. macOS has camera permission enabled for this process.")
            return

    detection = DetectionTrackingLayer(model_path=args.detector_model, known_classes=KNOWN_CLASSES)
    filtering = GateOutlierFilterLayer(known_classes=KNOWN_CLASSES)
    render = RenderAlertLayer()
    classifier = None
    llm_task = None

    if not args.skip_llm:
        from layer4_llm_classifier import LLMClassifierLayer

        classifier = LLMClassifierLayer(
            queue=filtering.llm_queue,
            known_classes=KNOWN_CLASSES,
            track_state_db=filtering.track_state_db,
        )
        llm_task = asyncio.create_task(classifier.run())

    # Processing Loop
    try:
        if mode == "graph":
            # Process single static image frame
            frame_data = ingestion.get_frame()
            if frame_data:
                tracks = detection.process(frame_data)
                await filtering.process(tracks)
                
                # Wait for LLM classification to finish for images when enabled.
                if llm_task is not None:
                    await filtering.llm_queue.join()
                
                render.process(filtering.track_state_db, frame_data["source_type"], raw_frame=frame_data["raw_frame"], tracks=tracks)
        else:
            # Video Mode Loop
            while True:
                frame_data = ingestion.get_frame()
                if frame_data is None:
                    await asyncio.sleep(0.01)
                    continue
                
                tracks = detection.process(frame_data)
                await filtering.process(tracks)
                
                render.process(filtering.track_state_db, frame_data["source_type"], raw_frame=frame_data["raw_frame"], tracks=tracks)
                
                # Press 'q' in the window or close it to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if render._window_created and cv2.getWindowProperty(render.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
                # Yield control to let asyncio tasks run (like the LLM classifier)
                await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        print("\n[MAIN] Shutting down...")
        if ingestion.cap:
            ingestion.cap.release()
        cv2.destroyAllWindows()

        # Graceful LLM shutdown with timeout
        if classifier is not None and llm_task is not None:
            classifier.shutdown()
            llm_task.cancel()
            try:
                await asyncio.wait_for(llm_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Release model memory
        try:
            if classifier is not None:
                del classifier.model
                del classifier.processor
        except Exception:
            pass
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[MAIN] Shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
