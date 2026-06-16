import asyncio
import cv2
import sys
import argparse
from pathlib import Path
from constants import COCO_CLASSES
from layer1_ingestion import IngestionLayer
from layer2_detection import DetectionTrackingLayer
from layer3_filtering import GateOutlierFilterLayer
from layer4_llm_classifier import LLMClassifierLayer
from layer5_render import RenderAlertLayer

KNOWN_CLASSES = COCO_CLASSES

async def main():
    parser = argparse.ArgumentParser(description="EdgeAnomalyCCTV Main Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["graph", "video"],
        help="Mode to run: 'graph' (benchmark plot) or 'video' (real-time camera)"
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

    # Initialization based on mode
    if mode == "graph":
        print("\nStarting Static Image Detection (Graph mode)...")
        image_path = str(Path(__file__).resolve().parents[2] / "benchmark_data" / "kitchen_person.jpg")
        ingestion = IngestionLayer(mode="IMAGE", source=image_path)
    else:
        print("\nStarting Real-time Camera Detection (Video mode)...")
        ingestion = IngestionLayer(mode="VIDEO", source="0", width=640, height=480)
        if ingestion.mode == "VIDEO" and (not ingestion.cap or not ingestion.cap.isOpened()):
            print("\nError: Camera source '0' could not be opened.")
            print("Please check if:")
            print("1. Another application is currently using your webcam.")
            print("2. The camera index (0) is correct (try '1' if you have an external webcam).")
            print("3. macOS has camera permission enabled for this process.")
            return

    detection = DetectionTrackingLayer(known_classes=KNOWN_CLASSES)
    filtering = GateOutlierFilterLayer(known_classes=KNOWN_CLASSES)
    classifier = LLMClassifierLayer(queue=filtering.llm_queue, known_classes=KNOWN_CLASSES, track_state_db=filtering.track_state_db)
    render = RenderAlertLayer()

    # Start LLM consumer in background
    llm_task = asyncio.create_task(classifier.run())

    # Processing Loop
    try:
        if mode == "graph":
            # Process single static image frame
            frame_data = ingestion.get_frame()
            if frame_data:
                tracks = detection.process(frame_data)
                await filtering.process(tracks)
                
                # Wait for LLM classification to finish for images
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
                
                # Press 'q' in the window to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                # Yield control to let asyncio tasks run (like the LLM classifier)
                await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if ingestion.cap:
            ingestion.cap.release()
        cv2.destroyAllWindows()
        llm_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
