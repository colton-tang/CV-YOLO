import asyncio
from constants import COCO_CLASSES
from layer1_ingestion import IngestionLayer
from layer2_detection import DetectionTrackingLayer
from layer3_filtering import GateOutlierFilterLayer
from layer4_llm_classifier import LLMClassifierLayer
from layer5_render import RenderAlertLayer

KNOWN_CLASSES = COCO_CLASSES

async def main():
    # Initialization
    ingestion = IngestionLayer(mode="IMAGE", source="benchmark_data/street_signs.jpg")
    detection = DetectionTrackingLayer(known_classes=KNOWN_CLASSES)
    filtering = GateOutlierFilterLayer(known_classes=KNOWN_CLASSES)
    classifier = LLMClassifierLayer(queue=filtering.llm_queue, known_classes=KNOWN_CLASSES, track_state_db=filtering.track_state_db)
    render = RenderAlertLayer()

    # Start LLM consumer in background
    llm_task = asyncio.create_task(classifier.run())

    # Mock Processing Loop
    try:
        frame_data = ingestion.get_frame()
        if frame_data:
            tracks = detection.process(frame_data)
            await filtering.process(tracks)
            
            # Wait for LLM classification to finish for images
            if frame_data["source_type"] == "IMAGE":
                await filtering.llm_queue.join()
            
            render.process(filtering.track_state_db, frame_data["source_type"], raw_frame=frame_data["raw_frame"])
    except KeyboardInterrupt:
        pass
    finally:
        llm_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
