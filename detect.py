#Detection script for YOLO11 Seat Belt Detection


from ultralytics import YOLO
import supervision as sv
import numpy as np

# Paths (update with your files)
MODEL_PATH = "runs/train/SeatBelt/weights/best.pt"
SOURCE_VIDEO_PATH = "input.mp4"
TARGET_VIDEO_PATH = "output.mp4"

# Load and fuse model
model = YOLO(MODEL_PATH)
model.fuse()

# Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

CONFIDENCE_THRESHOLD = 0.7 # This was used to improve model detection reliability

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Keep only high confidence
    mask = detections.confidence >= CONFIDENCE_THRESHOLD
    detections = sv.Detections(
        xyxy=detections.xyxy[mask],
        class_id=detections.class_id[mask],
        confidence=detections.confidence[mask],
    )

    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=[f"{conf:.2f}" for conf in detections.confidence],
    )
    return frame

if __name__ == "__main__":
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )
