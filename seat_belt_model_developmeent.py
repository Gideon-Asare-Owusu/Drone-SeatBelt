# Training YOLO11 Model

from ultralytics import YOLO


model = YOLO("yolo11x.pt")  # Pretrained weights


model.train(
    data="/path/to/data.yaml",  # Dataset YAML
    epochs=200,
    imgsz=1024,
    batch=16,
    lr0=0.001667,
    optimizer='AdamW',
    momentum=0.9,
    weight_decay=0.0005,
    cos_lr=True,
    project="/path/to/output/results",
    name="Seat Belt"
)


# Inferencing Model

from ultralytics import YOLO
import supervision as sv
import numpy as np

# Load trained model and fuse
MODEL_PATH = "/path/to/trained/model/best.pt"
model = YOLO(MODEL_PATH)
model.fuse()

# Input/output video paths
SOURCE_VIDEO_PATH = "/content/gdrive/MyDrive/Model 10 Replica/Asante 1.mp4"
TARGET_VIDEO_PATH = "/content/gdrive/MyDrive/Model 10/Video Out4.mp4"

# Get video metadata and frames
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

# Confidence Threshold
CONFIDENCE_THRESHOLD = 0.7

# Callback Function for Frame Annotation
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    high_conf_idx = detections.confidence >= CONFIDENCE_THRESHOLD
    detections = sv.Detections(
        xyxy=detections.xyxy[high_conf_idx],
        class_id=detections.class_id[high_conf_idx],
        confidence=detections.confidence[high_conf_idx],
    )

    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=[f"{conf:.2f}" for conf in detections.confidence]
    )
    return frame

# Process video
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)
