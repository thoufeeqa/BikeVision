import cv2
from ultralytics import YOLO
import numpy as np

# ---- CONFIG ----
MODEL_PATH = "yolo11n.pt"   # or yolo11s.pt, yolo11m.pt, etc.
SOURCE = "../data/videos/Oct13_frontMount_720p.mp4"                  # 0 = default webcam
CONFIDENCE = 0.4            # detection threshold
#CLASSES = ['person'] # restrict to key safety classes (optional)
CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'stop sign', 'traffic light']

model = YOLO(MODEL_PATH)

results = model.track(SOURCE, stream=True, conf=CONFIDENCE, tracker="bytetrack.yaml")

while True:
    for result in results:
        annotated = result.plot()
        cv2.imshow("Bike Vision – YOLOv11 Feed", annotated)
        cv2.setWindowProperty("Bike Vision – YOLOv11 Feed", cv2.WND_PROP_TOPMOST, 1)

        # Keep window visible on top (macOS)


    # macOS needs waitKey>0 to refresh windows
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
