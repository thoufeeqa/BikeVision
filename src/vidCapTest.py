import cv2
from ultralytics import YOLO
import numpy as np
import os

# Configure display for Raspberry Pi
os.environ["DISPLAY"] = ":0"
cv2.startWindowThread()

# ---- CONFIG ----
MODEL_PATH = "yolo11n.pt"   # or yolo11s.pt, yolo11m.pt, etc.
SOURCE = "data/videos/Oct13_frontMount_720p_trim.mp4"                  # 0 = default webcam
CONFIDENCE = 0.4            # detection threshold
#CLASSES = ['person'] # restrict to key safety classes (optional)
CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'stop sign', 'traffic light']

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    raise RuntimeError("Could not open webcam/source")

print("[INFO] Press 'q' to quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, stream=True, conf=CONFIDENCE, tracker="bytetrack.yaml")

    for result in results:
        annotated = result.plot()
        
        # Display frame without macOS-specific properties
        cv2.namedWindow("Bike Vision", cv2.WINDOW_NORMAL)
        cv2.imshow("Bike Vision", annotated)

    # Wait for key press (1ms)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
