import cv2
from ultralytics import YOLO
import numpy as np
import os
import sys
from picamera2 import Picamera2

# Debug info
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Display env: {os.environ.get('DISPLAY', 'Not set')}")

# Try to create a test window first
try:
    cv2.namedWindow("Test")
    cv2.destroyWindow("Test")
    print("OpenCV window creation test: SUCCESS")
except Exception as e:
    print(f"OpenCV window creation test: FAILED - {str(e)}")

# ---- CONFIG ----
MODEL_PATH = "yolo11n.pt"   # or yolo11s.pt, yolo11m.pt, etc.
SOURCE = "data/videos/Oct13_frontMount_720p_trim.mp4"                  # 0 = default webcam
CONFIDENCE = 0.4            # detection threshold
#CLASSES = ['person'] # restrict to key safety classes (optional)
CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'stop sign', 'traffic light']

model = YOLO(MODEL_PATH)

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

#cap = cv2.VideoCapture(0)




print("[INFO] Press 'q' to quit")

while True:
    print("in loop")

    frame = picam2.capture_array()
    if frame is None:
        print("Failed to capture frame")
        break

    results = model.track(frame, stream=True, conf=CONFIDENCE, tracker="bytetrack.yaml")

    for result in results:
        try:
            annotated = result.plot()
            
            # Create window with flags for RPi compatibility
            cv2.namedWindow("Bike Vision", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("Bike Vision", annotated)
            print("Frame displayed successfully", end="\r")
        except Exception as e:
            print(f"\nDisplay error: {str(e)}")
            break

    # Shorter wait time for RPi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
