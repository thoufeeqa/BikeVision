import cv2
from ultralytics import YOLO
import numpy as np
import os
import sys

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


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    raise RuntimeError("Could not open webcam/source")

print("[INFO] Press 'q' to quit")

while True:
    print("in loop")
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    cv2.namedWindow("Bike Vision", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Bike Vision", frame)
            

    # Shorter wait time for RPi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
