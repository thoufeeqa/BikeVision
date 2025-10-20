import cv2
from picamera2 import Picamera2
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


picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("Pi Camera 3 Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


