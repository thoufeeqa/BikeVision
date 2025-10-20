import cv2
from ultralytics import YOLO
import numpy as np
import os
import sys
import platform

# Debug info
print(f"Platform: {platform.system()}")
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Display env: {os.environ.get('DISPLAY', 'Not set')}")

# Platform-specific imports and setup
IS_RASPBERRY_PI = platform.system() == 'Linux' and os.path.exists('/sys/firmware/devicetree/base/model') and 'raspberry pi' in open('/sys/firmware/devicetree/base/model').read().lower()

try:
    if IS_RASPBERRY_PI:
        from picamera2 import Picamera2
        print("Running on Raspberry Pi with Picamera2")
    else:
        print("Running on macOS/other platform")
except ImportError:
    print("Warning: picamera2 not available")
    IS_RASPBERRY_PI = False

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

# Initialize camera based on platform
if IS_RASPBERRY_PI:
    try:
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (1280, 720)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.configure("preview")
        picam2.start()
        print("Initialized Raspberry Pi camera")
    except Exception as e:
        print(f"Failed to initialize Pi camera: {e}")
        IS_RASPBERRY_PI = False

if not IS_RASPBERRY_PI:
    # macOS or fallback to regular webcam
    cap = cv2.VideoCapture(0)
    if platform.system() == 'Darwin':  # macOS
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

print("[INFO] Press 'q' to quit")

try:
    while True:
        if IS_RASPBERRY_PI:
            frame = picam2.capture_array()
            if frame is None:
                print("Failed to capture frame from Pi camera")
                break
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam")
                break

        # Run model inference
        results = model.track(frame, stream=True, conf=CONFIDENCE, tracker="bytetrack.yaml")

        for result in results:
            try:
                annotated = result.plot()
                
                # Platform-specific window setup
                if IS_RASPBERRY_PI:
                    # RPi: Use simpler window flags
                    cv2.namedWindow("Bike Vision", cv2.WINDOW_NORMAL)
                else:
                    # macOS: Keep window on top
                    cv2.namedWindow("Bike Vision", cv2.WINDOW_NORMAL)
                    if platform.system() == 'Darwin':
                        cv2.setWindowProperty("Bike Vision", cv2.WND_PROP_TOPMOST, 1)
                
                cv2.imshow("Bike Vision", annotated)
            except Exception as e:
                print(f"\nDisplay error: {str(e)}")
                break

        # Platform-specific wait time
        wait_time = 1 if IS_RASPBERRY_PI else 30  # longer on macOS for smooth display
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    if not IS_RASPBERRY_PI:
        cap.release()
    else:
        picam2.stop()
    cv2.destroyAllWindows()
