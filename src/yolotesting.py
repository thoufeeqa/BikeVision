import cv2
from ultralytics import YOLO # type: ignore
from ultralytics.utils.plotting import plot_results

# ---- CONFIG ----
MODEL_PATH = "yolo11n.pt"   # or yolo11s.pt, yolo11m.pt, etc.
SOURCE = "../data/images/crowd.jpeg"                  # 0 = default webcam
CONFIDENCE = 0.4            # detection threshold
CLASSES = ['person', 'car'] # restrict to key safety classes (optional)

model = YOLO(MODEL_PATH)

# Run YOLOv11 inference
results = model.predict(SOURCE, conf=CONFIDENCE, verbose=False)


annotated = results[0].plot()

while True:
    cv2.imshow('frame', annotated)
    # macOS needs waitKey>0 to refresh windows
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
