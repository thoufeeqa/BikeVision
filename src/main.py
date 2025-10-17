import cv2
import numpy as np
from ultralytics import YOLO # pyright: ignore[reportPrivateImportUsage]



# ---- CONFIG ----
MODEL_PATH = "yolo11n.pt"   # or yolo11s.pt, yolo11m.pt, etc.
SOURCE = 0                  # 0 = default webcam
CONFIDENCE = 0.4            # detection threshold
CLASSES = ['person', 'car'] # restrict to key safety classes (optional)
#CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'stop sign', 'traffic light']

# --- DANGER ZONE MASK ---
POLY_DANGER_ZONE = [
    [0.2, 1.0],#botL
    [0.8, 1.0],#botR
    [0.60, 0.5],#topR
    [0.40, 0.5]#topL
]

POLY_BUFFER_LEFT =[
    [0.0,1.0],
    [0.2, 1.0],
    [0.2, 0.5],
    [0.1, 0.5]
]

POLY_BUFFER_RIGHT =[
    [0.85,1.0],
    [1.0, 1.0],
    [1.0, 0.5],
    [0.8, 0.5]
]

def pixelZoneConv(points, fw, fh):
    pts = np.array([[x*fw, y*fh] for x,y in points], np.int32)
    pts.reshape(-1,1,2) #reshapes to format needed by cv2.fillpoly()
    return pts

model = YOLO(MODEL_PATH)
VIDEO_PATH = "data/videos/Oct13_frontMount_720p_trim.mp4"  # path to your recorded file
cap = cv2.VideoCapture(VIDEO_PATH)

#cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)



if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("[INFO] Press 'q' to quit")
frame_count = 0
SKIP = 3  # process every 3rd frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP != 0:
        # skip YOLO on this frame
        continue

    h, w = frame.shape[:2]
    polyDangerZone = pixelZoneConv(POLY_DANGER_ZONE, w, h)
    polyBufferL = pixelZoneConv(POLY_BUFFER_LEFT, w, h)
    polyBufferR = pixelZoneConv(POLY_BUFFER_RIGHT, w, h)

    # Run YOLOv11 inference
    results = model.track(frame, conf=CONFIDENCE, persist=True, verbose=False, stream=True, tracker="bytetrack.yaml")
    #results = model.predict(frame, conf=CONFIDENCE, verbose=False, stream=True)
    #annotated = results[0].plot()
    boxId = 0
    result = {}
    for result in results:

        annotated = result.plot()
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [polyDangerZone], (0, 0, 255))
        cv2.fillPoly(overlay, [polyBufferL], (255,0,0))
        cv2.fillPoly(overlay, [polyBufferR], (255, 0, 0))
        annotated = cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0)

        boxes = result.boxes

        currentInside = set()
        #print(result)
        #print(result.boxes)
    #Filter by class if desired
        if CLASSES:
            filtered_boxes = []
            #print(result.boxes)
            for box in result.boxes: # pyright: ignore[reportOptionalIterable]
                #print(box)
                cls_name = model.names[int(box.cls)]
                #print(box.id)
                if cls_name in CLASSES:

                    filtered_boxes.append(box)
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx, cy = int((x1 + x2) / 2), int(y2)
                    inside = cv2.pointPolygonTest(polyDangerZone, (cx, cy), False)
                    insideBufferL = cv2.pointPolygonTest(polyBufferL, (cx, cy), False)
                    insideBufferR = cv2.pointPolygonTest(polyBufferR, (cx, cy), False)
                    #print(inside)
                    if box.id is not None:
                        boxId = int(box.id[0])

                    if inside > 0:
                        cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), 10)
                        print("ALERT", boxId)
                        print(currentInside)
                        currentInside.add(boxId)

                    if insideBufferL > 0  or insideBufferR > 0:
                        cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), 10)
                        print("IN BUFFER ZONE", boxId)
                        if cls_name in currentInside:
                            currentInside.remove(boxId)

            #result.boxes = filtered_boxes


    # Keep window visible on top (macOS)
    cv2.imshow("Bike Vision – YOLOv11 Feed", annotated)
    cv2.setWindowProperty("Bike Vision – YOLOv11 Feed", cv2.WND_PROP_TOPMOST, 1)

    # macOS needs waitKey>0 to refresh windows
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
