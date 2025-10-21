import argparse
import cv2
import importlib
import logging
import time
import numpy as np
import os
import sys
import platform

# Small helper: robust ultralytics import
try:
    from ultralytics import YOLO
except Exception:
    try:
        # older installs may expose differently
        yolom = importlib.import_module('ultralytics')
        YOLO = getattr(yolom, 'YOLO')
    except Exception:
        raise ImportError('Could not import YOLO from ultralytics. Please install ultralytics in the active environment.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('vidCapTest')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='yolo11n.pt', help='Path to YOLO model weights')
    p.add_argument('--source', default='0', help='Video source: integer for camera index or path to video file')
    p.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    p.add_argument('--skip', type=int, default=1, help='Process every Nth frame (1 = every frame)')
    return p.parse_args()


def is_raspberry_pi():
    try:
        return platform.system() == 'Linux' and os.path.exists('/sys/firmware/devicetree/base/model') and 'raspberry pi' in open('/sys/firmware/devicetree/base/model').read().lower()
    except Exception:
        return False


def main():
    args = parse_args()
    IS_RASPBERRY_PI = is_raspberry_pi()

    logger.info('Platform: %s', platform.system())
    logger.info('Python: %s', sys.version.splitlines()[0])
    logger.info('OpenCV: %s', cv2.__version__)
    logger.info('Using model: %s', args.model)

    # Load model
    model = YOLO(args.model)

    # Camera / source setup
    use_file = False
    try:
        src_int = int(args.source)
    except Exception:
        src_int = None
    if src_int is None:
        # treat as file path
        source_path = args.source
        if not os.path.exists(source_path):
            logger.error('Source file does not exist: %s', source_path)
            return
        use_file = True

    picam2 = None
    cap = None

    if IS_RASPBERRY_PI and not use_file:
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.preview_configuration.main.size = (1280, 720)
            picam2.preview_configuration.main.format = 'RGB888'
            picam2.configure('preview')
            picam2.start()
            logger.info('Initialized Picamera2')
        except Exception as e:
            logger.warning('Picamera2 not available or failed: %s', e)
            picam2 = None

    if picam2 is None:
        # fallback to OpenCV VideoCapture (camera index or file)
        if use_file:
            cap = cv2.VideoCapture(source_path)
        else:
            idx = src_int if src_int is not None else 0
            if platform.system() == 'Darwin':
                cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            else:
                cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            logger.error('Could not open video source')
            return

    # Prepare UI once
    window_name = 'Bike Vision'
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if platform.system() == 'Darwin':
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        logger.warning('Could not create named window (headless?)')

    logger.info("[INFO] Press 'q' to quit")

    frame_count = 0
    last_annotated = None
    skip = max(1, args.skip)
    fps_t0 = time.time()
    frames_shown = 0

    try:
        while True:
            frame_count += 1

            if picam2 is not None:
                frame = picam2.capture_array()
                if frame is None:
                    logger.error('Failed to read from Picamera2')
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.info('End of video or failed to read frame')
                    break

            # Only run heavy model inference every `skip` frames
            if frame_count % skip == 0:
                try:
                    results = model.track(frame, stream=True, conf=args.conf, tracker='bytetrack.yaml', verbose=False)
                    # model.track with stream yields a generator; take the last result if multiple
                    annotated = None
                    for res in results:
                        annotated = res.plot()
                    if annotated is None:
                        last_annotated = frame
                    else:
                        last_annotated = annotated
                except Exception as e:
                    logger.exception('Model failure: %s', e)
                    last_annotated = frame
            # show last annotated frame (or raw frame)
            out = last_annotated if last_annotated is not None else frame
            try:
                cv2.imshow(window_name, out)
            except Exception:
                # if headless, skip displaying
                pass

            frames_shown += 1
            # compute and log FPS every 120 frames
            if frames_shown % 120 == 0:
                dt = time.time() - fps_t0
                fps = frames_shown / dt if dt > 0 else 0.0
                logger.info('Displayed frames=%d, approx FPS=%.2f', frames_shown, fps)

            wait_time = 1 if IS_RASPBERRY_PI else 30
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info('Interrupted by user')
    finally:
        logger.info('Cleaning up')
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
