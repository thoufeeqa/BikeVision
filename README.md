# BikeVision

Small YOLOv11-based bike safety demo. Loads a local YOLO model and runs detection/tracking on images or video. Includes danger-zone polygon logic and macOS-friendly OpenCV display.

## Quick setup (macOS)

1. Create a Python virtualenv and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run an example:

- Image predict (annotated window):

```bash
python src/yolotesting.py
```

- Video track (uses model.track):

```bash
python src/vidCapTest.py
```

- Danger-zone demo (uses `data/videos/Oct13_frontMount_720p_trim.mp4` by default):

```bash
python src/main.py
```

## Model files

This repository expects `yolo11n.pt` model weights in the project root or `src/`/`models/` directories. The weights are large and should not be committed to git. Use Git LFS if you want to store them on GitHub.

## GitHub instructions

1. Initialize the repo locally (if not already):

```bash
git init
git add .
git commit -m "Initial commit: BikeVision"
```

2. Create a remote repo on GitHub (via web UI or `gh repo create`) and push:

```bash
git branch -M main
git remote add origin git@github.com:<your-user>/BikeVision.git
git push -u origin main
```

3. If you plan to include the model weights in the remote, enable Git LFS and track `*.pt` files:

```bash
# install git-lfs (if not already)
brew install git-lfs
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "Track model weights with Git LFS"
```

## Notes

- Keep model files out of regular git history unless you intentionally use Git LFS.
- The examples assume macOS (AVFoundation capture backend). Adjust `cv2.VideoCapture` call for other OSes.

