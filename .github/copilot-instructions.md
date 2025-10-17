## BikeVision — Copilot instructions for code edits

These focused instructions help an AI coding agent be productive in the BikeVision repo. Keep edits small and testable. Reference the files listed below for examples.

### Big picture
- Purpose: a small computer-vision project that runs YOLOv11-based detection/tracking against local images or video (see `src/` examples).
- Key flow: load a YOLO model (local file `yolo11n.pt`), run `model.predict()` or `model.track()` on frames, draw annotated frames with OpenCV, and optionally test danger-zone polygon logic in `src/main.py`.

### Important files and patterns
- `src/main.py` — realtime/video track demo with danger-zone polygons, shows use of `ultralytics.YOLO.track(...)`, OpenCV display (`cv2.imshow`), and frame skipping (`SKIP` variable).
- `src/vidCapTest.py` — simpler track example using `model.track(..., tracker="bytetrack.yaml")`. Use this when adding or changing tracking code.
- `src/yolotesting.py` — single-image predict example using `model.predict()` and `results[0].plot()` for annotated frames.
- Model artifact: top-level `yolo11n.pt` and copies under `models/` and `src/`. Code expects relative paths, e.g. `MODEL_PATH = "yolo11n.pt"` in `src/*` files.
- Data: `data/images/` and `data/videos/` contain sample inputs referenced by examples.

### Project-specific conventions
- Use local model files (no remote downloads). When changing model path, update all examples in `src/` for consistency.
- Tracking calls use `tracker="bytetrack.yaml"` string; the actual YAML is not committed — avoid changing tracker name unless also providing the config.
- UI behavior assumes macOS: `cv2.setWindowProperty(..., cv2.WND_PROP_TOPMOST, 1)` and `waitKey(1)` usage. Keep those lines when editing display code unless cross-platformizing.

### How to run and debug locally
- Create a Python virtualenv, install requirements from `requirements.txt` (contains `ultralytics`, `opencv-python`, `pyyaml`). Example (macOS zsh):

```bash
# create venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Quick runs (examples):
  - Image predict: `python src/yolotesting.py` (opens an annotated window for `../data/images/crowd.jpeg`).
  - Video track: `python src/vidCapTest.py` (uses `model.track()` and streaming).
  - Danger-zone demo: `python src/main.py` (loads `../data/videos/Oct13_LeftMountParallel_720p.mp4` by default).

### Editing guidance for AI agents
- Make one logical change per commit. Run the specific example script that exercises your change (see runs above).
- Avoid adding or renaming external tracker YAMLs unless you include the file and update all references (`bytetrack.yaml`).
- When changing model loading, keep defaults pointing at `yolo11n.pt` so examples continue to run.
- For changes to OpenCV display loops, ensure `cv2.waitKey(...)` remains >0 on macOS to keep windows responsive.

### Tests & validation
- There are no unit tests in repo. Validate changes by running the smallest example that touches your change (prefer `yolotesting.py` for image-level changes, `vidCapTest.py` for tracking changes).

### Integration and dependencies
- Primary runtime dependency: `ultralytics` (YOLOv11). Code relies on `YOLO.track(..., stream=True)` returning streaming results with `result.boxes`, `box.id`, and `.plot()`.
- OpenCV is used for display/geometry (`cv2.imshow`, `cv2.fillPoly`, `cv2.pointPolygonTest`). Use `numpy` for polygon and coordinate math.

### Examples to cite when making edits
- Filtering by class is implemented in `src/main.py` (look for CLASSES list and loop where `cls_name = model.names[int(box.cls)]`).
- Danger zone calculation: `pixelZoneConv()` in `src/main.py` — reuse when adding polygon logic.

### Pull request guidance
- Keep PRs small. Include a short runnable verification step in the PR description (e.g., `python src/yolotesting.py` shows an annotated window).
- If changing a tracker or model file, attach the file or provide exact instructions to reproduce the environment.

### If something is missing
- If you need a tracker config (bytetrack.yaml) or alternative model weights, add them under `models/` and update the example scripts to reference them by relative path.

---
If you want, I can expand sections (e.g., add example commands for macOS camera capture, or include a tiny smoke test script). What should I add or clarify?
