# Gender Detector Web App (YOLOv8 vs Haar Cascade)

A chic, interactive Flask website to compare your YOLOv8 model with a Haar Cascade face detector.
It supports **image uploads** and **webcam snapshots**.

## What's included
- `app.py` — Flask server with two endpoints:
  - `POST /detect-image` returns an annotated JPEG
  - `POST /detect-frame` returns JSON with a Base64 image (used by the webcam snapshot)
- `templates/index.html` — modern Tailwind UI, model selector, upload + webcam
- `models/` — where model files live (I copied what I found from your zip):
  - YOLO weights: `best.pt`
  - Haar XML: `HC-classifier.xml`

## Quickstart

1. Create a fresh Python 3.10+ env.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open http://localhost:7860

### Notes
- If your YOLO weights file has a different name, either rename it to match the default or start the app like:
  ```bash
  YOLO_WEIGHTS=yourfile.pt python app.py
  ```
- Haar Cascade labels boxes as **Face** only (no gender). Your YOLO model should expose gender classes via `results.names` (e.g., `male` / `female`). The app will display counts for whatever class names the model emits.
- For live video, the app uses *webcam snapshots*. If you want true streaming FPS, we can switch to WebSockets later.

---

*Generated from your uploaded project. Enjoy!*



## Troubleshooting

- **Camera not starting?**
  - Use a modern browser (Chrome/Edge). Open the app via **http://localhost:7860** (some browsers block camera on plain HTTP except localhost).
  - Click **Allow** on the camera permission prompt.
  - If using Safari, ensure you click **Start** first; autoplay might be blocked otherwise.

- **YOLO detection not working?**
  - Install **PyTorch** first (Ultralytics needs it). Example (CPU only):
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```
    Or follow the instructions at https://pytorch.org for your CUDA version.
  - Make sure your weights file is in `./models` (e.g., `best.pt`). You can check with:
    ```bash
    curl http://localhost:7860/health
    ```
  - If your weights have different class names, the UI will show those counts directly.

- **Haar not working?**
  - Ensure your XML cascade file is in `./models` and the filename matches `HAAR_XML` env var if you changed it.

