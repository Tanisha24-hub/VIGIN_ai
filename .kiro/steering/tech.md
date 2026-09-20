# Tech Stack

## Language
- Python 3.x

## Core Libraries
| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8 object detection and multi-object tracking |
| `opencv-python` | Video capture, frame processing, rendering, image transforms |
| `easyocr` | License plate text extraction (English, CPU mode) |
| `torch` / `torchvision` | PyTorch backend for YOLO and EasyOCR inference |
| `numpy` | Numerical operations (velocity calculations, image kernels) |

## Models
- `yolov8n.pt` — YOLOv8 nano model, stored at the project root. Used for both detection and tracking (`detector.track(..., persist=True)`).

## Configuration
All tunable constants live in `config/settings.py`:
- `CONF_THRESHOLD` — YOLO detection confidence threshold (default `0.35`)
- `VELOCITY_DROP_THRESHOLD` — pixel/sec velocity threshold for anomaly detection (default `120.0`)
- `YOLO_MODEL_NAME` — model filename
- `DATA_INPUT` / `DATA_OUTPUT` — resolved absolute paths to data directories

## Common Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the pipeline
```bash
python main.py
```
Place at least one `.mp4`, `.avi`, or `.mkv` file in `data/input/` before running. The first video found is processed automatically.

### Keyboard controls (during playback)
- `Space` — pause / resume
- `q` — quit
