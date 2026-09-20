# Project Structure

```
VIGIN_ai/
├── main.py                  # Entry point — orchestrates the full pipeline
├── yolov8n.pt               # YOLOv8 nano weights (project root, loaded at runtime)
├── requirements.txt         # Pip dependencies
│
├── config/                         
│   ├── settings.py          # Global constants and resolved data paths
│   └── __init__.py
│
├── modules/                 # Core AI pipeline components
│   ├── anomaly_detector.py  # IncidentDetector — velocity-based anomaly detection
│   ├── plate_ocr.py         # PlateOCR — EasyOCR wrapper for license plate reading
│   └── __init__.py
│
├── utils/                   # Stateless image processing helpers
│   ├── image_processing.py  # process_license_plate(), enhance_windshield_interior()
│   └── __init__.py
│
└── data/
    ├── input/               # Source video files (.mp4, .avi, .mkv)
    └── output/              # Reserved for future saved results / exports
```

## Architectural Patterns

- **`modules/`** holds stateful classes with domain logic (detectors, OCR engines). Each module is instantiated once in `main.py` and reused across frames.
- **`utils/`** holds pure, stateless functions for image manipulation. No class wrappers — just functions imported directly.
- **`config/settings.py`** is the single source of truth for all thresholds and paths. Import constants from here rather than hardcoding values in modules.
- The main loop in `main.py` drives everything: capture → detect/track → evaluate → render. New pipeline stages should be integrated there.
- COCO vehicle classes in use: `2` (car), `5` (bus), `7` (truck). Any new vehicle-class logic should reference these IDs consistently.
