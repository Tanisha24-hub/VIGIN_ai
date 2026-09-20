# VIGIN_ai — Product Overview

**VIGIN_ai** (Vehicle & Incident Intelligence Net) is a real-time CCTV video analysis system focused on vehicle surveillance and traffic incident detection.

## Core Capabilities

- **Vehicle tracking**: Detects and tracks cars, buses, and trucks (COCO classes 2, 5, 7) frame-by-frame using YOLOv8.
- **Anomaly/incident detection**: Flags vehicles exhibiting sudden high-velocity motion changes that indicate accidents or erratic driving.
- **License plate OCR**: Extracts and reads license plate text from cropped vehicle regions using EasyOCR.
- **Live inspection UI**: Renders annotated video in real time with bounding boxes (green = normal, red = alert) and per-vehicle VIGIN IDs.

## Intended Use

Processes pre-recorded CCTV footage (placed in `data/input/`) or can be extended to live camera feeds. Outputs visual alerts to the screen and console logs for flagged incidents.
