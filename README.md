# VIGIN_ai: AI-Powered Road Surveillance & Enforcement Pipeline

An end-to-end intelligent traffic surveillance, metric speed kinematics, ANPR, and occupant safety inspection system built with OpenCV, YOLOv8, ByteTrack, PaddleOCR, and RetinaFace.

## Core Features
- **Pillar 1 (Metric Speed Kinematics):** Solves camera perspective distortion using Piecewise Multi-Zone Homography to map 2D coordinates into real-world velocity (km/h).
- **Pillar 2 (High-Accuracy ANPR):** License plate localization paired with multi-frame temporal majority voting to eliminate motion blur misreads.
- **Pillar 3 (Cabin Safety & HUD):** Near-field candidate filtering and CLAHE windshield glare removal with RetinaFace occupant monitoring.
- **Performance:** Throttled cadence pipeline maintaining a continuous 30 FPS.

## Project Structure
- `modules/` - Speed estimation, ANPR pipeline, and incident detector
- `utils/` - Image processing, environmental enhancement, and calibration helpers
- `config/` - System thresholds and calibration parameters
- `main.py` - Central orchestrator pipeline 

## 🚀 Development Status & Roadmap

> **Status:** `Active Development / In Progress` (Current Milestone: Phase 2 - Advanced Robustness)

| Module / Milestone | Status | Description |
| :--- | :---: | :--- |
| **Vehicle Detection & Tracking** | Completed | YOLOv8 + ByteTrack persistent trajectory tracking across occlusions |
| **ANPR Pipeline** | Completed | Two-stage localization, PaddleOCR, and temporal majority consensus |
| **Metric Speed Kinematics** | Completed | Inverse Perspective Mapping (Homography) to convert 2D pixels to km/h |
| **Cabin Safety & HUD** | Completed | Near-field vehicle filtering, CLAHE reflection suppression, and PiP HUD |
| **Cadence Scheduling** | Completed | Multi-rate loop optimization to preserve 30 FPS throughput |
| **Environmental Robustness** | In Progress | Integrating Retinex gamma correction and Dark Channel Prior de-hazing |
| **Multi-Zone Homography** | In Progress | Dynamic matrix switching for sloped terrain, ramps, and flyovers |
| **Edge Hardware Optimization** | Planned | Model quantization (ONNX FP16 / TensorRT) for NVIDIA Jetson deployment |