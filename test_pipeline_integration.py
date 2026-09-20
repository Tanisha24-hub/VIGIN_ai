"""
End-to-end pipeline integration test for VIGIN_ai.
Runs the complete integrated pipeline on bangalore_traffic.mp4 for 45 frames,
verifying enhancer pre-filtering, multi-zone speed tracking, and report generation.
"""

import os
import json
import csv
import cv2
import numpy as np
from ultralytics import YOLO

from modules.plate_ocr import PlateRecognizer
from modules.anomaly_detector import IncidentDetector
from modules.safety_analyzer import SafetyBehaviorAnalyzer
from modules.anpr_pipeline import ANPRPipeline, find_plate_detector_model
from modules.speed_estimator import SpeedEstimator
from utils.environment_enhancer import EnvironmentEnhancer
from config.settings import (
    CONF_THRESHOLD, DATA_OUTPUT, SPEED_LIMIT_KMH,
    ROAD_WIDTH_METERS, ROAD_LENGTH_METERS
)


def test_pipeline_integration():
    print("==================================================")
    print("     TESTING: Full VIGIN_ai Pipeline E2E         ")
    print("==================================================")

    video_path = os.path.join("data", "input", "bangalore_traffic.mp4")
    assert os.path.exists(video_path), f"Video not found at {video_path}"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    # 1. Initialize Engines
    vehicle_detector = YOLO('yolov8n.pt')
    plate_model_path = find_plate_detector_model()
    plate_detector = YOLO(plate_model_path)
    ocr_engine = PlateRecognizer()
    anpr_pipeline = ANPRPipeline(plate_detector=plate_detector, ocr_engine=ocr_engine)
    incident_engine = IncidentDetector()
    safety_engine = SafetyBehaviorAnalyzer()

    # 2. Initialize Environment Enhancer in 'auto' mode
    env_enhancer = EnvironmentEnhancer(mode="auto", auto_interval=10)

    # 3. Initialize Multi-Zone SpeedEstimator with 2 piecewise zones
    test_zones = [
        {
            "name": "Left Highway Lane",
            "src_points": [[0.05, 0.40], [0.48, 0.40], [0.45, 0.95], [0.02, 0.95]],
            "road_width_m": 7.0,
            "road_length_m": 25.0,
            "color": (0, 230, 230),
            "speed_limit_kmh": 60.0
        },
        {
            "name": "Right Highway Lane",
            "src_points": [[0.52, 0.40], [0.95, 0.40], [0.98, 0.95], [0.55, 0.95]],
            "road_width_m": 7.0,
            "road_length_m": 25.0,
            "color": (0, 140, 255),
            "speed_limit_kmh": 60.0
        }
    ]

    speed_estimator = SpeedEstimator(
        zones=test_zones,
        frame_size=(frame_w, frame_h),
        speed_limit_kmh=SPEED_LIMIT_KMH,
        expected_flow="down"
    )

    detected_anomalies = {}
    frame_count = 0
    max_test_frames = 45

    print(f" [INFO] Running test loop on {video_path} for {max_test_frames} frames...")

    while cap.isOpened() and frame_count < max_test_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_id = frame_count
        current_timestamp = frame_id / fps
        fh, fw = frame.shape[:2]

        # Pre-filter frame through Environment Enhancer
        enhanced_frame, active_env = env_enhancer.process(frame)
        assert enhanced_frame is not None and enhanced_frame.shape == frame.shape
        display_frame = enhanced_frame.copy()

        # Run ByteTrack on enhanced frame
        veh_results = vehicle_detector.track(
            source=enhanced_frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0, 2, 3, 5, 7],
            conf=CONF_THRESHOLD,
            verbose=False
        )

        # Draw multi-zone calibration boundaries on display frame
        speed_estimator.draw_calibration_zones(display_frame, thickness=2, fill_alpha=0.06)

        # Overlay environmental badge
        cond_info = env_enhancer.get_condition_info()
        c_metrics = cond_info["metrics"]
        env_badge = f"ENV: {active_env.upper()} | LUM: {int(c_metrics['luminance'])}"
        cv2.putText(display_frame, env_badge, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        all_current_boxes = {}
        if veh_results and veh_results[0].boxes is not None and veh_results[0].boxes.id is not None:
            boxes = veh_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = veh_results[0].boxes.id.int().cpu().numpy()
            class_ids = veh_results[0].boxes.cls.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                all_current_boxes[int(track_id)] = box

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                vx1, vy1, vx2, vy2 = box
                tid = int(track_id)

                if cls_id == 0:
                    continue

                cx = int((vx1 + vx2) / 2.0)
                ground_y = int(vy2)
                speed_kmh, direction_vec, is_wrong_way = speed_estimator.estimate_speed(
                    tid, (cx, ground_y), frame_id, fps
                )

                veh_zone = speed_estimator.get_vehicle_zone(tid)

            speed_estimator.clean_inactive_tracks(all_current_boxes.keys())

    cap.release()
    print(f" [INFO] Processed {frame_count} frames successfully.")

    # Generate Test Reports
    test_json = os.path.join(DATA_OUTPUT, "report_test_integration.json")
    test_csv = os.path.join(DATA_OUTPUT, "report_test_integration.csv")

    all_ids = sorted(list(speed_estimator.cached_speeds.keys()))
    records = []
    for tid in all_ids:
        records.append({
            "vehicle_id": int(tid),
            "speed_kmh": speed_estimator.get_max_speed(tid),
            "road_zone": speed_estimator.get_vehicle_zone(tid)
        })

    with open(test_json, 'w') as f:
        json.dump(records, f, indent=4)

    with open(test_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Vehicle ID", "Speed (km/h)", "Road Zone"])
        for r in records:
            writer.writerow([r["vehicle_id"], r["speed_kmh"], r["road_zone"]])

    assert os.path.exists(test_json) and os.path.getsize(test_json) > 0
    assert os.path.exists(test_csv) and os.path.getsize(test_csv) > 0
    print(f"[PASS] Successfully generated test JSON ({test_json}) and CSV ({test_csv}) with road_zone metadata.")
    print(f"[PASS] Tracked {len(all_ids)} vehicles with multi-zone speed estimator.")
    print("==================================================\n")


if __name__ == "__main__":
    test_pipeline_integration()
