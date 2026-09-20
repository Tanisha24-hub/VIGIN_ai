import os
import sys
import shutil
import warnings
import glob
import json
import csv
import re
from collections import Counter, defaultdict, deque

import cv2
from ultralytics import YOLO

# Project Module Imports
from modules.plate_ocr import PlateRecognizer
from modules.anomaly_detector import IncidentDetector
from modules.safety_analyzer import SafetyBehaviorAnalyzer
from modules.anpr_pipeline import ANPRPipeline, find_plate_detector_model
from modules.speed_estimator import SpeedEstimator
from utils.environment_enhancer import EnvironmentEnhancer
from config.settings import (
    CONF_THRESHOLD, DATA_OUTPUT, SPEED_LIMIT_KMH,
    ROAD_WIDTH_METERS, ROAD_LENGTH_METERS, HOMOGRAPHY_SRC_POINTS,
    HOMOGRAPHY_ZONES, ENVIRONMENT_MODE, ENVIRONMENT_LUM_THRESHOLD,
    ENVIRONMENT_CONTRAST_THRESHOLD, ENVIRONMENT_HAZE_THRESHOLD
)
from utils.image_processing import crop_cabin_region, enhance_cabin_interior, analyze_cabin

# Disable Ultralytics/OpenCV verbose warnings
os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore")
cv2.setLogLevel(0)


def run_vigin_ai(video_source):
    print("==================================================")
    print("   VIGIN_ai: Full Vehicle, ANPR & Safety Net      ")
    print("==================================================")

    # 1. Initialize Engines
    vehicle_detector = YOLO('yolov8n.pt')
    plate_model_path = find_plate_detector_model()
    print(f"[INFO] Using Plate Model: {plate_model_path}")
    plate_detector = YOLO(plate_model_path)
    ocr_engine = PlateRecognizer()
    anpr_pipeline = ANPRPipeline(plate_detector=plate_detector, ocr_engine=ocr_engine)
    incident_engine = IncidentDetector()
    safety_engine = SafetyBehaviorAnalyzer()

    # Tracking storage with temporal buffers
    cabin_tracker = defaultdict(lambda: {"history": deque(maxlen=7), "last_frame": -999, "cached_result": None})
    detected_anomalies = {}         # track_id -> set of anomalies/violations
    saved_snapshots = set()         # track_ids already screenshotted for incidents
    saved_cabin_inspections = set() # track_ids already saved for cabin

    incident_dir = os.path.join(DATA_OUTPUT, "incidents")
    if os.path.exists(incident_dir):
        shutil.rmtree(incident_dir)
    os.makedirs(incident_dir, exist_ok=True)

    cabin_dir = os.path.join(DATA_OUTPUT, "cabin_inspections")
    os.makedirs(cabin_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    # Initialize Environment Enhancer for adverse weather & low-light
    env_enhancer = EnvironmentEnhancer(
        mode=ENVIRONMENT_MODE,
        luminance_threshold=ENVIRONMENT_LUM_THRESHOLD,
        contrast_threshold=ENVIRONMENT_CONTRAST_THRESHOLD,
        haze_threshold=ENVIRONMENT_HAZE_THRESHOLD,
        auto_interval=15
    )

    # Initialize Multi-Zone Piecewise Speed Estimator
    speed_estimator = SpeedEstimator(
        src_points=HOMOGRAPHY_SRC_POINTS,
        road_width_m=ROAD_WIDTH_METERS,
        road_length_m=ROAD_LENGTH_METERS,
        frame_size=(frame_w, frame_h),
        speed_limit_kmh=SPEED_LIMIT_KMH,
        expected_flow="down",
        zones=HOMOGRAPHY_ZONES
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_timestamp = frame_id / fps
        fh, fw = frame.shape[:2]

        # ── Pre-filter video frame for adverse weather / low-light ──
        enhanced_frame, active_env = env_enhancer.process(frame)

        # Use enhanced frame for display when low-light / rain boost is active
        display_frame = enhanced_frame.copy() if active_env in ["night", "rain"] else frame.copy()

        primary_vehicle_cabin = None
        primary_vehicle_id = None
        near_field_candidates = []

        # 2. Track Vehicles AND People (using ByteTrack on enhanced frame)
        # classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
        veh_results = vehicle_detector.track(
            source=enhanced_frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=[0, 2, 3, 5, 7], 
            conf=CONF_THRESHOLD, 
            verbose=False
        )

        # Draw calibrated road zone polygons on display frame
        speed_estimator.draw_calibration_zones(display_frame, thickness=2, fill_alpha=0.06)

        # Render Environmental Condition HUD Pill in upper-left
        cond_info = env_enhancer.get_condition_info()
        c_metrics = cond_info["metrics"]
        env_badge_text = f"ENV: {active_env.upper()} | LUM: {int(c_metrics['luminance'])} | CONTR: {int(c_metrics['contrast'])}"
        env_color = (0, 140, 255) if active_env == "night" else ((255, 180, 0) if active_env == "rain" else (0, 200, 0))
        (etw, eth), _ = cv2.getTextSize(env_badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.rectangle(display_frame, (12, 12), (20 + etw + 6, 12 + eth + 12), (20, 20, 25), -1)
        cv2.rectangle(display_frame, (12, 12), (20 + etw + 6, 12 + eth + 12), env_color, 1)
        cv2.putText(display_frame, env_badge_text, (18, 12 + eth + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, env_color, 1, cv2.LINE_AA)

        all_current_boxes = {}
        if veh_results and veh_results[0].boxes is not None and veh_results[0].boxes.id is not None:
            boxes = veh_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = veh_results[0].boxes.id.int().cpu().numpy()
            class_ids = veh_results[0].boxes.cls.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                all_current_boxes[int(track_id)] = box

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                vx1, vy1, vx2, vy2 = box
                vw, vh = vx2 - vx1, vy2 - vy1
                tid = int(track_id)

                # ── Person / Pedestrian (class 0) ──────────────────────
                if cls_id == 0:
                    ped_color = (255, 255, 0)  # Cyan in BGR
                    cv2.rectangle(display_frame, (vx1, vy1), (vx2, vy2), ped_color, 2)
                    cv2.putText(display_frame, f"Person #{tid}", (vx1, max(20, vy1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ped_color, 2, cv2.LINE_AA)
                    continue

                # ── Motorcycle Safety & Triple-Riding Check (class 3) ──
                if cls_id == 3:
                    bike_crop = frame[max(0, vy1):min(fh, vy2), max(0, vx1):min(fw, vx2)]
                    bike_violations = safety_engine.check_motorcycle_safety(bike_crop, vehicle_detector)
                    for bv in bike_violations:
                        if tid not in detected_anomalies:
                            detected_anomalies[tid] = set()
                        detected_anomalies[tid].add(bv)

                # ── Homography Speed & Motion Vector Estimation (classes 2, 3, 5, 7) ──
                # Bottom-center coordinate where tires contact the ground
                cx = int((vx1 + vx2) / 2.0)
                ground_y = int(vy2)
                speed_kmh, direction_vec, is_wrong_way = speed_estimator.estimate_speed(
                    tid, (cx, ground_y), frame_id, fps
                )

                # Flag speed & lane discipline violations
                speed_alerts = []
                is_overspeed = speed_estimator.is_overspeeding(tid)
                if is_overspeed:
                    speed_alerts.append(f"Overspeeding ({int(speed_kmh)} km/h)")
                if is_wrong_way:
                    speed_alerts.append("Wrong-Way Driving")

                # ── Vehicles Motion & Incident Analysis ──
                anomaly_alerts = incident_engine.evaluate_motion(
                    tid, (vx1, vy1, vx2, vy2), current_timestamp, all_boxes=all_current_boxes
                )
                
                combined_alerts = list(anomaly_alerts) + speed_alerts
                if combined_alerts:
                    if tid not in detected_anomalies:
                        detected_anomalies[tid] = set()
                    for alert in combined_alerts:
                        detected_anomalies[tid].add(alert)

                    # Save annotated incident evidence snapshot
                    if tid not in saved_snapshots:
                        snap_path = os.path.join(incident_dir, f"incident_veh_{tid}_{current_timestamp:.1f}s.jpg")
                        margin = 40
                        crop_x1 = max(0, vx1 - margin)
                        crop_y1 = max(0, vy1 - margin)
                        crop_x2 = min(fw, vx2 + margin)
                        crop_y2 = min(fh, vy2 + margin)
                        snap_img = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                        if snap_img.size > 0:
                            sh, sw = snap_img.shape[:2]
                            rx1, ry1 = vx1 - crop_x1, vy1 - crop_y1
                            rx2, ry2 = vx2 - crop_x1, vy2 - crop_y1
                            cv2.rectangle(snap_img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)

                            tag_text = f"INCIDENT: {', '.join(combined_alerts)} | VEH #{tid} @ {current_timestamp:.1f}s"
                            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                            banner_h = th + 14
                            cv2.rectangle(snap_img, (0, 0), (sw, banner_h), (0, 0, 180), -1)
                            cv2.putText(snap_img, tag_text, (6, th + 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

                            cv2.imwrite(snap_path, snap_img)
                            saved_snapshots.add(tid)
                            print(f"📸 [SNAPSHOT SAVED] Incident recorded: {snap_path}")

                # ── Candidate Gathering for Near-Field Cabin Inspection ──
                if cls_id == 2 and vw >= 190 and vh >= 130:
                    v_center = (vy1 + vy2) / 2.0
                    area = vw * vh
                    near_field_candidates.append((v_center, area, tid, box))

                # ── License Plate Recognition (ANPRPipeline with Throttling) ──
                best_plate, is_locked = anpr_pipeline.process_vehicle(frame, tid, (vx1, vy1, vx2, vy2), frame_id)
                p_box = anpr_pipeline.get_plate_box(tid)

                # ── Draw Vehicle Bounding Boxes & Clean Speed Badges ──
                active_violations = list(detected_anomalies.get(tid, []))
                has_violation = len(active_violations) > 0 or is_overspeed or is_wrong_way
                veh_color = (0, 0, 255) if has_violation else (0, 255, 0)
                cv2.rectangle(display_frame, (vx1, vy1), (vx2, vy2), veh_color, 2)

                # Format Speed Badge styling with active zone info
                veh_zone = speed_estimator.get_vehicle_zone(tid)
                zone_tag = f" | {veh_zone}" if (veh_zone and veh_zone not in ["Default", "Primary Road Zone"]) else ""

                if is_overspeed:
                    speed_badge_text = f"{int(speed_kmh)} km/h | OVERSPEED{zone_tag}"
                    badge_bg = (0, 0, 220)       # Bright Red
                    badge_fg = (255, 255, 255)
                elif is_wrong_way:
                    speed_badge_text = f"{int(speed_kmh)} km/h | WRONG WAY{zone_tag}"
                    badge_bg = (0, 140, 255)     # Bright Orange
                    badge_fg = (255, 255, 255)
                elif speed_kmh > 0:
                    speed_badge_text = f"{int(speed_kmh)} km/h{zone_tag}"
                    badge_bg = (0, 180, 0)       # Clean Vivid Green
                    badge_fg = (255, 255, 255)
                else:
                    speed_badge_text = f"0 km/h{zone_tag}"
                    badge_bg = (45, 45, 45)      # Neutral Slate
                    badge_fg = (200, 200, 200)

                # Render Speed Badge pill above bounding box
                (sbw, sbh), _ = cv2.getTextSize(speed_badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
                badge_h = sbh + 8
                badge_y2 = max(badge_h + 18, vy1 - 2)
                badge_y1 = badge_y2 - badge_h
                badge_x1 = vx1
                badge_x2 = min(fw, vx1 + sbw + 12)
                cv2.rectangle(display_frame, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_bg, -1)
                cv2.putText(display_frame, speed_badge_text, (badge_x1 + 6, badge_y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, badge_fg, 2, cv2.LINE_AA)

                # Vehicle ID & Violation Label stacked cleanly above speed badge
                veh_label = f"ID: {tid}"
                if active_violations:
                    veh_label += f" | {active_violations[0]}"
                id_y = max(14, badge_y1 - 4)
                cv2.putText(display_frame, veh_label, (vx1, id_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, veh_color, 2, cv2.LINE_AA)

                # ── Draw License Plate Box ─────────────────────────────
                if p_box is not None:
                    px1, py1, px2, py2 = p_box
                    gx1, gy1 = vx1 + px1, vy1 + py1
                    gx2, gy2 = vx1 + px2, vy1 + py2

                    plate_color = (0, 255, 0) if is_locked else (255, 0, 0)
                    cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), plate_color, 2)

                    p_label = f"{best_plate} [LOCKED]" if is_locked else (best_plate if best_plate != "UNKNOWN" else "PLATE")
                    (tw, th), _ = cv2.getTextSize(p_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                    cv2.rectangle(display_frame, (gx1, max(0, gy1 - th - 8)), (gx1 + tw + 6, gy1), plate_color, -1)
                    cv2.putText(display_frame, p_label, (gx1 + 3, max(12, gy1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

            # Free tracking queues for vehicles that left the scene
            speed_estimator.clean_inactive_tracks(all_current_boxes.keys())


        # ── Primary Near-Field Cabin Inspection & Temporal Smoothing ────
        if near_field_candidates:
            # Prioritize the closest vehicle: lowest vertical center, then largest area
            near_field_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            _, _, primary_id, primary_box = near_field_candidates[0]

            primary_entry = cabin_tracker[primary_id]

            # Throttled Cabin Inspection: run every 4 frames per vehicle
            if (frame_id - primary_entry["last_frame"] >= 4) or (primary_entry["cached_result"] is None):
                cabin_crop = crop_cabin_region(frame, primary_box, class_id=2)
                if cabin_crop is not None and cabin_crop.size > 0:
                    cabin_data = analyze_cabin(cabin_crop, vehicle_detector)
                    enhanced_crop, s_alerts = safety_engine.inspect_driver_safety(
                        cabin_data['enhanced'], primary_id, yolo_detector=vehicle_detector
                    )
                    cabin_data['enhanced'] = enhanced_crop
                    cabin_data['alerts'] = s_alerts

                    if s_alerts:
                        if primary_id not in detected_anomalies:
                            detected_anomalies[primary_id] = set()
                        for sa in s_alerts:
                            detected_anomalies[primary_id].add(sa)

                    # Update history buffer (deque maxlen=7)
                    primary_entry["history"].append({
                        "occupants": cabin_data["occupants"],
                        "driver_present": cabin_data["driver_present"]
                    })
                    primary_entry["cached_result"] = cabin_data
                    primary_entry["last_frame"] = frame_id

                    # Save cabin inspection snapshot once
                    if primary_id not in saved_cabin_inspections and cabin_data["driver_present"]:
                        cabin_path = os.path.join(cabin_dir, f"cabin_veh_{primary_id}_{current_timestamp:.1f}s.jpg")
                        cv2.imwrite(cabin_path, cabin_data['enhanced'])
                        saved_cabin_inspections.add(primary_id)
                        print(f"📸 [CABIN INSPECTION SAVED] Vehicle #{primary_id} crop saved: {cabin_path}")

            # Temporal Rolling Smoothing: majority vote across rolling buffer
            cached_res = primary_entry["cached_result"]
            if cached_res is not None and primary_entry["history"]:
                occ_votes = [h["occupants"] for h in primary_entry["history"]]
                driver_votes = [h["driver_present"] for h in primary_entry["history"]]
                smoothed_occ = Counter(occ_votes).most_common(1)[0][0]
                smoothed_driver = Counter(driver_votes).most_common(1)[0][0]

                primary_vehicle_cabin = {
                    "enhanced": cached_res["enhanced"],
                    "occupants": smoothed_occ,
                    "driver_present": smoothed_driver
                }
                primary_vehicle_id = primary_id

        # ── Render PiP Inset in Top-Right Corner ────────────────────────
        pip_w = min(320, max(160, int(fw * 0.45)))
        pip_h = max(100, int(pip_w * (180 / 320)))
        x_offset = max(10, fw - pip_w - 15)
        y_offset = 15

        # Header Banner
        cv2.rectangle(display_frame, (x_offset, y_offset), (x_offset + pip_w, y_offset + 25), (200, 0, 0), -1)
        cv2.putText(display_frame, "CABIN INSPECTION (HD)", (x_offset + 8, y_offset + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        if primary_vehicle_cabin is not None and primary_vehicle_cabin.get('enhanced') is not None:
            pip_img = primary_vehicle_cabin['enhanced']
            pip_resized = cv2.resize(pip_img, (pip_w, pip_h), interpolation=cv2.INTER_CUBIC) if pip_img.shape[:2] != (pip_h, pip_w) else pip_img

            # Draw enhanced windshield with boundary clipping
            y_start = y_offset + 25
            y_end = min(fh, y_start + pip_h)
            x_end = min(fw, x_offset + pip_w)
            h_slice = max(0, y_end - y_start)
            w_slice = max(0, x_end - x_offset)

            if h_slice > 0 and w_slice > 0:
                display_frame[y_start:y_end, x_offset:x_end] = pip_resized[:h_slice, :w_slice]

            # Check if there is an active safety violation on this car
            v_notes = list(detected_anomalies.get(primary_vehicle_id, []))
            active_alert_str = f" | {v_notes[0]}" if v_notes else ""
            driver_str = "Driver Active" if primary_vehicle_cabin.get('driver_present') else "Driver Inactive"
            status_str = f"[{driver_str} | Occ: {primary_vehicle_cabin.get('occupants', 0)}{active_alert_str}]"

            # Badge background: Red if alert exists, otherwise Dark Slate
            badge_bg = (0, 0, 180) if v_notes else (20, 20, 25)
            cv2.rectangle(display_frame, (x_offset, y_offset + 25 + pip_h - 22), (x_offset + pip_w, y_offset + 25 + pip_h), badge_bg, -1)
            cv2.putText(display_frame, status_str, (x_offset + 6, y_offset + 25 + pip_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Standby HUD when no vehicle satisfies vw >= 190
            cv2.rectangle(display_frame, (x_offset, y_offset + 25), (x_offset + pip_w, y_offset + 25 + pip_h), (28, 28, 35), -1)
            standby_text = "[AWAITING CLOSE-RANGE VEHICLE]"
            (stw, sth), _ = cv2.getTextSize(standby_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            st_x = x_offset + max(4, (pip_w - stw) // 2)
            st_y = y_offset + 25 + (pip_h + sth) // 2
            cv2.putText(display_frame, standby_text, (st_x, st_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

        # Border
        cv2.rectangle(display_frame, (x_offset, y_offset), (x_offset + pip_w, y_offset + 25 + pip_h), (255, 0, 0), 2)

        # ── Show Live Frame & Handle Keypress ───────────────────────────
        # Create a resizable window that auto-fits your screen
        cv2.namedWindow("VIGIN_ai Live Inspection Pipeline", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VIGIN_ai Live Inspection Pipeline", 1280, 720)

        # Resize the frame so the full view (traffic + PiP HUD) fits cleanly
        display_frame = cv2.resize(display_frame, (1280, 720))
        cv2.imshow("VIGIN_ai Live Inspection Pipeline", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Pipeline stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # 3. Generate Final Comprehensive Reports
    video_base = os.path.basename(video_source).split('.')[0]
    report_json = os.path.join(DATA_OUTPUT, f"report_{video_base}.json")
    report_csv = os.path.join(DATA_OUTPUT, f"report_{video_base}.csv")

    all_ids = sorted(list(set(
        list(anpr_pipeline.track_history.keys()) +
        list(detected_anomalies.keys()) +
        list(cabin_tracker.keys()) +
        list(speed_estimator.cached_speeds.keys())
    )))
    final_records = []

    for tid in all_ids:
        best_plate, _ = anpr_pipeline.compute_best_plate(tid)
        best_conf = anpr_pipeline.get_plate_conf(tid)
        anomalies = list(detected_anomalies.get(tid, []))

        c_entry = cabin_tracker.get(tid)
        if c_entry and c_entry["history"]:
            occ_votes = [h["occupants"] for h in c_entry["history"]]
            drv_votes = [h["driver_present"] for h in c_entry["history"]]
            best_occupants = Counter(occ_votes).most_common(1)[0][0]
            best_driver_present = Counter(drv_votes).most_common(1)[0][0]
        else:
            best_occupants = 0
            best_driver_present = False

        max_speed = round(float(speed_estimator.get_max_speed(tid)), 1)

        final_records.append({
            "vehicle_id": int(tid),
            "license_plate": best_plate,
            "confidence": round(float(best_conf), 2),
            "violations_and_anomalies": anomalies if anomalies else ["None"],
            "occupants": int(best_occupants),
            "driver_present": bool(best_driver_present),
            "speed_kmh": max_speed,
            "road_zone": speed_estimator.get_vehicle_zone(tid)
        })

    with open(report_json, 'w') as f:
        json.dump(final_records, f, indent=4)

    with open(report_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Vehicle ID", "License Plate", "Confidence", "OCR Confidence", "Violations/Anomalies", "Occupants", "Driver Present", "Speed (km/h)", "Road Zone"])
        for r in final_records:
            writer.writerow([r["vehicle_id"], r["license_plate"], r["confidence"], r["confidence"], "; ".join(r["violations_and_anomalies"]), r["occupants"], r["driver_present"], r["speed_kmh"], r["road_zone"]])

    print(f"\n==================================================")
    print(f" [SUCCESS] Processing Complete!")
    print(f" - JSON Report: {report_json}")
    print(f" - CSV Report:  {report_csv}")
    print(f" - Evidence:    {incident_dir}")
    print(f"==================================================")


if __name__ == "__main__":
    input_dir = os.path.join("data", "input")
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if os.path.exists(arg_path):
            video_path = arg_path
        elif os.path.exists(os.path.join(input_dir, arg_path)):
            video_path = os.path.join(input_dir, arg_path)
        else:
            video_path = os.path.join(input_dir, arg_path)
    else:
        video_path = os.path.join(input_dir, "bangalore_traffic.mp4")

    if os.path.exists(video_path):
        run_vigin_ai(video_path)
    else:
        print(f"File not found: {video_path}")