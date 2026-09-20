import cv2
import numpy as np

# Skeletal Pose Tracking via Ultralytics YOLOv8-Pose
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class SafetyBehaviorAnalyzer:
    def __init__(self, pose_model_name='yolov8n-pose.pt'):
        self.pose_model_name = pose_model_name
        self.pose_model = None
        self._init_pose_model()

        # Skeleton connections for upper body visualization (COCO 17 keypoint topology)
        # 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
        # 5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow, 9: L-Wrist, 10: R-Wrist
        self.SKELETON_EDGES = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6),                          # Shoulders
            (5, 7), (7, 9),                  # Left arm
            (6, 8), (8, 10),                 # Right arm
        ]

        # Vehicle tracking states for temporal analysis if needed
        self.inattention_counters = {}

    def _init_pose_model(self):
        """Initializes the YOLOv8-Pose model with graceful fallback."""
        if YOLO is not None:
            try:
                self.pose_model = YOLO(self.pose_model_name)
            except Exception as e:
                print(f"[WARN] Unable to load YOLO pose model '{self.pose_model_name}': {e}")
                self.pose_model = None
        else:
            self.pose_model = None

    def inspect_driver_safety(self, cabin_crop, track_id, yolo_detector=None):
        """
        Runs comprehensive driver presence and safety evaluation using YOLOv8-Pose:
        1. Skeletal Keypoint Verification: Verifies head + shoulders in driver zone
           (left half of crop for oncoming Indian RHD traffic).
        2. Inattention / Head Turn: Measures horizontal delta between nose and shoulder center.
        3. Distraction / Phone: Evaluates wrist-to-head/ear proximity + YOLO class 67 detection.
        """
        if cabin_crop is None or cabin_crop.size == 0:
            return cabin_crop, []

        alerts = []
        ch, cw = cabin_crop.shape[:2]

        driver_skeleton_verified = False

        # 1. YOLOv8n-Pose Inference
        if self.pose_model is not None and cw >= 80 and ch >= 50:
            try:
                pose_results = self.pose_model(cabin_crop, conf=0.25, verbose=False)
                if pose_results and len(pose_results) > 0 and pose_results[0].keypoints is not None:
                    kpts_obj = pose_results[0].keypoints
                    if hasattr(kpts_obj, 'data') and kpts_obj.data is not None:
                        kpts_data = kpts_obj.data.cpu().numpy()  # shape (N, 17, 3)

                        for kpts in kpts_data:
                            # Extract keypoints: [x, y, conf]
                            nose = kpts[0]
                            l_eye, r_eye = kpts[1], kpts[2]
                            l_ear, r_ear = kpts[3], kpts[4]
                            l_sh, r_sh = kpts[5], kpts[6]
                            l_wrist, r_wrist = kpts[9], kpts[10]

                            # ── 1a. Verify Driver Presence (Left Half for Oncoming Indian RHD) ──
                            driver_zone_bound = cw * 0.55
                            head_in_driver_zone = (
                                (nose[2] > 0.25 and nose[0] < driver_zone_bound) or
                                (l_eye[2] > 0.25 and l_eye[0] < driver_zone_bound) or
                                (r_eye[2] > 0.25 and r_eye[0] < driver_zone_bound)
                            )
                            shoulder_in_driver_zone = (
                                (l_sh[2] > 0.20 and l_sh[0] < driver_zone_bound) or
                                (r_sh[2] > 0.20 and r_sh[0] < driver_zone_bound)
                            )

                            if head_in_driver_zone and shoulder_in_driver_zone:
                                driver_skeleton_verified = True

                            # ── 1b. Inattention / Head Turn (Nose vs. Shoulder Center) ──
                            if nose[2] > 0.25 and (l_sh[2] > 0.20 and r_sh[2] > 0.20):
                                shoulder_cx = (l_sh[0] + r_sh[0]) / 2.0
                                shoulder_span = max(1.0, abs(r_sh[0] - l_sh[0]))
                                delta_x = abs(nose[0] - shoulder_cx)

                                # If nose significantly deviates from shoulder center axis
                                if shoulder_span > 20 and (delta_x / shoulder_span > 0.32 or delta_x > 25.0):
                                    alerts.append("Driver Inattention (Looking Away)")
                                elif shoulder_span <= 20 and delta_x > 20.0:
                                    alerts.append("Driver Inattention (Looking Away)")
                            elif nose[2] > 0.25 and (l_sh[2] > 0.20 or r_sh[2] > 0.20):
                                ref_sh = l_sh if l_sh[2] > 0.20 else r_sh
                                delta_x = abs(nose[0] - ref_sh[0])
                                if delta_x < 8.0 or delta_x > 45.0:
                                    alerts.append("Driver Inattention (Looking Away)")

                            # ── 1c. Phone / Handheld Device via Keypoint Proximity ──
                            head_pts = [pt for pt in [l_ear, r_ear, l_eye, r_eye, nose] if pt[2] > 0.20]
                            wrist_pts = [pt for pt in [l_wrist, r_wrist] if pt[2] > 0.20]

                            phone_proximity = False
                            for w_pt in wrist_pts:
                                for h_pt in head_pts:
                                    dist = np.hypot(w_pt[0] - h_pt[0], w_pt[1] - h_pt[1])
                                    if dist < 42.0:
                                        phone_proximity = True
                                        cv2.line(
                                            cabin_crop,
                                            (int(w_pt[0]), int(w_pt[1])),
                                            (int(h_pt[0]), int(h_pt[1])),
                                            (0, 0, 255),
                                            2,
                                        )
                                        break
                                if phone_proximity:
                                    break

                            if phone_proximity:
                                alerts.append("Distracted: Hand to Head / Phone")

                            # ── 1d. Render Skeleton Keypoints and Bone Segments ──
                            for p1_idx, p2_idx in self.SKELETON_EDGES:
                                if kpts[p1_idx][2] > 0.20 and kpts[p2_idx][2] > 0.20:
                                    pt1 = (int(kpts[p1_idx][0]), int(kpts[p1_idx][1]))
                                    pt2 = (int(kpts[p2_idx][0]), int(kpts[p2_idx][1]))
                                    cv2.line(cabin_crop, pt1, pt2, (0, 255, 255), 2)

                            for idx in range(min(11, len(kpts))):
                                if kpts[idx][2] > 0.20:
                                    cv2.circle(cabin_crop, (int(kpts[idx][0]), int(kpts[idx][1])), 3, (0, 255, 0), -1)
            except Exception as e:
                pass

        # 2. Standard YOLO Class 67 (Cell Phone) Detection
        if yolo_detector is not None and cw >= 80:
            try:
                phone_results = yolo_detector(cabin_crop, conf=0.18, classes=[67], verbose=False)
                if phone_results and phone_results[0].boxes is not None and len(phone_results[0].boxes) > 0:
                    alerts.append("Distracted: Phone in Use")
                    for box in phone_results[0].boxes.xyxy.cpu().numpy().astype(int):
                        cv2.rectangle(cabin_crop, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                        cv2.putText(
                            cabin_crop,
                            "PHONE",
                            (box[0], max(12, box[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
            except Exception:
                pass

        # De-duplicate alerts preserving order
        unique_alerts = list(dict.fromkeys(alerts))
        return cabin_crop, unique_alerts

    def check_motorcycle_safety(self, bike_crop, yolo_detector=None):
        """
        Evaluates motorcycle safety (Class 3):
        - Triple-riding violation: counts overlapping person / pose skeletons bound to bike.
        """
        if bike_crop is None or bike_crop.size == 0:
            return []

        violations = []
        rider_count = 0

        # 1. Check via pose detector if active
        if self.pose_model is not None:
            try:
                pose_res = self.pose_model(bike_crop, conf=0.22, verbose=False)
                if pose_res and pose_res[0].boxes is not None:
                    rider_count = max(rider_count, len(pose_res[0].boxes))
            except Exception:
                pass

        # 2. Check via YOLO COCO class 0 (person)
        if yolo_detector is not None:
            try:
                rider_results = yolo_detector(bike_crop, conf=0.20, classes=[0], verbose=False)
                if rider_results and rider_results[0].boxes is not None:
                    rider_count = max(rider_count, len(rider_results[0].boxes))
            except Exception:
                pass

        if rider_count >= 3:
            violations.append("Triple Riding Violation")

        return violations