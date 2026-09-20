import numpy as np

class IncidentDetector:
    def __init__(self):
        # track_id -> list of (timestamp, cx, cy, vw, vh)
        self.trajectories = {}
        self.logged_incidents = set()

    def evaluate_motion(self, track_id, box, timestamp, all_boxes=None):
        """
        Evaluates genuine road incidents:
        1. Bounding-box proximity conflicts / Collisions
        2. High-speed hard deceleration stops (> 2.0s confirmed history)
        """
        vx1, vy1, vx2, vy2 = box
        vw, vh = vx2 - vx1, vy2 - vy1
        cx, cy = (vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0

        # Ignore tiny background vehicles
        if vw < 120 or vh < 95:
            return []

        # Prevent duplicate alert spamming for the same vehicle
        if track_id in self.logged_incidents:
            return []

        # Track historical motion
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []
        self.trajectories[track_id].append((timestamp, cx, cy, vw, vh))

        if len(self.trajectories[track_id]) > 30:
            self.trajectories[track_id].pop(0)

        # Need at least 20 frames (~0.7s) to establish true velocity baseline
        if len(self.trajectories[track_id]) < 20:
            return []

        alerts = []
        traj = self.trajectories[track_id]

        # 1. Multi-Vehicle Collision / Intersecting Trajectory
        if all_boxes is not None:
            for other_id, other_box in all_boxes.items():
                if other_id == track_id:
                    continue
                ox1, oy1, ox2, oy2 = other_box
                # Calculate Intersection over Union (IoU)
                ix1 = max(vx1, ox1)
                iy1 = max(vy1, oy1)
                ix2 = min(vx2, ox2)
                iy2 = min(vy2, oy2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection = iw * ih
                if intersection > 0:
                    area_v = vw * vh
                    area_o = (ox2 - ox1) * (oy2 - oy1)
                    iou = intersection / float(area_v + area_o - intersection)
                    if iou > 0.15:  # Significant physical overlap
                        alerts.append("Traffic Collision Impact")
                        self.logged_incidents.add(track_id)
                        self.logged_incidents.add(other_id)
                        return alerts

        # 2. Extreme Emergency Braking
        t_start, x_start, y_start, w_start, _ = traj[0]
        t_mid, x_mid, y_mid, w_mid, _ = traj[10]
        t_end, x_end, y_end, w_end, _ = traj[-1]

        dt1 = max(0.01, t_mid - t_start)
        dt2 = max(0.01, t_end - t_mid)

        speed_early = (np.hypot(x_mid - x_start, y_mid - y_start) / w_mid) / dt1
        speed_late = (np.hypot(x_end - x_mid, y_end - y_mid) / w_end) / dt2

        # Must have been moving at substantial speed (> 1.8 widths/sec) and dropped near dead-stop (< 0.15 widths/sec)
        if speed_early > 1.8 and speed_late < 0.15:
            alerts.append("Emergency Collision Stop")
            self.logged_incidents.add(track_id)

        return alerts