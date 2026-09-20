import cv2
import numpy as np
from collections import deque, defaultdict


class SpeedZone:
    """Represents a discrete road plane segment with its own homography calibration."""
    def __init__(self, name, src_points, road_width_m=7.0, road_length_m=20.0,
                 dst_points=None, color=(0, 230, 230), speed_limit_kmh=60.0,
                 expected_flow="down"):
        self.name = name
        self.road_width_m = float(road_width_m)
        self.road_length_m = float(road_length_m)
        self.color = color
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.expected_flow = expected_flow.lower()

        self.src_points = np.array(src_points, dtype=np.float32)
        self.src_poly = self.src_points.astype(np.int32).reshape((-1, 1, 2))

        if dst_points is not None:
            self.dst_points = np.array(dst_points, dtype=np.float32)
        else:
            self.dst_points = np.array([
                [0.0, 0.0],
                [self.road_width_m, 0.0],
                [self.road_width_m, self.road_length_m],
                [0.0, self.road_length_m]
            ], dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.H_inv = np.linalg.inv(self.H)


class SpeedEstimator:
    """
    Multi-Zone Piecewise Homography-based Speed and Direction Estimator for VIGIN_ai.
    Supports complex road topologies (e.g. flat ground vs. flyover ramps/inclines).
    Transforms 2D camera-perspective vehicle contact points into real-world metric space (meters)
    using planar homography, dynamically picking the appropriate zone via pointPolygonTest.
    """

    def __init__(
        self,
        src_points=None,
        dst_points=None,
        road_width_m=7.0,
        road_length_m=20.0,
        frame_size=(1920, 1080),
        speed_limit_kmh=60.0,
        expected_flow="down",
        zones=None
    ):
        """
        Args:
            src_points: 4 source points for default single-zone setup.
            dst_points: 4 destination points in real-world meters.
            road_width_m: Real-world width in meters.
            road_length_m: Real-world depth/length in meters.
            frame_size: (width, height) of the video feed for scaling.
            speed_limit_kmh: Global default speed limit.
            expected_flow: Expected traffic flow ('down', 'up', 'left', 'right').
            zones: List of zone configuration dicts for piecewise multi-zone homography.
        """
        self.road_width_m = float(road_width_m)
        self.road_length_m = float(road_length_m)
        self.frame_width, self.frame_height = frame_size
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.expected_flow = expected_flow.lower()

        # Track history buffer: track_id -> deque of (frame_id, mx, my, px, py, zone_name)
        self.rolling_window = 10
        self.history = defaultdict(lambda: deque(maxlen=self.rolling_window))
        self.cached_speeds = {}       # track_id -> latest smoothed speed (km/h)
        self.cached_directions = {}   # track_id -> (dx, dy) in m/s
        self.cached_wrong_way = {}    # track_id -> bool
        self.max_speeds = {}          # track_id -> peak recorded speed (km/h)
        self.vehicle_zones = {}       # track_id -> current active zone name

        # Storage for calibrated zones
        self.zones = []
        self._setup_zones(zones, src_points, dst_points)

    def _setup_zones(self, zones_cfg=None, default_src=None, default_dst=None):
        """Initializes single or multiple homography zones."""
        self.zones = []
        fw, fh = self.frame_width, self.frame_height

        palette = [
            (0, 230, 230),   # Cyan for Zone 1 (Main Road)
            (0, 165, 255),   # Orange for Zone 2 (Ramp/Incline)
            (255, 128, 0),   # Sky Blue for Zone 3
            (0, 255, 128),   # Spring Green for Zone 4
        ]

        if zones_cfg and len(zones_cfg) > 0:
            for idx, zc in enumerate(zones_cfg):
                name = zc.get("name", f"Zone {idx + 1}")
                raw_src = np.array(zc["src_points"], dtype=np.float32)
                # Scale if normalized (0.0 to 1.05)
                if np.all(raw_src <= 1.05):
                    raw_src[:, 0] *= fw
                    raw_src[:, 1] *= fh

                w_m = float(zc.get("road_width_m", self.road_width_m))
                l_m = float(zc.get("road_length_m", self.road_length_m))
                dst = zc.get("dst_points", None)
                color = zc.get("color", palette[idx % len(palette)])
                sl_limit = float(zc.get("speed_limit_kmh", self.speed_limit_kmh))
                flow = zc.get("expected_flow", self.expected_flow)

                zone = SpeedZone(
                    name=name,
                    src_points=raw_src,
                    road_width_m=w_m,
                    road_length_m=l_m,
                    dst_points=dst,
                    color=color,
                    speed_limit_kmh=sl_limit,
                    expected_flow=flow
                )
                self.zones.append(zone)
        else:
            # Default single zone setup
            if default_src is None:
                src = np.array([
                    [int(fw * 0.28), int(fh * 0.42)],
                    [int(fw * 0.72), int(fh * 0.42)],
                    [int(fw * 0.95), int(fh * 0.96)],
                    [int(fw * 0.05), int(fh * 0.96)]
                ], dtype=np.float32)
            else:
                src = np.array(default_src, dtype=np.float32)
                if np.all(src <= 1.05):
                    src[:, 0] *= fw
                    src[:, 1] *= fh

            default_zone = SpeedZone(
                name="Primary Road Zone",
                src_points=src,
                road_width_m=self.road_width_m,
                road_length_m=self.road_length_m,
                dst_points=default_dst,
                color=(0, 230, 230),
                speed_limit_kmh=self.speed_limit_kmh,
                expected_flow=self.expected_flow
            )
            self.zones.append(default_zone)

        # Backward compatibility properties pointing to primary zone
        if self.zones:
            self.H = self.zones[0].H
            self.H_inv = self.zones[0].H_inv
            self.src_points = self.zones[0].src_points
            self.dst_points = self.zones[0].dst_points

    def add_zone(self, name, src_points, road_width_m=7.0, road_length_m=20.0,
                 dst_points=None, color=None, speed_limit_kmh=None, expected_flow=None):
        """Adds a new calibration zone to the multi-zone speed estimator."""
        fw, fh = self.frame_width, self.frame_height
        raw_src = np.array(src_points, dtype=np.float32)
        if np.all(raw_src <= 1.05):
            raw_src[:, 0] *= fw
            raw_src[:, 1] *= fh

        zone_color = color if color is not None else (0, 200, 255)
        sl = speed_limit_kmh if speed_limit_kmh is not None else self.speed_limit_kmh
        flow = expected_flow if expected_flow is not None else self.expected_flow

        new_zone = SpeedZone(
            name=name,
            src_points=raw_src,
            road_width_m=road_width_m,
            road_length_m=road_length_m,
            dst_points=dst_points,
            color=zone_color,
            speed_limit_kmh=sl,
            expected_flow=flow
        )
        self.zones.append(new_zone)
        return new_zone

    def update_calibration(self, src_points=None, dst_points=None, frame_size=None):
        """Updates primary zone calibration (preserves legacy API)."""
        if frame_size is not None:
            self.frame_width, self.frame_height = frame_size
        self._setup_zones(None, src_points, dst_points)

    def find_zone_for_point(self, point):
        """
        Dynamically selects the matching road zone for a (px, py) coordinate
        using cv2.pointPolygonTest.
        Returns:
            active_zone (SpeedZone): Matching zone object, or closest zone if outside boundaries.
        """
        if not self.zones:
            return None

        px, py = float(point[0]), float(point[1])
        target_pt = (px, py)

        # 1. Exact containment test (inside or on edge: score >= 0)
        for zone in self.zones:
            score = cv2.pointPolygonTest(zone.src_poly, target_pt, False)
            if score >= 0:
                return zone

        # 2. If point is slightly outside defined zones, pick the closest zone
        best_zone = self.zones[0]
        max_dist = -float('inf')  # cv2.pointPolygonTest returns negative distance outside polygon
        for zone in self.zones:
            dist = cv2.pointPolygonTest(zone.src_poly, target_pt, True)
            if dist > max_dist:
                max_dist = dist
                best_zone = zone

        return best_zone

    def project_to_ground(self, point, zone=None):
        """
        Projects a 2D camera pixel point (x, y) onto the metric ground plane (meters)
        using the designated or dynamically detected zone homography matrix H.
        """
        if zone is None:
            zone = self.find_zone_for_point(point)

        if zone is None:
            return 0.0, 0.0, None

        px, py = point
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, zone.H)
        mx, my = transformed[0][0]
        return float(mx), float(my), zone

    def estimate_speed(self, track_id, bbox_bottom_center, current_frame, fps=30.0):
        """
        Estimates real-world speed, motion direction vector, and wrong-way status
        with multi-zone piecewise homography.

        Args:
            track_id (int): Unique identifier for the vehicle
            bbox_bottom_center (tuple): (cx, y2) where vehicle tires contact ground
            current_frame (int): Monotonically increasing frame index
            fps (float): Video framerate (default 30.0)

        Returns:
            speed_kmh (float): Smoothed speed in km/h
            direction_vector (tuple): (vx, vy) metric ground velocity vector in m/s
            is_wrong_way (bool): True if movement opposes expected traffic flow
        """
        cx, y2 = bbox_bottom_center
        fps = float(fps) if (fps and fps > 0) else 30.0

        # Dynamically determine the active zone and transform coordinates
        mx, my, active_zone = self.project_to_ground((cx, y2))
        zone_name = active_zone.name if active_zone else "Default"
        self.vehicle_zones[track_id] = zone_name

        track_buffer = self.history[track_id]
        track_buffer.append((current_frame, mx, my, cx, y2, zone_name))

        # We need at least 4 frames to establish a reliable baseline against tracking jitter
        if len(track_buffer) < 4:
            cached_speed = self.cached_speeds.get(track_id, 0.0)
            cached_dir = self.cached_directions.get(track_id, (0.0, 0.0))
            cached_ww = self.cached_wrong_way.get(track_id, False)
            return cached_speed, cached_dir, cached_ww

        # Calculate delta between oldest and newest samples in rolling window
        f_old, mx_old, my_old, _, _, _ = track_buffer[0]
        f_new, mx_new, my_new, _, _, _ = track_buffer[-1]

        delta_frames = f_new - f_old
        if delta_frames <= 0:
            return (self.cached_speeds.get(track_id, 0.0),
                    self.cached_directions.get(track_id, (0.0, 0.0)),
                    self.cached_wrong_way.get(track_id, False))

        delta_time = delta_frames / fps
        if delta_time <= 0:
            return (self.cached_speeds.get(track_id, 0.0),
                    self.cached_directions.get(track_id, (0.0, 0.0)),
                    self.cached_wrong_way.get(track_id, False))

        dx = mx_new - mx_old
        dy = my_new - my_old

        # Physical Euclidean distance traveled in meters
        distance_meters = float(np.sqrt(dx ** 2 + dy ** 2))

        # Velocity in meters/second -> converted to km/h
        speed_mps = distance_meters / delta_time
        raw_speed_kmh = speed_mps * 3.6

        # Direction velocity vector in m/s
        vx_mps = round(dx / delta_time, 2)
        vy_mps = round(dy / delta_time, 2)
        direction_vector = (vx_mps, vy_mps)

        # Exponential Moving Average (EMA) to smooth out bounding box flickering
        prev_speed = self.cached_speeds.get(track_id, raw_speed_kmh)
        alpha = 0.35  # Smoothing factor
        smoothed_speed_kmh = round(alpha * raw_speed_kmh + (1 - alpha) * prev_speed, 1)

        # Noise filter: negligible drift under 3.0 km/h is treated as stationary
        if smoothed_speed_kmh < 3.0:
            smoothed_speed_kmh = 0.0

        # Wrong-way detection evaluated against the active zone's expected traffic flow
        zone_flow = active_zone.expected_flow if active_zone else self.expected_flow
        is_wrong_way = False
        if smoothed_speed_kmh > 8.0:
            if zone_flow == "down" and dy < -1.2:
                is_wrong_way = True
            elif zone_flow == "up" and dy > 1.2:
                is_wrong_way = True
            elif zone_flow == "right" and dx < -1.2:
                is_wrong_way = True
            elif zone_flow == "left" and dx > 1.2:
                is_wrong_way = True

        self.cached_speeds[track_id] = smoothed_speed_kmh
        self.cached_directions[track_id] = direction_vector
        self.cached_wrong_way[track_id] = is_wrong_way

        # Keep track of peak speed
        if track_id not in self.max_speeds or smoothed_speed_kmh > self.max_speeds[track_id]:
            self.max_speeds[track_id] = smoothed_speed_kmh

        return smoothed_speed_kmh, direction_vector, is_wrong_way

    def get_speed(self, track_id):
        """Returns the current smoothed speed in km/h for a track ID."""
        return self.cached_speeds.get(track_id, 0.0)

    def get_max_speed(self, track_id):
        """Returns the peak recorded speed in km/h for a track ID."""
        return self.max_speeds.get(track_id, self.get_speed(track_id))

    def get_direction(self, track_id):
        """Returns the current direction vector (vx, vy) in m/s."""
        return self.cached_directions.get(track_id, (0.0, 0.0))

    def is_wrong_way(self, track_id):
        """Returns True if vehicle is flagged for wrong-way movement."""
        return self.cached_wrong_way.get(track_id, False)

    def get_vehicle_zone(self, track_id):
        """Returns the active zone name for the vehicle."""
        return self.vehicle_zones.get(track_id, "Unknown")

    def is_overspeeding(self, track_id):
        """Returns True if vehicle exceeds speed limit for its current zone."""
        current_speed = self.get_speed(track_id)
        current_zone_name = self.vehicle_zones.get(track_id)
        limit = self.speed_limit_kmh
        for z in self.zones:
            if z.name == current_zone_name:
                limit = z.speed_limit_kmh
                break
        return current_speed > limit

    def clean_inactive_tracks(self, active_track_ids):
        """Frees memory for vehicles that have left the scene."""
        active_set = set(active_track_ids)
        stored_ids = list(self.history.keys())
        for tid in stored_ids:
            if tid not in active_set:
                if len(self.history[tid]) > 0:
                    del self.history[tid]

    def draw_calibration_zones(self, frame, thickness=2, fill_alpha=0.08):
        """
        Draws all calibrated road homography polygons on the frame with distinct colors,
        labels, and corner markers.
        """
        if not self.zones:
            return frame

        labels = ["TL", "TR", "BR", "BL"]

        for zone in self.zones:
            pts = zone.src_poly

            # Semi-transparent polygon fill
            if fill_alpha > 0:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], zone.color)
                cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

            # Polygon outline
            cv2.polylines(frame, [pts], isClosed=True, color=zone.color, thickness=thickness, lineType=cv2.LINE_AA)

            # Corner markers
            for i, pt in enumerate(zone.src_points):
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                cv2.putText(frame, labels[i], (px + 5, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # Zone label badge near centroid or top edge of the zone
            top_center_x = int((zone.src_points[0][0] + zone.src_points[1][0]) / 2.0)
            top_center_y = int((zone.src_points[0][1] + zone.src_points[1][1]) / 2.0)
            badge_text = f"[{zone.name} | {zone.road_width_m:.1f}m x {zone.road_length_m:.1f}m]"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bx1 = max(10, top_center_x - tw // 2)
            by1 = max(th + 6, top_center_y - 8)
            cv2.rectangle(frame, (bx1 - 4, by1 - th - 4), (bx1 + tw + 4, by1 + 4), (20, 20, 20), -1)
            cv2.putText(frame, badge_text, (bx1, by1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, zone.color, 1, cv2.LINE_AA)

        return frame

    def draw_calibration_zone(self, frame, color=(0, 255, 255), thickness=2, fill_alpha=0.08):
        """Backward compatible alias for drawing zones."""
        return self.draw_calibration_zones(frame, thickness=thickness, fill_alpha=fill_alpha)
