"""
Unit and dynamic zone switching tests for Multi-Zone SpeedEstimator.
"""

import numpy as np
import cv2
from modules.speed_estimator import SpeedEstimator


def test_multi_zone_speed():
    print("==================================================")
    print("    TESTING: Multi-Zone SpeedEstimator Module     ")
    print("==================================================")

    # 1. Define two discrete road zones:
    # Zone 1: Main Flat Highway (Left side of road, 7m x 30m)
    # Zone 2: Elevated Flyover Ramp (Right side of road, 5m x 25m)
    zones_config = [
        {
            "name": "Main Highway",
            "src_points": [
                [100, 300],   # TL
                [550, 300],   # TR
                [500, 900],   # BR
                [50, 900]     # BL
            ],
            "road_width_m": 7.0,
            "road_length_m": 30.0,
            "color": (0, 230, 230),
            "speed_limit_kmh": 80.0,
            "expected_flow": "down"
        },
        {
            "name": "Flyover Ramp",
            "src_points": [
                [650, 300],   # TL
                [1100, 300],  # TR
                [1150, 900],  # BR
                [700, 900]    # BL
            ],
            "road_width_m": 5.0,
            "road_length_m": 25.0,
            "color": (0, 140, 255),
            "speed_limit_kmh": 50.0,
            "expected_flow": "down"
        }
    ]

    estimator = SpeedEstimator(
        zones=zones_config,
        frame_size=(1280, 1000)
    )

    assert len(estimator.zones) == 2, f"Expected 2 zones, got {len(estimator.zones)}"
    print(f"[PASS] Initialized multi-zone estimator with {len(estimator.zones)} zones: {[z.name for z in estimator.zones]}")

    # 2. Test Dynamic Zone Lookup via cv2.pointPolygonTest
    # Test Point 1 inside Zone 1 (Main Highway)
    pt_zone1 = (300, 600)
    matched_zone1 = estimator.find_zone_for_point(pt_zone1)
    print(f" [Lookup Point {pt_zone1}] Matched: {matched_zone1.name}")
    assert matched_zone1.name == "Main Highway", f"Expected 'Main Highway', got '{matched_zone1.name}'"

    # Test Point 2 inside Zone 2 (Flyover Ramp)
    pt_zone2 = (900, 600)
    matched_zone2 = estimator.find_zone_for_point(pt_zone2)
    print(f" [Lookup Point {pt_zone2}] Matched: {matched_zone2.name}")
    assert matched_zone2.name == "Flyover Ramp", f"Expected 'Flyover Ramp', got '{matched_zone2.name}'"

    # Test Point 3 slightly outside boundaries -> should pick closest zone
    pt_outside = (20, 910)
    matched_outside = estimator.find_zone_for_point(pt_outside)
    print(f" [Lookup Out-of-Bounds {pt_outside}] Fallback: {matched_outside.name}")
    assert matched_outside is not None
    print("[PASS] cv2.pointPolygonTest dynamic zone resolution accurate across both zones.")

    # 3. Test Speed Estimation in Zone 1 (Moving down at ~50 km/h)
    # 50 km/h ~ 13.89 m/s. Over 10 frames @ 30 FPS (0.333s), travel ~4.63 meters.
    # In Zone 1 (length 30m, pixel span y from 300 to 900 => 600px = 30m => 20px/m approx)
    track_id_1 = 101
    y_start = 500
    for f in range(1, 12):
        cur_y = y_start + (f - 1) * 7  # Moving down ~7 pixels per frame
        speed, direction, is_wrong_way = estimator.estimate_speed(track_id_1, (300, cur_y), f, fps=30.0)

    print(f" [Zone 1 Vehicle #{track_id_1}] Calculated Speed: {speed:.1f} km/h | Zone: {estimator.get_vehicle_zone(track_id_1)}")
    assert estimator.get_vehicle_zone(track_id_1) == "Main Highway"
    assert 20.0 < speed < 90.0, f"Speed {speed} out of expected realistic range!"
    assert not is_wrong_way, "Vehicle traveling down should not be flagged as wrong way!"
    print("[PASS] Zone 1 vehicle speed calculation and tracking buffer verified.")

    # 4. Test Speed Estimation in Zone 2 with Overspeeding
    # Speed limit on Ramp is 50 km/h. Let's move fast (> 60 km/h)
    track_id_2 = 202
    y_start_2 = 400
    for f in range(1, 12):
        cur_y = y_start_2 + (f - 1) * 16  # Moving rapidly down
        speed2, direction2, is_wrong_way2 = estimator.estimate_speed(track_id_2, (900, cur_y), f, fps=30.0)

    print(f" [Zone 2 Vehicle #{track_id_2}] Calculated Speed: {speed2:.1f} km/h | Zone: {estimator.get_vehicle_zone(track_id_2)}")
    assert estimator.get_vehicle_zone(track_id_2) == "Flyover Ramp"
    assert speed2 > 50.0, f"Expected overspeed > 50 km/h, got {speed2}"
    assert estimator.is_overspeeding(track_id_2), "Vehicle #{track_id_2} should be flagged as overspeeding in Zone 2!"
    print("[PASS] Zone 2 vehicle overspeeding detected according to per-zone speed limits.")

    # 5. Test Wrong-Way Detection in Zone 2 (Vehicle moving upwards: y decreasing)
    track_id_3 = 303
    y_start_3 = 800
    for f in range(1, 12):
        cur_y = y_start_3 - (f - 1) * 12  # Moving upwards against traffic flow
        speed3, direction3, is_wrong_way3 = estimator.estimate_speed(track_id_3, (900, cur_y), f, fps=30.0)

    print(f" [Zone 2 Wrong-Way Vehicle #{track_id_3}] Speed: {speed3:.1f} km/h | Wrong-Way Flag: {is_wrong_way3}")
    assert is_wrong_way3, "Vehicle #{track_id_3} traveling backwards must be flagged as wrong-way!"
    print("[PASS] Wrong-way detection correctly triggered in multi-zone environment.")

    # 6. Test Multi-Zone Visualization
    test_canvas = np.zeros((1000, 1280, 3), dtype=np.uint8)
    drawn_canvas = estimator.draw_calibration_zones(test_canvas)
    assert drawn_canvas is not None
    assert np.any(drawn_canvas > 0), "Drawn canvas should contain non-zero overlay pixels!"
    print("[PASS] Multi-zone calibration overlay successfully drawn with distinct colors and badges.")

    print("==================================================\n")


if __name__ == "__main__":
    test_multi_zone_speed()
