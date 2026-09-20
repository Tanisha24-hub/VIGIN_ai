import numpy as np
import cv2
import os
import sys

from utils.image_processing import crop_cabin_region, enhance_cabin_interior, analyze_cabin
from modules.safety_analyzer import SafetyBehaviorAnalyzer

def run_tests():
    print("=== Testing Cabin & Safety Pipeline Upgrade ===")

    # Test 1: crop_cabin_region distance and optical resolution gating
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Far car (width 150 < 190) -> should return None
    far_box = (100, 200, 250, 310)
    assert crop_cabin_region(dummy_frame, far_box, class_id=2) is None, "Far car should be rejected"
    print("[PASS] Distance Gate: Far vehicle rejected.")

    # Wrong class (e.g. class_id=3 motorcycle) -> should return None
    assert crop_cabin_region(dummy_frame, (100, 200, 350, 400), class_id=3) is None, "Non-car rejected"
    print("[PASS] Class Gate: Non-car class rejected.")

    # Close car (width 220 >= 190, height 160 >= 130) -> should produce valid crop
    close_box = (200, 250, 420, 410)
    crop = crop_cabin_region(dummy_frame, close_box, class_id=2)
    assert crop is not None and crop.size > 0, "Close car should produce valid cabin crop"
    print(f"[PASS] Cabin Crop: Valid crop generated with shape {crop.shape}.")

    # Test 2: enhance_cabin_interior super-sampling and anti-glare
    test_crop = np.random.randint(40, 180, (60, 100, 3), dtype=np.uint8)
    enhanced = enhance_cabin_interior(test_crop, target_w=320, target_h=180)
    assert enhanced.shape == (180, 320, 3), f"Expected (180, 320, 3), got {enhanced.shape}"
    print(f"[PASS] Enhance Interior: Target shape (180, 320, 3) confirmed.")

    # Test 3: analyze_cabin return structure
    cabin_res = analyze_cabin(test_crop)
    assert "enhanced" in cabin_res and "occupants" in cabin_res and "driver_present" in cabin_res
    assert cabin_res["enhanced"].shape == (180, 320, 3)
    assert isinstance(cabin_res["occupants"], int)
    assert isinstance(cabin_res["driver_present"], bool)
    print(f"[PASS] analyze_cabin: Output format verified. Occupants: {cabin_res['occupants']}, Driver: {cabin_res['driver_present']}")

    # Test 4: SafetyBehaviorAnalyzer
    safety_engine = SafetyBehaviorAnalyzer()
    print(f"[INFO] Pose model loaded: {safety_engine.pose_model is not None}")
    
    annotated_crop, alerts = safety_engine.inspect_driver_safety(enhanced, track_id=1)
    assert annotated_crop.shape == (180, 320, 3)
    assert isinstance(alerts, list)
    print(f"[PASS] inspect_driver_safety: Processed crop without errors. Alerts: {alerts}")

    # Test 5: check_motorcycle_safety
    bike_crop = np.zeros((150, 100, 3), dtype=np.uint8)
    bike_violations = safety_engine.check_motorcycle_safety(bike_crop)
    assert isinstance(bike_violations, list)
    print(f"[PASS] check_motorcycle_safety: Output verified. Violations: {bike_violations}")

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    run_tests()
