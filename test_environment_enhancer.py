"""
Unit and benchmark tests for EnvironmentEnhancer.
"""

import time
import numpy as np
import cv2
from utils.environment_enhancer import EnvironmentEnhancer


def test_environment_enhancer():
    print("==================================================")
    print("     TESTING: EnvironmentEnhancer Module         ")
    print("==================================================")

    # 1. Test Modes Initialization
    enhancer_auto = EnvironmentEnhancer(mode="auto")
    enhancer_night = EnvironmentEnhancer(mode="night")
    enhancer_rain = EnvironmentEnhancer(mode="rain")
    enhancer_clear = EnvironmentEnhancer(mode="clear")

    assert enhancer_auto.mode == "auto"
    assert enhancer_night.mode == "night"
    assert enhancer_rain.mode == "rain"
    assert enhancer_clear.mode == "clear"
    print("[PASS] Initialized EnvironmentEnhancer across all 4 modes.")

    # 2. Test Low-Light / Night Enhancement
    # Create synthetic dark night scene (mean luminance ~35) with bright headlights
    dark_scene = np.ones((720, 1280, 3), dtype=np.uint8) * 35
    # Add headlights (bright circles > 230)
    cv2.circle(dark_scene, (600, 400), 25, (245, 245, 245), -1)
    cv2.circle(dark_scene, (680, 400), 25, (245, 245, 245), -1)

    initial_mean = float(np.mean(dark_scene))
    initial_headlight = float(dark_scene[400, 600, 0])

    night_enhanced = enhancer_night.enhance_night(dark_scene)
    enhanced_mean = float(np.mean(night_enhanced))
    enhanced_headlight = float(night_enhanced[400, 600, 0])

    print(f" [Night Enhancement] Initial Mean: {initial_mean:.1f} -> Enhanced Mean: {enhanced_mean:.1f}")
    print(f" [Headlight Preservation] Initial: {initial_headlight} -> Enhanced: {enhanced_headlight}")

    assert enhanced_mean > initial_mean + 15.0, "Night enhancer should boost dark scene brightness!"
    # Ensure highlight was preserved without extreme clipping distortion
    assert enhanced_headlight >= 210, "Headlights should remain bright without distortion!"
    print("[PASS] Low-light Retinex/gamma boost lifted shadows while preserving headlights.")
    
    # 3. Test Rain/Haze De-hazing (DCP + Bilateral filter)
    # Create hazy/rainy scene: blend with atmospheric white veil (mean dark channel high)
    clear_scene = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(clear_scene, (200, 200), (800, 600), (120, 80, 40), -1)
    # Add simulated atmospheric haze
    hazy_scene = cv2.addWeighted(clear_scene, 0.45, np.full_like(clear_scene, 210), 0.55, 0)
    # Add simulated rain streaks
    for _ in range(50):
        rx = np.random.randint(0, 1280)
        ry = np.random.randint(0, 700)
        cv2.line(hazy_scene, (rx, ry), (rx + 2, ry + 15), (255, 255, 255), 1)

    rain_enhanced = enhancer_rain.enhance_rain(hazy_scene)

    hazy_std = float(np.std(hazy_scene))
    enhanced_std = float(np.std(rain_enhanced))
    print(f" [Rain/Haze De-hazing] Hazy Contrast Std: {hazy_std:.1f} -> De-hazed Std: {enhanced_std:.1f}")

    assert enhanced_std > hazy_std, "DCP de-hazing should restore scene contrast!"
    assert rain_enhanced.shape == hazy_scene.shape
    print("[PASS] Dark Channel Prior (DCP) + Bilateral de-hazing successfully restored contrast.")

    # 4. Test Auto Condition Detection
    enhancer_auto.reset_history()
    det_night, metrics_night = enhancer_auto.analyze_environment(dark_scene)
    print(f" [Auto Detect Dark Scene] Detected: {det_night}, Metrics: {metrics_night}")
    assert det_night == "night", f"Expected 'night', got '{det_night}'"

    enhancer_auto.reset_history()
    det_rain, metrics_rain = enhancer_auto.analyze_environment(hazy_scene)
    print(f" [Auto Detect Hazy Scene] Detected: {det_rain}, Metrics: {metrics_rain}")
    assert det_rain == "rain", f"Expected 'rain', got '{det_rain}'"

    # Test clean daylight scene
    daylight = np.zeros((720, 1280, 3), dtype=np.uint8)
    daylight[:, :] = (130, 120, 110)
    cv2.rectangle(daylight, (100, 100), (1100, 600), (40, 180, 50), -1)
    enhancer_auto.reset_history()
    det_clear, metrics_clear = enhancer_auto.analyze_environment(daylight)
    print(f" [Auto Detect Daylight] Detected: {det_clear}, Metrics: {metrics_clear}")
    assert det_clear == "clear", f"Expected 'clear', got '{det_clear}'"
    print("[PASS] Auto condition sensing correctly classifies night, rain, and clear scenes.")

    # 5. Benchmark Latency for 30 FPS Pipeline
    test_frame = np.random.randint(40, 160, (720, 1280, 3), dtype=np.uint8)
    warmup_runs = 3
    for _ in range(warmup_runs):
        enhancer_auto.process(test_frame)

    timed_runs = 20
    t0 = time.perf_counter()
    for _ in range(timed_runs):
        enhancer_auto.process(test_frame)
    avg_latency_ms = ((time.perf_counter() - t0) / timed_runs) * 1000.0

    print(f"\n [Benchmark] Average Processing Latency: {avg_latency_ms:.2f} ms per 720p/1080p frame")
    assert avg_latency_ms < 25.0, f"Processing too slow ({avg_latency_ms:.2f} ms)! Must sustain 30 FPS (< 33 ms)."
    print("[PASS] Latency test passed! Ready for 30+ FPS real-time execution.")
    print("==================================================\n")


if __name__ == "__main__":
    test_environment_enhancer()
