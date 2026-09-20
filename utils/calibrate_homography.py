"""
Interactive Homography Road Plane Calibration Tool for VIGIN_ai.

Usage:
    python utils/calibrate_homography.py
    python utils/calibrate_homography.py --video data/input/sample3.mp4
    python utils/calibrate_homography.py --video data/input/sample_4.mp4 --frame 60

Controls:
    - Left Click: Select 4 road polygon corners in order:
        1. Top-Left (TL)     - Far left road boundary / lane marking
        2. Top-Right (TR)    - Far right road boundary / lane marking
        3. Bottom-Right (BR) - Near right road boundary / lane marking
        4. Bottom-Left (BL)  - Near left road boundary / lane marking
    - Key 'r': Reset / Clear clicked points
    - Key 's': Advance 30 frames forward
    - Key 'a': Go back 30 frames
    - Key 'q': Quit and print configuration
"""

import sys
import os
import argparse
import cv2
import numpy as np

# Ensure root directory is on Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

clicked_points = []
orig_frame = None
display_scale = 1.0


def mouse_callback(event, x, y, flags, param):
    global clicked_points, orig_frame, display_scale
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            # Map displayed coordinates back to full original resolution
            orig_x = int(round(x / display_scale))
            orig_y = int(round(y / display_scale))
            clicked_points.append((orig_x, orig_y))
            labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
            idx = len(clicked_points) - 1
            print(f" [+] Point {idx+1}/4 ({labels[idx]}): Pixel=({orig_x}, {orig_y})")


def find_video_source(requested_path):
    """Finds the video file, searching relative to cwd and BASE_DIR/data/input."""
    if requested_path and os.path.exists(requested_path):
        return requested_path

    # Check relative to BASE_DIR
    if requested_path:
        base_rel = os.path.join(BASE_DIR, requested_path)
        if os.path.exists(base_rel):
            return base_rel

    # Try common sample videos
    candidates = [
        os.path.join(BASE_DIR, "data", "input", "sample3.mp4"),
        os.path.join(BASE_DIR, "data", "input", "sample_4.mp4"),
        os.path.join(BASE_DIR, "data", "input", "bangalore_traffic.mp4"),
        os.path.join(BASE_DIR, "data", "input", "sample_cctv.mp4"),
        os.path.join(BASE_DIR, "data", "input", "comprehensive_test.mp4")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return requested_path


def run_calibration(video_path, frame_idx=50):
    global clicked_points, orig_frame, display_scale

    resolved_path = find_video_source(video_path)
    if not resolved_path or not os.path.exists(resolved_path):
        print(f"[ERROR] Video file not found: {video_path}")
        print(f"Looked in: {resolved_path}")
        return

    cap = cv2.VideoCapture(resolved_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read frame from video.")
        return

    orig_h, orig_w = frame.shape[:2]
    print("=" * 65)
    print("      VIGIN_ai: Homography Road Calibration Tool")
    print("=" * 65)
    print(f" Video Source:     {resolved_path}")
    print(f" Native Resolution: {orig_w} x {orig_h}")
    print("\n Instructions:")
    print(" Click 4 points on the road surface in order:")
    print("   1. [Top-Left]     - Far left boundary / lane marker")
    print("   2. [Top-Right]    - Far right boundary / lane marker")
    print("   3. [Bottom-Right] - Near right boundary / lane marker")
    print("   4. [Bottom-Left]  - Near left boundary / lane marker")
    print("\n Keyboard Controls:")
    print("   [r] Reset points | [s] Skip 30 frames | [a] Prev 30 frames | [q] Quit & Output")
    print("=" * 65)

    window_name = "VIGIN_ai Homography Calibrator (Click 4 Road Corners)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Compute display scaling so large 4K frames fit comfortably on screen
    max_disp_w, max_disp_h = 1280, 720
    scale_w = max_disp_w / orig_w
    scale_h = max_disp_h / orig_h
    display_scale = min(scale_w, scale_h, 1.0)
    disp_w = int(orig_w * display_scale)
    disp_h = int(orig_h * display_scale)
    cv2.resizeWindow(window_name, disp_w, disp_h)

    while True:
        vis_frame = frame.copy()

        # Draw selected points and connecting polygon lines
        labels = ["1. TL", "2. TR", "3. BR", "4. BL"]
        colors = [(0, 255, 255), (0, 255, 0), (0, 165, 255), (255, 0, 255)]

        for i, pt in enumerate(clicked_points):
            cv2.circle(vis_frame, pt, int(8 / display_scale), colors[i], -1)
            cv2.putText(vis_frame, f"{labels[i]} {pt}", (pt[0] + 15, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65 / display_scale, (255, 255, 255), 2, cv2.LINE_AA)

        if len(clicked_points) > 1:
            for i in range(len(clicked_points) - 1):
                cv2.line(vis_frame, clicked_points[i], clicked_points[i + 1], (0, 255, 255), max(2, int(2 / display_scale)))

        if len(clicked_points) == 4:
            # Close the polygon
            cv2.line(vis_frame, clicked_points[3], clicked_points[0], (0, 255, 255), max(2, int(2 / display_scale)))

            # Semi-transparent overlay over road zone
            overlay = vis_frame.copy()
            poly_pts = np.array(clicked_points, dtype=np.int32)
            cv2.fillPoly(overlay, [poly_pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.25, vis_frame, 0.75, 0, vis_frame)

            # Generate top-down warped Bird's Eye View preview
            src_np = np.array(clicked_points, dtype=np.float32)
            bev_w, bev_h = 300, 600
            dst_np = np.array([[0, 0], [bev_w, 0], [bev_w, bev_h], [0, bev_h]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(src_np, dst_np)
            warped = cv2.warpPerspective(frame, H, (bev_w, bev_h))
            cv2.imshow("Bird's Eye View (BEV Preview)", warped)

        # Instructions banner at top
        banner_text = f"Points: {len(clicked_points)}/4. [r]=Reset [s]=Next Frame [a]=Prev Frame [q]=Save & Exit"
        cv2.putText(vis_frame, banner_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80 / display_scale, (0, 255, 255), 2, cv2.LINE_AA)

        # Resize for display
        display_view = cv2.resize(vis_frame, (disp_w, disp_h))
        cv2.imshow(window_name, display_view)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            clicked_points = []
            print(" [INFO] Reset clicked points. Click 4 points again.")
            try:
                cv2.destroyWindow("Bird's Eye View (BEV Preview)")
            except Exception:
                pass
        elif key == ord('s'):
            cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos + 30)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                clicked_points = []
                print(f" [INFO] Advanced forward to frame {int(cur_pos + 30)}.")
        elif key == ord('a'):
            cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            target_f = max(0, cur_pos - 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                clicked_points = []
                print(f" [INFO] Rewound back to frame {int(target_f)}.")

    cap.release()
    cv2.destroyAllWindows()

    if len(clicked_points) == 4:
        norm_pts = [[round(p[0] / orig_w, 4), round(p[1] / orig_h, 4)] for p in clicked_points]
        print("\n" + "=" * 65)
        print("  🎉 CALIBRATION COMPLETE! COPY CONFIGURATION BELOW:")
        print("=" * 65)
        print("\n# 1. Absolute Pixel Coordinates (for native resolution):")
        print(f"HOMOGRAPHY_SRC_POINTS = {clicked_points}\n")
        print("# 2. Normalized Resolution-Independent Coordinates (0.0 to 1.0):")
        print(f"HOMOGRAPHY_SRC_POINTS_NORM = {norm_pts}\n")
        print("# 3. Drop-in usage for SpeedEstimator:")
        print(f"speed_estimator = SpeedEstimator(src_points={norm_pts}, frame_size=({orig_w}, {orig_h}))\n")
        print("=" * 65 + "\n")
    else:
        print("\n[INFO] Calibration window closed without selecting all 4 points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Homography Road Plane for VIGIN_ai")
    parser.add_argument("--video", type=str, default="data/input/sample3.mp4", help="Path to input video")
    parser.add_argument("--frame", type=int, default=50, help="Initial frame number to view")
    args = parser.parse_args()

    run_calibration(args.video, frame_idx=args.frame)
