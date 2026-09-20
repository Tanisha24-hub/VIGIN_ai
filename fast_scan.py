# pyrefly: ignore [missing-import]
import cv2
import sys
import os
from modules.plate_ocr import PlateRecognizer

def fast_scan_video(video_path, skip_frames=30):
    # Initialize PlateRecognizer
    recognizer = PlateRecognizer(ocr_confidence_threshold=0.5)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    print(f"Scanning '{video_path}' ({total_frames} frames @ {fps:.1f} FPS)...")
    print(f"Processing 1 frame every {skip_frames} frames (~{fps/skip_frames:.1f} Hz)...")
    
    frame_idx = 0
    while frame_idx < total_frames:
        # Quickly skip to the next target frame without decoding intermediate frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize full frame to 50% for faster OCR if scanning full screen text
        small_frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        
        # Run OCR
        timestamp = frame_idx / fps
        results, conf = recognizer.extract_text(small_frame)
        if results:
            print(f"[{timestamp:.1f}s] Frame {frame_idx} (Conf: {conf:.2f}): {results}")
            
        frame_idx += skip_frames
        
    cap.release()
    print("Done!")

if __name__ == "__main__":
    video = "data/input/test_compilation.mp4"
    # Change skip_frames=15 (to scan every 0.5s) or 30 (to scan every 1.0s)
    fast_scan_video(video, skip_frames=30)
