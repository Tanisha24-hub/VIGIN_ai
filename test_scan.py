import cv2
from datetime import timedelta
from ultralytics import YOLO
from modules.plate_ocr import PlateRecognizer

# 1. Initialize models
try:
    model = YOLO("yolov8n.pt")
except Exception:
    model = YOLO("models/yolov8n.pt")

recognizer = PlateRecognizer()

video_path = "data/input/test_compilation.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

detected_vehicles_cache = set()
frame_count = 0
FRAME_SKIP = 5  # Scan every 5th frame for fast processing

print("⚡ Starting fast video scan...")

while cap.isOpened():
    ret = cap.grab()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    ret, frame = cap.retrieve()
    if not ret:
        continue

    # Run YOLO vehicle tracker
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False, imgsz=640)

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        if boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(xyxy, track_ids):
                if track_id not in detected_vehicles_cache:
                    x1, y1, x2, y2 = box
                    car_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    
                    if car_crop.size > 0:
                        plate_text = recognizer.extract_text(car_crop)
                        if plate_text and plate_text != ('', 0.0):
                            timestamp = str(timedelta(seconds=frame_count / fps))
                            print(f"[{timestamp}] Vehicle ID #{track_id} Plate: {plate_text}")
                            detected_vehicles_cache.add(track_id)

cap.release()
print("✅ Full video scan completed!")
