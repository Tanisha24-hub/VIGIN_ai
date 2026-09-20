import cv2
import os
import re
import numpy as np
from ultralytics import YOLO
from modules.plate_ocr import PlateRecognizer

# 1. Initialize models
vehicle_detector = YOLO('yolov8n.pt')
ocr_engine = PlateRecognizer()

image_path = os.path.join("data", "input", "test_car.jpg")
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load {image_path}. Make sure data/input/test_car.jpg exists.")
    exit()

display_frame = frame.copy()

def restore_and_enhance_plate(crop):
    """Upscales and applies contrast sharpening to distant/blurry plates."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop
    
    # 1. Super-resolution 3x upscaling if the crop is small
    scale = 3.0 if w < 200 else 1.5
    upscaled = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale & Morphological Hat Filter to pop dark text on bright plates
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if len(upscaled.shape) == 3 else upscaled
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(cv2.subtract(gray, blackhat), tophat)
    
    # 3. Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)
    return enhanced

# 2. Detect all vehicles (conf=0.15 captures distant/blurry background cars)
results = vehicle_detector(frame, classes=[2, 3, 5, 7], conf=0.15, verbose=False)

detected_count = 0
for r in results:
    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            car_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            
            if car_crop.size > 0:
                plate_text, conf, plate_bbox = ocr_engine.extract_plate_and_box(car_crop)
                
                # If standard OCR missed it because it's blurry/distant, run enhancement
                if not plate_text or conf < 0.4:
                    vh, vw = car_crop.shape[:2]
                    lower_car = car_crop[int(vh * 0.4):, :]
                    if lower_car.size > 0:
                        restored = restore_and_enhance_plate(lower_car)
                        text_res, conf_res = ocr_engine.extract_text(restored)
                        if conf_res > conf and len(text_res) >= 4:
                            plate_text = text_res
                            conf = conf_res
                            plate_bbox = (0, int(vh * 0.4), vw, vh)

                # Draw tight blue box if plate is detected
                if plate_text and plate_bbox is not None:
                    detected_count += 1
                    px1, py1, px2, py2 = plate_bbox
                    gx1, gy1 = x1 + int(px1), y1 + int(py1)
                    gx2, gy2 = x1 + int(px2), y1 + int(py2)

                    print(f"✅ Found Plate: '{plate_text}' | Confidence: {conf:.2f}")

                    # Tight blue box around plate
                    cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
                    
                    # Blue badge label
                    (tw, th), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (gx1, max(0, gy1 - th - 8)), (gx1 + tw + 6, gy1), (255, 0, 0), -1)
                    cv2.putText(display_frame, plate_text, (gx1 + 3, max(12, gy1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Outer green vehicle bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

print(f"\nTotal Plates Detected: {detected_count}")

# 3. Save and display
os.makedirs("data/output", exist_ok=True)
out_path = os.path.join("data", "output", "detected_test_car.jpg")
cv2.imwrite(out_path, display_frame)
print(f"📁 Output saved to: {out_path}")

# Preview
h, w = display_frame.shape[:2]
preview = cv2.resize(display_frame, (1024, int(h * (1024 / w))))
cv2.imshow("Multi-Car & Distant Plate Detection (Press any key to close)", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()
