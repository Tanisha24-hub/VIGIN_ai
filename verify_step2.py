from utils.image_processing import crop_cabin_region, enhance_cabin_interior, analyze_cabin
from ultralytics import YOLO
import cv2
import os

print("==================================================")
print("       VIGIN_ai: Step 2 Verification Suite        ")
print("==================================================")

yolo_model = YOLO("yolov8n.pt")

image_path = os.path.join("data", "input", "test_car.jpg")
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image at {image_path}")
    exit(1)

# Step 2a: Crop cabin region using vehicle bbox
# Let's approximate a vehicle bbox for the car in test_car.jpg
results = yolo_model(frame, classes=[2, 3, 5, 7], verbose=False)
if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
    bbox = results[0].boxes.xyxy.cpu().numpy()[0]
    
    print(f"Detected Vehicle Bounding Box: {bbox}")
    
    # Crop
    cabin_crop = crop_cabin_region(frame, bbox)
    if cabin_crop is not None and cabin_crop.size > 0:
        print("✓ PASS: crop_cabin_region successfully extracted cabin crop.")
        
        # Step 2b: Enhance
        enhanced = enhance_cabin_interior(cabin_crop)
        if enhanced is not None and enhanced.size > 0:
            print("✓ PASS: enhance_cabin_interior successfully enhanced the crop.")
            
            # Step 2c: Analyze
            cabin_data = analyze_cabin(cabin_crop, yolo_model)
            print(f"✓ PASS: analyze_cabin returned: occupants={cabin_data['occupants']}, driver_present={cabin_data['driver_present']}")
            
            # Save enhanced crop
            os.makedirs("data/output", exist_ok=True)
            cv2.imwrite("data/output/test_cabin_enhanced.jpg", cabin_data['enhanced'])
            print("✓ PASS: Saved enhanced cabin crop to data/output/test_cabin_enhanced.jpg")
        else:
            print("✗ FAIL: enhance_cabin_interior returned empty/None crop.")
    else:
        print("✗ FAIL: crop_cabin_region returned empty/None crop.")
else:
    print("✗ FAIL: No vehicle detected in test_car.jpg to test cabin region.")
