import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from utils.image_processing import process_license_plate
from config.settings import PLATE_MODEL_NAME, BLUR_THRESHOLD

def preprocess_plate_image(plate_crop, blur_threshold=None):
    """
    Exposed helper function using CLAHE and sharpening to clean up blurry or dark
    license plate cropped images before passing them to OCR. Only applies sharpening
    kernel if the crop is detected as blurry.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
        
    if len(plate_crop.shape) == 2 or plate_crop.shape[2] == 1:
        gray = plate_crop
    else:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
    if blur_threshold is None:
        blur_threshold = BLUR_THRESHOLD

    # Measure blurriness using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = laplacian_var < blur_threshold
    
    # Apply CLAHE (use a gentler clipLimit of 2.0 to minimize noise)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    if is_blurry:
        print(f"[INFO] Blurry plate crop detected (Variance: {laplacian_var:.2f} < Threshold: {blur_threshold}). Applying sharpening kernel.")
        # Apply sharpening filter
        sharpen_kernel = np.array([[0, -1, 0], 
                                   [-1, 5, -1], 
                                   [0, -1, 0]])
        processed = cv2.filter2D(enhanced, -1, sharpen_kernel)
    else:
        processed = enhanced
        
    # Resize small plate crops to a standard height of 64 pixels to aid neural network OCR
    ph, pw = processed.shape[:2]
    if ph < 64 and ph > 0:
        target_h = 64
        target_w = int(pw * (target_h / ph))
        processed = cv2.resize(processed, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
    return processed

class PlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, plate_crop):
        if plate_crop.size == 0:
            return ""

        cleaned_crop = process_license_plate(plate_crop)
        results = self.reader.readtext(cleaned_crop)

        extracted_text = []
        for _, text, conf in results:
            if conf > 0.25:
                extracted_text.append(text.upper())

        return " ".join(extracted_text)


class PlateRecognizer:
    def __init__(self, yolo_model_path='yolov8n.pt', ocr_confidence_threshold=0.25, strict_mode=True):
        """
        Initializes the PlateRecognizer with a YOLOv8 model for vehicle tracking,
        a YOLOv8 model for license plate detection, and PaddleOCR for license plate reading.
        """
        self.detector = YOLO(yolo_model_path)
        self.strict_mode = strict_mode
        
        # Load YOLOv8 license plate detector
        from huggingface_hub import hf_hub_download
        try:
            if PLATE_MODEL_NAME.endswith(".pt"):
                model_path = PLATE_MODEL_NAME
            else:
                print(f"[INFO] Fetching plate detector from Hugging Face repo '{PLATE_MODEL_NAME}'...")
                model_path = hf_hub_download(repo_id=PLATE_MODEL_NAME, filename="best.pt")
                
            self.plate_detector = YOLO(model_path)
            print(f"[INFO] Loaded plate detector: {model_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load plate detector '{PLATE_MODEL_NAME}': {e}. Falling back to default.")
            try:
                fallback_path = hf_hub_download(repo_id="Koushim/yolov8-license-plate-detection", filename="best.pt")
                self.plate_detector = YOLO(fallback_path)
            except Exception as fallback_err:
                print(f"[ERROR] Failed to load fallback model: {fallback_err}")
                self.plate_detector = YOLO("yolov8n.pt")
            
        import logging
        logging.getLogger('ppocr').setLevel(logging.WARNING)
        logging.getLogger('paddleocr').setLevel(logging.WARNING)
        self.ocr = PaddleOCR(use_textline_orientation=False, use_doc_orientation_classify=False, use_doc_unwarping=False, lang='en', enable_mkldnn=False)
        self.ocr_confidence_threshold = ocr_confidence_threshold


    def perspective_transform(self, image, pts):
        """
        Applies perspective transformation to obtain a bird's-eye view of the plate.
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sort points based on x and y coordinates
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        (tl, tr, br, bl) = rect
        
        # Calculate new image width
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # Calculate new image height
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth < 20 or maxHeight < 10:
            return None
            
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def localize_plate_with_box(self, vehicle_crop):
        """
        Localize license plate within vehicle crop using YOLOv8 plate detector.
        Falls back to traditional contour-based computer vision if YOLO fails.
        Returns: (plate_crop, plate_box) where plate_box is (x1, y1, x2, y2) relative to vehicle_crop.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, None

        # 1. Try YOLOv8 License Plate Detection
        try:
            results = self.plate_detector(vehicle_crop, verbose=False, conf=0.25)
            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                best_box = None
                best_conf = -1.0
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box
                
                if best_box is not None:
                    x1, y1, x2, y2 = map(int, best_box)
                    h, w = vehicle_crop.shape[:2]
                    
                    # Add 6% padding to preserve character edges
                    pad_w = int((x2 - x1) * 0.06)
                    pad_h = int((y2 - y1) * 0.06)
                    px1 = max(0, x1 - pad_w)
                    py1 = max(0, y1 - pad_h)
                    px2 = min(w, x2 + pad_w)
                    py2 = min(h, y2 + pad_h)
                    
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    if plate_crop.size > 0:
                        return plate_crop, (px1, py1, px2, py2)
        except Exception as e:
            print(f"[WARNING] YOLO plate localization failed: {e}. Using traditional CV fallback.")

        # 2. Fallback: Traditional CV-based localization
        h, w = vehicle_crop.shape[:2]
        ymin, ymax = int(h * 0.45), int(h * 0.95)
        xmin, xmax = int(w * 0.15), int(w * 0.85)
        search_region = vehicle_crop[ymin:ymax, xmin:xmax]
        
        if search_region.size == 0:
            search_region = vehicle_crop
            ymin, xmin = 0, 0

        # Preprocessing for contour analysis
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blurred, 30, 200)
        
        # Connect vertical and horizontal components of the license plate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        
        plate_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            
            if len(approx) == 4:
                x, y, w_box, h_box = cv2.boundingRect(approx)
                aspect_ratio = w_box / float(h_box)
                
                # License plates are rectangular with aspect ratio typically between 2.0 and 6.0
                if 2.0 <= aspect_ratio <= 6.0 and (0.005 * search_region.size <= w_box * h_box <= 0.25 * search_region.size):
                    plate_contour = approx
                    break
                    
        if plate_contour is not None:
            # Shift coordinates back to vehicle crop coordinate space
            plate_contour_shifted = plate_contour.copy()
            plate_contour_shifted[:, 0, 0] += xmin
            plate_contour_shifted[:, 0, 1] += ymin
            
            x, y, w_box, h_box = cv2.boundingRect(plate_contour_shifted)
            warped = self.perspective_transform(vehicle_crop, plate_contour_shifted.reshape(4, 2))
            if warped is not None and warped.size > 0:
                return warped, (x, y, x + w_box, y + h_box)
                
        # Fallback 1: Use the bounding box of the largest candidate contour
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            aspect_ratio = w_box / float(h_box)
            if 2.0 <= aspect_ratio <= 6.0 and (0.005 * search_region.size <= w_box * h_box <= 0.3 * search_region.size):
                crop = search_region[y:y+h_box, x:x+w_box]
                px1 = xmin + x
                py1 = ymin + y
                px2 = xmin + x + w_box
                py2 = ymin + y + h_box
                return crop, (px1, py1, px2, py2)

        # Fallback 2: Crop a fixed lower-middle region of the vehicle
        fh, fw = vehicle_crop.shape[:2]
        crop_y1, crop_y2 = int(fh * 0.55), int(fh * 0.85)
        crop_x1, crop_x2 = int(fw * 0.25), int(fw * 0.75)
        return vehicle_crop[crop_y1:crop_y2, crop_x1:crop_x2], (crop_x1, crop_y1, crop_x2, crop_y2)

    def localize_plate(self, vehicle_crop):
        """
        Localize license plate within vehicle crop using YOLOv8 plate detector.
        Falls back to traditional contour-based computer vision if YOLO fails.
        """
        plate_crop, _ = self.localize_plate_with_box(vehicle_crop)
        return plate_crop

    def preprocess_plate(self, plate_crop):
        """
        Applies OpenCV CLAHE and a sharpening filter to handle blurry or low-contrast plates.
        """
        return preprocess_plate_image(plate_crop)

    def enhance_blurry_crop(self, plate_crop):
        """
        Enhances a blurry or distant license plate crop to improve OCR readability.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        # 1. Upscale if width < 150px
        h, w = plate_crop.shape[:2]
        if w < 150:
            plate_crop = cv2.resize(plate_crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

        # 2. Convert to grayscale
        if len(plate_crop.shape) == 3 and plate_crop.shape[2] == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()

        # 3. Top-Hat / Black-Hat morphological filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        enhanced = cv2.add(gray, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)

        # 4. Bilateral filtering to smooth noise while keeping character edges sharp
        filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # 5. Adaptive Gaussian thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    def standardize_indian_plate(self, text):
        """
        Standardizes Indian license plate format by swapping common OCR confusions:
        'O' <-> '0', 'I' <-> '1', 'B' <-> '8', 'S' <-> '5' based on character index.
        """
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        n = len(text)
        if n < 7 or n > 11:
            return text

        char_list = list(text)

        def to_letter(c):
            mapping = {'0': 'O', '1': 'I', '8': 'B', '5': 'S'}
            return mapping.get(c, c)

        def to_digit(c):
            mapping = {'O': '0', 'I': '1', 'B': '8', 'S': '5'}
            return mapping.get(c, c)

        # 1. State Code (First two chars) -> Letters
        char_list[0] = to_letter(char_list[0])
        char_list[1] = to_letter(char_list[1])

        # 2. District Code (Next two chars) -> Digits
        char_list[2] = to_digit(char_list[2])
        char_list[3] = to_digit(char_list[3])

        # 3. Unique ID (Last four chars) -> Digits
        for i in range(max(4, n - 4), n):
            char_list[i] = to_digit(char_list[i])

        # 4. Middle characters (index 4 to N-5) -> Letters
        for i in range(4, max(4, n - 4)):
            char_list[i] = to_letter(char_list[i])

        return "".join(char_list)

    def normalize_plate_text(self, raw_text):
        """
        Clean up and normalize plate text by disambiguating character misreads
        for Indian standard plates and Bharat (BH) series plates.
        """
        # Clean up OCR text (strip spaces, hyphens, dots, special symbols, and convert to uppercase)
        text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        n = len(text)
        if n < 4:
            return text

        to_letter = {'0': 'O', '1': 'I', '8': 'B', '5': 'S'}
        to_digit = {'O': '0', 'I': '1', 'S': '5', 'B': '8'}

        # 1. Normalize as a Standard plate candidate
        std_list = list(text)
        # First 2 characters -> letters
        if n >= 1:
            std_list[0] = to_letter.get(std_list[0], std_list[0])
        if n >= 2:
            std_list[1] = to_letter.get(std_list[1], std_list[1])
        # District digits (positions 2-3) -> digits
        if n >= 3:
            std_list[2] = to_digit.get(std_list[2], std_list[2])
        if n >= 4:
            std_list[3] = to_digit.get(std_list[3], std_list[3])
        # Final 4 digits -> digits
        for i in range(max(4, n - 4), n):
            std_list[i] = to_digit.get(std_list[i], std_list[i])
        standard_candidate = "".join(std_list)

        # 2. Normalize as a BH series candidate
        # BH format: ^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$
        bh_list = list(text)
        # Positions 0-1 are digits: fix to digits
        if n >= 1:
            bh_list[0] = to_digit.get(bh_list[0], bh_list[0])
        if n >= 2:
            bh_list[1] = to_digit.get(bh_list[1], bh_list[1])
        # Positions 2-3 are letters (BH): fix to letters
        if n >= 3:
            bh_list[2] = to_letter.get(bh_list[2], bh_list[2])
        if n >= 4:
            bh_list[3] = to_letter.get(bh_list[3], bh_list[3])
        # Positions 4-7 are digits: fix to digits
        for i in range(4, min(8, n)):
            bh_list[i] = to_digit.get(bh_list[i], bh_list[i])
        # Last 1 or 2 are letters: fix to letters
        for i in range(max(4, n - 2), n):
            bh_list[i] = to_letter.get(bh_list[i], bh_list[i])
        bh_candidate = "".join(bh_list)

        # Patterns
        standard_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
        bh_pattern = r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$'

        # Check which one matches either regex
        if re.match(standard_pattern, standard_candidate):
            return standard_candidate
        if re.match(bh_pattern, bh_candidate):
            return bh_candidate
        if re.match(bh_pattern, standard_candidate):
            return standard_candidate
        if re.match(standard_pattern, bh_candidate):
            return bh_candidate

        # Fallback to standard normalization candidate
        return standard_candidate

    def should_discard_text(self, text):
        """
        Discards any text containing noise, timestamps, or watermarks.
        """
        text_upper = text.upper()
        discard_words = ['VENTURES', 'YOUTUBE', 'KM/H', '2023', '2024', '2025', '2026']
        for word in discard_words:
            if word in text_upper:
                return True
        return False

    def is_valid_indian_plate(self, text):
        """
        Checks if the text matches standard or BH-series Indian license plate format.
        """
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        standard_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
        bh_pattern = r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$'
        
        if re.match(standard_pattern, cleaned) or re.match(bh_pattern, cleaned):
            return True
        return False

    def get_alphanumeric_score(self, text, confidence):
        """
        Calculates alphanumeric score: (count_of_alphanumeric_chars, confidence).
        """
        alnum_count = sum(1 for c in text if c.isalnum())
        return (alnum_count, confidence)

    def _ocr_and_parse(self, bgr_crop, scale_factor=1.0, threshold=0.45):
        """
        Runs PaddleOCR on a crop, parses results, and scales polygon points back to original crop size.
        """
        try:
            results = self.ocr.ocr(bgr_crop)
        except Exception as e:
            print(f"[Error] PaddleOCR failed on crop: {e}")
            return "", 0.0, []

        if not results or len(results) == 0:
            return "", 0.0, []

        all_texts = []
        all_confs = []
        all_points = []

        first_result = results[0]

        if isinstance(first_result, dict):
            texts = first_result.get('rec_texts', [])
            scores = first_result.get('rec_scores', [])
            polys = first_result.get('dt_polys', [])
            for text, conf, poly in zip(texts, scores, polys):
                if conf >= threshold:
                    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if cleaned:
                        all_texts.append(cleaned)
                        all_confs.append(conf)
                        for pt in poly:
                            all_points.append([pt[0] / scale_factor, pt[1] / scale_factor])

        elif isinstance(first_result, list):
            for line in first_result:
                if (isinstance(line, list) and len(line) >= 2
                        and isinstance(line[1], (list, tuple))
                        and len(line[1]) >= 2):
                    poly = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    if conf >= threshold:
                        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if cleaned:
                            all_texts.append(cleaned)
                            all_confs.append(conf)
                            for pt in poly:
                                all_points.append([pt[0] / scale_factor, pt[1] / scale_factor])

        plate_text = ''.join(all_texts)
        avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
        return plate_text, avg_conf, all_points


    def extract_text(self, preprocessed_plate):
        """
        Runs PaddleOCR and extracts plate text above the confidence threshold.
        Returns the concatenated text and the average confidence.
        """
        if preprocessed_plate is None or preprocessed_plate.size == 0:
            return "", 0.0
            
        # Convert grayscale back to BGR for PaddleOCR if it is single-channel
        if len(preprocessed_plate.shape) == 2 or preprocessed_plate.shape[2] == 1:
            bgr_plate = cv2.cvtColor(preprocessed_plate, cv2.COLOR_GRAY2BGR)
        else:
            bgr_plate = preprocessed_plate
        
        try:
            results = self.ocr.ocr(bgr_plate)
        except Exception as e:
            print(f"[Error] PaddleOCR failed during inference: {e}")
            return "", 0.0
            
        extracted_text = []
        confidences = []
        
        if results and len(results) > 0:
            # Handle new dictionary-based structure (PaddleOCR 3.x / PaddleX)
            if isinstance(results[0], dict):
                res_dict = results[0]
                texts = res_dict.get('rec_texts', [])
                scores = res_dict.get('rec_scores', [])
                for text, conf in zip(texts, scores):
                    if conf >= self.ocr_confidence_threshold:
                        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if cleaned:
                            extracted_text.append(cleaned)
                            confidences.append(conf)
            # Handle legacy list-based structure (PaddleOCR 2.x)
            elif isinstance(results[0], list):
                for line in results[0]:
                    if isinstance(line, list) and len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        text = line[1][0]
                        conf = line[1][1]
                        if conf >= self.ocr_confidence_threshold:
                            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                            if cleaned:
                                extracted_text.append(cleaned)
                                confidences.append(conf)
                                
        if extracted_text:
            raw_text = "".join(extracted_text)
            avg_conf = float(np.mean(confidences))
            
            # Normalize plate text using the new helper
            normalized = self.normalize_plate_text(raw_text)
            
            # Check validation
            standard_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
            bh_pattern = r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$'
            
            is_strict_valid = (re.match(standard_pattern, normalized) is not None) or \
                              (re.match(bh_pattern, normalized) is not None)
                              
            if is_strict_valid:
                return normalized, avg_conf
            elif not self.strict_mode:
                cleaned = re.sub(r'[^A-Z0-9]', '', normalized.upper())
                if 6 <= len(cleaned) <= 10 and avg_conf >= 0.50:
                    return normalized, avg_conf
            
        return "", 0.0

    def extract_plate_and_box(self, vehicle_crop):
        """
        Isolates the exact license plate bounding box inside a vehicle crop.

        Performs OCR on both the raw crop and an upscaled CLAHE-enhanced lower-half crop.
        Only returns non-empty plate text if it strictly passes the regex validation
        (or fallback to high-confidence 6–10 character alphanumeric string if strict mode is disabled).
        
        Returns (plate_text, confidence, (px1, py1, px2, py2)) when valid
        text is found, otherwise ('', 0.0, None).
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return ('', 0.0, None)

        vh, vw = vehicle_crop.shape[:2]

        # 1. Localize the license plate region and get its box in vehicle_crop
        plate_crop, plate_box = self.localize_plate_with_box(vehicle_crop)
        bx1, by1, bx2, by2 = plate_box if plate_box is not None else (0, 0, vw, vh)

        # --- Candidate 1: Raw Crop OCR ---
        text1, conf1, points1 = "", 0.0, []
        if plate_crop is not None and plate_crop.size > 0:
            if len(plate_crop.shape) == 2 or plate_crop.shape[2] == 1:
                raw_bgr = cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2BGR)
            else:
                raw_bgr = plate_crop
            text1, conf1, points1 = self._ocr_and_parse(raw_bgr, scale_factor=1.0, threshold=0.25)
        
        # Clean up OCR text first (strip spaces, hyphens, dots, special symbols, and convert to uppercase)
        text1_cleaned = re.sub(r'[^A-Z0-9]', '', text1.upper())
        text1_normalized = self.normalize_plate_text(text1_cleaned)
        print(f"[DEBUG-OCR] Read candidate: '{text1_normalized}' (conf: {conf1:.2f})")

        # --- Candidate 2: Upscaled CLAHE-enhanced lower-half crop OCR ---
        text2, conf2, points2 = "", 0.0, []
        lower_half_y = int(vh / 2)
        lower_half = vehicle_crop[lower_half_y:, :]
        if lower_half.size > 0:
            # Upscale 3x using cv2.INTER_CUBIC
            lh_h, lh_w = lower_half.shape[:2]
            upscaled_lh = cv2.resize(lower_half, (lh_w * 3, lh_h * 3), interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            if len(upscaled_lh.shape) == 3 and upscaled_lh.shape[2] == 3:
                lh_gray = cv2.cvtColor(upscaled_lh, cv2.COLOR_BGR2GRAY)
            else:
                lh_gray = upscaled_lh.copy()
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lh_enhanced = clahe.apply(lh_gray)
            
            # Convert back to BGR for PaddleOCR
            lh_bgr = cv2.cvtColor(lh_enhanced, cv2.COLOR_GRAY2BGR)
            
            text2, conf2, points2 = self._ocr_and_parse(lh_bgr, scale_factor=3.0, threshold=0.25)
        
        # Clean up OCR text first (strip spaces, hyphens, dots, special symbols, and convert to uppercase)
        text2_cleaned = re.sub(r'[^A-Z0-9]', '', text2.upper())
        text2_normalized = self.normalize_plate_text(text2_cleaned)
        print(f"[DEBUG-OCR] Read candidate: '{text2_normalized}' (conf: {conf2:.2f})")

        # --- Evaluation & Validation ---
        def evaluate(normalized_text, conf):
            # Check discard words list to ignore timestamps / watermarks / dashboard text
            if self.should_discard_text(normalized_text):
                return 0, 0.0, False
                
            standard_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
            bh_pattern = r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$'
            
            is_strict_valid = (re.match(standard_pattern, normalized_text) is not None) or \
                              (re.match(bh_pattern, normalized_text) is not None)
                              
            if is_strict_valid:
                return len(normalized_text), conf, True
            elif not self.strict_mode:
                cleaned = re.sub(r'[^A-Z0-9]', '', normalized_text.upper())
                if 6 <= len(cleaned) <= 10 and conf >= 0.50:
                    return len(cleaned), conf, True
            return 0, 0.0, False

        len1, score_conf1, is_valid1 = evaluate(text1_normalized, conf1)
        len2, score_conf2, is_valid2 = evaluate(text2_normalized, conf2)

        score1 = (len1, score_conf1) if is_valid1 else (0, 0.0)
        score2 = (len2, score_conf2) if is_valid2 else (0, 0.0)

        if score2 > score1 and score2 > (0, 0.0):
            chosen_text = text2_normalized
            chosen_conf = conf2
            chosen_points = points2
            is_candidate2 = True
        elif score1 > (0, 0.0):
            chosen_text = text1_normalized
            chosen_conf = conf1
            chosen_points = points1
            is_candidate2 = False
        else:
            return ('', 0.0, None)

        # 6. Map the points back to vehicle_crop coordinates and calculate bounding box
        if chosen_points:
            pts = np.array(chosen_points, dtype=np.float32)
            if is_candidate2:
                px1 = int(np.min(pts[:, 0]))
                py1 = int(np.min(pts[:, 1]) + lower_half_y)
                px2 = int(np.max(pts[:, 0]))
                py2 = int(np.max(pts[:, 1]) + lower_half_y)
            else:
                px1 = int(np.min(pts[:, 0]) + bx1)
                py1 = int(np.min(pts[:, 1]) + by1)
                px2 = int(np.max(pts[:, 0]) + bx1)
                py2 = int(np.max(pts[:, 1]) + by1)
        else:
            if is_candidate2:
                px1, py1, px2, py2 = 0, lower_half_y, vw, vh
            else:
                px1, py1, px2, py2 = bx1, by1, bx2, by2

        # Discard if bounding box spans more than 75% of vehicle width
        plate_width = px2 - px1
        if plate_width > 0.75 * vw:
            return ('', 0.0, None)

        # Clip values to ensure they lie within vehicle_crop dimensions
        px1 = max(0, min(vw - 1, px1))
        py1 = max(0, min(vh - 1, py1))
        px2 = max(0, min(vw - 1, px2))
        py2 = max(0, min(vh - 1, py2))

        return (chosen_text, chosen_conf, (px1, py1, px2, py2))

    def process_frame(self, frame):
        """
        Runs multi-vehicle tracking, crops bounding boxes, crops plates, preprocesses,
        and extracts plate text.
        Returns a list of structured dictionaries for each vehicle.
        """
        # Run vehicle detection/tracking (classes: 2=car, 5=bus, 7=truck)
        results = self.detector.track(frame, persist=True, classes=[2, 5, 7], verbose=False)
        output = []
        
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                
                # Retrieve tracking IDs if available, else fallback to indices
                if r.boxes.id is not None:
                    track_ids = r.boxes.id.int().cpu().numpy()
                else:
                    track_ids = list(range(len(boxes)))
                    
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure bbox bounds are within frame limits
                    h_frame, w_frame = frame.shape[:2]
                    x1 = max(0, min(w_frame, x1))
                    y1 = max(0, min(h_frame, y1))
                    x2 = max(0, min(w_frame, x2))
                    y2 = max(0, min(h_frame, y2))
                    
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size == 0:
                        continue
                        
                    # 1. License Plate localization and warping
                    plate_crop = self.localize_plate(vehicle_crop)
                    if plate_crop is None or plate_crop.size == 0:
                        continue
                        
                    # 2. Image Preprocessing (CLAHE + Sharpening)
                    preprocessed = self.preprocess_plate(plate_crop)
                    
                    # 3. PaddleOCR extraction
                    plate_text, confidence = self.extract_text(preprocessed)
                    
                    if confidence >= self.ocr_confidence_threshold:
                        output.append({
                            "vehicle_id": int(track_id),
                            "plate_text": plate_text,
                            "confidence": confidence
                        })
                        
        return output

    def process_vehicle_crops(self, frame, bounding_boxes):
        """
        Processes license plate recognition for multiple vehicle bounding boxes.
        bounding_boxes: list of dicts/tuples: [{"id": track_id, "bbox": (x1, y1, x2, y2)}, ...]
                        or list of tuples: [(track_id, (x1, y1, x2, y2)), ...]
        """
        output = []
        for item in bounding_boxes:
            track_id = -1
            bbox = None
            
            # Handle list of dicts
            if isinstance(item, dict):
                track_id = item.get("id", -1)
                bbox = item.get("bbox", None)
            # Handle tuple of (track_id, bbox)
            elif isinstance(item, tuple) and len(item) == 2:
                track_id, bbox = item
            else:
                continue
                
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure boundaries are within frame limits
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, min(w_frame, x1))
            y1 = max(0, min(h_frame, y1))
            x2 = max(0, min(w_frame, x2))
            y2 = max(0, min(h_frame, y2))
            
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue
                
            # 1. License Plate localization and warping
            plate_crop = self.localize_plate(vehicle_crop)
            if plate_crop is None or plate_crop.size == 0:
                continue
                
            # 2. Image Preprocessing (CLAHE + Sharpening)
            preprocessed = self.preprocess_plate(plate_crop)
            
            # 3. PaddleOCR extraction
            plate_text, confidence = self.extract_text(preprocessed)
            
            if confidence >= self.ocr_confidence_threshold:
                output.append({
                    "vehicle_id": int(track_id),
                    "plate_text": plate_text,
                    "confidence": confidence
                })
                
        return output
