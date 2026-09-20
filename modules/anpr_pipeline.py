import cv2
import numpy as np
import re
import os
import glob
from collections import defaultdict, Counter
from ultralytics import YOLO

# Standard Indian / International vehicle plate regex:
# 2 alpha (State) + 2 digits (RTO) + optional 1-2 alpha (Series) + 4 digits (Registration number)
PLATE_STANDARD_REGEX = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')
PLATE_GENERAL_REGEX = re.compile(r'^[A-Z0-9]{6,11}$')


def find_plate_detector_model():
    """Locates cached license plate detection weights or returns default."""
    cache_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Koushim--yolov8-license-plate-detection/snapshots/*/best.pt"
    )
    matches = glob.glob(cache_pattern)
    if matches and os.path.exists(matches[0]):
        return matches[0]
    return 'yolov8n.pt'


def preprocess_plate_crop(plate_crop):
    """
    Cleans up plate crop using bilateral filtering and adaptive thresholding
    to enhance character edges and suppress road grime and reflections.
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    ph, pw = plate_crop.shape[:2]
    # Resize small crops to boost OCR character resolution
    if pw < 180 or ph < 45:
        scale = max(2, int(200 / max(1, pw)))
        plate_crop = cv2.resize(plate_crop, (pw * scale, ph * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop

    # Bilateral filter to smooth noise while preserving edge boundaries
    denoised = cv2.bilateralFilter(gray, 9, 65, 65)

    # Adaptive threshold to maximize contrast between characters and background
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Convert back to 3-channel for OCR engine compatibility
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


class ANPRPipeline:
    """
    Adaptive ANPR Pipeline with inference throttling and temporal rolling smoothing:
    - Stores structured track history per vehicle.
    - Locks onto high-confidence, structural majority-voted plates.
    - Throttles heavy inference to run only every N frames per track ID.
    - Uses length-weighted majority voting to favor complete reads over partial fragments.
    """

    def __init__(self, plate_detector=None, ocr_engine=None):
        if plate_detector is not None:
            self.plate_detector = plate_detector
        else:
            model_path = find_plate_detector_model()
            self.plate_detector = YOLO(model_path)

        if ocr_engine is not None:
            self.ocr_engine = ocr_engine
            try:
                from modules.plate_ocr import PlateRecognizer
            except ImportError:
                from plate_ocr import PlateRecognizer
            self.ocr_engine = PlateRecognizer()

        # Structured tracking history: track_id -> dict
        self.track_history = defaultdict(lambda: {
            "plates": [],                 # list of (plate_str, conf)
            "last_inferred_frame": -999,
            "locked_plate": None,
            "last_box": None,             # (px1, py1, px2, py2) relative to vehicle
            "last_conf": 0.0
        })

    def _score_candidate(self, candidate_str, conf=0.5):
        """
        Length-weighted scoring function:
        - Awards high bonus for standard structural pattern (2 alpha + 2 digit + 1-2 alpha + 4 digit).
        - Awards length bonus favoring 9-10 character complete plates over partials.
        """
        clean = re.sub(r'[^A-Z0-9]', '', candidate_str.upper())
        length = len(clean)

        if length < 4:
            return 0.0

        score = 1.0 + float(conf)

        # Standard plate regex match bonus
        if PLATE_STANDARD_REGEX.match(clean):
            score += 15.0
        elif PLATE_GENERAL_REGEX.match(clean):
            score += 4.0

        # Length weighting: 9-10 chars is standard full Indian plate
        if length in (9, 10):
            score += 8.0
        elif length in (7, 8):
            score += 3.0
        elif length > 10:
            score += 1.0

        return score

    def compute_best_plate(self, track_id):
        """
        Computes the best consensus plate using length-weighted majority voting.
        Locks the plate if the leading candidate appears >= 3 times with standard structure.
        """
        history = self.track_history[track_id]
        if history["locked_plate"]:
            return history["locked_plate"], True

        records = history["plates"]
        if not records:
            return "UNKNOWN", False

        # Aggregate weighted scores and frequencies
        candidate_scores = defaultdict(float)
        candidate_counts = Counter()
        candidate_confs = defaultdict(list)

        for text, conf in records:
            s = self._score_candidate(text, conf)
            candidate_scores[text] += s
            candidate_counts[text] += 1
            candidate_confs[text].append(conf)

        if not candidate_scores:
            return "UNKNOWN", False

        # Sort by aggregate score descending
        best_candidate = max(candidate_scores, key=lambda c: (candidate_scores[c], candidate_counts[c]))
        best_count = candidate_counts[best_candidate]
        avg_conf = (
            sum(candidate_confs[best_candidate]) / len(candidate_confs[best_candidate])
            if candidate_confs[best_candidate] else 0.0
        )
        history["last_conf"] = avg_conf

        # Locking check: appears >= 3 times with consistent structure
        is_standard_struct = bool(PLATE_STANDARD_REGEX.match(best_candidate))
        if best_count >= 3 and (is_standard_struct or (len(best_candidate) >= 8 and best_count >= 4)):
            history["locked_plate"] = best_candidate
            print(f"🎯 [PLATE LOCKED] Vehicle #{track_id} -> '{best_candidate}' (Count: {best_count}, Conf: {avg_conf:.2f})")
            return best_candidate, True

        return best_candidate, False

    def process_vehicle(self, frame, track_id, bbox, frame_idx):
        """
        Processes vehicle license plate with adaptive inference throttling:
        1. If locked_plate exists -> returns immediately without OCR.
        2. If (frame_idx - last_inferred_frame) < 5 -> throttles, returns current consensus.
        3. Else -> runs YOLO plate detector + bilateral/adaptive threshold + PaddleOCR.
        Returns:
            (best_plate: str, is_locked: bool)
        """
        history = self.track_history[track_id]

        # 1. Immediate return if plate is already locked
        if history["locked_plate"] is not None:
            return history["locked_plate"], True

        # 2. Inference Throttling (Every 5 frames per track ID)
        if (frame_idx - history["last_inferred_frame"]) < 5:
            consensus, is_locked = self.compute_best_plate(track_id)
            return consensus, is_locked

        # 3. Crop vehicle region from frame
        fh, fw = frame.shape[:2]
        vx1, vy1, vx2, vy2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(fw, vx2), min(fh, vy2)

        car_crop = frame[vy1:vy2, vx1:vx2]
        if car_crop.size == 0 or (vx2 - vx1) < 40 or (vy2 - vy1) < 30:
            consensus, is_locked = self.compute_best_plate(track_id)
            return consensus, is_locked

        # Update last inferred frame timestamp
        history["last_inferred_frame"] = frame_idx

        # 4. Secondary YOLO Plate Detection
        plate_box = None
        best_plate_crop = None

        try:
            plate_res = self.plate_detector(car_crop, conf=0.15, verbose=False)
            if plate_res and len(plate_res) > 0 and plate_res[0].boxes is not None and len(plate_res[0].boxes) > 0:
                # Find highest confidence box
                boxes = plate_res[0].boxes.xyxy.cpu().numpy().astype(int)
                confs = plate_res[0].boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                px1, py1, px2, py2 = boxes[best_idx]

                crop_h, crop_w = car_crop.shape[:2]
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(crop_w, px2), min(crop_h, py2)

                if (px2 - px1) > 15 and (py2 - py1) > 8:
                    plate_box = (px1, py1, px2, py2)
                    best_plate_crop = car_crop[py1:py2, px1:px2]
                    history["last_box"] = plate_box
        except Exception:
            pass

        # 5. Preprocessing & OCR
        if best_plate_crop is not None and best_plate_crop.size > 0:
            preprocessed_crop = preprocess_plate_crop(best_plate_crop)

            # Extract text via PaddleOCR
            text, conf = self.ocr_engine.extract_text(preprocessed_crop)
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

            # Fallback to original crop if thresholding yielded nothing
            if len(clean_text) < 4:
                text_raw, conf_raw = self.ocr_engine.extract_text(best_plate_crop)
                clean_raw = re.sub(r'[^A-Z0-9]', '', text_raw.upper())
                if len(clean_raw) > len(clean_text):
                    clean_text, conf = clean_raw, conf_raw

            if len(clean_text) >= 4:
                history["plates"].append((clean_text, float(conf)))
                # Keep maximum 10 recent observations
                if len(history["plates"]) > 10:
                    history["plates"].pop(0)

        # 6. Length-Weighted Majority Voting & Lock Verification
        best_plate, is_locked = self.compute_best_plate(track_id)
        return best_plate, is_locked

    def get_plate_box(self, track_id):
        """Returns the most recent relative plate bounding box for track_id."""
        return self.track_history[track_id].get("last_box")

    def get_plate_conf(self, track_id):
        """Returns the consensus confidence for track_id."""
        return self.track_history[track_id].get("last_conf", 0.0)

    def is_locked(self, track_id):
        """Returns True if plate for track_id has been locked."""
        return self.track_history[track_id]["locked_plate"] is not None

    def get_consensus_plate(self, track_id):
        """Returns current consensus plate string."""
        plate, _ = self.compute_best_plate(track_id)
        return plate
