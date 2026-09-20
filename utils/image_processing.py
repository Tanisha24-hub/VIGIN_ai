import cv2
import numpy as np

# RetinaFace MobileNet Face Detector
try:
    from retinaface import RetinaFace
except Exception:
    RetinaFace = None


def process_license_plate(plate_crop):
    """Super-resolution and contrast enhancement for blurry plates."""
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    ph, pw = plate_crop.shape[:2]
    if pw < 200:
        scale = max(2, int(200 / max(1, pw)))
        plate_crop = cv2.resize(plate_crop, (pw * scale, ph * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    return cv2.bilateralFilter(contrast, 7, 50, 50)


def enhance_cabin_interior(cabin_crop, target_w=320, target_h=180):
    """
    High-Definition super-sampling, shadow lifting, and anti-glare sharpening:
    - High-quality bicubic upscaling to target (320, 180) dimensions
    - Inverse gamma LUT (gamma = 1.6) to lift dark cabin shadows
    - Multi-scale CLAHE (clipLimit=2.8, tileGridSize=(6, 6)) on Luminance channel
    - Bilateral filter denoising (d=5, sigmaColor=35, sigmaSpace=35)
    - Gentle unsharp mask to make driver contours crisp
    """
    if cabin_crop is None or cabin_crop.size == 0:
        return cabin_crop

    # 1. Resize with cv2.INTER_CUBIC to target (320, 180)
    upscaled = cv2.resize(cabin_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 2. Apply inverse gamma LUT (gamma = 1.6) to lift dark cabin shadows
    inv_gamma = 1.0 / 1.6
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corr = cv2.LUT(upscaled, table)

    # 3. Apply CLAHE (clipLimit=2.8, tileGridSize=(6, 6)) on the Luminance channel in LAB color space
    lab = cv2.cvtColor(gamma_corr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(6, 6))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

    # 4. Denoise with cv2.bilateralFilter(d=5, sigmaColor=35, sigmaSpace=35)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=35)

    # 5. Crisp high-pass sharpening to penetrate windshield reflections
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    crisp = cv2.addWeighted(denoised, 1.35, gaussian, -0.35, 0)

    return crisp


def crop_cabin_region(frame, box, class_id=2):
    """
    Extracts windshield glass ONLY when vehicle class is car/SUV (class_id == 2)
    and distance optical resolution viability is satisfied (vw >= 190, vh >= 130).
    Adjusts crop window upward from wiper/hood boundary into true windshield glass.
    """
    if class_id != 2:
        return None

    vx1, vy1, vx2, vy2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    vh = vy2 - vy1
    vw = vx2 - vx1

    # Strict optical resolution gate: windshield inspection requires >= 190w and >= 130h
    if vw < 190 or vh < 130:
        return None

    frame_h, frame_w = frame.shape[:2]

    # Adjusted crop window capturing true windshield glass:
    cy1 = int(max(0, vy1 - int(vh * 0.25)))
    cy2 = int(min(frame_h, vy1 + int(vh * 0.35)))
    cx1 = int(max(0, vx1 + int(vw * 0.12)))
    cx2 = int(min(frame_w, vx2 - int(vw * 0.12)))

    crop_h = cy2 - cy1
    crop_w = cx2 - cx1

    if crop_h < 35 or crop_w < 60:
        return None

    return frame[cy1:cy2, cx1:cx2]


def analyze_cabin(cabin_crop, detector_model=None):
    """
    Analyzes cabin interior with RetinaFace detection and shadow/reflection penetration.
    Returns:
        {"enhanced": enhanced_img, "occupants": count, "driver_present": bool(count > 0)}
    """
    if cabin_crop is None or cabin_crop.size == 0:
        return {"enhanced": None, "occupants": 0, "driver_present": False}

    # 1. Enhance and super-sample cabin interior to (320, 180)
    enhanced = enhance_cabin_interior(cabin_crop, target_w=320, target_h=180)
    target_h, target_w = enhanced.shape[:2]

    occupants = 0
    driver_present = False

    # 2. RetinaFace occupant detection
    if RetinaFace is not None:
        try:
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            faces = RetinaFace.detect_faces(enhanced_rgb)
            if isinstance(faces, dict):
                for _, face_data in faces.items():
                    score = face_data.get('score', 0.0)
                    if score >= 0.30:
                        occupants += 1
                        driver_present = True
                        area = face_data.get('facial_area', [])
                        if len(area) == 4:
                            fx1, fy1, fx2, fy2 = [int(v) for v in area]
                            cv2.rectangle(enhanced, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                            cv2.putText(
                                enhanced,
                                f"Occupant ({score:.2f})",
                                (fx1, max(12, fy1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.40,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )
        except Exception:
            pass

    # 3. Fallback driver zone verification (Indian RHD vehicle oncoming: driver is on the left half of crop)
    if not driver_present:
        driver_roi = enhanced[int(target_h * 0.15):int(target_h * 0.85), 0:int(target_w * 0.52)]
        if driver_roi.size > 0:
            gray_roi = cv2.cvtColor(driver_roi, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            mean_lum = np.mean(gray_roi)

            # Silhouette presence check under tinted glass
            if 35.0 < lap_var < 360.0 and mean_lum < 235:
                occupants = max(1, occupants)
                driver_present = True
                cv2.rectangle(
                    enhanced,
                    (10, int(target_h * 0.15)),
                    (int(target_w * 0.50), int(target_h * 0.85)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    enhanced,
                    "Driver Active",
                    (15, int(target_h * 0.30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

    return {
        "enhanced": enhanced,
        "occupants": occupants,
        "driver_present": bool(occupants > 0 or driver_present),
    }