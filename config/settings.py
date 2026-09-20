import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_INPUT = os.path.join(BASE_DIR, "data", "input")
DATA_OUTPUT = os.path.join(BASE_DIR, "data", "output")

YOLO_MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.35
VELOCITY_DROP_THRESHOLD = 120.0

# License Plate Detection Model
PLATE_MODEL_NAME = "Koushim/yolov8-license-plate-detection"

# Scanning skips (number of frames)
FAST_SCAN_SKIP = 30
ACTIVE_SCAN_SKIP = 1

# Blurry plate threshold (Laplacian variance)
BLUR_THRESHOLD = 80.0

# Anomaly detection thresholds
ANOMALY_SPEED_THRESHOLD = 300.0   # Speed threshold in pixels per second
ANOMALY_DECEL_THRESHOLD = 500.0   # Deceleration threshold for sudden crash/stop
ANOMALY_WEAVE_THRESHOLD = 1.5     # Erratic lateral deviation / weaving threshold

# Metric Speed & Homography settings
SPEED_LIMIT_KMH = 60.0            # Real-world speed limit in km/h
ROAD_WIDTH_METERS = 7.0           # Standard 2-lane road width
ROAD_LENGTH_METERS = 20.0         # Depth of monitored road segment in meters
HOMOGRAPHY_SRC_POINTS = None      # None uses auto-calibrated trapezoid; or specify 4 [x,y] points

# Multi-Zone Piecewise Homography configuration (e.g. main flat road vs flyover ramps/inclines)
# If None, SpeedEstimator falls back to HOMOGRAPHY_SRC_POINTS or default auto trapezoid
HOMOGRAPHY_ZONES = None

# Environment Enhancer settings ('auto', 'night', 'rain', 'clear')
ENVIRONMENT_MODE = "auto"
ENVIRONMENT_LUM_THRESHOLD = 75.0
ENVIRONMENT_CONTRAST_THRESHOLD = 38.0
ENVIRONMENT_HAZE_THRESHOLD = 0.28


