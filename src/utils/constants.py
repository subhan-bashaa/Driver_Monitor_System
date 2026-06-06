"""
Configuration constants for the Driver Monitoring System.
All thresholds and parameters are centralized here for easy tuning.
"""

# Eye Aspect Ratio (EAR) thresholds
EAR_THRESHOLD = 0.25  # Below this value, eyes are considered closed
EYE_CLOSED_FRAMES = 20  # Number of consecutive frames to trigger drowsiness alert

# Mouth Aspect Ratio (MAR) thresholds
MAR_THRESHOLD = 0.6  # Above this value, mouth is considered open (yawning)
YAWN_FRAMES = 15  # Number of consecutive frames to confirm yawning

# YOLO Configuration
YOLO_CONFIDENCE = 0.5  # Minimum confidence for phone detection
YOLO_MODEL_PATH = "models/yolov8n.pt"

# Cell phone class ID in COCO dataset (used by YOLOv8)
CELL_PHONE_CLASS_ID = 67

# Display Configuration
WINDOW_NAME = "Driver Monitoring System"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

# MediaPipe Face Mesh landmarks indices
# Left eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Right eye landmarks
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks for MAR calculation
MOUTH_LANDMARKS = [61, 291, 0, 17, 269, 405]

# Alarm configuration
ALARM_PATH = "assets/alarm.wav"
ALARM_COOLDOWN = 2.0  # Seconds between alarm plays to avoid spam
