"""
Phone detection module using YOLOv8.
"""

from ultralytics import YOLO
import cv2
from src.utils.constants import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE,
    CELL_PHONE_CLASS_ID
)


class PhoneDetector:
    """
    Detects cell phone usage using YOLOv8 object detection.
    """
    
    def __init__(self):
        """Initialize the YOLO model for phone detection."""
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.phone_detected = False
            self.detection_confidence = 0.0
            self.phone_bbox = None
            print(f"✓ YOLOv8 model loaded successfully from {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            print(f"  Please ensure yolov8n.pt is in the {YOLO_MODEL_PATH} directory")
            raise
    
    def detect_phone(self, frame):
        """
        Detect cell phone in the given frame.
        
        Args:
            frame (numpy.ndarray): Input frame from webcam
            
        Returns:
            dict: Detection results with keys:
                - 'phone_detected': Boolean indicating if phone is detected
                - 'confidence': Detection confidence score
                - 'bbox': Bounding box coordinates (x1, y1, x2, y2) or None
        """
        # Run YOLO inference
        results = self.model(frame, conf=YOLO_CONFIDENCE, verbose=False)
        
        # Reset detection state
        self.phone_detected = False
        self.detection_confidence = 0.0
        self.phone_bbox = None
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get class ID
                class_id = int(box.cls[0])
                
                # Check if detected object is a cell phone
                if class_id == CELL_PHONE_CLASS_ID:
                    self.phone_detected = True
                    self.detection_confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    self.phone_bbox = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Break after first phone detection (can be modified for multiple phones)
                    break
            
            if self.phone_detected:
                break
        
        return {
            'phone_detected': self.phone_detected,
            'confidence': self.detection_confidence,
            'bbox': self.phone_bbox
        }
    
    def draw_detection(self, frame):
        """
        Draw bounding box on frame if phone is detected.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Frame with bounding box drawn
        """
        if self.phone_detected and self.phone_bbox:
            x1, y1, x2, y2 = self.phone_bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw label
            label = f"Phone: {self.detection_confidence:.2f}"
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        return frame
    
    def reset(self):
        """Reset the detector state."""
        self.phone_detected = False
        self.detection_confidence = 0.0
        self.phone_bbox = None
