"""
Yawn detection module using Mouth Aspect Ratio (MAR).
"""

import numpy as np
from src.yawn_detection.mouth_aspect_ratio import calculate_mouth_aspect_ratio
from src.utils.constants import (
    MAR_THRESHOLD,
    YAWN_FRAMES,
    MOUTH_LANDMARKS
)


class YawnDetector:
    """
    Detects yawning based on mouth aspect ratio.
    """
    
    def __init__(self):
        """Initialize the yawn detector with frame counter."""
        self.yawn_frames_counter = 0
        self.is_yawning = False
        self.current_mar = 0.0
        self.mouth_open = False
    
    def extract_mouth_landmarks(self, face_landmarks, frame_width, frame_height):
        """
        Extract mouth landmark coordinates from MediaPipe face mesh.
        
        Args:
            face_landmarks: MediaPipe face landmarks list (new API format)
            frame_width (int): Width of the video frame
            frame_height (int): Height of the video frame
            
        Returns:
            list: List of mouth landmark coordinates
        """
        mouth_coords = []
        for idx in MOUTH_LANDMARKS:
            landmark = face_landmarks[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            mouth_coords.append((x, y))
        
        return mouth_coords
    
    def detect_yawn(self, face_landmarks, frame_width, frame_height):
        """
        Detect yawning based on mouth aspect ratio.
        
        Args:
            face_landmarks: MediaPipe face landmarks list (new API format)
            frame_width (int): Width of the video frame
            frame_height (int): Height of the video frame
            
        Returns:
            dict: Detection results with keys:
                - 'mar': Current mouth aspect ratio
                - 'mouth_open': Boolean indicating if mouth is open
                - 'yawning': Boolean indicating if driver is yawning
                - 'yawn_frames': Number of consecutive yawn frames
        """
        # Extract mouth landmarks
        mouth_coords = self.extract_mouth_landmarks(
            face_landmarks, frame_width, frame_height
        )
        
        # Calculate mouth aspect ratio
        mar = calculate_mouth_aspect_ratio(mouth_coords)
        self.current_mar = mar
        
        # Check if mouth is open (potential yawn)
        if mar > MAR_THRESHOLD:
            self.yawn_frames_counter += 1
            self.mouth_open = True
            
            # Check if yawn threshold is reached
            if self.yawn_frames_counter >= YAWN_FRAMES:
                self.is_yawning = True
            else:
                self.is_yawning = False
        else:
            # Mouth is closed, reset counter
            self.yawn_frames_counter = 0
            self.mouth_open = False
            self.is_yawning = False
        
        return {
            'mar': self.current_mar,
            'mouth_open': self.mouth_open,
            'yawning': self.is_yawning,
            'yawn_frames': self.yawn_frames_counter
        }
    
    def reset(self):
        """Reset the detector state."""
        self.yawn_frames_counter = 0
        self.is_yawning = False
        self.current_mar = 0.0
        self.mouth_open = False
