"""
Eye detection and drowsiness monitoring module.
"""

import numpy as np
from src.eye_detection.eye_aspect_ratio import calculate_average_ear
from src.utils.constants import (
    EAR_THRESHOLD, 
    EYE_CLOSED_FRAMES,
    LEFT_EYE,
    RIGHT_EYE
)


class EyeDetector:
    """
    Detects eye closure and monitors drowsiness based on consecutive closed frames.
    """
    
    def __init__(self):
        """Initialize the eye detector with frame counter."""
        self.closed_frames_counter = 0
        self.is_drowsy = False
        self.current_ear = 0.0
        self.eyes_closed = False
    
    def extract_eye_landmarks(self, face_landmarks, frame_width, frame_height):
        """
        Extract eye landmark coordinates from MediaPipe face mesh.
        
        Args:
            face_landmarks: MediaPipe face landmarks list (new API format)
            frame_width (int): Width of the video frame
            frame_height (int): Height of the video frame
            
        Returns:
            tuple: (left_eye_coords, right_eye_coords)
        """
        # Extract left eye landmarks
        left_eye_coords = []
        for idx in LEFT_EYE:
            landmark = face_landmarks[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            left_eye_coords.append((x, y))
        
        # Extract right eye landmarks
        right_eye_coords = []
        for idx in RIGHT_EYE:
            landmark = face_landmarks[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            right_eye_coords.append((x, y))
        
        return left_eye_coords, right_eye_coords
    
    def detect_drowsiness(self, face_landmarks, frame_width, frame_height):
        """
        Detect drowsiness based on eye aspect ratio.
        
        Args:
            face_landmarks: MediaPipe face landmarks list (new API format)
            frame_width (int): Width of the video frame
            frame_height (int): Height of the video frame
            
        Returns:
            dict: Detection results with keys:
                - 'ear': Current eye aspect ratio
                - 'eyes_closed': Boolean indicating if eyes are closed
                - 'drowsy': Boolean indicating if driver is drowsy
                - 'closed_frames': Number of consecutive closed frames
        """
        # Extract eye landmarks
        left_eye, right_eye = self.extract_eye_landmarks(
            face_landmarks, frame_width, frame_height
        )
        
        # Calculate eye aspect ratio
        ear = calculate_average_ear(left_eye, right_eye)
        self.current_ear = ear
        
        # Check if eyes are closed
        if ear < EAR_THRESHOLD:
            self.closed_frames_counter += 1
            self.eyes_closed = True
            
            # Check if drowsiness threshold is reached
            if self.closed_frames_counter >= EYE_CLOSED_FRAMES:
                self.is_drowsy = True
            else:
                self.is_drowsy = False
        else:
            # Eyes are open, reset counter
            self.closed_frames_counter = 0
            self.eyes_closed = False
            self.is_drowsy = False
        
        return {
            'ear': self.current_ear,
            'eyes_closed': self.eyes_closed,
            'drowsy': self.is_drowsy,
            'closed_frames': self.closed_frames_counter
        }
    
    def reset(self):
        """Reset the detector state."""
        self.closed_frames_counter = 0
        self.is_drowsy = False
        self.current_ear = 0.0
        self.eyes_closed = False
