"""
Eye Aspect Ratio (EAR) calculation.
Based on the paper "Real-Time Eye Blink Detection using Facial Landmarks" by Soukupová and Čech.
"""

import numpy as np
from scipy.spatial import distance


def calculate_eye_aspect_ratio(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) for a single eye.
    
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Where:
    - p1, p4 are the horizontal eye landmarks
    - p2, p3, p5, p6 are the vertical eye landmarks
    
    Args:
        eye_landmarks (list): List of 6 (x, y) coordinates for eye landmarks
        
    Returns:
        float: Eye Aspect Ratio value
    """
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    vertical_dist_1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    vertical_dist_2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    horizontal_dist = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    # Calculate the eye aspect ratio
    ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
    
    return ear


def calculate_average_ear(left_eye_landmarks, right_eye_landmarks):
    """
    Calculate the average Eye Aspect Ratio for both eyes.
    
    Args:
        left_eye_landmarks (list): List of 6 (x, y) coordinates for left eye
        right_eye_landmarks (list): List of 6 (x, y) coordinates for right eye
        
    Returns:
        float: Average EAR value for both eyes
    """
    # Calculate EAR for each eye
    left_ear = calculate_eye_aspect_ratio(left_eye_landmarks)
    right_ear = calculate_eye_aspect_ratio(right_eye_landmarks)
    
    # Return the average
    return (left_ear + right_ear) / 2.0
