"""
Mouth Aspect Ratio (MAR) calculation for yawn detection.
"""

import numpy as np
from scipy.spatial import distance


def calculate_mouth_aspect_ratio(mouth_landmarks):
    """
    Calculate the Mouth Aspect Ratio (MAR) for yawn detection.
    
    MAR = ||p2 - p6|| / ||p1 - p4||
    
    A higher MAR indicates a more open mouth (yawning).
    
    Args:
        mouth_landmarks (list): List of 6 (x, y) coordinates for mouth landmarks
                               [left, right, top1, bottom1, top2, bottom2]
        
    Returns:
        float: Mouth Aspect Ratio value
    """
    # Calculate vertical distances (mouth height at different points)
    vertical_dist_1 = distance.euclidean(mouth_landmarks[2], mouth_landmarks[3])
    vertical_dist_2 = distance.euclidean(mouth_landmarks[4], mouth_landmarks[5])
    
    # Calculate horizontal distance (mouth width)
    horizontal_dist = distance.euclidean(mouth_landmarks[0], mouth_landmarks[1])
    
    # Avoid division by zero
    if horizontal_dist == 0:
        return 0.0
    
    # Calculate the mouth aspect ratio
    mar = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
    
    return mar
