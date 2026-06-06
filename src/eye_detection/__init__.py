"""
Eye detection module for Driver Monitoring System.
"""

from .eye_detector import EyeDetector
from .eye_aspect_ratio import calculate_eye_aspect_ratio, calculate_average_ear

__all__ = ['EyeDetector', 'calculate_eye_aspect_ratio', 'calculate_average_ear']
