"""
Yawn detection module for Driver Monitoring System.
"""

from .yawn_detector import YawnDetector
from .mouth_aspect_ratio import calculate_mouth_aspect_ratio

__all__ = ['YawnDetector', 'calculate_mouth_aspect_ratio']
