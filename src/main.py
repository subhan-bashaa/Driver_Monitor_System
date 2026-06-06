"""
Main application for the Driver Monitoring System.
Real-time detection of drowsiness, yawning, and phone usage.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eye_detection import EyeDetector
from src.yawn_detection import YawnDetector
from src.phone_detection import PhoneDetector
from src.utils.alarm import play_alarm
from src.utils.constants import (
    WINDOW_NAME,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    COLOR_WHITE
)


class DriverMonitoringSystem:
    """
    Main class orchestrating the driver monitoring system.
    """
    
    def __init__(self):
        """Initialize all components of the monitoring system."""
        print("=" * 60)
        print("🚗 DRIVER MONITORING SYSTEM - INITIALIZING")
        print("=" * 60)
        
        # Initialize detectors
        self.eye_detector = EyeDetector()
        self.yawn_detector = YawnDetector()
        self.phone_detector = PhoneDetector()
        
        # Initialize MediaPipe Face Landmarker (new API)
        base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Error: Could not open webcam")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # System state
        self.is_running = False
        self.frame_count = 0
        
        print("✓ Eye Detector initialized")
        print("✓ Yawn Detector initialized")
        print("✓ Phone Detector initialized")
        print("✓ MediaPipe Face Mesh initialized")
        print("✓ Webcam initialized")
        print("=" * 60)
        print("📹 Press 'q' to quit")
        print("=" * 60)
    
    def process_frame(self, frame):
        """
        Process a single frame for all detections.
        
        Args:
            frame (numpy.ndarray): Input frame from webcam
            
        Returns:
            tuple: (processed_frame, detection_results)
        """
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Initialize detection results
        results = {
            'eyes': None,
            'yawn': None,
            'phone': None,
            'face_detected': False
        }
        
        # Detect phone using YOLO
        phone_result = self.phone_detector.detect_phone(frame)
        results['phone'] = phone_result
        
        # Draw phone detection
        frame = self.phone_detector.draw_detection(frame)
        
        # Process face landmarks with MediaPipe
        face_results = self.face_landmarker.detect(mp_image)
        
        if face_results.face_landmarks:
            results['face_detected'] = True
            face_landmarks = face_results.face_landmarks[0]
            
            # Detect eye closure and drowsiness
            eye_result = self.eye_detector.detect_drowsiness(
                face_landmarks, frame_width, frame_height
            )
            results['eyes'] = eye_result
            
            # Detect yawning
            yawn_result = self.yawn_detector.detect_yawn(
                face_landmarks, frame_width, frame_height
            )
            results['yawn'] = yawn_result
        
        return frame, results
    
    def draw_status(self, frame, results):
        """
        Draw status information on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            results (dict): Detection results
            
        Returns:
            numpy.ndarray: Frame with status overlay
        """
        # Panel dimensions
        panel_height = 200
        panel_width = frame.shape[1]
        
        # Create semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        # Status text position
        y_offset = 40
        line_height = 35
        
        # Title
        cv2.putText(frame, "DRIVER MONITORING SYSTEM", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)
        y_offset += line_height + 10
        
        # Face detection status
        if results['face_detected']:
            # Eye status
            if results['eyes']:
                eye_status = "CLOSED" if results['eyes']['eyes_closed'] else "OPEN"
                eye_color = COLOR_RED if results['eyes']['eyes_closed'] else COLOR_GREEN
                ear_value = results['eyes']['ear']
                
                cv2.putText(frame, f"Eyes: {eye_status} (EAR: {ear_value:.2f})", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
                y_offset += line_height
            
            # Yawn status
            if results['yawn']:
                yawn_status = "YES" if results['yawn']['yawning'] else "NO"
                yawn_color = COLOR_YELLOW if results['yawn']['yawning'] else COLOR_GREEN
                mar_value = results['yawn']['mar']
                
                cv2.putText(frame, f"Yawning: {yawn_status} (MAR: {mar_value:.2f})", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2)
                y_offset += line_height
        else:
            cv2.putText(frame, "No Face Detected", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)
            y_offset += line_height * 2
        
        # Phone detection status
        if results['phone']:
            phone_status = "DETECTED" if results['phone']['phone_detected'] else "NOT DETECTED"
            phone_color = COLOR_RED if results['phone']['phone_detected'] else COLOR_GREEN
            
            cv2.putText(frame, f"Phone: {phone_status}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phone_color, 2)
            y_offset += line_height
        
        # Overall status
        status = self.determine_overall_status(results)
        status_color = self.get_status_color(status)
        
        cv2.putText(frame, f"Status: {status}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # FPS counter
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (panel_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        
        return frame
    
    def determine_overall_status(self, results):
        """
        Determine overall driver status.
        
        Args:
            results (dict): Detection results
            
        Returns:
            str: Overall status
        """
        if results['phone'] and results['phone']['phone_detected']:
            return "DISTRACTED"
        
        if results['eyes'] and results['eyes']['drowsy']:
            return "DROWSY"
        
        if results['yawn'] and results['yawn']['yawning']:
            return "FATIGUED"
        
        return "SAFE"
    
    def get_status_color(self, status):
        """
        Get color for status text.
        
        Args:
            status (str): Status text
            
        Returns:
            tuple: BGR color
        """
        if status == "SAFE":
            return COLOR_GREEN
        elif status == "FATIGUED":
            return COLOR_YELLOW
        else:  # DROWSY or DISTRACTED
            return COLOR_RED
    
    def trigger_alerts(self, results):
        """
        Trigger appropriate alerts based on detection results.
        
        Args:
            results (dict): Detection results
        """
        alert_triggered = False
        
        # Alert for drowsiness
        if results['eyes'] and results['eyes']['drowsy']:
            alert_triggered = True
        
        # Alert for yawning
        if results['yawn'] and results['yawn']['yawning']:
            alert_triggered = True
        
        # Alert for phone detection
        if results['phone'] and results['phone']['phone_detected']:
            alert_triggered = True
        
        # Play alarm if any alert is triggered
        if alert_triggered:
            play_alarm()
    
    def run(self):
        """Main loop for the driver monitoring system."""
        self.is_running = True
        
        try:
            while self.is_running:
                # Read frame from camera
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Error: Failed to read frame from webcam")
                    break
                
                self.frame_count += 1
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Draw status overlay
                output_frame = self.draw_status(processed_frame, results)
                
                # Trigger alerts if necessary
                self.trigger_alerts(results)
                
                # Display the frame
                cv2.imshow(WINDOW_NAME, output_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n📴 Shutting down...")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up resources...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Driver Monitoring System terminated successfully")


def main():
    """Main entry point."""
    try:
        # Create and run the monitoring system
        dms = DriverMonitoringSystem()
        dms.run()
    
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
