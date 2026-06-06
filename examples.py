"""
Example script demonstrating how to use individual modules.
This is useful for testing and understanding each component separately.
"""

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from src.eye_detection import EyeDetector
from src.yawn_detection import YawnDetector
from src.phone_detection import PhoneDetector
from src.utils import constants


def example_eye_detection():
    """Example: Test eye detection with webcam."""
    print("\n=== Eye Detection Example ===")
    print("This will show real-time eye detection.")
    print("Press 'q' to quit.\n")
    
    import mediapipe as mp
    
    cap = cv2.VideoCapture(0)
    detector = EyeDetector()
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                eye_result = detector.detect_drowsiness(face_landmarks, w, h)
                
                # Display results
                status = "CLOSED" if eye_result['eyes_closed'] else "OPEN"
                color = (0, 0, 255) if eye_result['eyes_closed'] else (0, 255, 0)
                
                cv2.putText(frame, f"Eyes: {status}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"EAR: {eye_result['ear']:.3f}", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Frames: {eye_result['closed_frames']}", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if eye_result['drowsy']:
                    cv2.putText(frame, "DROWSY!", (20, 160),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            cv2.imshow("Eye Detection Example", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def example_yawn_detection():
    """Example: Test yawn detection with webcam."""
    print("\n=== Yawn Detection Example ===")
    print("This will show real-time yawn detection.")
    print("Press 'q' to quit.\n")
    
    import mediapipe as mp
    
    cap = cv2.VideoCapture(0)
    detector = YawnDetector()
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                yawn_result = detector.detect_yawn(face_landmarks, w, h)
                
                # Display results
                status = "YAWNING" if yawn_result['yawning'] else "NORMAL"
                color = (0, 255, 255) if yawn_result['yawning'] else (0, 255, 0)
                
                cv2.putText(frame, f"Mouth: {status}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"MAR: {yawn_result['mar']:.3f}", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Frames: {yawn_result['yawn_frames']}", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Yawn Detection Example", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def example_phone_detection():
    """Example: Test phone detection with webcam."""
    print("\n=== Phone Detection Example ===")
    print("This will show real-time phone detection using YOLO.")
    print("Hold a phone in front of the camera to test.")
    print("Press 'q' to quit.\n")
    
    cap = cv2.VideoCapture(0)
    detector = PhoneDetector()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect phone
            result = detector.detect_phone(frame)
            
            # Draw detection
            frame = detector.draw_detection(frame)
            
            # Display status
            status = "DETECTED" if result['phone_detected'] else "NOT DETECTED"
            color = (0, 0, 255) if result['phone_detected'] else (0, 255, 0)
            
            cv2.putText(frame, f"Phone: {status}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if result['phone_detected']:
                cv2.putText(frame, f"Confidence: {result['confidence']:.2f}", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Phone Detection Example", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def example_test_constants():
    """Example: Display all constants."""
    print("\n=== System Constants ===")
    print(f"EAR Threshold: {constants.EAR_THRESHOLD}")
    print(f"Eye Closed Frames: {constants.EYE_CLOSED_FRAMES}")
    print(f"MAR Threshold: {constants.MAR_THRESHOLD}")
    print(f"Yawn Frames: {constants.YAWN_FRAMES}")
    print(f"YOLO Confidence: {constants.YOLO_CONFIDENCE}")
    print(f"Cell Phone Class ID: {constants.CELL_PHONE_CLASS_ID}")
    print()


def main():
    """Main menu for examples."""
    print("\n" + "=" * 60)
    print("  DRIVER MONITORING AI - MODULE EXAMPLES")
    print("=" * 60)
    
    while True:
        print("\nSelect an example to run:")
        print("  1. Eye Detection (Drowsiness)")
        print("  2. Yawn Detection")
        print("  3. Phone Detection")
        print("  4. Show Constants")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            example_eye_detection()
        elif choice == '2':
            example_yawn_detection()
        elif choice == '3':
            example_phone_detection()
        elif choice == '4':
            example_test_constants()
        elif choice == '5':
            print("\nExiting examples. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
