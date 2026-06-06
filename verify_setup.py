"""
System verification script for Driver Monitoring AI.
Checks if all dependencies and files are properly installed.
"""

import sys
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python version must be 3.8 or higher")
        return False


def check_dependencies():
    """Check if all required packages are installed."""
    print_header("Checking Dependencies")
    
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'ultralytics': 'ultralytics',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'playsound': 'playsound'
    }
    
    all_installed = True
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
    
    return all_installed


def check_directory_structure():
    """Check if all required directories exist."""
    print_header("Checking Directory Structure")
    
    required_dirs = [
        'src',
        'src/eye_detection',
        'src/yawn_detection',
        'src/phone_detection',
        'src/utils',
        'models',
        'assets'
    ]
    
    all_exist = True
    
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"✓ {directory}/")
        else:
            print(f"✗ {directory}/ - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_source_files():
    """Check if all required source files exist."""
    print_header("Checking Source Files")
    
    required_files = [
        'src/main.py',
        'src/eye_detection/eye_detector.py',
        'src/eye_detection/eye_aspect_ratio.py',
        'src/yawn_detection/yawn_detector.py',
        'src/yawn_detection/mouth_aspect_ratio.py',
        'src/phone_detection/phone_detector.py',
        'src/utils/alarm.py',
        'src/utils/constants.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    
    for file in required_files:
        if os.path.isfile(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_model_file():
    """Check if YOLO model exists."""
    print_header("Checking YOLO Model")
    
    model_path = 'models/yolov8n.pt'
    
    if os.path.isfile(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ {model_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"✗ {model_path} - NOT FOUND")
        print("\nThe model will be downloaded automatically on first run.")
        print("Or download manually with:")
        print("  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
        return False


def check_alarm_file():
    """Check if alarm sound exists."""
    print_header("Checking Alarm Sound")
    
    alarm_path = 'assets/alarm.wav'
    
    if os.path.isfile(alarm_path):
        size_kb = os.path.getsize(alarm_path) / 1024
        print(f"✓ {alarm_path} ({size_kb:.2f} KB)")
        return True
    else:
        print(f"✗ {alarm_path} - NOT FOUND")
        print("\nGenerate alarm sound with:")
        print("  python generate_alarm.py")
        return False


def test_camera():
    """Test if camera is accessible."""
    print_header("Testing Camera Access")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("✓ Camera is accessible and working")
                print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                return True
            else:
                print("✗ Camera opened but cannot read frames")
                return False
        else:
            print("✗ Cannot open camera")
            print("  Make sure no other application is using the camera")
            return False
    
    except Exception as e:
        print(f"✗ Error testing camera: {e}")
        return False


def test_mediapipe():
    """Test MediaPipe initialization."""
    print_header("Testing MediaPipe Face Mesh")
    
    try:
        import mediapipe as mp
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✓ MediaPipe Face Mesh initialized successfully")
        return True
    
    except Exception as e:
        print(f"✗ Error initializing MediaPipe: {e}")
        return False


def test_yolo():
    """Test YOLO model loading."""
    print_header("Testing YOLO Model Loading")
    
    try:
        from ultralytics import YOLO
        
        if os.path.isfile('models/yolov8n.pt'):
            model = YOLO('models/yolov8n.pt')
            print("✓ YOLO model loaded successfully")
            return True
        else:
            print("⚠ YOLO model not found (will download on first run)")
            return True  # Not a critical error
    
    except Exception as e:
        print(f"✗ Error loading YOLO: {e}")
        return False


def print_summary(results):
    """Print verification summary."""
    print_header("Verification Summary")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready to run.")
        print("\nStart the system with:")
        print("  python src/main.py")
        print("\nOr on Windows:")
        print("  run.bat")
    else:
        print("\n⚠️  Some checks failed. Please resolve the issues above.")
        print("\nRefer to SETUP.md for detailed installation instructions.")
    
    print("\n" + "=" * 60)


def main():
    """Main verification routine."""
    print("\n" + "=" * 60)
    print("  DRIVER MONITORING AI - SYSTEM VERIFICATION")
    print("=" * 60)
    
    results = {}
    
    # Run all checks
    results['Python Version'] = check_python_version()
    results['Dependencies'] = check_dependencies()
    results['Directory Structure'] = check_directory_structure()
    results['Source Files'] = check_source_files()
    results['YOLO Model'] = check_model_file()
    results['Alarm Sound'] = check_alarm_file()
    results['Camera Access'] = test_camera()
    results['MediaPipe'] = test_mediapipe()
    results['YOLO Loading'] = test_yolo()
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
