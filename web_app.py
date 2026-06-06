"""
Web-based Driver Monitoring System
Access from browser: http://localhost:5000
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os

# Add src directory to path
sys.path.insert(0, 'src')

from src.eye_detection import EyeDetector
from src.yawn_detection import YawnDetector
from src.phone_detection import PhoneDetector
from src.utils.alarm import play_alarm

app = Flask(__name__)

# Global variables
camera = None
face_landmarker = None
eye_detector = None
yawn_detector = None
phone_detector = None
current_status = {
    'eyes': 'UNKNOWN',
    'yawning': 'UNKNOWN',
    'phone': 'NOT DETECTED',
    'status': 'INITIALIZING',
    'ear': 0.0,
    'mar': 0.0
}


def initialize_system():
    """Initialize all detection components."""
    global camera, face_landmarker, eye_detector, yawn_detector, phone_detector
    
    print("🚗 Initializing Driver Monitoring System...")
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize detectors
    eye_detector = EyeDetector()
    yawn_detector = YawnDetector()
    phone_detector = PhoneDetector()
    
    # Initialize MediaPipe Face Landmarker
    base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    
    print("✓ System initialized successfully!")


def process_frame(frame):
    """Process a single frame and return annotated frame + results."""
    global current_status
    
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect phone
    phone_result = phone_detector.detect_phone(frame)
    frame = phone_detector.draw_detection(frame)
    
    # Detect face landmarks
    face_results = face_landmarker.detect(mp_image)
    
    face_detected = False
    if face_results.face_landmarks:
        face_detected = True
        face_landmarks = face_results.face_landmarks[0]
        
        # Detect eyes
        eye_result = eye_detector.detect_drowsiness(face_landmarks, frame_width, frame_height)
        
        # Detect yawn
        yawn_result = yawn_detector.detect_yawn(face_landmarks, frame_width, frame_height)
        
        # Update status
        current_status['eyes'] = 'CLOSED' if eye_result['eyes_closed'] else 'OPEN'
        current_status['yawning'] = 'YES' if yawn_result['yawning'] else 'NO'
        current_status['ear'] = round(eye_result['ear'], 3)
        current_status['mar'] = round(yawn_result['mar'], 3)
        
        # Determine overall status
        if phone_result['phone_detected']:
            current_status['status'] = 'DISTRACTED'
            current_status['phone'] = 'DETECTED'
            play_alarm()
        elif eye_result['drowsy']:
            current_status['status'] = 'DROWSY'
            play_alarm()
        elif yawn_result['yawning']:
            current_status['status'] = 'FATIGUED'
            play_alarm()
        else:
            current_status['status'] = 'SAFE'
            current_status['phone'] = 'NOT DETECTED'
        
        # Draw status on frame
        color = (0, 255, 0) if current_status['status'] == 'SAFE' else (0, 0, 255)
        cv2.putText(frame, f"Eyes: {current_status['eyes']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Yawning: {current_status['yawning']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Status: {current_status['status']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        current_status['status'] = 'NO FACE'
        cv2.putText(frame, "No Face Detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame


def generate_frames():
    """Generate video frames for streaming."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame
        frame = process_frame(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    """Return current detection status as JSON."""
    return jsonify(current_status)


if __name__ == '__main__':
    try:
        initialize_system()
        print("\n" + "="*60)
        print("🌐 WEB INTERFACE STARTING")
        print("="*60)
        print("📱 Access from browser:")
        print("   Local:   http://localhost:5000")
        print("   Network: http://<your-ip>:5000")
        print("\n💡 Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
        if camera:
            camera.release()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
