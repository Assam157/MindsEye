import os
import base64
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import urllib.request

# Force CPU mode
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['OPENCV_OPENCL_RUNTIME'] = ''

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---------- Download model if missing ----------
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        MODEL_PATH
    )
    print("✅ Download complete.")

# ---------- MediaPipe setup ----------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

# ---------- Helper functions ----------
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def aspect_ratio(landmarks, indices):
    """EAR (6 points) or MAR (4 points)"""
    if len(indices) == 6:
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
        vert1 = distance(p2, p6)
        vert2 = distance(p3, p5)
        hor = distance(p1, p4)
        return (vert1 + vert2) / (2.0 * hor) if hor != 0 else 0
    elif len(indices) == 4:
        top, bottom, left, right = [landmarks[i] for i in indices]
        vert = distance(top, bottom)
        hor = distance(left, right)
        return vert / hor if hor != 0 else 0
    return 0

# ---------- Lightweight feature extraction ----------
def extract_light_features(landmarks):
    """Extract only the most important metrics for stress."""
    ear_left = aspect_ratio(landmarks, [33, 160, 158, 133, 153, 144])
    ear_right = aspect_ratio(landmarks, [362, 385, 387, 263, 373, 380])
    ear = (ear_left + ear_right) / 2.0
    mar = aspect_ratio(landmarks, [13, 14, 78, 308])
    # Gaze (iris position relative to eye corners)
    left_iris = landmarks[468]
    right_iris = landmarks[474]
    left_corner = landmarks[33]
    right_corner = landmarks[263]
    gaze_x = ((left_iris.x - left_corner.x) + (right_iris.x - right_corner.x)) / 2.0
    gaze_y = ((left_iris.y - (landmarks[159].y + landmarks[145].y)/2) +
              (right_iris.y - (landmarks[386].y + landmarks[374].y)/2)) / 2.0
    return {'ear': ear, 'mar': mar, 'gaze_x': gaze_x, 'gaze_y': gaze_y}

# ---------- State & calibration ----------
state = {
    'prev_nose': None,
    'last_blink_time': time.time(),
    'blink_count': 0,
    'blink_rate': 0.0,
    'head_speed': 0.0,
    'stress': 0.0,
}

class BaselineCalibrator:
    def __init__(self, frames=30):
        self.frames = frames
        self.count = 0
        self.baselines = {'ear': [], 'mar': []}
        self.calibrated = False

    def update(self, ear, mar):
        if self.calibrated:
            return
        self.count += 1
        self.baselines['ear'].append(ear)
        self.baselines['mar'].append(mar)
        if self.count >= self.frames:
            self.calibrated = True
            self.baselines['ear'] = sum(self.baselines['ear']) / len(self.baselines['ear'])
            self.baselines['mar'] = sum(self.baselines['mar']) / len(self.baselines['mar'])
            print("✅ Calibration complete.")

    def get(self, key):
        return self.baselines.get(key, 0) if self.calibrated else 0

calibrator = BaselineCalibrator(frames=30)

# ---------- Stress history for smoothing ----------
stress_history = []

def compute_stress(ear, mar, blink_rate, head_speed, gaze_y):
    # Baseline adjustment
    ear_base = calibrator.get('ear') or 0.22
    mar_base = calibrator.get('mar') or 0.15

    ear_dev = abs(ear - ear_base) * 2.0
    mar_dev = abs(mar - mar_base) * 2.0
    blink_factor = min(1.0, blink_rate * 2.0)
    head_factor = min(1.0, head_speed * 0.3)
    gaze_down = min(1.0, abs(gaze_y) * 3.0)

    raw = (ear_dev * 25) + (mar_dev * 20) + (blink_factor * 30) + (head_factor * 15) + (gaze_down * 10)
    return min(100, raw)

# ---------- Socket event: receive frame ----------
@socketio.on('frame')
def handle_frame(data):
    try:
        img_data = base64.b64decode(data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            features = extract_light_features(landmarks)
            nose = landmarks[1]

            # Blink rate
            current_time = time.time()
            if features['ear'] < 0.2 and (current_time - state['last_blink_time']) > 0.15:
                state['blink_count'] += 1
                state['last_blink_time'] = current_time

            elapsed = current_time - state['last_blink_time']
            state['blink_rate'] = state['blink_count'] / elapsed if elapsed > 1 else 0.0
            if elapsed > 5.0:
                state['blink_count'] = 0

            # Head speed
            if state['prev_nose'] and nose:
                dx = abs(nose.x - state['prev_nose'].x)
                dy = abs(nose.y - state['prev_nose'].y)
                state['head_speed'] = (dx + dy) * 5
            state['prev_nose'] = nose

            # Calibrate (first 30 frames)
            if not calibrator.calibrated:
                calibrator.update(features['ear'], features['mar'])
                socketio.emit('biofeedback', {
                    'stress': 0,
                    'blink_rate': 0,
                    'mouth_tension': 0,
                    'tip': '🔧 Calibrating... please look naturally.'
                })
                return

            # Compute stress
            raw_stress = compute_stress(
                features['ear'],
                features['mar'],
                state['blink_rate'],
                state['head_speed'],
                features['gaze_y']
            )

            # Smoothing (moving average)
            stress_history.append(raw_stress)
            if len(stress_history) > 10:
                stress_history.pop(0)
            smooth_stress = sum(stress_history) / len(stress_history)

            # Tip generation
            if smooth_stress > 70:
                tip = "😰 High stress – consider a short break."
            elif smooth_stress > 50:
                tip = "🌿 Moderate stress – gentle stretching could help."
            elif smooth_stress > 30:
                tip = "😌 You're managing well – stay with this."
            else:
                tip = "🧘 You seem calm – continue breathing steadily."

            socketio.emit('biofeedback', {
                'stress': round(smooth_stress, 1),
                'blink_rate': round(state['blink_rate'], 2),
                'mouth_tension': round(features['mar'], 3),
                'tip': tip
            })
        else:
            socketio.emit('biofeedback', {
                'stress': 0,
                'blink_rate': 0,
                'mouth_tension': 0,
                'tip': 'No face detected – please look at the camera.'
            })
    except Exception as e:
        print(f"Error: {e}")

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
