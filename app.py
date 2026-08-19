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

# ---------- MediaPipe setup (GLOBAL) ----------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)   # <-- now global

# ---------- Landmark indices ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def aspect_ratio(landmarks, indices):
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

# ---------- Global state ----------
state = {
    'stress': 0.0,
    'prev_nose': None,
    'last_blink_time': time.time(),
    'blink_count': 0,
    'blink_rate': 0.0,
    'frame_count': 0
}

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
        detection_result = detector.detect(mp_image)   # detector is now global

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            ear_left = aspect_ratio(landmarks, LEFT_EYE)
            ear_right = aspect_ratio(landmarks, RIGHT_EYE)
            ear = (ear_left + ear_right) / 2.0
            mar = aspect_ratio(landmarks, MOUTH)
            nose = landmarks[1]

            current_time = time.time()
            if ear < 0.2 and (current_time - state['last_blink_time']) > 0.15:
                state['blink_count'] += 1
                state['last_blink_time'] = current_time

            elapsed = current_time - state['last_blink_time']
            state['blink_rate'] = state['blink_count'] / elapsed if elapsed > 1 else 0.0
            if elapsed > 5.0:
                state['blink_count'] = 0

            head_speed = 0
            if state['prev_nose'] and nose:
                dx = abs(nose.x - state['prev_nose'].x)
                dy = abs(nose.y - state['prev_nose'].y)
                head_speed = (dx + dy) * 5
            state['prev_nose'] = nose

            eye_strain = abs(ear - 0.22) * 2.0
            mouth_tension = abs(mar - 0.15) * 2.0
            blink_factor = min(1.0, state['blink_rate'] * 2.0)
            head_factor = min(1.0, head_speed * 0.2)

            raw_stress = (eye_strain * 25) + (mouth_tension * 20) + (blink_factor * 30) + (head_factor * 25)
            target_stress = min(100, raw_stress)
            state['stress'] = 0.9 * state['stress'] + 0.1 * target_stress

            tip = "You seem calm – keep breathing steadily."
            if state['stress'] > 80:
                tip = "Take a break – step away from the screen for a minute."
            elif state['stress'] > 60:
                tip = "Close your eyes for a moment and focus on your breath."
            elif state['stress'] > 40:
                tip = "Try a slow, deep breath – in for 4, out for 6."
            elif state['stress'] > 20:
                tip = "Notice any tension? Roll your shoulders gently."

            socketio.emit('biofeedback', {
                'stress': round(state['stress'], 1),
                'blink_rate': round(state['blink_rate'], 2),
                'mouth_tension': round(mouth_tension, 2),
                'tip': tip
            })

            state['frame_count'] += 1
            if state['frame_count'] % 30 == 0:
                print(f"📤 Stress: {state['stress']:.1f}%")
        else:
            socketio.emit('biofeedback', {
                'stress': 0,
                'blink_rate': 0,
                'mouth_tension': 0,
                'tip': 'No face detected – please look at the camera.'
            })
    except Exception as e:
        print(f"💥 Frame processing error: {e}")
        import traceback
        traceback.print_exc()

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
