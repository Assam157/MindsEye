 import os
import urllib.request
from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time

# Force CPU mode (no GPU libraries needed)
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---------- Download the face landmarker model if not present ----------
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        MODEL_PATH
    )
    print("✅ Download complete.")

# ---------- Create the landmarker ----------
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

# Landmark indices
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

# ---------- Background loop ----------
@socketio.on('connect')
def handle_connect():
    print("Client connected – starting biofeedback loop")
    socketio.start_background_task(emit_loop)

def emit_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed to open")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    stress = 0.0
    prev_nose = None
    last_blink_time = time.time()
    blink_count = 0
    blink_rate = 0.0
    frame_count = 0
    emit_every = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            ear_left = aspect_ratio(landmarks, LEFT_EYE)
            ear_right = aspect_ratio(landmarks, RIGHT_EYE)
            ear = (ear_left + ear_right) / 2.0
            mar = aspect_ratio(landmarks, MOUTH)
            nose = landmarks[1]

            current_time = time.time()
            if ear < 0.2 and (current_time - last_blink_time) > 0.15:
                blink_count += 1
                last_blink_time = current_time

            elapsed = current_time - last_blink_time
            blink_rate = blink_count / elapsed if elapsed > 1 else 0.0
            if elapsed > 5.0:
                blink_count = 0

            head_speed = 0
            if prev_nose and nose:
                dx = abs(nose.x - prev_nose.x)
                dy = abs(nose.y - prev_nose.y)
                head_speed = (dx + dy) * 5
            prev_nose = nose

            eye_strain = abs(ear - 0.22) * 2.0
            mouth_tension = abs(mar - 0.15) * 2.0
            blink_factor = min(1.0, blink_rate * 2.0)
            head_factor = min(1.0, head_speed * 0.2)

            raw_stress = (eye_strain * 25) + (mouth_tension * 20) + (blink_factor * 30) + (head_factor * 25)
            target_stress = min(100, raw_stress)
            stress = 0.9 * stress + 0.1 * target_stress

            if stress < 20:
                tip = "You seem calm – keep breathing steadily."
            elif stress < 40:
                tip = "Notice any tension? Roll your shoulders gently."
            elif stress < 60:
                tip = "Try a slow, deep breath – in for 4, out for 6."
            elif stress < 80:
                tip = "Close your eyes for a moment and focus on your breath."
            else:
                tip = "Take a break – step away from the screen for a minute."

            if frame_count % emit_every == 0:
                socketio.emit('biofeedback', {
                    'stress': round(stress, 1),
                    'blink_rate': round(blink_rate, 2),
                    'mouth_tension': round(mouth_tension, 2),
                    'tip': tip
                })

        time.sleep(0.1)

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
