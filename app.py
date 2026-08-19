 from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---------- MediaPipe Face Landmarker (tasks API) ----------
mp_path = os.path.dirname(mp.__file__)
model_path = os.path.join(mp_path, 'modules', 'face_landmarker', 'face_landmarker.task')
if not os.path.exists(model_path):
    # fallback to local file if you downloaded manually
    model_path = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
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

# ------- Background task -------
@socketio.on('connect')
def handle_connect():
    print("Client connected – starting biofeedback loop")
    socketio.start_background_task(emit_loop)

def emit_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed to open.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    stress = 0.0
    prev_nose = None
    last_blink_time = time.time()
    blink_count = 0
    blink_rate = 0.0   # blinks per second

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

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
            # Blink detection
            if ear < 0.2 and (current_time - last_blink_time) > 0.15:
                blink_count += 1
                last_blink_time = current_time

            # Blink rate (average over last 5 seconds)
            elapsed = current_time - last_blink_time
            if elapsed > 0:
                blink_rate = blink_count / elapsed if elapsed > 1 else 0.0
            if elapsed > 5.0:
                blink_count = 0

            # Head movement speed
            head_speed = 0
            if prev_nose and nose:
                dx = abs(nose.x - prev_nose.x)
                dy = abs(nose.y - prev_nose.y)
                head_speed = (dx + dy) * 5
            prev_nose = nose

            # Stress components
            # 1. Eye strain: very wide eyes (ear > 0.3) or very narrow (ear < 0.15)
            eye_strain = abs(ear - 0.22) * 2.0
            # 2. Mouth tension: tightly closed (mar < 0.05) or open (mar > 0.3)
            mouth_tension = abs(mar - 0.15) * 2.0
            # 3. Blink rate: > 0.5 blinks/sec is stress
            blink_factor = min(1.0, blink_rate * 2.0)
            # 4. Head speed
            head_factor = min(1.0, head_speed * 0.2)

            raw_stress = (eye_strain * 25) + (mouth_tension * 20) + (blink_factor * 30) + (head_factor * 25)
            target_stress = min(100, raw_stress)
            stress = 0.9 * stress + 0.1 * target_stress   # smooth

            # Generate a calming tip based on stress level
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

            # Emit to frontend
            socketio.emit('biofeedback', {
                'stress': round(stress, 1),
                'blink_rate': round(blink_rate, 2),
                'mouth_tension': round(mouth_tension, 2),
                'tip': tip
            })

            # Overlay on webcam window (optional debugging)
            cv2.putText(frame, f"Stress: {int(stress)}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink/s: {blink_rate:.1f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Mind Mirror – Webcam Feed (close to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)