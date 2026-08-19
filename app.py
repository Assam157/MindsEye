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

# Force CPU mode
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['OPENCV_OPENCL_RUNTIME'] = ''

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---------- MediaPipe setup (same as before) ----------
# (include model download and detector creation)
# ... your existing MediaPipe initialization code ...

# ---------- Frame processing ----------
@socketio.on('frame')
def handle_frame(data):
    """
    Receives base64-encoded JPEG frame from client,
    decodes, processes with MediaPipe, and emits biofeedback.
    """
    try:
        # Decode base64 to bytes
        img_data = base64.b64decode(data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Flip horizontally for natural mirror view
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            # ---- Compute metrics (same as before) ----
            # ... your EAR, MAR, blink rate, head speed, stress, tip ...

            # Emit back to the client
            socketio.emit('biofeedback', {
                'stress': round(stress, 1),
                'blink_rate': round(blink_rate, 2),
                'mouth_tension': round(mouth_tension, 2),
                'tip': tip
            })
        else:
            # No face detected – send zeros
            socketio.emit('biofeedback', {
                'stress': 0,
                'blink_rate': 0,
                'mouth_tension': 0,
                'tip': 'No face detected – please look at the camera.'
            })
    except Exception as e:
        print(f"Frame processing error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
