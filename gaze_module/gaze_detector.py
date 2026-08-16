from flask import Flask, Response, jsonify
import cv2
import numpy as np
import time
from collections import deque
import atexit
from flask_cors import CORS

# =====================================================
# Flask Setup
# =====================================================

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"]
)

cap = cv2.VideoCapture(0)

@atexit.register
def cleanup():
    if cap.isOpened():
        cap.release()

# =====================================================
# Haar Detectors (Improved Settings)
# =====================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# =====================================================
# Buffers & Timers
# =====================================================

blink_events = deque(maxlen=60)
eye_missing_frames = 0
head_buffer = deque(maxlen=30)

BLINK_MISS_FRAMES = 4
BLINK_EVAL_DELAY = 5.0
BLINK_HIGH_THRESHOLD = 8

STATE_DELAY = 3.0
CURRENT_STATE = "ATTENTIVE"
STATE_START = time.time()

LAST_PERSON_TIME = time.time()
PERSON_TIMEOUT = 6.0   # Increased to avoid false NO STUDENT

LATEST = {}

# =====================================================
# Vision Functions
# =====================================================

def eye_gaze(face_center_x, frame_w):
    center_margin = frame_w * 0.18
    return "CENTER" if abs(face_center_x - frame_w//2) < center_margin else "AWAY"

def blink_rate(eyes_found):
    global eye_missing_frames

    if not eyes_found:
        eye_missing_frames += 1
    else:
        if eye_missing_frames >= BLINK_MISS_FRAMES:
            blink_events.append(time.time())
        eye_missing_frames = 0

    recent = [t for t in blink_events if time.time()-t < BLINK_EVAL_DELAY]
    return "HIGH" if len(recent) >= BLINK_HIGH_THRESHOLD else "NORMAL"

def head_posture(y, h, frame_h):
    """
    Improved head down detection.
    Checks vertical face position instead of width/height ratio.
    """
    face_bottom = y + h
    down = face_bottom > frame_h * 0.80

    head_buffer.append(1 if down else 0)

    if sum(head_buffer) > 18:
        return "DOWN"

    return "UPRIGHT"

# =====================================================
# Attention Logic
# =====================================================

def raw_state(gaze, blink, head):
    if blink == "HIGH" and head == "DOWN":
        return "DROWSY"
    if gaze == "AWAY":
        return "DISTRACTED"
    return "ATTENTIVE"

def stabilize(new_state):
    global CURRENT_STATE, STATE_START
    now = time.time()

    if new_state != CURRENT_STATE:
        if now - STATE_START >= STATE_DELAY:
            CURRENT_STATE = new_state
            STATE_START = now
    else:
        STATE_START = now

    return CURRENT_STATE

# =====================================================
# Video Stream
# =====================================================

@app.route("/")
def video():
    def generate():
        global LAST_PERSON_TIME, LATEST

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640,480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Improved detection parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80,80)
            )

            now = time.time()

            if len(faces) > 0:
                LAST_PERSON_TIME = now
                student_present = True
            else:
                student_present = (now - LAST_PERSON_TIME) < PERSON_TIMEOUT

            if student_present:

                gaze = blink = head = "NORMAL"

                if len(faces) > 0:

                    x,y,w,h = faces[0]
                    face_gray = gray[y:y+h, x:x+w]

                    eyes = eye_cascade.detectMultiScale(
                        face_gray,
                        scaleFactor=1.1,
                        minNeighbors=3
                    )

                    gaze = eye_gaze(x + w//2, frame.shape[1])
                    blink = blink_rate(len(eyes) > 0)
                    head = head_posture(y, h, frame.shape[0])

                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                final = stabilize(raw_state(gaze, blink, head))

            else:
                gaze = blink = head = "NA"
                final = "No student"

            LATEST = {
                "state": final,
                "gaze": gaze,
                "blink": blink,
                "head": head
            }

            # Overlay
            cv2.putText(frame,f"STATE: {final}",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
            cv2.putText(frame,f"Gaze: {gaze}",(20,90),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.putText(frame,f"Blink: {blink}",(20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.putText(frame,f"Head: {head}",(20,150),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

            _, buffer = cv2.imencode(".jpg",frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")

            time.sleep(0.03)

    return Response(generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

# =====================================================
# LMS API
# =====================================================

@app.route("/attention-state")
def attention_state():
    return jsonify(LATEST)

# =====================================================
# Run
# =====================================================

if __name__=="__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        threaded=True,
        debug=False,
        use_reloader=False
    )