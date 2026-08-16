from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque
import atexit

# =====================================================
# Flask App
# =====================================================
app = Flask(__name__)
CORS(app)
cap = cv2.VideoCapture(0)

@atexit.register
def cleanup():
    if cap.isOpened():
        cap.release()

# =====================================================
# Detectors
# =====================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# =====================================================
# Emotion Model (Demo)
# =====================================================
emotion_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(48,48,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(7, activation="softmax")
])

EMOTIONS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
cached_emotion = "Neutral"
last_emotion_time = 0

def predict_emotion(face_gray):
    global cached_emotion, last_emotion_time
    if time.time() - last_emotion_time > 1:
        face = cv2.resize(face_gray,(48,48))/255.0
        face = face.reshape(1,48,48,1)
        pred = emotion_model(face, training=False)
        cached_emotion = EMOTIONS[int(np.argmax(pred))]
        last_emotion_time = time.time()
    return cached_emotion

# =====================================================
# Buffers & Timers
# =====================================================
blink_events = deque(maxlen=30)
eye_missing_frames = 0
head_buffer = deque(maxlen=30)

BLINK_MISS_FRAMES = 3
BLINK_EVAL_DELAY = 2.0
BLINK_HIGH_THRESHOLD = 4

STATE_DELAY = 4.0
CURRENT_STATE = "ATTENTIVE"
STATE_START = time.time()

LAST_PERSON_TIME = time.time()
PERSON_TIMEOUT = 4.0

# =====================================================
# Vision Signal Functions
# =====================================================
def eye_gaze(face_center_x, frame_w):
    return "CENTER" if abs(face_center_x-frame_w//2) < frame_w*0.20 else "AWAY"

def face_orientation(face_w, frame_w):
    return "FRONT" if face_w > frame_w*0.18 else "SIDE"

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

def head_posture(y, h, w, frame_h):
    down = (y+h) > frame_h*0.75
    tilt = h > w*1.4
    head_buffer.append(1 if (down or tilt) else 0)
    if sum(head_buffer) > 18:
        return "DOWN" if down else "TILT"
    return "UPRIGHT"

# =====================================================
# Attention Logic
# =====================================================
def raw_state(gaze, blink, head, face):
    # Strong drowsiness signal
    if blink == "HIGH" and head in ["DOWN", "TILT"]:
        return "DROWSY"

    # Clear distraction
    if gaze == "AWAY" and face == "SIDE":
        return "DISTRACTED"

    # Mild distraction (needs persistence)
    if gaze == "AWAY" or head in ["DOWN", "TILT"]:
        return "DISTRACTED"

    return "ATTENTIVE"


def stabilize(new_state):
    global CURRENT_STATE, STATE_START
    now = time.time()

    # If attentive → immediately reset
    if new_state == "ATTENTIVE":
        CURRENT_STATE = "ATTENTIVE"
        STATE_START = now
        return CURRENT_STATE

    # If distracted/drowsy → require persistence
    if new_state != CURRENT_STATE:
        if now - STATE_START >= STATE_DELAY:
            CURRENT_STATE = new_state
            STATE_START = now
    return CURRENT_STATE


# =====================================================
# Shared LMS Data
# =====================================================
LATEST = {}

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

            frame = cv2.resize(frame,(640,480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(60,60)
)

            now = time.time()

            if len(faces)>0:
                LAST_PERSON_TIME = now
                student_present = True
            else:
                student_present = (now-LAST_PERSON_TIME) < PERSON_TIMEOUT

            if student_present:
                gaze = blink = head = face = "NORMAL"
                emotion = cached_emotion

                if len(faces)>0:
                    x,y,w,h = faces[0]
                    face_gray = gray[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(face_gray,1.1,3)

                    gaze = eye_gaze(x+w//2, frame.shape[1])
                    blink = blink_rate(len(eyes)>0)
                    head = head_posture(y,h,w,frame.shape[0])
                    face = face_orientation(w, frame.shape[1])
                    emotion = predict_emotion(face_gray)

                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                else:
                    gaze="AWAY"
                    head="TILT"
                    face="SIDE"
                    blink="NORMAL"

                final = stabilize(raw_state(gaze,blink,head,face))
            else:
                gaze=blink=head=face=emotion="NA"
                final="NO STUDENT"

            LATEST={
                "state":final,
                "gaze":gaze,
                "blink":blink,
                "head":head,
                "face":face,
                "emotion":emotion
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
            cv2.putText(frame,f"Face: {face}",(20,180),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.putText(frame,f"Emotion: {emotion}",(20,210),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)

            _, buffer=cv2.imencode(".jpg",frame)
            yield(b"--frame\r\n"
                  b"Content-Type: image/jpeg\r\n\r\n"+
                  buffer.tobytes()+b"\r\n")

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
