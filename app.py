import os
import shutil

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import json
import moviepy.video.io.VideoFileClip as mp
from groq import Groq
import cv2
import numpy as np
import time
from collections import deque

# =====================================================
# FFmpeg configuration
# =====================================================

FFMPEG_PATH = r"C:\Users\BHARATH\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build-shared\bin\ffmpeg.exe"

if os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] = (
        os.path.dirname(FFMPEG_PATH)
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    print("FFmpeg found:", FFMPEG_PATH)
else:
    print("WARNING: FFmpeg not found:", FFMPEG_PATH)


app = Flask(__name__)

# =====================================================
# BROWSER-BASED ATTENTION DETECTOR
# =====================================================
# This replaces the old standalone gaze_detector.py webcam
# process. The browser sends camera frames to /attention-frame.
# IMPORTANT: do NOT use cv2.VideoCapture(0) on Render.

blink_events = deque(maxlen=60)
eye_missing_frames = 0
head_buffer = deque(maxlen=30)

BLINK_MISS_FRAMES = 4
BLINK_EVAL_DELAY = 5.0
BLINK_HIGH_THRESHOLD = 8

STATE_DELAY = 3.0
CURRENT_STATE = "ATTENTIVE"
STATE_START = time.time()

LATEST_ATTENTION = {
    "state": "ATTENTIVE",
    "gaze": "NA",
    "blink": "NORMAL",
    "head": "UPRIGHT",
    "face": "ABSENT",
    "emotion": "NEUTRAL"
}


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def eye_gaze(face_center_x, frame_w):
    center_margin = frame_w * 0.18
    return "CENTER" if abs(face_center_x - frame_w // 2) < center_margin else "AWAY"


def blink_rate(eyes_found):
    global eye_missing_frames

    if not eyes_found:
        eye_missing_frames += 1
    else:
        if eye_missing_frames >= BLINK_MISS_FRAMES:
            blink_events.append(time.time())
        eye_missing_frames = 0

    now = time.time()
    recent = [
        t for t in blink_events
        if now - t < BLINK_EVAL_DELAY
    ]

    return (
        "HIGH"
        if len(recent) >= BLINK_HIGH_THRESHOLD
        else "NORMAL"
    )


def head_posture(y, h, frame_h):
    face_bottom = y + h
    down = face_bottom > frame_h * 0.80

    head_buffer.append(1 if down else 0)

    if sum(head_buffer) > 18:
        return "DOWN"

    return "UPRIGHT"


def raw_attention_state(gaze, blink, head):
    if blink == "HIGH" and head == "DOWN":
        return "DROWSY"

    if gaze == "AWAY":
        return "DISTRACTED"

    return "ATTENTIVE"


def stabilize_attention(new_state):
    global CURRENT_STATE, STATE_START

    now = time.time()

    if new_state != CURRENT_STATE:
        if now - STATE_START >= STATE_DELAY:
            CURRENT_STATE = new_state
            STATE_START = now
    else:
        STATE_START = now

    return CURRENT_STATE


def process_attention_frame(frame_bytes):
    """
    Process one JPEG/PNG frame sent by the student's browser.
    Returns the same attention fields previously exposed by
    gaze_detector.py.
    """
    global LATEST_ATTENTION

    if not frame_bytes:
        return LATEST_ATTENTION

    array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            **LATEST_ATTENTION,
            "error": "Invalid image frame"
        }

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        # Keep the last state briefly instead of immediately
        # marking the student distracted.
        LATEST_ATTENTION = {
            "state": CURRENT_STATE,
            "gaze": "NA",
            "blink": "NORMAL",
            "head": "NA",
            "face": "ABSENT",
            "emotion": "NEUTRAL"
        }
        return LATEST_ATTENTION

    x, y, w, h = faces[0]
    face_gray = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=3
    )

    gaze = eye_gaze(
        x + w // 2,
        frame.shape[1]
    )

    blink = blink_rate(len(eyes) > 0)

    head = head_posture(
        y,
        h,
        frame.shape[0]
    )

    final_state = stabilize_attention(
        raw_attention_state(
            gaze,
            blink,
            head
        )
    )

    LATEST_ATTENTION = {
        "state": final_state,
        "gaze": gaze,
        "blink": blink,
        "head": head,
        "face": "PRESENT",
        "emotion": "NEUTRAL"
    }

    return LATEST_ATTENTION


@app.route("/attention-state")
def attention_state():
    return jsonify(LATEST_ATTENTION)


@app.route("/attention-frame", methods=["POST"])
def attention_frame():
    """
    Browser sends a camera frame here.

    Accepts:
      1. raw image bytes with Content-Type image/jpeg
      2. multipart/form-data field named 'frame'
    """
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    frame_bytes = None

    uploaded = request.files.get("frame")
    if uploaded:
        frame_bytes = uploaded.read()
    else:
        frame_bytes = request.get_data()

    try:
        result = process_attention_frame(frame_bytes)
        return jsonify(result)
    except Exception as e:
        print("Attention frame error:", e)
        return jsonify({
            "error": "Attention processing failed"
        }), 500


# ================= GROK AI CONFIG =================
# ================= GROK AI CONFIG =================
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_URL = "https://api.x.ai/v1/chat/completions"
app.secret_key = "college_lms_secret"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# ================= DB =================
def get_db():
    return sqlite3.connect("database.db")
# ================= CREATE TABLES (RUN ONCE) =================
def init_db():
    db = get_db()
    cur = db.cursor()

    # =====================================================
    # USERS
    # =====================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT,
        child_email TEXT
    )
    """)

    # =====================================================
    # COURSES
    # =====================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_name TEXT,
        teacher_id INTEGER,
        video_path TEXT,
        video_topics TEXT,
        video_summary TEXT
    )
    """)

    # =====================================================
    # ENROLLMENTS
    # =====================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS enrollments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        course_id INTEGER
    )
    """)

    # =====================================================
    # QUIZZES
    # =====================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS quizzes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER,
        title TEXT,
        deadline TEXT
    )
    """)

    # =====================================================
    # QUIZ QUESTIONS
    # =====================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS quiz_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        quiz_id INTEGER,
        question TEXT,
        a TEXT,
        b TEXT,
        c TEXT,
        d TEXT,
        correct TEXT
    )
    """)

    # =====================================================
    # QUIZ RESULTS
    # =====================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        quiz_id INTEGER,
        score INTEGER
    )
    """)

    # =====================================================
    # VIDEO PROGRESS
    # =====================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS video_progress (
        student_id INTEGER,
        course_id INTEGER,
        watched_seconds INTEGER,
        total_seconds INTEGER,
        percentage REAL,
        PRIMARY KEY (student_id, course_id)
    )
    """)

    # =====================================================
    # LECTURE AI DATA
    # =====================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS lecture_ai (
        course_id INTEGER PRIMARY KEY,
        transcript TEXT,
        summary TEXT,
        topics TEXT
    )
    """)

    # =====================================================
    # ATTENTION LOGS
    # =====================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attention_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        course_id INTEGER,
        state TEXT,
        gaze REAL,
        blink REAL,
        head REAL,
        face REAL,
        emotion TEXT,
        timestamp TEXT
    )
    """)

    # =====================================================
    # SAVE CHANGES
    # =====================================================
    db.commit()
    db.close()


# ================= DB MIGRATION (RUN ONCE) =================
# ================= DB MIGRATION =================
def migrate_db():
    db = get_db()
    cur = db.cursor()

    try:
        # Get existing columns in courses table
        cur.execute("PRAGMA table_info(courses)")
        columns = [col[1] for col in cur.fetchall()]

        # Add video_path if missing
        if "video_path" not in columns:
            print("Adding video_path column...")
            cur.execute(
                "ALTER TABLE courses ADD COLUMN video_path TEXT"
            )

        # Add video_topics if missing
        if "video_topics" not in columns:
            print("Adding video_topics column...")
            cur.execute(
                "ALTER TABLE courses ADD COLUMN video_topics TEXT"
            )

        # Add video_summary if missing
        if "video_summary" not in columns:
            print("Adding video_summary column...")
            cur.execute(
                "ALTER TABLE courses ADD COLUMN video_summary TEXT"
            )

        db.commit()
        print("Database migration completed successfully.")

    except Exception as e:
        db.rollback()
        print("Migration error:", e)

    finally:
        db.close()


# ================= LANDING PAGE =================
@app.route("/")
def home():
    return render_template("home.html")


# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        db = get_db()
        cur = db.cursor()

        cur.execute("""
            SELECT * FROM users 
            WHERE username=? AND password=?
        """, (username, password))

        user = cur.fetchone()

        if user:
            # Save session info
            session["user_id"] = user[0]
            session["name"] = user[1]
            session["username"] = user[2]
            session["role"] = user[4]   # VERY IMPORTANT

            # Redirect based on role
            if user[4] == "teacher":
                return redirect("/teacher")

            elif user[4] == "student":
                return redirect("/student")

            elif user[4] == "parent":
                return redirect("/parent/dashboard")

        return "Invalid Login"

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        username = request.form["email"]   # ✅ FIXED
        password = request.form["password"]
        role = request.form["role"]

        # parent extra field
        child_email = request.form.get("child_email")

        db = get_db()
        cur = db.cursor()

        try:
            cur.execute("""
                INSERT INTO users (name, username, password, role, child_email)
                VALUES (?, ?, ?, ?, ?)
            """, (name, username, password, role, child_email))

            db.commit()

        except sqlite3.IntegrityError:
            return "User already exists"

        return redirect("/login")

    return render_template("register.html")

# ================= TEACHER =================
@app.route("/teacher")
def teacher_dashboard():
    if session.get("role") != "teacher":
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT * FROM courses WHERE teacher_id=?", (session["user_id"],))
    courses = cur.fetchall()

    cur.execute("SELECT id,name FROM users WHERE role='student'")
    students = cur.fetchall()

    return render_template("teacher_dashboard.html",
                           name=session["name"],
                           courses=courses,
                           students=students)

@app.route("/create-course", methods=["POST"])
def create_course():
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO courses(course_name,teacher_id) VALUES (?,?)",
                (request.form["course"], session["user_id"]))
    db.commit()
    return redirect("/teacher")

@app.route("/enroll", methods=["POST"])
def enroll():
    if session.get("role") != "teacher":
        return redirect("/")

    student_id = request.form.get("student")
    course_id = request.form.get("course_id")

    # Safety check
    if not student_id or not course_id:
        return "Invalid enrollment data", 400

    db = get_db()
    cur = db.cursor()

    cur.execute(
        "INSERT INTO enrollments (student_id, course_id) VALUES (?, ?)",
        (student_id, course_id)
    )
    db.commit()

    return redirect("/teacher")



# ================= STUDENT =================
# ================= STUDENT =================
@app.route("/student")
def student_dashboard():

    if session.get("role") != "student":
        return redirect("/")

    student_id = session["user_id"]

    db = get_db()
    cur = db.cursor()

    # =====================================================
    # COURSES
    # =====================================================

    cur.execute("""
        SELECT courses.id, courses.course_name
        FROM courses
        JOIN enrollments
            ON courses.id = enrollments.course_id
        WHERE enrollments.student_id = ?
    """, (student_id,))

    courses = cur.fetchall()


    # =====================================================
    # VIDEO PROGRESS
    # =====================================================

    cur.execute("""
        SELECT COALESCE(AVG(percentage), 0)
        FROM video_progress
        WHERE student_id = ?
    """, (student_id,))

    video_progress = cur.fetchone()[0] or 0


    # =====================================================
    # ATTENTION
    # =====================================================

    cur.execute("""
        SELECT
            COUNT(*),
            SUM(
                CASE
                    WHEN state = 'ATTENTIVE'
                    THEN 1
                    ELSE 0
                END
            )
        FROM attention_logs
        WHERE student_id = ?
    """, (student_id,))

    attention_row = cur.fetchone()

    total_attention_checks = attention_row[0] or 0
    attentive_checks = attention_row[1] or 0

    if total_attention_checks > 0:

        attention_percentage = (
            attentive_checks /
            total_attention_checks
        ) * 100

    else:

        attention_percentage = 0


    # =====================================================
    # QUIZ PERFORMANCE
    # =====================================================

    cur.execute("""
        SELECT
            qr.quiz_id,
            qr.score
        FROM quiz_results qr
        WHERE qr.student_id = ?
    """, (student_id,))

    quiz_results = cur.fetchall()

    total_correct = 0
    total_questions = 0

    for quiz_id, score in quiz_results:

        cur.execute("""
            SELECT COUNT(*)
            FROM quiz_questions
            WHERE quiz_id = ?
        """, (quiz_id,))

        question_count = cur.fetchone()[0] or 0

        total_correct += int(score or 0)
        total_questions += question_count


    if total_questions > 0:

        quiz_percentage = (
            total_correct /
            total_questions
        ) * 100

    else:

        quiz_percentage = 0


    # =====================================================
    # QUIZ COUNT
    # =====================================================

    cur.execute("""
        SELECT COUNT(*)
        FROM quiz_results
        WHERE student_id = ?
    """, (student_id,))

    quizzes_completed = cur.fetchone()[0] or 0


    # =====================================================
    # VIDEO COUNT
    # =====================================================

    cur.execute("""
        SELECT COUNT(*)
        FROM video_progress
        WHERE student_id = ?
          AND percentage >= 90
    """, (student_id,))

    videos_watched = cur.fetchone()[0] or 0


    db.close()


    # =====================================================
    # SEND DATA TO DASHBOARD
    # =====================================================

    return render_template(
        "student_dashboard.html",

        name=session["name"],

        courses=courses,

        attention_percentage=round(
            attention_percentage,
            1
        ),

        video_progress=round(
            video_progress,
            1
        ),

        quiz_percentage=round(
            quiz_percentage,
            1
        ),

        videos_watched=videos_watched,

        quizzes_completed=quizzes_completed
    )




# ==========================================================
# AI STUDY COACH
# ==========================================================

@app.route("/student/ai-coach")
def student_ai_coach():

    if session.get("role") != "student":
        return jsonify({
            "error": "Unauthorized"
        }), 401

    student_id = session["user_id"]
    student_name = session["name"]

    db = get_db()
    cur = db.cursor()

    # ======================================================
    # ATTENTION
    # ======================================================

    cur.execute("""
        SELECT
            COUNT(*),
            SUM(
                CASE
                    WHEN state = 'ATTENTIVE'
                    THEN 1
                    ELSE 0
                END
            )
        FROM attention_logs
        WHERE student_id = ?
    """, (student_id,))

    attention_row = cur.fetchone()

    total_checks = attention_row[0] or 0
    attentive_checks = attention_row[1] or 0

    if total_checks > 0:

        attention = round(
            (attentive_checks / total_checks) * 100,
            1
        )

    else:

        attention = 0


    # ======================================================
    # VIDEO
    # ======================================================

    cur.execute("""
        SELECT COALESCE(AVG(percentage), 0)
        FROM video_progress
        WHERE student_id = ?
    """, (student_id,))

    video = round(
        cur.fetchone()[0] or 0,
        1
    )


    # ======================================================
    # QUIZ
    # ======================================================

    cur.execute("""
        SELECT
            qr.quiz_id,
            qr.score
        FROM quiz_results qr
        WHERE qr.student_id = ?
    """, (student_id,))

    quiz_results = cur.fetchall()

    correct = 0
    questions = 0

    for quiz_id, score in quiz_results:

        cur.execute("""
            SELECT COUNT(*)
            FROM quiz_questions
            WHERE quiz_id = ?
        """, (quiz_id,))

        q_count = cur.fetchone()[0] or 0

        correct += int(score or 0)
        questions += q_count


    if questions > 0:

        quiz = round(
            (correct / questions) * 100,
            1
        )

    else:

        quiz = 0


    db.close()


    # ======================================================
    # CHOOSE LEARNING MOOD
    # ======================================================

    if attention < 50:

        mood = "focus"

    elif quiz < 50:

        mood = "improve"

    elif video < 50:

        mood = "learning"

    elif attention >= 80 and quiz >= 80:

        mood = "achievement"

    else:

        mood = "motivation"


    # ======================================================
    # GROQ PROMPT
    # ======================================================

    prompt = f"""
You are an encouraging AI study coach inside a college LMS.

Student name: {student_name}

Current learning statistics:

Attention: {attention}%
Video completion: {video}%
Quiz performance: {quiz}%

Generate a short personalized study coaching message.

The response MUST be valid JSON with exactly these fields:

{{
    "greeting": "...",
    "message": "...",
    "goal": "...",
    "tip": "...",
    "mood": "{mood}"
}}

Rules:

- Be encouraging and natural.
- Do NOT sound robotic.
- Do NOT use exaggerated motivational quotes.
- Mention the student's actual performance when useful.
- Give one realistic goal.
- Give one practical study tip.
- Keep the message short.
- Do not mention that you are an AI.
- Do not use markdown.
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful college study coach."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.7,

            max_tokens=300
        )


        content = response.choices[0].message.content.strip()

        # Remove accidental markdown code fences
        content = content.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

        data = json.loads(content)

        return jsonify({

            "success": True,

            "greeting": data.get(
                "greeting",
                f"Hey {student_name}! 👋"
            ),

            "message": data.get(
                "message",
                "Keep learning and stay consistent."
            ),

            "goal": data.get(
                "goal",
                "Complete one focused learning session today."
            ),

            "tip": data.get(
                "tip",
                "Try studying without distractions."
            ),

            "mood": data.get(
                "mood",
                mood
            ),

            "attention": attention,

            "video": video,

            "quiz": quiz
        })


    except Exception as e:

        print("AI Coach Error:", e)

        # Fallback so the dashboard still works
        return jsonify({

            "success": True,

            "greeting": f"Hey {student_name}! 👋",

            "message":
                "Keep building your learning routine. "
                "Small consistent sessions make a big difference.",

            "goal":
                "Complete one focused learning session today.",

            "tip":
                "Keep distractions away while watching your lecture.",

            "mood": mood,

            "attention": attention,

            "video": video,

            "quiz": quiz
        })



@app.route("/parent/dashboard")
def parent_dashboard():

    if session.get("role") != "parent":
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    # ======================================================
    # GET LINKED CHILD
    # ======================================================

    cur.execute(
        "SELECT child_email FROM users WHERE username=?",
        (session["username"],)
    )

    child = cur.fetchone()

    if not child or not child[0]:
        db.close()
        return "No child linked"

    child_email = child[0]

    # ======================================================
    # GET CHILD DETAILS
    # ======================================================

    cur.execute(
        "SELECT id, name FROM users WHERE username=?",
        (child_email,)
    )

    child_row = cur.fetchone()

    if not child_row:
        db.close()
        return "Linked child not found"

    child_id = child_row[0]
    child_name = child_row[1]

    # ======================================================
    # GET UNIQUE COURSES
    # ======================================================

    cur.execute("""
        SELECT DISTINCT
            c.id,
            c.course_name
        FROM enrollments e
        JOIN courses c
            ON c.id = e.course_id
        WHERE e.student_id = ?
        ORDER BY c.course_name
    """, (child_id,))

    courses = cur.fetchall()

    course_data = []

    # ======================================================
    # COURSE-WISE ANALYTICS
    # ======================================================

    for course_id, course_name in courses:

        # --------------------------------------------------
        # VIDEO PROGRESS
        # --------------------------------------------------

        cur.execute("""
            SELECT ROUND(AVG(percentage), 2)
            FROM video_progress
            WHERE student_id = ?
              AND course_id = ?
        """, (child_id, course_id))

        video_result = cur.fetchone()

        video_percentage = (
            video_result[0]
            if video_result and video_result[0] is not None
            else 0
        )

        # --------------------------------------------------
        # ATTENTION
        # --------------------------------------------------

        cur.execute("""
            SELECT
                COUNT(*),
                SUM(
                    CASE
                        WHEN state = 'ATTENTIVE'
                        THEN 1
                        ELSE 0
                    END
                )
            FROM attention_logs
            WHERE student_id = ?
              AND course_id = ?
        """, (child_id, course_id))

        attention_result = cur.fetchone()

        total_checks = attention_result[0] or 0
        attentive_checks = attention_result[1] or 0

        if total_checks > 0:

            attention_percentage = round(
                (attentive_checks / total_checks) * 100,
                2
            )

        else:

            attention_percentage = 0

        # --------------------------------------------------
        # QUIZ RESULTS
        # --------------------------------------------------

        cur.execute("""
            SELECT
                qr.quiz_id,
                qr.score
            FROM quiz_results qr
            JOIN quizzes q
                ON q.id = qr.quiz_id
            WHERE qr.student_id = ?
              AND q.course_id = ?
        """, (child_id, course_id))

        quiz_results = cur.fetchall()

        total_correct = 0
        total_questions = 0

        for quiz_id, score in quiz_results:

            cur.execute("""
                SELECT COUNT(*)
                FROM quiz_questions
                WHERE quiz_id = ?
            """, (quiz_id,))

            question_count = cur.fetchone()[0] or 0

            total_correct += int(score or 0)
            total_questions += question_count

        if total_questions > 0:

            quiz_percentage = round(
                (total_correct / total_questions) * 100,
                2
            )

        else:

            quiz_percentage = 0

        # --------------------------------------------------
        # STORE COURSE DATA
        # --------------------------------------------------

        course_data.append({
            "course_id": course_id,
            "course_name": course_name,
            "video": video_percentage,
            "attention": attention_percentage,
            "quiz": quiz_percentage
        })

    # ======================================================
    # OVERALL SUMMARY
    # ======================================================

    total_courses = len(course_data)

    if total_courses > 0:

        avg_attention = round(
            sum(c["attention"] for c in course_data)
            / total_courses,
            2
        )

        avg_score = round(
            sum(c["quiz"] for c in course_data)
            / total_courses,
            2
        )

        avg_video = round(
            sum(c["video"] for c in course_data)
            / total_courses,
            2
        )

    else:

        avg_attention = 0
        avg_score = 0
        avg_video = 0

    # ======================================================
    # LOW FOCUS ALERTS
    # ======================================================

    alerts = [
        c["course_name"]
        for c in course_data
        if c["attention"] < 60
    ]

    # ======================================================
    # MOST DISTRACTED TIME
    # ======================================================

    cur.execute("""
        SELECT
            strftime('%H', timestamp),
            COUNT(*)
        FROM attention_logs
        WHERE student_id = ?
          AND state = 'DISTRACTED'
        GROUP BY strftime('%H', timestamp)
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """, (child_id,))

    fatigue = cur.fetchone()

    if fatigue:

        fatigue_hour = fatigue[0] + ":00"

    else:

        fatigue_hour = "No data"

    db.close()

    # ======================================================
    # RENDER
    # ======================================================

    return render_template(
        "parent_dashboard.html",

        child_name=child_name,

        child_id=child_id,

        course_data=course_data,

        avg_video=avg_video,

        avg_attention=avg_attention,

        avg_score=avg_score,

        alerts=alerts,

        fatigue_hour=fatigue_hour
    )






# ==========================================================
# 🤖 AI PARENT LEARNING INSIGHT
# ==========================================================

@app.route("/parent/ai-insight/<int:student_id>")
def parent_ai_insight(student_id):

    if session.get("role") != "parent":
        return jsonify({
            "success": False,
            "error": "Unauthorized"
        }), 403

    db = get_db()
    cur = db.cursor()

    try:

        # ==================================================
        # VERIFY THIS IS THE PARENT'S LINKED CHILD
        # ==================================================

        cur.execute(
            "SELECT child_email FROM users WHERE username=?",
            (session["username"],)
        )

        parent_row = cur.fetchone()

        if not parent_row or not parent_row[0]:

            return jsonify({
                "success": False,
                "error": "No child linked to this parent."
            }), 403

        child_username = parent_row[0]

        cur.execute("""
            SELECT id, name
            FROM users
            WHERE username=?
        """, (child_username,))

        child_row = cur.fetchone()

        if not child_row:

            return jsonify({
                "success": False,
                "error": "Linked child not found."
            }), 404

        linked_child_id = child_row[0]
        child_name = child_row[1]

        # IMPORTANT:
        # Prevent parent from requesting another student's data.

        if int(student_id) != int(linked_child_id):

            return jsonify({
                "success": False,
                "error": "You are not authorized to view this student."
            }), 403

        # ==================================================
        # GET COURSE DATA
        # ==================================================

        cur.execute("""
            SELECT DISTINCT
                c.id,
                c.course_name
            FROM enrollments e
            JOIN courses c
                ON c.id = e.course_id
            WHERE e.student_id = ?
            ORDER BY c.course_name
        """, (student_id,))

        courses = cur.fetchall()

        course_data = []

        for course_id, course_name in courses:

            # ----------------------------------------------
            # VIDEO
            # ----------------------------------------------

            cur.execute("""
                SELECT AVG(percentage)
                FROM video_progress
                WHERE student_id=?
                  AND course_id=?
            """, (student_id, course_id))

            video_row = cur.fetchone()

            video = round(
                float(video_row[0] or 0),
                1
            )

            # ----------------------------------------------
            # ATTENTION
            # ----------------------------------------------

            cur.execute("""
                SELECT
                    COUNT(*),
                    SUM(
                        CASE
                            WHEN state='ATTENTIVE'
                            THEN 1
                            ELSE 0
                        END
                    )
                FROM attention_logs
                WHERE student_id=?
                  AND course_id=?
            """, (student_id, course_id))

            attention_row = cur.fetchone()

            total_checks = attention_row[0] or 0
            attentive_checks = attention_row[1] or 0

            if total_checks > 0:

                attention = round(
                    (attentive_checks / total_checks) * 100,
                    1
                )

            else:

                attention = 0

            # ----------------------------------------------
            # QUIZ
            # ----------------------------------------------

            cur.execute("""
                SELECT
                    qr.quiz_id,
                    qr.score
                FROM quiz_results qr
                JOIN quizzes q
                    ON q.id = qr.quiz_id
                WHERE qr.student_id=?
                  AND q.course_id=?
            """, (student_id, course_id))

            results = cur.fetchall()

            correct = 0
            questions = 0

            for quiz_id, score in results:

                cur.execute("""
                    SELECT COUNT(*)
                    FROM quiz_questions
                    WHERE quiz_id=?
                """, (quiz_id,))

                count = cur.fetchone()[0] or 0

                correct += int(score or 0)
                questions += count

            if questions > 0:

                quiz = round(
                    (correct / questions) * 100,
                    1
                )

            else:

                quiz = 0

            course_data.append({
                "course": course_name,
                "video": video,
                "attention": attention,
                "quiz": quiz
            })

        # ==================================================
        # OVERALL VALUES
        # ==================================================

        total_courses = len(course_data)

        if total_courses > 0:

            avg_video = round(
                sum(c["video"] for c in course_data)
                / total_courses,
                1
            )

            avg_attention = round(
                sum(c["attention"] for c in course_data)
                / total_courses,
                1
            )

            avg_quiz = round(
                sum(c["quiz"] for c in course_data)
                / total_courses,
                1
            )

        else:

            avg_video = 0
            avg_attention = 0
            avg_quiz = 0

        # ==================================================
        # COURSE DATA FOR GROQ
        # ==================================================

        course_text = ""

        for c in course_data:

            course_text += (
                f"Course: {c['course']}\n"
                f"Video completion: {c['video']}%\n"
                f"Attention: {c['attention']}%\n"
                f"Quiz performance: {c['quiz']}%\n\n"
            )

        # ==================================================
        # GROQ PROMPT
        # ==================================================

        prompt = f"""
You are an AI learning assistant for parents.

Analyze the following learning data for a student.

Student name:
{child_name}

Overall video completion:
{avg_video}%

Overall attention:
{avg_attention}%

Overall quiz performance:
{avg_quiz}%

Course-wise data:

{course_text}

Create a short, supportive parent-friendly learning update.

IMPORTANT:
- Do not use technical AI terminology.
- Do not criticize or shame the student.
- Do not make medical or psychological claims.
- Do not invent information.
- Base the response only on the provided data.
- Be encouraging and practical.

Return ONLY valid JSON in exactly this format:

{{
    "summary": "2-3 sentence parent-friendly summary",
    "strength": "One positive observation",
    "attention": "One observation about attention",
    "recommendation": "One practical suggestion for the parent",
    "encouragement": "One short encouraging sentence"
}}
"""

        # ==================================================
        # GROQ REQUEST
        # ==================================================

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "system",
                    "content":
                    "You are a supportive educational assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.4,

            max_tokens=500
        )

        ai_text = response.choices[0].message.content.strip()

        # ==================================================
        # REMOVE MARKDOWN CODE FENCES IF PRESENT
        # ==================================================

        if ai_text.startswith("```"):

            ai_text = ai_text.replace(
                "```json",
                ""
            ).replace(
                "```",
                ""
            ).strip()

        # ==================================================
        # PARSE JSON
        # ==================================================

        try:

            ai_data = json.loads(ai_text)

        except json.JSONDecodeError:

            # Fallback if Groq returns plain text

            ai_data = {
                "summary": ai_text,
                "strength":
                    "The student's learning activity is being tracked.",
                "attention":
                    f"Current attention level is {avg_attention}%.",
                "recommendation":
                    "Encourage regular study sessions and a distraction-free environment.",
                "encouragement":
                    "Keep supporting consistent learning habits."
            }

        # ==================================================
        # RETURN RESPONSE
        # ==================================================

        return jsonify({

            "success": True,

            "student": child_name,

            "avg_video": avg_video,

            "avg_attention": avg_attention,

            "avg_quiz": avg_quiz,

            "summary":
                ai_data.get(
                    "summary",
                    ""
                ),

            "strength":
                ai_data.get(
                    "strength",
                    ""
                ),

            "attention":
                ai_data.get(
                    "attention",
                    ""
                ),

            "recommendation":
                ai_data.get(
                    "recommendation",
                    ""
                ),

            "encouragement":
                ai_data.get(
                    "encouragement",
                    ""
                )

        })

    except Exception as e:

        print(
            "Parent AI Insight Error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error":
                "Unable to generate AI learning update."

        }), 500

    finally:

        db.close()






# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")
@app.route("/course/<int:course_id>")
def course_home(course_id):
    # Only students can access
    if session.get("role") != "student":
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    # 🔹 Check student is enrolled in this course
    cur.execute("""
        SELECT 1 FROM enrollments
        WHERE student_id = ? AND course_id = ?
    """, (session["user_id"], course_id))

    enrolled = cur.fetchone()
    if not enrolled:
        return "You are not enrolled in this course", 403

    # 🔹 Fetch course name
    cur.execute("SELECT course_name FROM courses WHERE id = ?", (course_id,))
    course = cur.fetchone()

    # 🔹 Fetch quizzes assigned to this course
    cur.execute("""
    SELECT q.id, q.title, q.deadline,
           CASE 
             WHEN qr.student_id IS NOT NULL THEN 1
             ELSE 0
           END AS attempted
    FROM quizzes q
    LEFT JOIN quiz_results qr
      ON q.id = qr.quiz_id AND qr.student_id = ?
    WHERE q.course_id = ?
""", (session["user_id"], course_id))
    
    quizzes = cur.fetchall()

    return render_template(
        "course_home.html",
        course_id=course_id,
        course_name=course[0],
        quizzes=quizzes,
        name=session["name"]
    )

@app.route("/video-progress", methods=["POST"])
def video_progress():
    data = request.json
    db = get_db()
    cur = db.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO video_progress
        (student_id, course_id, watched_seconds, total_seconds, percentage)
        VALUES (?, ?, ?, ?, ?)
    """, (
        session["user_id"],
        data["course_id"],
        data["watched"],
        data["total"],
        data["percentage"]
    ))

    db.commit()
    return {"status": "ok"}
# ================= COURSE HOME =================

import os
from werkzeug.utils import secure_filename
#from video_ai import process_video   # ⭐ AI VIDEO ANALYSIS
UPLOAD_FOLDER = "static/videos"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================
# 🎬 TEACHER VIDEO UPLOAD + AI ANALYSIS
# =====================================================
import os
from werkzeug.utils import secure_filename
#from video_ai import process_video   # AI video analyzer

UPLOAD_FOLDER = "static/videos"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/teacher/upload-video/<int:course_id>", methods=["POST"])
def teacher_upload_video(course_id):

    # 🔐 allow only teachers
    if session.get("role") != "teacher":
        return redirect("/")

    # ❌ no file selected
    if "video" not in request.files:
        return "No video selected"

    file = request.files["video"]

    if file.filename == "":
        return "No video selected"

    # =====================================================
    # 💾 SAVE VIDEO FILE
    # =====================================================
    filename = secure_filename(file.filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    print("Video saved at:", save_path)

    # =====================================================
    # 🤖 STEP 1 — AI VIDEO ANALYSIS (AUTO RUN)
    # =====================================================
    
    topics = "Video AI analysis will be enabled later."
    summary = "Video uploaded successfully. AI analysis is temporarily disabled."

    print("Video AI skipped - Phase 1 Render deployment")

    # =====================================================
    # 💾 STEP 2 — SAVE VIDEO PATH + AI DATA IN DATABASE
    # =====================================================
    db = get_db()
    cur = db.cursor()

    cur.execute("""
        UPDATE courses 
        SET video_path = ?, video_topics = ?, video_summary = ?
        WHERE id = ?
    """, (
        "/static/videos/" + filename,
        topics,
        summary,
        course_id
    ))

    db.commit()
    db.close()

    # =====================================================
    # 🔙 RETURN TO COURSE PAGE
    # =====================================================
    return redirect(f"/teacher/course/{course_id}")

# ================= STUDENT WATCH COURSE VIDEO =================
@app.route("/course/<int:course_id>/video")
def course_video(course_id):

    # 🔐 Only student can access
    if session.get("role") != "student":
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    # ================= GET COURSE + TEACHER + AI DATA =================
    cur.execute("""
        SELECT 
            c.course_name,
            u.name,
            COALESCE(c.video_path, ''),
            COALESCE(c.video_topics, ''),
            COALESCE(c.video_summary, '')
        FROM courses c
        JOIN users u ON c.teacher_id = u.id
        WHERE c.id = ?
    """, (course_id,))

    course = cur.fetchone()

    # 🚨 Safety check (course deleted or invalid id)
    if not course:
        return "Course not found"

    course_name   = course[0]
    teacher_name  = course[1]
    video_path    = course[2]
    video_topics  = course[3]
    video_summary = course[4]

    return render_template(
        "course_video.html",
        course_id=course_id,
        course_name=course_name,
        teacher_name=teacher_name,
        video_path=video_path,
        video_topics=video_topics,
        video_summary=video_summary,
        name=session.get("name")
    )


@app.route("/teacher/course/<int:course_id>")
def teacher_course(course_id):

    if session.get("role") != "teacher":
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    # =====================================================
    # COURSE
    # =====================================================

    cur.execute(
        "SELECT course_name FROM courses WHERE id=?",
        (course_id,)
    )

    course = cur.fetchone()

    if not course:
        db.close()
        return "Course not found", 404

    # =====================================================
    # GET STUDENTS ENROLLED IN THIS COURSE
    # =====================================================

    cur.execute("""
        SELECT u.id, u.name
        FROM enrollments e
        JOIN users u
            ON e.student_id = u.id
        WHERE e.course_id = ?
        ORDER BY u.name
    """, (course_id,))

    student_rows = cur.fetchall()

    students = []

    # =====================================================
    # PROCESS EACH STUDENT
    # =====================================================

    for student_id, name in student_rows:

        # =================================================
        # VIDEO PROGRESS
        # =================================================

        cur.execute("""
            SELECT percentage
            FROM video_progress
            WHERE student_id = ?
              AND course_id = ?
            LIMIT 1
        """, (student_id, course_id))

        video_row = cur.fetchone()

        video_percentage = round(
            video_row[0] if video_row and video_row[0] is not None else 0,
            1
        )

        # =================================================
        # ATTENTION %
        # =================================================

        cur.execute("""
            SELECT
                COUNT(*),
                SUM(
                    CASE
                        WHEN state = 'ATTENTIVE'
                        THEN 1
                        ELSE 0
                    END
                )
            FROM attention_logs
            WHERE student_id = ?
              AND course_id = ?
        """, (student_id, course_id))

        attention_row = cur.fetchone()

        total_checks = attention_row[0] or 0
        attentive_checks = attention_row[1] or 0

        if total_checks > 0:

            attention_percentage = round(
                (attentive_checks / total_checks) * 100,
                1
            )

        else:

            attention_percentage = 0

        # =================================================
        # QUIZ ANALYTICS
        #
        # IMPORTANT:
        # Only quizzes ATTEMPTED by this student are counted.
        # =================================================

        cur.execute("""
            SELECT
                qr.quiz_id,
                qr.score
            FROM quiz_results qr
            JOIN quizzes q
                ON qr.quiz_id = q.id
            WHERE qr.student_id = ?
              AND q.course_id = ?
        """, (student_id, course_id))

        quiz_results = cur.fetchall()

        total_correct = 0
        total_questions = 0

        # -------------------------------------------------
        # Count only questions from attempted quizzes
        # -------------------------------------------------

        for quiz_id, score in quiz_results:

            # Correct answers obtained in this quiz
            total_correct += int(score or 0)

            # Actual number of questions in this quiz
            cur.execute("""
                SELECT COUNT(*)
                FROM quiz_questions
                WHERE quiz_id = ?
            """, (quiz_id,))

            question_row = cur.fetchone()

            question_count = (
                question_row[0]
                if question_row
                else 0
            )

            total_questions += question_count

        # =================================================
        # QUIZ SCORE
        # =================================================

        if total_questions > 0:

            quiz_percentage = round(
                (total_correct / total_questions) * 100,
                1
            )

            quiz_score = (
                f"{total_correct}/{total_questions}"
            )

        else:

            quiz_percentage = 0
            quiz_score = "N/A"

        # =================================================
        # STATUS
        # =================================================

        if attention_percentage >= 75:

            status = "Good Focus"

        elif attention_percentage >= 50:

            status = "Moderate Focus"

        else:

            status = "Needs Focus"

        # =================================================
        # ADD STUDENT
        # =================================================

        students.append({

            "student_id": student_id,

            "name": name,

            "video": video_percentage,

            "attention": attention_percentage,

            "quiz_score": quiz_score,

            "quiz_percentage": quiz_percentage,

            "status": status
        })

    db.close()

    # =====================================================
    # SEND TO TEMPLATE
    # =====================================================

    return render_template(
        "teacher_course.html",

        course_name=course[0],

        course_id=course_id,

        students=students
    )
@app.route("/teacher/course/<int:course_id>/add-quiz", methods=["POST"])
def add_quiz(course_id):
    db = get_db()
    cur = db.cursor()

    cur.execute("""
        INSERT INTO quizzes
        (course_id, question, a, b, c, d, correct, deadline)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        course_id,
        request.form["question"],
        request.form["a"],
        request.form["b"],
        request.form["c"],
        request.form["d"],
        request.form["correct"],
        request.form["deadline"]
    ))

    db.commit()
    return redirect(f"/teacher/course/{course_id}")

@app.route("/teacher/course/<int:course_id>/create-quiz", methods=["POST"])
def create_quiz(course_id):
    db = get_db()
    cur = db.cursor()

    cur.execute("""
        INSERT INTO quizzes (course_id, title, deadline)
        VALUES (?, ?, ?)
    """, (
        course_id,
        request.form["title"],
        request.form["deadline"]
    ))
    db.commit()

    quiz_id = cur.lastrowid
    return redirect(f"/teacher/quiz/{quiz_id}/add-questions")
@app.route("/teacher/quiz/<int:quiz_id>/add-questions", methods=["GET","POST"])
def add_questions(quiz_id):
    db = get_db()
    cur = db.cursor()

    # ➜ Get course_id of this quiz (needed for Back button)
    cur.execute("SELECT course_id FROM quizzes WHERE id=?", (quiz_id,))
    result = cur.fetchone()

    if not result:
        return "Quiz not found"

    course_id = result[0]

    # ➜ When teacher adds a question
    if request.method == "POST":
        cur.execute("""
            INSERT INTO quiz_questions
            (quiz_id, question, a, b, c, d, correct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            quiz_id,
            request.form["question"],
            request.form["a"],
            request.form["b"],
            request.form["c"],
            request.form["d"],
            request.form["correct"]
        ))
        db.commit()

    # ➜ Send course_id to template (for Back button)
    return render_template(
        "add_questions.html",
        quiz_id=quiz_id,
        course_id=course_id
    )

@app.route("/course/<int:course_id>/quiz/<int:quiz_id>", methods=["GET", "POST"])
def student_quiz(course_id, quiz_id):
    if session.get("role") != "student":
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    # Fetch quiz metadata
    cur.execute(
        "SELECT title, deadline FROM quizzes WHERE id = ?",
        (quiz_id,)
    )
    quiz = cur.fetchone()

    # Fetch ALL questions for this quiz
    cur.execute(
        "SELECT id, question, a, b, c, d FROM quiz_questions WHERE quiz_id = ?",
        (quiz_id,)
    )
    questions = cur.fetchall()

    if not questions:
        return "No questions added to this quiz yet.", 400

    # Handle submission
    if request.method == "POST":
        score = 0

        for q in questions:
            qid = q[0]
            selected = request.form.get(f"q{qid}")

            cur.execute(
                "SELECT correct FROM quiz_questions WHERE id = ?",
                (qid,)
            )
            correct = cur.fetchone()[0]

            if selected == correct:
                score += 1

        # Save result
        cur.execute("""
            INSERT INTO quiz_results (student_id, quiz_id, score)
            VALUES (?, ?, ?)
        """, (session["user_id"], quiz_id, score))

        db.commit()

        return redirect(f"/course/{course_id}")

    return render_template(
        "student_quiz.html",
        quiz=quiz,
        questions=questions,
        course_id=course_id
    )
from datetime import datetime
from flask import jsonify

@app.route("/attention-log", methods=["POST"])
def attention_log():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json

    db = get_db()
    cur = db.cursor()

    cur.execute("""
        INSERT INTO attention_logs
        (student_id, course_id, state, gaze, blink, head, face, emotion, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session["user_id"],
        data.get("course_id"),
        data.get("state"),
        data.get("gaze"),
        data.get("blink"),
        data.get("head"),
        data.get("face"),
        data.get("emotion"),
        datetime.now()
    ))

    db.commit()
    return jsonify({"status": "logged"})


# ==========================================================
# TEACHER GLOBAL FOCUS ANALYTICS (ALL COURSES)
# ==========================================================
@app.route("/teacher/focus-analytics")
def focus_analytics():

    if session.get("role") != "teacher":
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    # ======================================================
    # GET FOCUS % PER STUDENT (ALL COURSES COMBINED)
    # ======================================================
    cur.execute("""
        SELECT 
            u.name,
            ROUND(
                SUM(CASE WHEN al.state='ATTENTIVE' THEN 1 ELSE 0 END)
                *100.0 / COUNT(*), 1
            ) AS focus_percentage,
            COUNT(*) AS total_checks
        FROM attention_logs al
        JOIN users u ON al.student_id = u.id
        GROUP BY al.student_id
        ORDER BY focus_percentage DESC
    """)

    rows = cur.fetchall()

    # ===== Prepare graph arrays =====
    names = []
    focus_scores = []
    checks = []

    for r in rows:
        names.append(r[0])
        focus_scores.append(r[1] if r[1] else 0)
        checks.append(r[2])

    # ======================================================
    # COURSE WISE FOCUS TREND (AVG PER COURSE)
    # ======================================================
    cur.execute("""
        SELECT 
            c.course_name,
            ROUND(
                SUM(CASE WHEN al.state='ATTENTIVE' THEN 1 ELSE 0 END)
                *100.0 / COUNT(*), 1
            ) AS avg_focus
        FROM attention_logs al
        JOIN courses c ON al.course_id = c.id
        GROUP BY al.course_id
        ORDER BY avg_focus DESC
    """)

    course_rows = cur.fetchall()

    course_names = []
    course_focus = []

    for r in course_rows:
        course_names.append(r[0])
        course_focus.append(r[1] if r[1] else 0)

    # ======================================================
    # SEND GRAPH DATA TO TEMPLATE
    # ======================================================
    return render_template(
        "focus_analytics.html",
        names=names,
        focus_scores=focus_scores,
        checks=checks,
        course_names=course_names,
        course_focus=course_focus
    )

# ==========================================================
# TEACHER COURSE ANALYTICS (PER COURSE - GRAPH READY)
# ==========================================================
# ==========================================================
# TEACHER COURSE ANALYTICS
# ==========================================================
# ==========================================================
# TEACHER COURSE ANALYTICS
# ==========================================================
@app.route("/teacher/analytics/<int:course_id>")
def teacher_analytics(course_id):

    if session.get("role") != "teacher":
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    # =====================================================
    # GET UNIQUE STUDENTS ENROLLED IN THIS COURSE
    # DISTINCT prevents duplicate students appearing twice
    # =====================================================
    cur.execute("""
        SELECT DISTINCT u.id, u.name
        FROM enrollments e
        JOIN users u
            ON e.student_id = u.id
        WHERE e.course_id = ?
        ORDER BY u.name
    """, (course_id,))

    students = cur.fetchall()

    names = []
    video_scores = []
    attention_scores = []
    quiz_scores = []

    for student_id, student_name in students:

        names.append(student_name)

        # =================================================
        # VIDEO PROGRESS
        # =================================================
        cur.execute("""
            SELECT percentage
            FROM video_progress
            WHERE student_id = ?
              AND course_id = ?
            ORDER BY rowid DESC
            LIMIT 1
        """, (student_id, course_id))

        vp = cur.fetchone()

        if vp and vp[0] is not None:
            video_percentage = round(float(vp[0]), 1)
        else:
            video_percentage = 0.0

        video_scores.append(video_percentage)

        # =================================================
        # ATTENTION %
        # =================================================
        cur.execute("""
            SELECT
                COUNT(*) AS total_checks,
                SUM(
                    CASE
                        WHEN state = 'ATTENTIVE'
                        THEN 1
                        ELSE 0
                    END
                ) AS attentive_checks
            FROM attention_logs
            WHERE student_id = ?
              AND course_id = ?
        """, (student_id, course_id))

        att = cur.fetchone()

        total_checks = att[0] or 0
        attentive_checks = att[1] or 0

        if total_checks > 0:
            attention_percentage = round(
                (attentive_checks / total_checks) * 100,
                1
            )
        else:
            attention_percentage = 0.0

        attention_scores.append(attention_percentage)

        # =================================================
        # QUIZ SCORE
        #
        # IMPORTANT:
        # We calculate each attempted quiz separately.
        #
        # This prevents:
        #
        # score × number_of_questions
        #
        # from happening because of SQL JOIN multiplication.
        # =================================================

        cur.execute("""
            SELECT
                qr.quiz_id,
                qr.score
            FROM quiz_results qr
            JOIN quizzes q
                ON qr.quiz_id = q.id
            WHERE qr.student_id = ?
              AND q.course_id = ?
        """, (student_id, course_id))

        quiz_results = cur.fetchall()

        total_correct = 0
        total_questions = 0

        for quiz_id, score in quiz_results:

            # ---------------------------------------------
            # Get the actual number of questions
            # ONLY for this attempted quiz
            # ---------------------------------------------
            cur.execute("""
                SELECT COUNT(*)
                FROM quiz_questions
                WHERE quiz_id = ?
            """, (quiz_id,))

            question_result = cur.fetchone()

            quiz_question_count = question_result[0] or 0

            # ---------------------------------------------
            # Add this quiz ONCE
            # ---------------------------------------------
            total_correct += int(score or 0)
            total_questions += quiz_question_count

        # =================================================
        # FINAL QUIZ PERCENTAGE
        # =================================================
        if total_questions > 0:

            quiz_percentage = round(
                (total_correct / total_questions) * 100,
                1
            )

        else:

            quiz_percentage = 0.0

        quiz_scores.append(quiz_percentage)

    db.close()

    # =====================================================
    # SEND DATA TO ANALYTICS PAGE
    # =====================================================
    return render_template(
        "teacher_analytics.html",
        names=names,
        video_scores=video_scores,
        attention_scores=attention_scores,
        quiz_scores=quiz_scores,
        course_id=course_id
    )




# ==========================================================
# AI TEACHER CLASS INSIGHTS
# ==========================================================
@app.route("/teacher/ai-insight/<int:course_id>")
def teacher_ai_insight(course_id):

    if session.get("role") != "teacher":
        return jsonify({"error": "Unauthorized"}), 401

    db = get_db()
    cur = db.cursor()

    # =====================================================
    # COURSE NAME
    # =====================================================

    cur.execute("""
        SELECT course_name
        FROM courses
        WHERE id = ?
    """, (course_id,))

    course = cur.fetchone()

    if not course:
        db.close()
        return jsonify({"error": "Course not found"}), 404

    course_name = course[0]

    # =====================================================
    # STUDENTS
    # =====================================================

    cur.execute("""
        SELECT DISTINCT u.id, u.name
        FROM enrollments e
        JOIN users u
            ON e.student_id = u.id
        WHERE e.course_id = ?
        ORDER BY u.name
    """, (course_id,))

    students = cur.fetchall()

    student_data = []

    # =====================================================
    # COLLECT REAL STUDENT PERFORMANCE
    # =====================================================

    for student_id, student_name in students:

        # ---------------- VIDEO ----------------

        cur.execute("""
            SELECT percentage
            FROM video_progress
            WHERE student_id = ?
              AND course_id = ?
            ORDER BY rowid DESC
            LIMIT 1
        """, (student_id, course_id))

        video_row = cur.fetchone()

        video = round(
            float(video_row[0])
            if video_row and video_row[0] is not None
            else 0,
            1
        )

        # ---------------- ATTENTION ----------------

        cur.execute("""
            SELECT
                COUNT(*),
                SUM(
                    CASE
                        WHEN state = 'ATTENTIVE'
                        THEN 1
                        ELSE 0
                    END
                )
            FROM attention_logs
            WHERE student_id = ?
              AND course_id = ?
        """, (student_id, course_id))

        attention_row = cur.fetchone()

        total_checks = attention_row[0] or 0
        attentive_checks = attention_row[1] or 0

        if total_checks > 0:

            attention = round(
                (attentive_checks / total_checks) * 100,
                1
            )

        else:

            attention = 0

        # ---------------- QUIZ ----------------

        cur.execute("""
            SELECT
                qr.quiz_id,
                qr.score
            FROM quiz_results qr
            JOIN quizzes q
                ON qr.quiz_id = q.id
            WHERE qr.student_id = ?
              AND q.course_id = ?
        """, (student_id, course_id))

        quiz_results = cur.fetchall()

        total_correct = 0
        total_questions = 0

        for quiz_id, score in quiz_results:

            cur.execute("""
                SELECT COUNT(*)
                FROM quiz_questions
                WHERE quiz_id = ?
            """, (quiz_id,))

            question_count = cur.fetchone()[0] or 0

            total_correct += int(score or 0)
            total_questions += question_count

        if total_questions > 0:

            quiz = round(
                (total_correct / total_questions) * 100,
                1
            )

        else:

            quiz = 0

        student_data.append({
            "name": student_name,
            "video": video,
            "attention": attention,
            "quiz": quiz
        })

    # =====================================================
    # CLASS AVERAGES
    # =====================================================

    total_students = len(student_data)

    if total_students > 0:

        avg_video = round(
            sum(s["video"] for s in student_data)
            / total_students,
            1
        )

        avg_attention = round(
            sum(s["attention"] for s in student_data)
            / total_students,
            1
        )

        avg_quiz = round(
            sum(s["quiz"] for s in student_data)
            / total_students,
            1
        )

    else:

        avg_video = 0
        avg_attention = 0
        avg_quiz = 0

    # =====================================================
    # STUDENTS NEEDING ATTENTION
    # =====================================================

    students_needing_attention = [
        s for s in student_data
        if s["attention"] < 50
        or s["quiz"] < 50
    ]

    # Sort by lowest attention
    students_needing_attention.sort(
        key=lambda x: x["attention"]
    )

    # Keep only the most relevant students
    students_needing_attention = (
        students_needing_attention[:10]
    )

    db.close()

    # =====================================================
    # PREPARE DATA FOR GROQ
    # =====================================================

    performance_text = "\n".join(
        [
            f"- {s['name']}: "
            f"Video {s['video']}%, "
            f"Attention {s['attention']}%, "
            f"Quiz {s['quiz']}%"
            for s in student_data
        ]
    )

    struggling_text = "\n".join(
        [
            f"- {s['name']}: "
            f"Attention {s['attention']}%, "
            f"Quiz {s['quiz']}%"
            for s in students_needing_attention
        ]
    )

    # =====================================================
    # GROQ PROMPT
    # =====================================================

    prompt = f"""
You are an AI teaching assistant inside a college LMS.

Course:
{course_name}

Number of enrolled students:
{total_students}

Class averages:
Video completion: {avg_video}%
Attention: {avg_attention}%
Quiz performance: {avg_quiz}%

Student performance:

{performance_text}

Students who may need additional attention:

{struggling_text if struggling_text else "No students currently show significantly low performance."}

Analyze this class performance and provide useful guidance to the teacher.

Return ONLY valid JSON with exactly these fields:

{{
    "summary": "...",
    "attention": "...",
    "performance": "...",
    "recommendations": [
        "...",
        "...",
        "..."
    ]
}}

Rules:

- Use only the information provided.
- Do not invent lecture topics, modules, chapters, or concepts.
- Do not invent student problems that are not supported by the data.
- Mention actual percentages when useful.
- Keep the summary concise.
- Identify the most important class-level issue.
- Give 2 or 3 practical recommendations.
- Use professional but natural language.
- Do not use markdown.
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "system",
                    "content":
                        "You are an educational analytics assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.4,

            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        # Remove accidental markdown fences
        content = content.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

        data = json.loads(content)

        return jsonify({
            "success": True,
            "course": course_name,
            "students": total_students,
            "avg_video": avg_video,
            "avg_attention": avg_attention,
            "avg_quiz": avg_quiz,
            "summary": data.get(
                "summary",
                "Class performance has been analyzed."
            ),
            "attention": data.get(
                "attention",
                "Review students with lower attention scores."
            ),
            "performance": data.get(
                "performance",
                "Continue monitoring quiz and video performance."
            ),
            "recommendations": data.get(
                "recommendations",
                [
                    "Review students with lower attention.",
                    "Encourage consistent lecture participation.",
                    "Use revision activities where necessary."
                ]
            )
        })

    except Exception as e:

        print("AI Teacher Insight Error:", e)

        # =================================================
        # FALLBACK
        # =================================================

        if avg_attention < avg_quiz:

            summary = (
                f"Quiz performance ({avg_quiz}%) is currently "
                f"higher than class attention ({avg_attention}%). "
                "The main area to monitor is lecture engagement."
            )

        else:

            summary = (
                f"Class attention is {avg_attention}% and "
                f"quiz performance is {avg_quiz}%. "
                "Continue monitoring both learning engagement "
                "and assessment performance."
            )

        return jsonify({
            "success": True,
            "course": course_name,
            "students": total_students,
            "avg_video": avg_video,
            "avg_attention": avg_attention,
            "avg_quiz": avg_quiz,
            "summary": summary,
            "attention":
                "Monitor students with consistently low attention.",
            "performance":
                "Use quiz results together with video and attention data.",
            "recommendations": [
                "Review students with low attention scores.",
                "Encourage students to stay engaged during lectures.",
                "Consider a short revision activity when quiz performance is low."
            ]
        })




# ==========================================
# SAVE ATTENTION DATA FROM STUDENT VIDEO PAGE
# ==========================================
@app.route("/save-attention", methods=["POST"])
def save_attention():

    if "user_id" not in session:
        return "Not logged in", 401

    data = request.get_json()

    student_id = session["user_id"]
    course_id = data["course_id"]

    db = get_db()
    cur = db.cursor()

    cur.execute("""
    INSERT INTO attention_logs
    (student_id, course_id, state, gaze, blink, head, face, emotion, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
""", (
    student_id,
    course_id,
    data.get("state"),
    data.get("gaze"),
    data.get("blink"),
    data.get("head"),
    data.get("face"),      # 🔥 SAFE
    data.get("emotion")
))

    db.commit()
    return {"status": "saved"}

import random
import requests
from flask import request, jsonify

# =====================================================
# MICRO-INTERVENTION ENGINE (SIMPLIFIED VERSION)
# =====================================================

@app.route("/generate-summary", methods=["POST"])
def generate_summary():
    try:
        data = request.get_json()
        course_id = data.get("course_id")

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        # Get video path from DB
        cur.execute("SELECT video_path FROM courses WHERE id=?", (course_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Course not found"})

        video_path = row[0]

        # Convert relative path to absolute path
        
        video_path = video_path.lstrip("/")
        video_path = os.path.join(app.root_path, video_path)
        
        print("FINAL VIDEO PATH:", video_path)
        # 1️⃣ Extract audio
        audio_path = video_path.replace(".mp4", ".wav")

        
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()

        # 2️⃣ Transcribe using Whisper
        import whisper
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]

        # 3️⃣ Concept-based summary
        prompt = f"""
You are a computer science professor.

From this transcript:

{transcript}

Identify the main topic and explain it clearly in 5–6 lines.
Do not mention the lecture.
Focus only on the core concept.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        summary = response.choices[0].message.content.strip()

        return jsonify({"summary": summary})

    except Exception as e:
        print("Summary Error:", e)
        return jsonify({"error": "Failed"})
    
#Motivational message
@app.route("/motivate", methods=["POST"])
def motivate():
    try:
        # 🔐 Put your NEW regenerated Groq key here
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        prompt = """
You are an energetic AI learning coach inside an online class.

The student is distracted.

Generate ONE short motivational sentence that:
• Encourages the student to refocus immediately
• Sounds confident and powerful
• Uses EXACTLY 2 motivational emojis
• Maximum 10 words
• No extra explanation
• No quotes
• No hashtags

The sentence MUST include emojis.
Return only the plain sentence without quotes.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0   # Higher creativity
        )

        message = response.choices[0].message.content.strip()

        return jsonify({
            "msg": message
        })

    except Exception as e:
        print("Motivation Error:", e)

        return jsonify({
            "msg": "Stay focused and keep going!"
        })

#Quiz generation route
@app.route("/generate-quiz", methods=["POST"])
def generate_quiz():
    try:
        data = request.get_json()
        summary = data.get("summary")

        prompt = f"""
You are an AI professor.

Based on this topic summary:

{summary}

Generate exactly 3 multiple-choice questions.

Rules:
- Each question must have 4 options
- Only one correct answer
- Return STRICT JSON format
- No explanations

Format:

{{
  "questions": [
    {{
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "answer": "Correct option text"
    }}
  ]
}}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        content = response.choices[0].message.content.strip()

        quiz_data = json.loads(content)

        # Save correct answers in session
        session["correct_answers"] = [q["answer"] for q in quiz_data["questions"]]

        return jsonify({"questions": quiz_data["questions"]})

    except Exception as e:
        print("Quiz Error:", e)
        return jsonify({"error": "Failed to generate quiz"})
    

#submitting quiz
@app.route("/submit-quiz", methods=["POST"])
def submit_quiz():
    data = request.get_json()
    user_answers = data.get("answers")

    correct_answers = session.get("correct_answers", [])

    score = 0
    for i in range(len(correct_answers)):
        if user_answers[i] == correct_answers[i]:
            score += 1

    return jsonify({"score": score})
# ==========================================================
# STUDENT ANALYTICS DASHBOARD (WITH CHART DATA)
# ==========================================================
@app.route("/student/analytics")
def student_analytics():

    if session.get("role") != "student":
        return redirect("/")

    student_id = session["user_id"]

    db = get_db()
    cur = db.cursor()

    # ==========================================================
    # VIDEO PROGRESS
    # ==========================================================
    cur.execute("""
        SELECT ROUND(AVG(percentage), 2)
        FROM video_progress
        WHERE student_id = ?
    """, (student_id,))

    avg_video = cur.fetchone()[0] or 0


    # ==========================================================
    # ATTENTION
    # ==========================================================
    cur.execute("""
        SELECT
            ROUND(
                CASE
                    WHEN COUNT(*) = 0 THEN 0
                    ELSE
                        SUM(
                            CASE
                                WHEN state = 'ATTENTIVE'
                                THEN 1
                                ELSE 0
                            END
                        ) * 100.0 / COUNT(*)
                END,
                2
            )
        FROM attention_logs
        WHERE student_id = ?
    """, (student_id,))

    row = cur.fetchone()

    avg_attention = (
        row[0]
        if row and row[0] is not None
        else 0
    )


    # ==========================================================
    # OVERALL QUIZ SCORE
    #
    # ONLY ATTEMPTED QUIZZES ARE INCLUDED
    # ==========================================================

    # Get every quiz actually attempted by this student
    cur.execute("""
        SELECT
            qr.quiz_id,
            qr.score
        FROM quiz_results qr
        WHERE qr.student_id = ?
    """, (student_id,))

    quiz_results = cur.fetchall()

    total_correct = 0
    total_questions = 0

    for quiz_id, score in quiz_results:

        # Get actual number of questions in this quiz
        cur.execute("""
            SELECT COUNT(*)
            FROM quiz_questions
            WHERE quiz_id = ?
        """, (quiz_id,))

        result = cur.fetchone()

        question_count = result[0] or 0

        # Add this quiz ONCE
        total_correct += int(score or 0)
        total_questions += question_count


    if total_questions > 0:

        avg_quiz = round(
            (total_correct / total_questions) * 100,
            2
        )

    else:

        avg_quiz = 0


    # ==========================================================
    # COURSE-WISE ANALYTICS
    # ==========================================================

    cur.execute("""
        SELECT DISTINCT
            c.id,
            c.course_name
        FROM enrollments e
        JOIN courses c
            ON e.course_id = c.id
        WHERE e.student_id = ?
        ORDER BY c.course_name
    """, (student_id,))

    course_list = cur.fetchall()

    courses = []

    for course_id, course_name in course_list:

        # ------------------------------------------------------
        # VIDEO
        # ------------------------------------------------------
        cur.execute("""
            SELECT ROUND(AVG(percentage), 2)
            FROM video_progress
            WHERE student_id = ?
              AND course_id = ?
        """, (student_id, course_id))

        video_result = cur.fetchone()

        video_percentage = (
            video_result[0]
            if video_result and video_result[0] is not None
            else 0
        )


        # ------------------------------------------------------
        # ATTENTION
        # ------------------------------------------------------
        cur.execute("""
            SELECT
                COUNT(*),
                SUM(
                    CASE
                        WHEN state = 'ATTENTIVE'
                        THEN 1
                        ELSE 0
                    END
                )
            FROM attention_logs
            WHERE student_id = ?
              AND course_id = ?
        """, (student_id, course_id))

        attention_result = cur.fetchone()

        total_checks = attention_result[0] or 0
        attentive_checks = attention_result[1] or 0

        if total_checks > 0:

            attention_percentage = round(
                (attentive_checks / total_checks) * 100,
                2
            )

        else:

            attention_percentage = 0


        # ------------------------------------------------------
        # QUIZ SCORE FOR THIS COURSE
        #
        # ONLY ATTEMPTED QUIZZES
        # ------------------------------------------------------

        cur.execute("""
            SELECT
                qr.quiz_id,
                qr.score
            FROM quiz_results qr
            JOIN quizzes q
                ON qr.quiz_id = q.id
            WHERE qr.student_id = ?
              AND q.course_id = ?
        """, (student_id, course_id))

        course_quiz_results = cur.fetchall()

        course_correct = 0
        course_questions = 0

        for quiz_id, score in course_quiz_results:

            cur.execute("""
                SELECT COUNT(*)
                FROM quiz_questions
                WHERE quiz_id = ?
            """, (quiz_id,))

            question_result = cur.fetchone()

            question_count = question_result[0] or 0

            course_correct += int(score or 0)
            course_questions += question_count


        if course_questions > 0:

            course_quiz_percentage = round(
                (course_correct / course_questions) * 100,
                2
            )

        else:

            course_quiz_percentage = 0


        courses.append((
            course_name,
            round(video_percentage, 2),
            round(course_quiz_percentage, 2),
            round(attention_percentage, 2)
        ))


    # ==========================================================
    # PREPARE CHART DATA
    # ==========================================================

    course_names = []
    video_data = []
    quiz_data = []
    attention_data = []

    for course in courses:

        course_names.append(
            course[0] if course[0] else "Course"
        )

        video_data.append(
            float(course[1] or 0)
        )

        quiz_data.append(
            float(course[2] or 0)
        )

        attention_data.append(
            float(course[3] or 0)
        )


    # ==========================================================
    # NO DATA PROTECTION
    # ==========================================================

    if len(course_names) == 0:

        course_names = ["No Data"]
        video_data = [0]
        quiz_data = [0]
        attention_data = [0]


    # ==========================================================
    # SEND DATA TO TEMPLATE
    # ==========================================================

    return render_template(
        "student_analytics.html",

        name=session["name"],

        avg_video=round(avg_video, 2),

        avg_attention=round(avg_attention, 2),

        avg_quiz=round(avg_quiz, 2),

        courses=courses,

        course_names=course_names,

        video_data=video_data,

        attention_data=attention_data,

        quiz_data=quiz_data
    )
# ==========================================================
# PARENT DASHBOARD ANALYTICS (FINAL)
# ==========================================================
@app.route("/parent/child-analytics/<int:student_id>")
def parent_child_analytics(student_id):

    if session.get("role") != "parent":
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    # ==========================================================
    # STUDENT NAME
    # ==========================================================
    cur.execute(
        "SELECT name FROM users WHERE id=?",
        (student_id,)
    )

    student_row = cur.fetchone()

    if not student_row:
        db.close()
        return "Student not found", 404

    student_name = student_row[0]


    # ==========================================================
    # OVERALL VIDEO
    # ==========================================================
    cur.execute("""
        SELECT ROUND(AVG(percentage), 1)
        FROM video_progress
        WHERE student_id=?
    """, (student_id,))

    avg_video = cur.fetchone()[0] or 0


    # ==========================================================
    # OVERALL ATTENTION
    # ==========================================================
    cur.execute("""
        SELECT
            CASE
                WHEN COUNT(*) = 0 THEN 0
                ELSE
                    SUM(
                        CASE
                            WHEN state='ATTENTIVE'
                            THEN 1
                            ELSE 0
                        END
                    ) * 100.0 / COUNT(*)
            END
        FROM attention_logs
        WHERE student_id=?
    """, (student_id,))

    avg_attention = cur.fetchone()[0] or 0
    avg_attention = round(avg_attention, 1)


    # ==========================================================
    # OVERALL QUIZ SCORE
    #
    # ONLY ATTEMPTED QUIZZES
    # ==========================================================

    cur.execute("""
        SELECT
            qr.quiz_id,
            qr.score
        FROM quiz_results qr
        WHERE qr.student_id=?
    """, (student_id,))

    quiz_results = cur.fetchall()

    total_correct = 0
    total_questions = 0

    for quiz_id, score in quiz_results:

        # Actual number of questions in this quiz
        cur.execute("""
            SELECT COUNT(*)
            FROM quiz_questions
            WHERE quiz_id=?
        """, (quiz_id,))

        question_count = cur.fetchone()[0] or 0

        # Add this attempted quiz ONCE
        total_correct += int(score or 0)
        total_questions += question_count


    if total_questions > 0:

        avg_quiz = round(
            (total_correct / total_questions) * 100,
            1
        )

    else:

        avg_quiz = 0


    # ==========================================================
    # COURSE-WISE ANALYTICS
    # ==========================================================

    cur.execute("""
        SELECT DISTINCT
            c.id,
            c.course_name
        FROM enrollments e
        JOIN courses c
            ON e.course_id=c.id
        WHERE e.student_id=?
        ORDER BY c.course_name
    """, (student_id,))

    rows = cur.fetchall()


    course_names = []
    video_data = []
    attention_data = []
    quiz_data = []


    for course_id, course_name in rows:

        course_names.append(course_name)


        # ======================================================
        # VIDEO
        # ======================================================

        cur.execute("""
            SELECT ROUND(AVG(percentage),1)
            FROM video_progress
            WHERE student_id=?
              AND course_id=?
        """, (student_id, course_id))

        video_result = cur.fetchone()

        video_percentage = (
            video_result[0]
            if video_result and video_result[0] is not None
            else 0
        )

        video_data.append(video_percentage)


        # ======================================================
        # ATTENTION
        # ======================================================

        cur.execute("""
            SELECT
                COUNT(*),
                SUM(
                    CASE
                        WHEN state='ATTENTIVE'
                        THEN 1
                        ELSE 0
                    END
                )
            FROM attention_logs
            WHERE student_id=?
              AND course_id=?
        """, (student_id, course_id))

        attention_result = cur.fetchone()

        total_checks = attention_result[0] or 0
        attentive_checks = attention_result[1] or 0

        if total_checks > 0:

            attention_percentage = round(
                (attentive_checks / total_checks) * 100,
                1
            )

        else:

            attention_percentage = 0

        attention_data.append(attention_percentage)


        # ======================================================
        # QUIZ SCORE FOR THIS COURSE
        #
        # ONLY ATTEMPTED QUIZZES
        # ======================================================

        cur.execute("""
            SELECT
                qr.quiz_id,
                qr.score
            FROM quiz_results qr
            JOIN quizzes q
                ON qr.quiz_id=q.id
            WHERE qr.student_id=?
              AND q.course_id=?
        """, (student_id, course_id))

        course_quiz_results = cur.fetchall()

        course_correct = 0
        course_questions = 0


        for quiz_id, score in course_quiz_results:

            # Count actual questions for this quiz
            cur.execute("""
                SELECT COUNT(*)
                FROM quiz_questions
                WHERE quiz_id=?
            """, (quiz_id,))

            question_count = cur.fetchone()[0] or 0

            course_correct += int(score or 0)
            course_questions += question_count


        if course_questions > 0:

            course_quiz_percentage = round(
                (course_correct / course_questions) * 100,
                1
            )

        else:

            course_quiz_percentage = 0


        quiz_data.append(course_quiz_percentage)


    # ==========================================================
    # SEND TO PARENT ANALYTICS
    # ==========================================================

    db.close()

    return render_template(
        "parent_analytics.html",

        student_name=student_name,

        avg_video=round(avg_video, 1),

        avg_attention=avg_attention,

        avg_quiz=avg_quiz,

        course_names=course_names,

        video_data=video_data,

        attention_data=attention_data,

        quiz_data=quiz_data
    )
init_db()      # create tables
migrate_db()   # update old tables
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )

