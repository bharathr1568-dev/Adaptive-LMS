# ============================================================
# VIDEO AI — Speech → Transcript → Topics + Summary
# Works with Grok API OR Smart Offline AI (Never fails)
# ============================================================

import whisper
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import re

# ============================================================
# 🔐 FORCE LOAD .env EVEN AFTER FLASK AUTO-RELOAD
# ============================================================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


# ============================================================
# 🎤 LOAD WHISPER MODEL (cached after first run)
# ============================================================
model = whisper.load_model("base")


# ============================================================
# 🧠 SMART OFFLINE AI (REAL LECTURE SUMMARIZER)
# ============================================================
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def offline_ai_analysis(transcript):

    print("⚠️ Using Smart Offline Lecture AI...")

    # ---------- CLEAN TEXT ----------
    transcript = transcript.replace("\n", " ")
    transcript = re.sub(r'\s+', ' ', transcript)

    if len(transcript) < 30:
        return "• No speech detected", "No speech found in video."

    # ---------- NLP SUMMARY ----------
    parser = PlaintextParser.from_string(transcript, Tokenizer("english"))
    summarizer = LsaSummarizer()

    summary_sentences = summarizer(parser.document, 4)
    summary = " ".join(str(s) for s in summary_sentences)

    # ---------- TOPIC EXTRACTION ----------
    words = transcript.lower().split()

    important_words = [
        "algorithm","sorting","merge","divide","linked","list",
        "array","stack","queue","tree","graph","search",
        "complexity","recursion","performance","data","structure"
    ]

    keywords = []
    for w in words:
        if w in important_words and w not in keywords:
            keywords.append(w)

    if len(keywords) == 0:
        keywords = ["Core Concepts", "Examples", "Applications"]

    topics = "\n".join([f"• {k.title()}" for k in keywords[:6]])

    return topics, summary


# ============================================================
# 🎬 MAIN FUNCTION CALLED AFTER VIDEO UPLOAD
# ============================================================
def process_video(video_path):

    # ========================================================
    # 🎤 STEP 1 — TRANSCRIBE VIDEO
    # ========================================================
    try:
        print("\n🎤 Extracting speech from video...")
        result = model.transcribe(video_path)
        transcript = result["text"]

        print("\n📝 TRANSCRIPT EXTRACTED (preview):")
        print(transcript[:400])

    except Exception as e:
        print("❌ Whisper failed:", e)
        return offline_ai_analysis("")

    # ========================================================
    # 🔑 STEP 2 — GET GROK KEY
    # ========================================================
    GROK_API_KEY = os.getenv("GROK_API_KEY")

    if not GROK_API_KEY:
        print("⚠️ GROK_API_KEY missing → Using Offline AI")
        return offline_ai_analysis(transcript)

    # ========================================================
    # 🤖 STEP 3 — TRY GROK (OPTIONAL)
    # ========================================================
    try:
        prompt = f"""
You are an AI Teaching Assistant.

From this lecture transcript:
1) Write a short lecture summary (5 lines)
2) Extract key lecture topics (bullet points)

Transcript:
{transcript}
"""

        url = "https://api.x.ai/v1/responses"

        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "grok",   # if not enabled → fallback auto triggers
            "input": prompt,
            "max_output_tokens": 400
        }

        print("\n🤖 Sending transcript to GROK...")
        res = requests.post(url, headers=headers, json=payload)
        data = res.json()

        print("\n🔎 GROK RAW RESPONSE:")
        print(data)

        if "output" not in data:
            raise Exception("Grok model not enabled")

        ai_text = data["output"][0]["content"][0]["text"]

        # Try splitting topics & summary
        if "•" in ai_text:
            parts = ai_text.split("•", 1)
            summary = parts[0]
            topics = "•" + parts[1]
        else:
            summary = ai_text
            topics = ai_text

        print("\n✅ GROK SUCCESS")
        return topics.strip(), summary.strip()

    # ========================================================
    # 🧠 STEP 4 — FALLBACK (ALWAYS WORKS)
    # ========================================================
    except Exception as e:
        print("⚠️ Grok failed → switching to Offline AI:", e)
        return offline_ai_analysis(transcript)
