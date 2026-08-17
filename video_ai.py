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
# 🔐 LOAD ENVIRONMENT VARIABLES
# ============================================================

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


# ============================================================
# 🎤 WHISPER MODEL
#
# Do NOT load Whisper when the Flask server starts.
# It will be loaded only when video processing is requested.
# This reduces Render startup memory usage.
# ============================================================

model = None


def get_whisper_model():
    """
    Load Whisper only when it is actually needed.
    Keep the model cached after the first load.
    """

    global model

    if model is None:
        print("🎤 Loading Whisper base model...")

        model = whisper.load_model("base")

        print("✅ Whisper model loaded successfully.")

    return model


# ============================================================
# 🧠 SMART OFFLINE AI
# ============================================================

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer


def offline_ai_analysis(transcript):

    print("⚠️ Using Smart Offline Lecture AI...")

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    transcript = transcript.replace("\n", " ")

    transcript = re.sub(
        r"\s+",
        " ",
        transcript
    )

    # --------------------------------------------------------
    # CHECK EMPTY TRANSCRIPT
    # --------------------------------------------------------

    if len(transcript.strip()) < 30:

        return (
            "• No speech detected",
            "No speech found in video."
        )

    # --------------------------------------------------------
    # NLP SUMMARY
    # --------------------------------------------------------

    try:

        parser = PlaintextParser.from_string(
            transcript,
            Tokenizer("english")
        )

        summarizer = LsaSummarizer()

        summary_sentences = summarizer(
            parser.document,
            4
        )

        summary = " ".join(
            str(sentence)
            for sentence in summary_sentences
        )

    except Exception as e:

        print(
            "⚠️ Offline summarization failed:",
            e
        )

        summary = transcript[:500]

    # --------------------------------------------------------
    # TOPIC EXTRACTION
    # --------------------------------------------------------

    words = transcript.lower().split()

    important_words = [
        "algorithm",
        "sorting",
        "merge",
        "divide",
        "linked",
        "list",
        "array",
        "stack",
        "queue",
        "tree",
        "graph",
        "search",
        "complexity",
        "recursion",
        "performance",
        "data",
        "structure"
    ]

    keywords = []

    for word in words:

        # Remove basic punctuation
        clean_word = re.sub(
            r"[^a-zA-Z]",
            "",
            word
        )

        if (
            clean_word in important_words
            and clean_word not in keywords
        ):
            keywords.append(clean_word)

    # --------------------------------------------------------
    # DEFAULT TOPICS
    # --------------------------------------------------------

    if len(keywords) == 0:

        keywords = [
            "Core Concepts",
            "Examples",
            "Applications"
        ]

    # --------------------------------------------------------
    # CREATE TOPICS
    # --------------------------------------------------------

    topics = "\n".join(
        f"• {keyword.title()}"
        for keyword in keywords[:6]
    )

    return topics, summary


# ============================================================
# 🎬 MAIN VIDEO PROCESSING FUNCTION
# ============================================================

def process_video(video_path):

    print("\n========================================")
    print("🎬 VIDEO AI PROCESSING STARTED")
    print("========================================")

    # ========================================================
    # STEP 1 — LOAD WHISPER ONLY WHEN REQUIRED
    # ========================================================

    try:

        whisper_model = get_whisper_model()

    except Exception as e:

        print(
            "❌ Failed to load Whisper:",
            e
        )

        return (
            "• Whisper unavailable",
            "Unable to load the speech recognition model."
        )

    # ========================================================
    # STEP 2 — TRANSCRIBE VIDEO
    # ========================================================

    try:

        print(
            "\n🎤 Extracting speech from video..."
        )

        result = whisper_model.transcribe(
            video_path
        )

        transcript = result.get(
            "text",
            ""
        )

        print(
            "\n📝 TRANSCRIPT EXTRACTED (preview):"
        )

        print(
            transcript[:400]
        )

    except Exception as e:

        print(
            "❌ Whisper transcription failed:",
            e
        )

        return offline_ai_analysis("")

    # ========================================================
    # STEP 3 — CHECK TRANSCRIPT
    # ========================================================

    if not transcript.strip():

        print(
            "⚠️ No transcript detected."
        )

        return offline_ai_analysis("")

    # ========================================================
    # STEP 4 — GET GROK API KEY
    # ========================================================

    GROK_API_KEY = os.getenv(
        "GROK_API_KEY"
    )

    # ========================================================
    # STEP 5 — FALLBACK TO OFFLINE AI
    # ========================================================

    if not GROK_API_KEY:

        print(
            "⚠️ GROK_API_KEY missing → "
            "Using Offline AI"
        )

        return offline_ai_analysis(
            transcript
        )

    # ========================================================
    # STEP 6 — TRY GROK
    # ========================================================

    try:

        prompt = f"""
You are an AI Teaching Assistant.

From this lecture transcript:

1) Write a short lecture summary in approximately 5 lines.
2) Extract the key lecture topics as bullet points.

Transcript:

{transcript}
"""

        url = (
            "https://api.x.ai/v1/responses"
        )

        headers = {

            "Authorization":
                f"Bearer {GROK_API_KEY}",

            "Content-Type":
                "application/json"
        }

        payload = {

            "model": "grok",

            "input": prompt,

            "max_output_tokens": 400
        }

        print(
            "\n🤖 Sending transcript to GROK..."
        )

        response = requests.post(

            url,

            headers=headers,

            json=payload,

            timeout=60
        )

        # ----------------------------------------------------
        # CHECK HTTP RESPONSE
        # ----------------------------------------------------

        response.raise_for_status()

        data = response.json()

        print(
            "\n🔎 GROK RESPONSE RECEIVED"
        )

        # ----------------------------------------------------
        # CHECK OUTPUT
        # ----------------------------------------------------

        if "output" not in data:

            raise Exception(
                "Grok response does not contain output."
            )

        output = data["output"]

        if not output:

            raise Exception(
                "Grok returned empty output."
            )

        # ----------------------------------------------------
        # EXTRACT TEXT
        # ----------------------------------------------------

        ai_text = ""

        try:

            ai_text = (
                output[0]
                ["content"][0]
                ["text"]
            )

        except (
            KeyError,
            IndexError,
            TypeError
        ):

            # Some API responses can have
            # a different structure.

            if isinstance(
                output,
                list
            ):

                ai_text = str(
                    output
                )

            else:

                ai_text = str(
                    output
                )

        ai_text = ai_text.strip()

        if not ai_text:

            raise Exception(
                "Grok returned empty text."
            )

        # ====================================================
        # SPLIT SUMMARY AND TOPICS
        # ====================================================

        if "•" in ai_text:

            parts = ai_text.split(
                "•",
                1
            )

            summary = parts[0].strip()

            topics = (
                "•" +
                parts[1]
            ).strip()

        else:

            summary = ai_text

            topics = ai_text

        print(
            "\n✅ GROK SUCCESS"
        )

        return (
            topics,
            summary
        )

    # ========================================================
    # STEP 7 — GROK FAILURE → OFFLINE AI
    # ========================================================

    except Exception as e:

        print(
            "\n⚠️ Grok failed → "
            "switching to Offline AI:"
        )

        print(e)

        return offline_ai_analysis(
            transcript
        )