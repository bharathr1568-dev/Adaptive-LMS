// =====================================================
// AI ATTENTION TRACKER + AUTO VIDEO CONTROL (STABLE)
// =====================================================

let distractionSeconds = 0;
let videoPausedByAI = false;
let quizActive = false;
const DISTRACTION_LIMIT = 5; // seconds before pause
let trackingInterval = null;

// =====================================================
// POPUP NOTIFICATION
// =====================================================
function showMotivation(message) {
    const box = document.getElementById("motivationBox");
    if (!box) return;

    box.innerText = message;
    box.classList.add("show");

    setTimeout(() => {
        box.classList.remove("show");
    }, 5000);
}

// =====================================================
// CALL GROK BACKEND FOR MESSAGE
// =====================================================
async function getAIMotivation(state) {
    try {
        const res = await fetch("http://127.0.0.1:5002/motivate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ state: state })
        });

        const data = await res.json();
        return data.msg;

    } catch (err) {
        console.log("Motivation API error:", err);
        return "Stay focused! Resume learning 💪";
    }
}

// =====================================================
// UPDATE STATUS PANEL
// =====================================================
function updateLiveState(data) {
    const box = document.getElementById("attentionStatus");
    if (!box) return;

    box.innerHTML =
        "🧠 <b>State:</b> " + data.state +
        "<br>👁️ <b>Gaze:</b> " + data.gaze +
        "<br>😉 <b>Blink:</b> " + data.blink +
        "<br>🧍 <b>Head:</b> " + data.head +
        "<br>🙂 <b>Emotion:</b> " + data.emotion;
}

// =====================================================
// NORMALIZE STATE (IMPORTANT)
// Handles all backend variations safely
// =====================================================
function normalizeState(state) {
    if (!state) return "UNKNOWN";

    state = state.toUpperCase().trim();

    if (state.includes("DISTRACT")) return "DISTRACTED";
    if (state.includes("NO")) return "NO_STUDENT";
    if (state.includes("ATTENT")) return "ATTENTIVE";

    return state;
}

// =====================================================
// MAIN TRACKER
// =====================================================
function startAttentionTracking(courseId) {

    const video = document.getElementById("video");

    if (!video) {
        console.error("❌ Video element not found!");
        return;
    }

    // prevent multiple intervals
    if (trackingInterval) {
        clearInterval(trackingInterval);
    }

    trackingInterval = setInterval(async () => {

        try {
            const res = await fetch("http://127.0.0.1:5000/attention-state");
            const data = await res.json();

            const state = normalizeState(data.state);

            updateLiveState(data);
            // 🔥 SAVE ATTENTION TO DATABASE
fetch("/save-attention", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        course_id: courseId,
        state: state,
        gaze: data.gaze,
        blink: data.blink,
        head: data.head,
        face: data.face,
        emotion: data.emotion
    })
}).catch(err => console.log("Save attention error:", err));

            console.log("Current State:", state);
            console.log("Distraction Seconds:", distractionSeconds);

            // =====================================================
            // DISTRACTION TIMER
            // =====================================================
            if (state === "DISTRACTED" || state === "NO_STUDENT") {
                distractionSeconds += 1;
            } else {
                distractionSeconds = 0;
            }

            // =====================================================
            // AUTO PAUSE
            // =====================================================
            // =====================================================
// AUTO PAUSE + SUMMARY TRIGGER
// =====================================================
            let summaryTriggered = false;

if (distractionSeconds >= DISTRACTION_LIMIT && !videoPausedByAI && !summaryTriggered
    &&!quizActive
) {

    video.pause();
    videoPausedByAI = true;
    summaryTriggered = true;
    quizActive = true; 

    // 1️⃣ Motivation
    const msg = await getAIMotivation(state);
    showMotivation(msg);

    // 2️⃣ Generate Summary
    try {
        const summaryRes = await fetch("http://127.0.0.1:5002/generate-summary", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ course_id: courseId })
        });

        const summaryData = await summaryRes.json();

        if (summaryData.summary) {
            displaySummary(summaryData.summary);
        }

    } catch (err) {
        console.log("Summary error:", err);
    }
}

            // =====================================================
            // AUTO RESUME
            // =====================================================
            if ( videoPausedByAI && !quizActive) {

                console.log("▶ Resuming video - student attentive again");

                video.play();
                videoPausedByAI = false;
                distractionSeconds = 0;
            }

        } catch (err) {
            console.error("AI connection error:", err);

            const box = document.getElementById("attentionStatus");
            if (box) {
                box.innerHTML = "❌ Camera AI not connected";
            }
        }

    }, 2000); // check every 1 second
}

// =====================================================
// DISPLAY SUMMARY BELOW VIDEO
// =====================================================
function displaySummary(summary) {

    window.currentSummary = summary;  // store summary
    quizActive = true;
    const container = document.getElementById("dynamicSummaryContainer");

    container.innerHTML = `
        <div class="card" id="summaryCard" style="margin-top:20px;">
            <h3>🧠 AI Lecture Summary</h3>
            <p>${summary}</p>

            <button onclick="startQuiz()" 
                style="margin-top:15px;padding:10px 20px;
                       background:#2563eb;color:white;
                       border:none;border-radius:8px;">
                Take Quiz
            </button>
        </div>
    `;
}

async function startQuiz() {

    document.getElementById("summaryCard").style.display = "none";

    const quizContainer = document.getElementById("quizContainer");
    quizContainer.innerHTML = "<p>Generating AI Quiz...</p>";

    const res = await fetch("http://127.0.0.1:5002/generate-quiz", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ summary: window.currentSummary })
    });

    const data = await res.json();

    renderQuiz(data.questions);
}
function renderQuiz(questions) {

    window.currentQuiz = questions;

    const quizContainer = document.getElementById("quizContainer");
    if (!quizContainer) return;

    let html = `
        <div class="quiz-card" id="quizCard">
            <h3>📝 Quick Quiz</h3>
    `;

    questions.forEach((q, index) => {

        html += `
            <div class="quiz-question">
                <p><b>Q${index + 1}. ${q.question}</b></p>
        `;

        q.options.forEach(option => {

            html += `
                <label class="quiz-option-row">
                    <input type="radio" name="q${index}" value="${option}">
                    <span>${option}</span>
                </label>
            `;
        });

        html += `</div>`;
    });

    html += `
        <button onclick="submitQuiz()" class="quiz-submit-btn">
            Submit Quiz
        </button>
        </div>
    `;

    quizContainer.innerHTML = html;
}
async function submitQuiz() {

    if (!window.currentQuiz) return;

    const answers = [];

    for (let i = 0; i < window.currentQuiz.length; i++) {
        const selected = document.querySelector(`input[name="q${i}"]:checked`);

        if (!selected) {
            alert("⚠ Please answer all questions before submitting.");
            return;
        }

        answers.push(selected.value);
    }

    try {

        const res = await fetch("http://127.0.0.1:5002/submit-quiz", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ answers: answers })
        });

        const data = await res.json();

        const quizCard = document.getElementById("quizCard");
        if (!quizCard) return;

        quizCard.innerHTML = `
            <h3>🎯 Your Score: ${data.score}/${window.currentQuiz.length}</h3>
            <p>Resuming video in <span id="countdown">15</span> seconds...</p>
        `;

        let seconds = 15;

        const interval = setInterval(() => {

            seconds--;

            const countdownEl = document.getElementById("countdown");
            if (countdownEl) {
                countdownEl.innerText = seconds;
            }

            if (seconds <= 0) {

                clearInterval(interval);

                const quizContainer = document.getElementById("quizContainer");
                const summaryContainer = document.getElementById("dynamicSummaryContainer");

                if (quizContainer) quizContainer.innerHTML = "";
                if (summaryContainer) summaryContainer.innerHTML = "";

                // 🔥 unlock system
                quizActive = false;

                const video = document.getElementById("video");
                if (video) {
                    video.play();
                    videoPausedByAI = false;
                    distractionSeconds = 0;
                }
            }

        }, 1000);

    } catch (err) {
        console.error("Quiz submit error:", err);
        alert("Something went wrong while submitting quiz.");
    }
}
const video = document.getElementById("video");

if (video) {
    video.addEventListener("play", function () {
        if (quizActive) {
            video.pause();
        }
    });
}