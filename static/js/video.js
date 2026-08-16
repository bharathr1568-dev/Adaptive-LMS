function initVideoTracking(courseId) {

    const video = document.getElementById("video");
    const progressBar = document.getElementById("progressBar");
    const progressText = document.getElementById("progressText");

    if (!video) return;

    let lastSavedPercent = 0;

    video.addEventListener("timeupdate", async () => {

        if (!video.duration) return;

        let watched = video.currentTime;
        let total = video.duration;

        let percent = (watched / total) * 100;

        // UI update
        progressBar.style.width = percent + "%";
        progressText.innerText = percent.toFixed(1) + "% completed";

        // 🔥 SAVE ONLY IF PROGRESS INCREASED BY 5%
        if (percent - lastSavedPercent < 5) return;

        lastSavedPercent = percent;

        try {
            await fetch("/video-progress", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    course_id: courseId,
                    watched: watched,
                    total: total,
                    percentage: percent
                })
            });

        } catch (err) {
            console.log("Progress save error:", err);
        }

    });

}
