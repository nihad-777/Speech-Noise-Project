
const pages = document.querySelectorAll(".page");
const navLinks = document.querySelectorAll(".nav-link");
const breadcrumb = document.getElementById("page-breadcrumb");

const pageNames = {
    dashboard: "Overview",
    transcribe: "Speech transcription",
    analysis: "Noise analysis"
};

function showPage(pageId) {
    pages.forEach(page => {
        page.classList.toggle("active-page", page.id === pageId);
    });

    navLinks.forEach(link => {
        link.classList.toggle("active", link.dataset.page === pageId);
    });

    if (breadcrumb) {
        breadcrumb.textContent = pageNames[pageId] || "Overview";
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
}

navLinks.forEach(link => {
    link.addEventListener("click", () => showPage(link.dataset.page));
});

document.querySelectorAll("[data-go]").forEach(button => {
    button.addEventListener("click", () => showPage(button.dataset.go));
});

// Charts use the existing noise experiment results.
if (window.Chart) {
    Chart.defaults.color = "#91a0b9";
    Chart.defaults.font.family = "'DM Sans', sans-serif";

    const labels = ["5 dB", "10 dB", "20 dB"];
    const averageWer = [49.40, 41.43, 36.14];
    const medianWer = [27.88, 20.83, 9.55];

    function createWerChart(canvasId, showMedian = false) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        new Chart(canvas, {
            type: "line",
            data: {
                labels,
                datasets: [
                    {
                        label: "Average WER (%)",
                        data: averageWer,
                        borderColor: "#829cff",
                        backgroundColor: "rgba(109, 145, 255, 0.12)",
                        fill: true,
                        tension: 0.35,
                        borderWidth: 2.5,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    },
                    ...(showMedian ? [{
                        label: "Median WER (%)",
                        data: medianWer,
                        borderColor: "#57dfd0",
                        backgroundColor: "transparent",
                        tension: 0.35,
                        borderWidth: 2,
                        pointRadius: 4
                    }] : [])
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: "index"
                },
                plugins: {
                    legend: {
                        display: showMedian,
                        position: "bottom",
                        labels: {
                            usePointStyle: true,
                            boxWidth: 7,
                            padding: 20
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: context =>
                                `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        border: { display: false }
                    },
                    y: {
                        beginAtZero: true,
                        suggestedMax: 60,
                        title: {
                            display: true,
                            text: "Word Error Rate (%)"
                        },
                        grid: {
                            color: "rgba(145, 160, 185, 0.12)"
                        },
                        border: { display: false },
                        ticks: {
                            callback: value => `${value}%`
                        }
                    }
                }
            }
        });
    }

    createWerChart("overviewChart");
    createWerChart("analysisChart", true);
}

// Audio upload and preview.
const fileInput = document.getElementById("audio-file");
const uploadZone = document.getElementById("upload-zone");
const filePreview = document.getElementById("file-preview");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const audioPreview = document.getElementById("audio-preview");
const transcribeButton = document.getElementById("transcribe-btn");
const removeFileButton = document.getElementById("remove-file");

const emptyOutput = document.getElementById("empty-output");
const transcriptResult = document.getElementById("transcript-result");
const transcriptText = document.getElementById("transcript-text");
const outputStatus = document.getElementById("output-status");
const progressArea = document.getElementById("progress-area");
const errorMessage = document.getElementById("error-message");
const werBox = document.getElementById("wer-box");

// Create the accent result panel without requiring an HTML edit.
let accentResult = document.getElementById("accent-result");
let accentName;
let accentConfidence;

if (transcriptResult && !accentResult) {
    accentResult = document.createElement("div");
    accentResult.id = "accent-result";
    accentResult.className = "accent-result";
    accentResult.hidden = true;

    const heading = document.createElement("h3");
    heading.textContent = "Detected accent";

    accentName = document.createElement("p");
    accentName.id = "accent-name";

    accentConfidence = document.createElement("p");
    accentConfidence.id = "accent-confidence";

    accentResult.append(heading, accentName, accentConfidence);
    transcriptResult.appendChild(accentResult);
} else if (accentResult) {
    accentName = document.getElementById("accent-name");
    accentConfidence = document.getElementById("accent-confidence");
}

let selectedFile = null;
let audioObjectUrl = null;

function clearOutput() {
    if (emptyOutput) emptyOutput.hidden = false;
    if (transcriptResult) transcriptResult.hidden = true;
    if (progressArea) progressArea.hidden = true;
    if (errorMessage) errorMessage.hidden = true;
    if (werBox) werBox.hidden = true;
    if (accentResult) accentResult.hidden = true;

    if (outputStatus) {
        outputStatus.textContent = "Waiting for audio";
    }
}

function showError(message) {
    if (errorMessage) {
        errorMessage.textContent = message;
        errorMessage.hidden = false;
    }

    if (outputStatus) outputStatus.textContent = "Error";
}

function setFile(file) {
    if (!file) return;

    const allowedExtensions = /\.(wav|mp3|m4a|flac|ogg|webm)$/i;

    if (!allowedExtensions.test(file.name)) {
        showError(
            "Unsupported file type. Please choose WAV, MP3, M4A, FLAC, OGG or WEBM audio."
        );
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        showError(
            "This file exceeds the 20 MB upload limit. Please choose a smaller recording."
        );
        return;
    }

    selectedFile = file;

    if (fileName) fileName.textContent = file.name;
    if (fileSize) {
        fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    }

    if (filePreview) filePreview.hidden = false;
    if (audioPreview) audioPreview.hidden = false;
    if (transcribeButton) transcribeButton.disabled = false;
    if (uploadZone) uploadZone.hidden = true;

    clearOutput();

    if (audioObjectUrl) {
        URL.revokeObjectURL(audioObjectUrl);
    }

    audioObjectUrl = URL.createObjectURL(file);

    if (audioPreview) {
        audioPreview.src = audioObjectUrl;
    }
}

function resetUpload() {
    selectedFile = null;

    if (fileInput) fileInput.value = "";
    if (filePreview) filePreview.hidden = true;
    if (uploadZone) uploadZone.hidden = false;

    if (audioPreview) {
        audioPreview.hidden = true;
        audioPreview.removeAttribute("src");
        audioPreview.load();
    }

    if (transcribeButton) transcribeButton.disabled = true;

    if (audioObjectUrl) {
        URL.revokeObjectURL(audioObjectUrl);
    }

    audioObjectUrl = null;
    clearOutput();
}

if (fileInput) {
    fileInput.addEventListener("change", event => {
        setFile(event.target.files[0]);
    });
}

if (removeFileButton) {
    removeFileButton.addEventListener("click", resetUpload);
}

if (uploadZone) {
    ["dragenter", "dragover"].forEach(eventName => {
        uploadZone.addEventListener(eventName, event => {
            event.preventDefault();
            uploadZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        uploadZone.addEventListener(eventName, event => {
            event.preventDefault();
            uploadZone.classList.remove("dragover");
        });
    });

    uploadZone.addEventListener("drop", event => {
        setFile(event.dataTransfer.files[0]);
    });
}

// Transcription and accent prediction.
if (transcribeButton) {
    transcribeButton.addEventListener("click", async () => {
        if (!selectedFile) return;

        const fileToUpload = selectedFile;
        const formData = new FormData();
        formData.append("audio", fileToUpload);

        if (emptyOutput) emptyOutput.hidden = true;
        if (transcriptResult) transcriptResult.hidden = true;
        if (errorMessage) errorMessage.hidden = true;
        if (progressArea) progressArea.hidden = false;
        if (werBox) werBox.hidden = true;
        if (accentResult) accentResult.hidden = true;

        if (outputStatus) outputStatus.textContent = "Processing";
        transcribeButton.disabled = true;

        try {
            
const response = await fetch("/api/transcribe", {
    method: "POST",
    body: formData
});

const responseText = await response.text();
let data = {};

try {
    data = responseText ? JSON.parse(responseText) : {};
} catch {
    throw new Error(
        `Server returned an invalid response (HTTP ${response.status}). Check the Render logs.`
    );
}

if (!response.ok) {
    throw new Error(
        data.error || `Request failed with status ${response.status}.`
    );
}

// Continue with your existing success-handling code here.


            if (transcriptText) {
                transcriptText.value = data.transcript || "";
            }

            if (transcriptResult) transcriptResult.hidden = false;
            if (emptyOutput) emptyOutput.hidden = true;

            if (data.accent && accentName) {
                accentName.textContent = data.accent;

                if (
                    typeof data.accent_confidence === "number" &&
                    Number.isFinite(data.accent_confidence)
                ) {
                    accentConfidence.textContent =
                        `Estimated model confidence: ${data.accent_confidence.toFixed(1)}%`;
                } else {
                    accentConfidence.textContent =
                        "Confidence score unavailable.";
                }

                accentResult.hidden = false;
            } else if (accentResult) {
                accentResult.hidden = true;
            }

            if (data.wer !== undefined && data.wer !== null) {
                const werValue = document.getElementById("wer-value");

                if (werValue) {
                    werValue.textContent = `${(data.wer * 100).toFixed(2)}%`;
                }

                if (werBox) werBox.hidden = false;
            }

            if (outputStatus) {
                outputStatus.textContent = data.language
                    ? `Language: ${data.language}`
                    : "Completed";
            }

            if (data.accent_error) {
                console.warn(data.accent_error);
            }
        } catch (error) {
            showError(
                error.message.includes("Failed to fetch")
                    ? "Cannot reach the backend. Start the Flask server and try again."
                    : error.message
            );
        } finally {
            if (progressArea) progressArea.hidden = true;
            transcribeButton.disabled = !selectedFile;
        }
    });
}

// Copy transcript.
const copyButton = document.getElementById("copy-btn");

if (copyButton && transcriptText) {
    copyButton.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(transcriptText.value);
            copyButton.textContent = "Copied!";

            setTimeout(() => {
                copyButton.textContent = "Copy text";
            }, 1500);
        } catch {
            transcriptText.select();
            document.execCommand("copy");
        }
    });
}

// Download transcript.
const downloadButton = document.getElementById("download-btn");

if (downloadButton && transcriptText) {
    downloadButton.addEventListener("click", () => {
        const blob = new Blob([transcriptText.value], {
            type: "text/plain;charset=utf-8"
        });

        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = url;
        link.download = "transcription.txt";
        link.click();

        URL.revokeObjectURL(url);
    });
}
