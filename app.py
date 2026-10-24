from pathlib import Path
import os
import tempfile

import joblib
import librosa
import numpy as np
from flask import Flask, jsonify, render_template, request
from faster_whisper import WhisperModel

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
ACCENT_MODEL_PATH = BASE_DIR / "models" / "accent_classifier.joblib"

ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"
}

_whisper_model = None
_accent_bundle = None


@app.get("/")
def home():
    return render_template("index.html")


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Audio file is too large. Maximum size is 20 MB."
    }), 413


def get_whisper_model():
    """Load the speech recognition model only once."""
    global _whisper_model

    if _whisper_model is None:
        _whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
        )

    return _whisper_model


def get_accent_bundle():
    """Load the saved accent classifier."""
    global _accent_bundle

    if _accent_bundle is None:
        if not ACCENT_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Accent model not found: {ACCENT_MODEL_PATH}"
            )

        bundle = joblib.load(ACCENT_MODEL_PATH)

        if not isinstance(bundle, dict) or "model" not in bundle:
            raise ValueError("Invalid accent classifier file.")

        _accent_bundle = bundle

    return _accent_bundle


def extract_accent_features(audio_path, sample_rate=16000, max_seconds=8):
    """Extract the same 48 features used during model training."""
    audio, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        duration=max_seconds,
    )

    if audio.size == 0:
        raise ValueError("The uploaded audio contains no readable samples.")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    features = [
        *np.mean(mfcc, axis=1),
        *np.std(mfcc, axis=1),
    ]

    extra_features = [
        librosa.feature.spectral_centroid(y=audio, sr=sr),
        librosa.feature.spectral_bandwidth(y=audio, sr=sr),
        librosa.feature.zero_crossing_rate(audio),
        librosa.feature.rms(y=audio),
    ]

    for feature in extra_features:
        features.extend([
            float(np.mean(feature)),
            float(np.std(feature)),
        ])

    return np.asarray(features, dtype=float).reshape(1, -1)


def get_display_label(class_value, classes, labels):
    """Convert the classifier's class value into a readable accent name."""
    if isinstance(class_value, (str, np.str_)):
        return str(class_value)

    if isinstance(labels, dict):
        if class_value in labels:
            return str(labels[class_value])
        if str(class_value) in labels:
            return str(labels[str(class_value)])

    if classes is not None and labels is not None:
        for index, value in enumerate(classes):
            if value == class_value and index < len(labels):
                return str(labels[index])

    return str(class_value)


def predict_accent(audio_path):
    """Predict an accent supported by the trained classifier."""
    bundle = get_accent_bundle()
    classifier = bundle["model"]

    features = extract_accent_features(
        audio_path,
        sample_rate=bundle.get("sample_rate", 16000),
        max_seconds=bundle.get("max_seconds", 8),
    )

    expected_count = bundle.get("feature_count", 48)
    if features.shape[1] != expected_count:
        raise ValueError(
            f"Expected {expected_count} features, got {features.shape[1]}."
        )

    classes = getattr(classifier, "classes_", None)
    labels = bundle.get("labels")

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))

        if classes is not None:
            class_value = classes[best_index]
        else:
            class_value = classifier.predict(features)[0]

        confidence = round(float(probabilities[best_index]) * 100, 2)
    else:
        class_value = classifier.predict(features)[0]
        confidence = None

    return {
        "accent": get_display_label(class_value, classes, labels),
        "accent_confidence": confidence,
    }


@app.post("/api/transcribe")
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "Please upload an audio file."}), 400

    audio = request.files["audio"]

    if not audio.filename:
        return jsonify({"error": "No audio file was selected."}), 400

    suffix = Path(audio.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported audio format."}), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp:
            temp_path = temp.name
            audio.save(temp_path)

        # Speech transcription
        model = get_whisper_model()
        segments, info = model.transcribe(temp_path)

        transcript = " ".join(
            segment.text.strip() for segment in segments
        ).strip()

        result = {
            "transcript": transcript,
            "language": info.language,
        }

        # Accent detection is independent of transcription.
        try:
            result.update(predict_accent(temp_path))
        except Exception:
            app.logger.exception("Accent prediction failed")
            result["accent"] = None
            result["accent_confidence"] = None
            result["accent_error"] = (
                "Accent prediction was unavailable for this recording."
            )

        return jsonify(result)

    except Exception:
        app.logger.exception("Transcription failed")
        return jsonify({
            "error": (
                "Transcription failed. Check the audio file "
                "and the server terminal."
            )
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)