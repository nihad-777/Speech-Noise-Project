
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

_model = None
_accent_bundle = None


@app.get("/")
def home():
    return render_template("index.html")


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Audio file is too large. Maximum size is 20 MB."
    }), 413


def get_model():
    """Load the Whisper transcription model once."""
    global _model

    if _model is None:
        _model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
        )

    return _model


def get_accent_bundle():
    """Load the trained accent classifier once."""
    global _accent_bundle

    if _accent_bundle is None:
        if not ACCENT_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Accent classifier not found: {ACCENT_MODEL_PATH}"
            )

        _accent_bundle = joblib.load(ACCENT_MODEL_PATH)

        if not isinstance(_accent_bundle, dict):
            raise ValueError(
                "Unexpected accent model format. "
                "Expected a dictionary containing the model and labels."
            )

        if "model" not in _accent_bundle:
            raise ValueError("The saved accent bundle has no 'model' entry.")

    return _accent_bundle


def extract_accent_features(
    audio_path,
    sample_rate=16000,
    max_seconds=8,
):
    """
    Extract the 48 features used by the training script:
    40 MFCC statistics and 8 additional audio statistics.
    """
    y, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        duration=max_seconds,
    )

    if y.size == 0:
        raise ValueError("No readable audio samples were found.")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = [
        *np.mean(mfcc, axis=1),
        *np.std(mfcc, axis=1),
    ]

    extra_features = [
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(y=y),
    ]

    for feature in extra_features:
        features.extend([
            float(np.mean(feature)),
            float(np.std(feature)),
        ])

    return np.asarray(features, dtype=float).reshape(1, -1)


def resolve_label(prediction, classes, labels):
    """Support classifiers trained on either text labels or integer labels."""
    if isinstance(prediction, (str, np.str_)):
        return str(prediction)

    index = int(prediction)

    if isinstance(labels, dict):
        if index in labels:
            return str(labels[index])
        if str(index) in labels:
            return str(labels[str(index)])

    if labels is not None and 0 <= index < len(labels):
        return str(labels[index])

    # Fallback for classifiers whose class values are the actual labels.
    if classes is not None and 0 <= index < len(classes):
        return str(classes[index])

    return str(prediction)


def predict_accent(audio_path):
    """Predict one of the accent classes supported by the trained model."""
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
            f"Feature mismatch: expected {expected_count}, "
            f"got {features.shape[1]}."
        )

    prediction = classifier.predict(features)[0]
    classes = getattr(classifier, "classes_", None)
    labels = bundle.get("labels")

    accent = resolve_label(prediction, classes, labels)

    confidence = None
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        confidence = round(float(probabilities[best_index]) * 100, 2)

        # For classifiers trained directly on text labels, use the
        # class associated with the highest probability.
        if classes is not None:
            accent = str(classes[best_index])
        elif labels is not None:
            accent = resolve_label(best_index, None, labels)

    result = {"accent": accent}
    result["accent_confidence"] = confidence
    return result


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

        # Transcribe the uploaded recording.
        model = get_model()
        segments, info = model.transcribe(temp_path)

        transcript = " ".join(
            segment.text.strip() for segment in segments
        ).strip()

        result = {
            "transcript": transcript,
            "language": info.language,
        }

        # Keep transcription available even if accent detection fails.
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
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )
