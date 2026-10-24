
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEECH_DIR = PROJECT_ROOT / "data" / "raw" / "speech"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "accent_classifier.joblib"

SAMPLE_RATE = 16000
MAX_SECONDS = 8
MAX_CLIPS_PER_ACCENT = 100

ACCENT_LABELS = {
    "british-english": "British English",
    "southern-american-english": "Southern American English",
}


# ============================================================
# 2. AUDIO FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):
    """Extract 48 numerical audio features."""

    audio, sr = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True,
        duration=MAX_SECONDS,
    )

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    features = []

    # MFCC: 20 means + 20 standard deviations
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20,
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
    )
    features.extend([np.mean(centroid), np.std(centroid)])

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
    )
    features.extend([np.mean(bandwidth), np.std(bandwidth)])

    # Zero-crossing rate: do not pass sr here
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features.extend([np.mean(zcr), np.std(zcr)])

    # Root mean square energy
    rms = librosa.feature.rms(y=audio)
    features.extend([np.mean(rms), np.std(rms)])

    result = np.asarray(features, dtype=np.float64)

    if result.shape != (48,):
        raise ValueError(
            f"Expected 48 features, received {result.shape}"
        )

    if not np.all(np.isfinite(result)):
        raise ValueError("Features contain invalid numeric values.")

    return result


# ============================================================
# 3. LOAD BOTH ACCENT DATASETS
# ============================================================

def load_dataset():
    feature_rows = []
    labels = []
    speaker_groups = []
    failed_files = 0

    if not SPEECH_DIR.is_dir():
        raise FileNotFoundError(
            f"Speech dataset directory not found: {SPEECH_DIR}"
        )

    for folder_key, accent_label in ACCENT_LABELS.items():

        dataset_dirs = [
            p for p in SPEECH_DIR.iterdir()
            if p.is_dir() and folder_key in p.name.lower()
        ]

        if not dataset_dirs:
            print(f"\nWARNING: No folder found for {accent_label}")
            continue

        for dataset_dir in dataset_dirs:
            tsv_path = dataset_dir / "train.tsv"
            clips_dir = dataset_dir / "clips"

            if not tsv_path.is_file():
                print(f"WARNING: Missing {tsv_path}")
                continue

            print(f"\nLoading {accent_label}")
            print(f"Dataset: {dataset_dir.name}")

            try:
                # Avoid chunk-based mixed-type inference warnings.
                metadata = pd.read_csv(
                    tsv_path,
                    sep="\t",
                    encoding="utf-8",
                    low_memory=False,
                    on_bad_lines="skip",
                )
            except Exception as exc:
                print(f"Could not read metadata: {exc}")
                continue

            if "path" not in metadata.columns:
                print("WARNING: Metadata has no 'path' column.")
                continue

            if "client_id" not in metadata.columns:
                metadata["client_id"] = None

            metadata = metadata.sample(
                frac=1,
                random_state=42,
            ).reset_index(drop=True)

            successful = 0

            for _, row in metadata.iterrows():
                if successful >= MAX_CLIPS_PER_ACCENT:
                    break

                filename = row.get("path")

                if pd.isna(filename):
                    continue

                audio_path = clips_dir / str(filename)

                if not audio_path.is_file():
                    continue

                try:
                    features = extract_features(audio_path)
                except Exception as exc:
                    failed_files += 1
                    print(
                        f"Skipping {audio_path.name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

                client_id = row.get("client_id")

                if pd.isna(client_id) or not str(client_id).strip():
                    # Do not pretend that a filename is a known speaker.
                    # Flag it separately so the split limitation is visible.
                    speaker_group = (
                        f"unknown_{accent_label}_{audio_path.stem}"
                    )
                else:
                    # Include the dataset folder to avoid ID collisions.
                    speaker_group = (
                        f"{dataset_dir.name}_{str(client_id)}"
                    )

                feature_rows.append(features)
                labels.append(accent_label)
                speaker_groups.append(speaker_group)

                successful += 1

            print(
                f"Successfully extracted {successful} "
                f"recordings for {accent_label}"
            )

    print(f"\nFailed audio files: {failed_files}")
    print(f"Total successful recordings: {len(feature_rows)}")

    if not feature_rows:
        raise RuntimeError(
            "No features extracted. Check the dataset folders "
            "and the audio errors above."
        )

    X = np.vstack(feature_rows)
    y = np.asarray(labels)
    groups = np.asarray(speaker_groups)

    if len(np.unique(y)) != 2:
        raise RuntimeError(
            "Both accent classes must contain usable audio. "
            f"Found: {np.unique(y).tolist()}"
        )

    for accent in ACCENT_LABELS.values():
        count = int(np.sum(y == accent))
        print(f"{accent}: {count} recordings")

    return X, y, groups


# ============================================================
# 4. SPEAKER-INDEPENDENT STRATIFIED SPLIT
# ============================================================

def make_train_test_split(X, y, groups):
    """Find a split with both accents in each set and no shared speakers."""

    # Four folds produce roughly 25% test data per fold.
    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=42,
    )

    candidates = []

    for train_idx, test_idx in splitter.split(X, y, groups):
        if (
            len(np.unique(y[train_idx])) < 2
            or len(np.unique(y[test_idx])) < 2
        ):
            continue

        # Prefer splits whose test class counts are closest to balanced.
        test_counts = np.bincount(
            (y[test_idx] == ACCENT_LABELS["southern-american-english"])
            .astype(int),
            minlength=2,
        )

        ratio = min(test_counts) / max(test_counts)

        candidates.append((ratio, train_idx, test_idx))

    if not candidates:
        raise RuntimeError(
            "Could not find a valid split containing both accents. "
            "Check the number of unique speakers per accent and "
            "whether client_id values are available."
        )

    # Select the most balanced valid candidate.
    _, train_idx, test_idx = max(
        candidates,
        key=lambda item: item[0],
    )

    # Explicitly verify there is no speaker overlap.
    train_speakers = set(groups[train_idx])
    test_speakers = set(groups[test_idx])

    if train_speakers.intersection(test_speakers):
        raise RuntimeError("Speaker leakage detected in the split.")

    return train_idx, test_idx


# ============================================================
# 5. TRAIN, EVALUATE, AND SAVE
# ============================================================

def main():
    print("=" * 60)
    print("BRITISH vs SOUTHERN AMERICAN ACCENT CLASSIFIER")
    print("=" * 60)

    X, y, groups = load_dataset()

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Unique speaker groups: {len(np.unique(groups))}")

    train_idx, test_idx = make_train_test_split(X, y, groups)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("\nTraining class distribution:")
    print(pd.Series(y_train).value_counts().to_string())

    print("\nTesting class distribution:")
    print(pd.Series(y_test).value_counts().to_string())

    print(f"\nTraining recordings: {len(train_idx)}")
    print(f"Testing recordings: {len(test_idx)}")
    print(f"Training speakers: {len(set(groups[train_idx]))}")
    print(f"Testing speakers: {len(set(groups[test_idx]))}")

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(ACCENT_LABELS.values()),
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "labels": list(ACCENT_LABELS.values()),
            "sample_rate": SAMPLE_RATE,
            "max_seconds": MAX_SECONDS,
            "feature_count": X.shape[1],
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Training completed.")


if __name__ == "__main__":
    main()
