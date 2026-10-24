
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel
from jiwer import wer
from tqdm import tqdm

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_DIR / "metadata"

INPUT_FILE = METADATA_DIR / "validated_metadata.csv"
OUTPUT_FILE = METADATA_DIR / "clean_evaluation_results.csv"

# Load validated dataset
df = pd.read_csv(INPUT_FILE)

# Find the transcript column automatically
possible_columns = [
    "reference_text",
    "transcript",
    "transcription",
    "sentence",
    "text",
]

reference_column = next(
    (col for col in possible_columns if col in df.columns),
    None,
)

if reference_column is None:
    raise ValueError(
        "Could not find the transcript column. "
        f"Available columns: {list(df.columns)}"
    )

if "clean_path" not in df.columns:
    raise ValueError("The dataset must contain a clean_path column.")

# Clean and validate the reference text
df["clean_path"] = df["clean_path"].fillna("").astype(str).str.strip()
df[reference_column] = (
    df[reference_column].fillna("").astype(str).str.strip()
)

# Exclude missing paths and very short references such as "Conv."
df = df[
    (df["clean_path"] != "")
    & (df[reference_column].str.len() > 10)
].copy()

# Evaluate each original audio file only once
df = df.drop_duplicates(subset=["clean_path"]).reset_index(drop=True)

print(f"Unique clean recordings to evaluate: {len(df)}")

# Load the same Whisper model used for noisy audio
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
)

results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Clean audio"):
    audio_path = Path(row["clean_path"])
    reference = row[reference_column]

    try:
        if not audio_path.is_absolute():
            audio_path = PROJECT_DIR / audio_path

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
        )

        prediction = " ".join(
            segment.text.strip() for segment in segments
        ).strip()

        error_rate = wer(reference, prediction)

        result = {
            "clean_path": str(audio_path),
            "reference_text": reference,
            "predicted_text": prediction,
            "wer": error_rate,
            "wer_percent": error_rate * 100,
            "status": "success",
        }

    except Exception as exc:
        print(f"\nError processing {audio_path}: {exc}")

        result = {
            "clean_path": str(audio_path),
            "reference_text": reference,
            "predicted_text": "",
            "wer": None,
            "wer_percent": None,
            "status": f"error: {exc}",
        }

    results.append(result)

    # Save progress after each recording
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

# Summarize successful transcriptions
results_df = pd.DataFrame(results)
successful = results_df[
    (results_df["status"] == "success")
    & results_df["wer"].notna()
]

print("\n=== CLEAN SPEECH EVALUATION ===")
print(f"Unique recordings: {len(df)}")
print(f"Successful recordings: {len(successful)}")
print(f"Failed recordings: {len(df) - len(successful)}")

if not successful.empty:
    print(f"Average WER: {successful['wer_percent'].mean():.2f}%")
    print(f"Median WER: {successful['wer_percent'].median():.2f}%")

print(f"\nDetailed results saved to: {OUTPUT_FILE}")
print("Lower WER means better transcription accuracy.")
