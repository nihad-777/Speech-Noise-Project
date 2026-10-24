
from pathlib import Path
import pandas as pd
from faster_whisper import WhisperModel
from jiwer import wer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "metadata" / "validated_metadata.csv"

df = pd.read_csv(METADATA_FILE)
df["reference_text"] = df["reference_text"].fillna("").astype(str)

# Choose a sentence with a useful transcript.
valid_records = df[df["reference_text"].str.len() > 10]
first_record = valid_records.iloc[0]

# Test all noise levels for the same speech recording.
same_speech = df[
    df["clean_path"] == first_record["clean_path"]
].sort_values("snr_db")

print("Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")

results = []

for _, record in same_speech.iterrows():
    audio_path = PROJECT_ROOT / str(record["noisy_path"]).replace("\\", "/")

    if not audio_path.is_file():
        print(f"Skipping missing file: {audio_path}")
        continue

    print(f"\nTranscribing: {audio_path.name}")

    segments, info = model.transcribe(str(audio_path))
    predicted_text = " ".join(
        segment.text.strip() for segment in segments
    ).strip()

    reference = record["reference_text"]
    error_rate = wer(reference.lower(), predicted_text.lower())

    results.append({
        "SNR (dB)": record["snr_db"],
        "Reference": reference,
        "Prediction": predicted_text,
        "WER (%)": round(error_rate * 100, 2),
    })

if results:
    results_df = pd.DataFrame(results)
    print("\n=== RESULTS BY NOISE LEVEL ===")
    print(results_df.to_string(index=False))

    output_file = PROJECT_ROOT / "metadata" / "snr_experiment_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
else:
    print("No audio files were successfully transcribed.")
