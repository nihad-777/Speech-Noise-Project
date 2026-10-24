
from pathlib import Path
import pandas as pd
from faster_whisper import WhisperModel
from jiwer import wer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "metadata" / "validated_metadata.csv"
OUTPUT_FILE = PROJECT_ROOT / "metadata" / "full_evaluation_results.csv"
SUMMARY_FILE = PROJECT_ROOT / "metadata" / "snr_summary.csv"

def main():
    # Load and validate the dataset
    if not METADATA_FILE.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    df = pd.read_csv(METADATA_FILE)
    required = ["noisy_path", "snr_db", "reference_text"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["reference_text"] = (
        df["reference_text"].fillna("").astype(str).str.strip()
    )

    # Exclude empty reference transcripts, but retain short nonempty ones.
    df = df[df["reference_text"] != ""].copy()

    # Resume safely if the script is interrupted.
    if OUTPUT_FILE.exists():
        results = pd.read_csv(OUTPUT_FILE)
        completed = set(results["noisy_path"].astype(str))
    else:
        results = pd.DataFrame(columns=[
            "noisy_path", "snr_db", "reference_text",
            "predicted_text", "wer", "status"
        ])
        completed = set()

    pending = df[~df["noisy_path"].astype(str).isin(completed)]

    print(f"Total records with references: {len(df)}")
    print(f"Already evaluated: {len(completed)}")
    print(f"Remaining: {len(pending)}")

    if pending.empty:
        print("All records have already been evaluated.")
    else:
        print("\nLoading Whisper Base model...")
        model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

        for number, (_, row) in enumerate(pending.iterrows(), start=1):
            relative_path = str(row["noisy_path"]).replace("\\", "/")
            audio_path = PROJECT_ROOT / relative_path
            reference = row["reference_text"]

            result = {
                "noisy_path": str(row["noisy_path"]),
                "snr_db": row["snr_db"],
                "reference_text": reference,
                "predicted_text": "",
                "wer": None,
                "status": "error"
            }

            print(
                f"\n[{number}/{len(pending)}] "
                f"SNR={row['snr_db']} dB: {audio_path.name}"
            )

            try:
                if not audio_path.is_file():
                    raise FileNotFoundError(str(audio_path))

                segments, info = model.transcribe(str(audio_path))
                prediction = " ".join(
                    segment.text.strip() for segment in segments
                ).strip()

                result["predicted_text"] = prediction
                result["wer"] = wer(reference, prediction)
                result["status"] = "success"

                print(f"WER: {result['wer']:.2%}")

            except Exception as error:
                print(f"Error: {error}")

            # Save after every record so progress isn't lost.
            results = pd.concat(
                [results, pd.DataFrame([result])],
                ignore_index=True
            )
            results.to_csv(OUTPUT_FILE, index=False)

    # Summarize only successful evaluations.
    results = pd.read_csv(OUTPUT_FILE)
    successful = results[
        (results["status"] == "success") &
        results["wer"].notna()
    ].copy()

    if successful.empty:
        print("\nNo recordings were evaluated successfully.")
        return

    summary = (
        successful.groupby("snr_db")
        .agg(
            recordings=("wer", "count"),
            average_wer=("wer", "mean"),
            median_wer=("wer", "median")
        )
        .reset_index()
        .sort_values("snr_db")
    )

    summary["average_wer_percent"] = (
        summary["average_wer"] * 100
    ).round(2)
    summary["median_wer_percent"] = (
        summary["median_wer"] * 100
    ).round(2)

    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n=== EVALUATION SUMMARY ===")
    print(
        summary[
            [
                "snr_db",
                "recordings",
                "average_wer_percent",
                "median_wer_percent"
            ]
        ].to_string(index=False)
    )

    print(f"\nDetailed results: {OUTPUT_FILE}")
    print(f"SNR summary: {SUMMARY_FILE}")
    print(
        "\nNote: WER can exceed 100% when extra words are inserted. "
        "Lower WER is better."
    )

if __name__ == "__main__":
    main()
