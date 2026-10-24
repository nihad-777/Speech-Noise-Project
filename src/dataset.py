
from pathlib import Path

import pandas as pd


# Project root: the folder containing data/, metadata/, and src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = PROJECT_ROOT / "metadata" / "metadata.csv"
SPEECH_DIR = PROJECT_ROOT / "data" / "raw" / "speech"


def load_metadata():
    """Load the dataset metadata CSV."""
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    df = pd.read_csv(METADATA_FILE)

    required_columns = [
        "clean_path",
        "noisy_path",
        "noise_path",
        "snr_db",
        "accent",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    return df


def load_transcripts():
    """Build a mapping from audio filename to reference sentence."""
    transcript_files = list(SPEECH_DIR.rglob("*.tsv"))
    transcripts = {}

    for tsv_file in transcript_files:
        table = pd.read_csv(tsv_file, sep="\t")

        if "path" not in table.columns or "sentence" not in table.columns:
            continue

        for _, row in table[["path", "sentence"]].dropna().iterrows():
            filename = Path(str(row["path"])).name
            sentence = str(row["sentence"]).strip()

            if filename not in transcripts and sentence:
                transcripts[filename] = sentence

    return transcripts


def main():
    df = load_metadata()
    transcripts = load_transcripts()

    print("\n=== DATASET SUMMARY ===")
    print(f"Total records: {len(df)}")
    print("\nRecords by SNR:")
    print(df["snr_db"].value_counts().sort_index().to_string())
    print("\nRecords by accent label:")
    print(df["accent"].value_counts().to_string())

    missing_clean = []
    missing_noisy = []
    missing_noise = []

    for _, row in df.iterrows():
        for column, missing_list in [
            ("clean_path", missing_clean),
            ("noisy_path", missing_noisy),
            ("noise_path", missing_noise),
        ]:
            relative_path = str(row[column]).replace("\\", "/")
            full_path = PROJECT_ROOT / relative_path

            if not full_path.is_file():
                missing_list.append(relative_path)

    print("\n=== FILE VALIDATION ===")
    print(f"Missing clean files: {len(missing_clean)}")
    print(f"Missing noisy files: {len(missing_noisy)}")
    print(f"Missing noise files: {len(missing_noise)}")

    df["clean_filename"] = df["clean_path"].apply(
        lambda path: Path(str(path).replace("\\", "/")).name
    )
    df["reference_text"] = df["clean_filename"].map(transcripts)

    missing_transcripts = df["reference_text"].isna().sum()

    print("\n=== TRANSCRIPT VALIDATION ===")
    print(f"Transcripts found: {len(df) - missing_transcripts}")
    print(f"Transcripts missing: {missing_transcripts}")

    print("\n=== EXAMPLE RECORDS ===")
    print(
        df[
            ["clean_filename", "snr_db", "reference_text"]
        ].head(5).to_string(index=False)
    )

    output_file = PROJECT_ROOT / "metadata" / "validated_metadata.csv"
    df.to_csv(output_file, index=False)
    print(f"\nValidated metadata saved to: {output_file}")


if __name__ == "__main__":
    main()
