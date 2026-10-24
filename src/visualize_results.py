
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_DIR / "metadata"
PLOT_DIR = PROJECT_DIR / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load the CSV files
noisy = pd.read_csv(METADATA_DIR / "full_evaluation_results.csv")
clean = pd.read_csv(METADATA_DIR / "clean_evaluation_results.csv")

# Convert WER ratios to percentages
noisy["wer"] = pd.to_numeric(noisy["wer"], errors="coerce")
noisy["snr_db"] = pd.to_numeric(noisy["snr_db"], errors="coerce")
noisy["wer_percent"] = noisy["wer"] * 100

clean["wer_percent"] = pd.to_numeric(
    clean["wer_percent"], errors="coerce"
)

# Keep valid, successful results
noisy = noisy[
    (noisy["status"] == "success")
    & noisy["wer_percent"].notna()
    & noisy["snr_db"].notna()
].copy()

clean = clean[
    (clean["status"] == "success")
    & clean["wer_percent"].notna()
].copy()

# Summarize noisy speech by SNR
summary = (
    noisy.groupby("snr_db")["wer_percent"]
    .agg(["mean", "median", "count"])
    .reset_index()
    .sort_values("snr_db")
)

clean_average = clean["wer_percent"].mean()
clean_median = clean["wer_percent"].median()

labels = ["Clean"] + [
    f"{int(snr)} dB" for snr in summary["snr_db"]
]

# Graph 1: Average WER
averages = [clean_average] + summary["mean"].tolist()

plt.figure(figsize=(9, 5))
plt.bar(labels, averages)
plt.title("Average Word Error Rate: Clean vs Noisy Speech")
plt.xlabel("Audio condition")
plt.ylabel("Average WER (%)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "average_wer.png", dpi=300)
plt.close()

# Graph 2: Median WER
medians = [clean_median] + summary["median"].tolist()

plt.figure(figsize=(9, 5))
plt.bar(labels, medians)
plt.title("Median Word Error Rate: Clean vs Noisy Speech")
plt.xlabel("Audio condition")
plt.ylabel("Median WER (%)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "median_wer.png", dpi=300)
plt.close()

# Graph 3: Distribution of WER at each noise level
snr_values = sorted(noisy["snr_db"].unique())

groups = [
    noisy.loc[noisy["snr_db"] == snr, "wer_percent"].values
    for snr in snr_values
]

group_labels = [f"{int(snr)} dB" for snr in snr_values]

plt.figure(figsize=(9, 5))
plt.boxplot(groups, tick_labels=group_labels)
plt.title("WER Distribution at Different Noise Levels")
plt.xlabel("Signal-to-Noise Ratio (SNR)")
plt.ylabel("WER (%)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "wer_distribution.png", dpi=300)
plt.close()

print("\n=== GRAPH GENERATION COMPLETE ===")
print(f"Clean recordings evaluated: {len(clean)}")
print(f"Clean average WER: {clean_average:.2f}%")
print(f"Clean median WER: {clean_median:.2f}%")
print("\nNoisy speech summary:")
print(summary.to_string(index=False))
print(f"\nGraphs saved to: {PLOT_DIR}")
