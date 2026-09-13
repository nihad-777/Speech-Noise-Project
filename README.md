# Speech Noise Dataset Project

This project creates a speech-in-noise dataset by combining speech recordings with background-noise recordings.

The generated dataset can be used for experiments in speech processing, speech enhancement, noise robustness, and audio classification.

## Project Overview

The project currently uses:

* **Speech dataset:** Southern American English speech recordings
* **Noise dataset:** Background environmental noise recordings
* **SNR levels:** 5 dB, 10 dB, and 20 dB
* **Initial speech samples:** 20
* **Generated noisy audio samples:** 60

Each noisy audio sample contains a clean speech recording mixed with a background-noise recording at a specified signal-to-noise ratio.

## Project Structure

```text
Speech-Noise-Project/
│
├── data/
│   ├── raw/
│   │   ├── speech/
│   │   └── noise/
│   │
│   └── processed/
│       └── noisy/
│
├── metadata/
│   └── metadata.csv
│
├── data_preparation.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Metadata

The `metadata.csv` file records information about each generated sample, including:

* Clean speech file path
* Noisy speech file path
* Noise file path
* Signal-to-noise ratio
* Accent group

## How to Reproduce the Dataset

1. Download the original speech dataset.

2. Download the background-noise dataset.

3. Place the speech files inside:

   ```text
   data/raw/speech/
   ```

4. Place the noise files inside:

   ```text
   data/raw/noise/
   ```

5. Open `data_preparation.ipynb` in Jupyter Notebook or VS Code.

6. Run the notebook cells in order.

7. The generated noisy audio files will be saved in:

   ```text
   data/processed/noisy/
   ```

8. The metadata will be saved in:

   ```text
   metadata/metadata.csv
   ```

## Important Note

The original audio files are not included in this repository because of file size and dataset licensing restrictions.

The notebook and metadata are included so that the dataset can be recreated using the original source datasets.

Please follow the licenses and usage conditions of the original datasets.

## Technologies Used

* Python
* Jupyter Notebook
* NumPy
* Pandas
* SoundFile
* Librosa
* tqdm

## Future Improvements

* Add speech datasets from other accents.
* Add more background-noise categories.
* Include more SNR levels.
* Increase the number of speech samples.
* Add data validation and audio-quality checks.
* Develop speech-enhancement and noise-classification models.

