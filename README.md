# VoxLab: Speech Transcription with Different Accents and Background Noise

## Overview

VoxLab is a speech-processing project that transcribes English speech,
predicts between two trained English accent categories, and evaluates
how background noise affects transcription accuracy.

The project uses Whisper for speech recognition and a machine-learning
classifier for accent prediction.

## Features

- Upload audio files through a Flask web interface.
- Convert speech into text using Whisper Base.
- Predict British English or Southern American English.
- Display the classifier's estimated confidence score.
- Evaluate transcription quality using Word Error Rate (WER).
- Compare results at different signal-to-noise ratios (SNR).
- Visualize noise experiment results through charts.

## Technologies

- Python
- Flask
- faster-whisper
- librosa
- NumPy
- scikit-learn
- joblib
- HTML, CSS and JavaScript
- Chart.js

## Project Structure

```text
Speech-Noise-Project/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── train_accent_model.py
│   ├── transcribe.py
│   ├── dataset.py
│   ├── evaluate_dataset.py
│   ├── evaluate_clean.py
│   └── visualize_results.py
├── models/
│   └── accent_classifier.joblib
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── data/
└── metadata/