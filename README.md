# VoxLab🎙️: Speech Transcription with Different Accents and Background Noise

A machine learning project that explores speech transcription across different accents and background-noise conditions. The project provides a web interface for uploading audio and viewing transcription results, along with accent prediction.

## 🌐 Live Demo

**Try the application:** https://speech-noise-project.onrender.com/

## 📌 Project Overview

Speech recognition systems can face challenges when processing different accents and audio recordings containing background noise. This project aims to explore these challenges by combining speech transcription with accent prediction in a web-based application.

## 🎯 Objectives

- Transcribe speech from uploaded audio files.
- Explore speech recognition performance across different accents.
- Investigate the effect of background noise on transcription.
- Predict accents using an audio classification model.
- Provide a simple, user-friendly web interface for testing audio.

## ✨ Features

- Audio file upload through a web interface.
- Automatic speech transcription.
- Accent prediction using a trained machine learning model.
- Support for experimenting with different audio conditions.
- Web deployment for remote access.

## 🛠️ Technologies Used

- **Python** — core programming language.
- **Flask** — web application framework.
- **Faster-Whisper** — speech transcription.
- **Librosa** — audio processing and feature extraction.
- **Scikit-learn** — machine learning and accent classification.
- **HTML, CSS, JavaScript** — frontend interface.
- **Git and GitHub** — version control and source code hosting.
- **Render** — web application deployment.

## 📂 Project Structure

```text
Speech-Noise-Project/
├── app.py
├── requirements.txt
├── .python-version
├── models/
│   └── accent_classifier.joblib
├── src/
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/nihad-777/Speech-Noise-Project.git
```

### 2. Open the project directory

```bash
cd Speech-Noise-Project
```

### 3. Create a virtual environment

Python 3.11 is recommended for this project.

```bash
py -3.11 -m venv .venv
```

### 4. Activate the environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

Open the local URL displayed in your terminal, commonly `http://127.0.0.1:5000/`.

## 🧪 Testing

You can experiment with:

- Audio recordings with different accents.
- Clean speech recordings.
- Speech recordings containing background noise.
- Short audio clips to examine transcription output.

Compare the output across these conditions to investigate recognition challenges. Actual accuracy should be measured using labelled test audio and documented results.

## 🚀 Deployment

The application is deployed using Render.

**Live application:** https://speech-noise-project.onrender.com/

## 🔮 Future Improvements

- Evaluate transcription accuracy using Word Error Rate (WER).
- Compare transcription performance under different noise levels.
- Expand accent training data.
- Add noise reduction and audio enhancement.
- Display evaluation metrics and comparison charts.
- Improve error handling and user feedback.

## 👨‍🎓 Project Information

**Project type:** MCA Academic Project  
**Domain:** Machine Learning, Speech Processing and Audio Classification  
**Language:** Python

## 📄 License

A license has not yet been specified. Add a suitable open-source license if you intend to allow others to reuse or modify this project.
