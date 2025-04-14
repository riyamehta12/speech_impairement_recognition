# Audio Anomaly Detection (Speech Analysis Tool)

This is a simple Python tool that analyzes `.wav` audio files to detect unusual or inconsistent speech patterns. It combines audio signal features with speech transcription using OpenAI’s Whisper model and some basic machine learning.

The audio files used are **user-generated or simulated samples** for testing purposes.

---

## 💡 What It Does

For each audio file you provide, the script:

- Loads the audio and displays the waveform
- Transcribes the speech using OpenAI's Whisper
- Detects:
  - Filler words like "um", "uh", "like", etc.
  - Repetitions of words
  - Long pauses in speech
- Extracts audio features like:
  - Pitch and pitch variability
  - MFCCs (mel-frequency cepstral coefficients)
  - Spectral features (centroid, rolloff, bandwidth, contrast)
  - Zero Crossing Rate
- Calculates a manual anomaly score
- Uses 3 machine learning models to detect anomalies:
  - One-Class SVM
  - Isolation Forest
  - Decision Tree
- Averages all scores and flags files as `Normal` or `Anomalous`
- Shows a **final plot** to visualize the verdict

---

## Libraries Used

- `librosa` – audio processing and feature extraction
- `matplotlib` – for plotting waveforms and final verdict
- `whisper` – OpenAI’s model for speech-to-text
- `pandas`, `numpy` – for handling data
- `scikit-learn` – for ML models (SVM, IsolationForest, DecisionTree)
- `re`, `collections` – for text pattern matching and counting

---

##  How to Run

### 1. Install required libraries:

```bash
pip install openai-whisper librosa matplotlib pandas scikit-learn
run script-> python detect_audio_anomalies.py
enter file path on prompt->Enter paths to audio files (comma-separated): ./audio/sample1.wav, ./audio/sample2.wav


