MemoTag: Voice-Based Cognitive Decline Detection (Proof of Concept)
This project analyzes voice recordings to detect early signs of cognitive stress or decline using a combination of audio processing, natural language features, and anomaly detection models.

Overview
Audio samples were simulated using Bark.ai to reflect realistic speech patterns under varying cognitive loads. Each .wav file is processed to extract both acoustic and linguistic features relevant to cognitive health.
Code used to generate samples using bark.ai:https://colab.research.google.com/drive/1i2SlgdYCbMpRiSnDXWpumkz2E17mxwpq?usp=sharing
Other audio samples used can be accessed through:https://drive.google.com/drive/folders/1IM6eCvccYgeXdu3-8hf9WezuFUpwdwnc?usp=sharing
Key Libraries Used:
librosa – audio analysis
whisper – speech-to-text transcription
sklearn – ML models (SVM, Isolation Forest, Decision Tree)
matplotlib, pandas, numpy, re – visualization and processing

Feature Extraction
Audio-based: MFCC, RMS, pitch variability, spectral features

Text-based: filler word count, repeated words, pause frequency, speech rate

Transcription is performed using OpenAI Whisper

Machine Learning Approaches
1. Rule-Based Scoring
Flags known risk signals (e.g., >3 fillers, slow speech rate)
Simple, interpretable baseline

2. One-Class SVM
Detects outliers in feature space
Ideal for small, unlabeled datasets

3. Isolation Forest
Randomly isolates anomalies
Fast and effective for high-dimensional data

4. Decision Tree Classifier
Supervised model trained on basic anomaly labels
Transparent decision logic

Final Verdict
Each model contributes to a combined score. A sample is flagged if 2 or more models (including the rule-based score) identify it as risky.The final verdicts of all files are plotted based on the risk score.

