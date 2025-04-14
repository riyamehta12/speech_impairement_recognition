**Voice-Based Cognitive Decline Detection**

This project analyzes voice recordings to detect early signs of cognitive stress or decline using audio processing, natural language features, and simple anomaly detection models.

**Overview**

Audio samples were simulated using Bark.ai to reflect realistic speech patterns under different cognitive loads.

Code to generate samples using Bark.ai:  
https://colab.research.google.com/drive/1i2SlgdYCbMpRiSnDXWpumkz2E17mxwpq?usp=sharing

Other sample audios used:  
https://drive.google.com/drive/folders/1IM6eCvccYgeXdu3-8hf9WezuFUpwdwnc?usp=sharing

Each .wav file is processed to extract both audio and text features.

**Key Libraries Used**

librosa - for audio feature extraction  
whisper - for transcription  
sklearn - for ML models like SVM, Isolation Forest, and Decision Tree  
matplotlib, pandas, numpy, re - for data handling and visualization

**Feature Extraction**

Audio features: MFCC, RMS, pitch variability, spectral features  
Text features: filler word count, repeated words, pause detection, speech rate  
Transcription is done using OpenAI Whisper

**Machine Learning Approaches**

Rule-Based Scoring - Flags samples with too many fillers or slow speech  
One-Class SVM - Finds outliers in audio and text features  
Isolation Forest - Efficient for detecting anomalies in feature data  
Decision Tree - Simple classifier trained on labeled anomaly scores

**Final Verdict**

Each method votes on whether a file is at risk.  
If 2 or more methods (including rule-based) detect risk, the file is flagged.  
Final risk scores are visualized in a bar chart for all files.



