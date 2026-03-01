# AURAFORGE  
## AI-Powered Music Authenticity & Forensic Analysis Engine

AURAFORGE is an AI-driven prototype designed to detect AI-generated music and analyze authenticity signals using advanced audio feature extraction and forensic signal processing.

Built for the AMD Slingshot AI Innovation Challenge.

---

## 🚀 Problem Statement

With the rapid rise of generative AI music platforms, streaming services and distributors lack a reliable system to verify whether uploaded tracks are AI-generated or human-created.

This creates:

- Intellectual property disputes  
- Copyright violations  
- Revenue leakage  
- Legal and regulatory risks  

AURAFORGE introduces a scalable authenticity verification layer for the modern music ecosystem.

---

## 🧠 Solution Overview

AURAFORGE processes audio files and extracts advanced spectral and temporal features to evaluate authenticity probability.

### Core Capabilities:

- MP3 audio ingestion
- RMS energy analysis
- Zero-crossing rate extraction
- Spectral centroid analysis
- Spectral flatness detection
- Heuristic AI probability scoring
- Forensic-style authenticity report generation
- Waveform visualization
- Spectrogram analysis
- Feature distribution plotting

This prototype validates the signal-processing backbone of a larger deep learning authenticity infrastructure.

---

## 🏗️ Architecture Vision

Production-scale system designed to include:

- Stem-level separation
- Generative model artifact detection
- Supervised deep learning classification
- AMD Instinct GPU accelerated training
- ROCm-based compute optimization
- Real-time API integration for streaming platforms

---

## ⚙️ Tech Stack

- Python 3.10
- NumPy
- Librosa
- Matplotlib
- SoundFile
- Audioread

---

## 📊 How It Works

1. User uploads an MP3 file.
2. Audio is processed and key acoustic features are extracted.
3. Heuristic scoring estimates AI vs Human probability.
4. A forensic authenticity report is generated.
5. Visual artifacts are saved:
   - `waveform.png`
   - `spectrogram.png`
   - `feature_plot.png`

---

## 🖥️ How to Run

### 1️⃣ Clone Repository


git clone https://github.com/Mayerust/auraforge-ai-authenticity.git

cd auraforge-ai-authenticity


### 2️⃣ Create Virtual Environment


python3.10 -m venv venv
source venv/bin/activate


### 3️⃣ Install Dependencies


pip install -r requirements.txt


### 4️⃣ Run Prototype


python demo.py


Enter the full path to an MP3 file when prompted.

---

## 📈 Sample Output

The system generates:

- Authenticity probability report
- Feature extraction metrics
- Saved waveform visualization
- Spectrogram mapping
- Feature distribution scatter plot

---

## 🔒 Future Scope

- Large-scale dataset training
- Supervised CNN spectrogram classification
- Generative model fingerprint detection
- Blockchain-based provenance tracking
- Streaming platform API deployment
- Enterprise IP protection tooling

---

## 🏆 Vision

AURAFORGE is designed to evolve into scalable AI-powered authenticity infrastructure for the global music industry.

Securing creativity in the generative era.

---

## 👤 Author

Mayank Verma  
Music Producer | AI Developer | Cybersecurity Enthusiast  

---

## 📜 License

Prototype developed for hackathon demonstration purposes.
