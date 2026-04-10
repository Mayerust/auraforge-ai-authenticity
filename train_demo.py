

import os
import numpy as np
import soundfile as sf
from pathlib import Path
import subprocess
import sys


def make_synthetic_ai_audio(path: str, sr: int = 22050, duration: int = 10):

    """
    Simulate AI-generated audio characteristics:
    - Very regular, quantised rhythm
    - Smooth, narrow spectral envelope (no natural variation)
    - Consistent amplitude (over-compressed)
    """

    t = np.linspace(0, duration, sr * duration)
    #Perfectly regular harmonic stack — no breath, no imperfection
    freqs = [110, 220, 330, 440, 550, 660]
    audio = sum(0.15 * np.sin(2 * np.pi * f * t) for f in freqs)
    #Add machine-regular LFO
    lfo = 0.3 * np.sin(2 * np.pi * 2.0 * t)
    audio = audio * (1 + lfo)
    #Normalize tightly (AI masters are over-compressed)
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.95
    audio = audio.astype(np.float32)
    sf.write(path, audio, sr)


def make_synthetic_human_audio(path: str, sr: int = 22050, duration: int = 10):

    """
    Simulate human-played audio characteristics:
    - Slight timing irregularities
    - Natural spectral variation (breath, room noise)
    - Dynamic amplitude (not compressed to a brick)
    """

    t = np.linspace(0, duration, sr * duration)
    np.random.seed(hash(path) % 2**31)

    #Fundamental with natural slight detuning
    detune = 1 + np.random.uniform(-0.003, 0.003)
    audio = 0.4 * np.sin(2 * np.pi * 261.63 * detune * t)

    #Harmonics with varying levels (natural timbre)
    for h in [2, 3, 4]:
        amp = np.random.uniform(0.05, 0.25)
        audio += amp * np.sin(2 * np.pi * 261.63 * h * t)

    #Room noise / breath noise
    audio += np.random.randn(len(t)) * 0.015

    #Humanised amplitude envelope (slight swell, not a brick)
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 0.7 * t + np.random.uniform(0, np.pi))
    audio = audio * envelope

    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.75
    audio = audio.astype(np.float32)
    sf.write(path, audio, sr)


def generate_dataset(n_per_class: int = 40):
    ai_dir = Path("data/ai")
    human_dir = Path("data/human")
    ai_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_per_class} synthetic AI tracks")
    for i in range(n_per_class):
        make_synthetic_ai_audio(str(ai_dir / f"ai_track_{i:03d}.wav"))

    print(f"Generating {n_per_class} synthetic human tracks")
    for i in range(n_per_class):
        make_synthetic_human_audio(str(human_dir / f"human_track_{i:03d}.wav"))

    print(f"Dataset ready: {n_per_class} AI + {n_per_class} human tracks\n")


if __name__ == "__main__":
    #Install deps if missing
    try:
        import soundfile
        import librosa
        import sklearn
        import joblib
    except ImportError:
        print("Installing dependencies")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    generate_dataset(n_per_class=40)

    #Run the real training pipeline
    from ml.train import load_dataset, train_model, evaluate_model
    import joblib
    from sklearn.model_selection import train_test_split

    X, y, _ = load_dataset("data/ai", "data/human")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}")
    pipeline = train_model(X_train, y_train, model_type="random_forest")
    pipeline.fit(X_train, y_train)
    evaluate_model(pipeline, X_test, y_test)

    joblib.dump(pipeline, "model.pkl")
    print("\nmodel.pkl saved")
