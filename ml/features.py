"""
AURAFORGE Feature Extraction
Extracts audio fingerprint features for ML classification.

"""

import numpy as np
import librosa
import soundfile as sf
from typing import Optional
import logging

logger = logging.getLogger("auraforge.features")


def extract_features(
    audio_path: str,
    max_seconds: Optional[int] = 30,
    sr: int = 22050
) -> np.ndarray:
    """
    Extract a rich feature vector from an audio file.
    
    Features extracted:
    - MFCC (13 coefficients, mean + std) = 26 features
    - Spectral Centroid (mean + std) = 2 features
    - Spectral Flatness (mean + std) = 2 features
    - Spectral Contrast (7 bands, mean) = 7 features
    - Chroma Features (12, mean) = 12 features
    - Zero Crossing Rate (mean + std) = 2 features
    - RMS Energy (mean + std) = 2 features
    - Tempo = 1 feature
    
    Total: 54 features
    
    Args:
        audio_path: Path to audio file (MP3/WAV)
        max_seconds: Max duration to process (default: 30s for speed)
        sr: Sample rate
    
    Returns:
        np.ndarray of shape (54,)
    """
    try:
        # Load audio, truncate to max_seconds for performance
        duration = max_seconds if max_seconds else None
        y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    except Exception as e:
        raise ValueError(f"Could not load audio file: {e}")

    features = []

    # 1. MFCC - 13 coefficients × (mean + std) = 26 features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. Spectral Centroid - mean + std = 2 features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))

    # 3. Spectral Flatness - mean + std = 2 features
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    features.append(np.mean(spectral_flatness))
    features.append(np.std(spectral_flatness))

    # 4. Spectral Contrast - 7 bands, mean = 7 features
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(spectral_contrast, axis=1))

    # 5. Chroma - 12 bins, mean = 12 features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # 6. Zero Crossing Rate - mean + std = 2 features
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 7. RMS Energy - mean + std = 2 features
    rms = librosa.feature.rms(y=y)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # 8. Tempo = 1 feature
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    feature_vector = np.array(features, dtype=np.float32)
    
    logger.debug(f"Extracted {len(feature_vector)} features from {audio_path}")
    
    return feature_vector


def validate_feature_vector(features: np.ndarray) -> bool:
    """Sanity check the feature vector."""
    if features.shape[0] != 54:
        return False
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return False
    return True
