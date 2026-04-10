# 🎙️ AURAFORGE

> **"Cloudflare for Audio Authenticity"**  
> Detect AI-generated music in milliseconds via API.

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```
data/
  ai/        ← AI-generated audio (MP3/WAV) — get from Suno, Udio, MusicGen
  human/     ← Human-made audio (MP3/WAV) — from GTZAN, FMA, your own beats
```

### 3. Train Model
```bash
python ml/train.py \
  --ai_dir data/ai \
  --human_dir data/human \
  --output model.pkl
```

### 4. Run API
```bash
uvicorn main:app --reload --port 8000
```

### 5. Test
```bash
curl -X POST http://localhost:8000/analyze-audio \
  -H "X-API-Key: demo-key-123" \
  -F "file=@my_track.mp3"
```

**Response:**
```json
{
  "ai_probability": 0.8312,
  "decision": "BLOCK",
  "confidence": "HIGH",
  "file_name": "my_track.mp3",
  "audio_fingerprint": "a3f9c21...",
  "processing_time_ms": 243.5
}
```

---

## Docker

```bash
docker build -t auraforge .
docker run -p 8000:8000 -v $(pwd)/model.pkl:/app/model.pkl auraforge
```

---

## Architecture

```
Client (Beat22 / Spotify / etc.)
        ↓
   AURAFORGE API  (FastAPI)
        ↓
   Feature Extraction  (librosa — 54 features, first 30s)
        ↓
   ML Model  (RandomForest sklearn pipeline)
        ↓
   Decision Engine  (ALLOW / FLAG / BLOCK)
        ↓
   JSON Response
```

---

## Feature Engineering

| Feature | Count | Why |
|---------|-------|-----|
| MFCC (mean+std) | 26 | Timbral texture — AI sounds "too smooth" |
| Spectral Centroid | 2 | Brightness/darkness pattern |
| Spectral Flatness | 2 | Noise-like vs tonal — AI is unnaturally tonal |
| Spectral Contrast | 7 | Peak/valley dynamics |
| Chroma | 12 | Harmonic content |
| ZCR | 2 | Noisiness proxy |
| RMS Energy | 2 | Dynamic range — AI often over-compressed |
| Tempo | 1 | Rhythmic regularity |

---

## Datasets

### AI Audio
- [Suno.com](https://suno.com) — download generated tracks
- [Udio.com](https://udio.com) — download generated tracks
- HuggingFace: `facebook/musicgen-small` outputs

### Human Audio
- [GTZAN Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) — 1000 tracks, 10 genres
- [FMA](https://github.com/mdeff/fma) — Free Music Archive, 25k tracks
- Your own production stems

**MVP target:** 50 AI + 50 Human (enough for proof-of-concept)  
**Production target:** 500+ per class

---

## Decision Thresholds

| Score | Decision | Meaning |
|-------|----------|---------|
| < 0.40 | ALLOW | Likely human, pass through |
| 0.40 – 0.69 | FLAG | Ambiguous, queue for review |
| ≥ 0.70 | BLOCK | Likely AI-generated, reject |

Thresholds are configurable per platform via environment variables.

---

## API Key Management

Set valid keys via environment variable:
```bash
export AURAFORGE_API_KEYS="key1,key2,key3"
export REQUIRE_API_KEY=true
```

---

## Roadmap

- [x] MVP: RandomForest + FastAPI
- [ ] CNN model on mel-spectrograms (higher accuracy)
- [ ] Redis cache (hash → result, skip reprocessing)
- [ ] Dashboard with analytics
- [ ] Per-platform threshold configuration
- [ ] Webhook callbacks for async processing
- [ ] Rate limiting per API key
