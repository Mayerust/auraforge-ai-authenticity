"""
AURAFORGE — Audio Authenticity Detection API
"Cloudflare for Audio" — B2B SaaS ML pipeline

Run locally:
    uvicorn main:app --reload --port 8000

API docs:
    http://localhost:8000/docs
"""

import os
import hashlib
import time
import logging
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from ml.inference import AuraForgeModel
from ml.features import extract_features
from core.decision import decide, confidence_label
from core.auth import validate_api_key
from core.logger import log_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("auraforge")

app = FastAPI(
    title="AURAFORGE API",
    description=(
        "AI-generated audio detection API.\n\n"
        "Upload an MP3 or WAV file and receive a decision: **ALLOW**, **FLAG**, or **BLOCK**.\n\n"
        "Built for B2B integration with music platforms."
    ),
    version="1.0.0",
    contact={"name": "AURAFORGE", "email": "admin@auraforge.io"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static demo page
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pre-load model once at startup
model = AuraForgeModel()

REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"


@app.on_event("startup")
async def startup_event():
    logger.info("AURAFORGE API starting up ...")
    model.load()
    if model.is_loaded:
        logger.info("Model loaded and ready")
    else:
        logger.warning("model.pkl not found — run: python ml/train.py first")


@app.get("/", tags=["info"])
def root():
    return {
        "service": "AURAFORGE",
        "version": "1.0.0",
        "status": "operational",
        "tagline": "Cloudflare for Audio Authenticity",
        "docs": "/docs",
        "demo": "/static/demo.html",
    }


@app.get("/health", tags=["info"])
def health():
    return {
        "status": "ok",
        "model_loaded": model.is_loaded,
        "api_key_required": REQUIRE_API_KEY,
    }


@app.post("/analyze-audio", tags=["detection"])
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file — MP3 or WAV, max 20 MB"),
    x_api_key: str = Header(None, description="Your AURAFORGE API key"),
):
    """
    Analyze an audio file for AI generation probability.

    Returns ai_probability (0-1), decision (ALLOW/FLAG/BLOCK),
    confidence (HIGH/MEDIUM/LOW), and processing latency.
    """
    if REQUIRE_API_KEY:
        validate_api_key(x_api_key)

    content = await file.read()

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 20 MB.")

    if len(content) < 1024:
        raise HTTPException(status_code=400, detail="File too small — must be valid audio.")

    ext = Path(file.filename or "track.mp3").suffix.lower()
    if ext not in {".mp3", ".wav"}:
        raise HTTPException(status_code=415, detail=f"Unsupported format '{ext}'. Use .mp3 or .wav")

    start = time.perf_counter()
    audio_hash = hashlib.md5(content).hexdigest()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path, max_seconds=30)
        score = model.predict(features)
        decision = decide(score)
        confidence = confidence_label(score)
        processing_ms = round((time.perf_counter() - start) * 1000, 1)

        result = {
            "ai_probability": round(float(score), 4),
            "decision": decision,
            "confidence": confidence,
            "file_name": file.filename,
            "audio_fingerprint": audio_hash,
            "processing_time_ms": processing_ms,
        }

        log_analysis(result)
        return JSONResponse(content=result)

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal analysis error.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/batch-analyze", tags=["detection"])
async def batch_analyze(
    files: list[UploadFile] = File(...),
    x_api_key: str = Header(None),
):
    """Analyze up to 10 audio files in one request."""
    if REQUIRE_API_KEY:
        validate_api_key(x_api_key)

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch.")

    results = []
    for f in files:
        r = await analyze_audio(file=f, x_api_key=x_api_key)
        results.append(r)

    return {"batch_results": results, "total": len(results)}
