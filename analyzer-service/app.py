"""Mood Analyzer Service — standalone HTTP API for audio mood analysis.

Uses essentia-tensorflow with Discogs-EffNet embeddings to extract:
- Mood scores: happy, sad, relaxed, aggressive, party
- Danceability, BPM, energy

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os

import mutagen
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mood Analyzer", version="1.0.0")

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

# Lazy-loaded essentia
_es = None


def _load_essentia():
    global _es
    if _es is None:
        import essentia.standard as es
        _es = es
    return _es


class AnalyzeRequest(BaseModel):
    file_path: str


@app.get("/health")
def health():
    try:
        _load_essentia()
        effnet = os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb")
        return {"status": "ok", "models_available": os.path.exists(effnet)}
    except ImportError:
        return {"status": "error", "message": "essentia not installed"}


@app.post("/api/analysis/file")
def analyze_file(req: AnalyzeRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    es = _load_essentia()
    results = {}

    # Read metadata
    title, artist, album = "", "", ""
    try:
        tags = mutagen.File(req.file_path)
        if tags:
            title = str((tags.get("\xa9nam") or tags.get("title") or [""])[0])
            artist = str((tags.get("\xa9ART") or tags.get("artist") or [""])[0])
            album = str((tags.get("\xa9alb") or tags.get("album") or [""])[0])
    except Exception:
        pass
    if not title:
        title = os.path.splitext(os.path.basename(req.file_path))[0]

    # BPM
    try:
        audio_44k = es.MonoLoader(filename=req.file_path, sampleRate=44100)()
        bpm = es.RhythmExtractor2013(method="multifeature")(audio_44k)[0]
        results["bpm"] = round(float(bpm), 1)
    except Exception as e:
        logger.warning(f"BPM failed for {req.file_path}: {e}")
        results["bpm"] = 0.0

    # Energy (RMS)
    try:
        results["energy"] = round(float(np.sqrt(np.mean(audio_44k ** 2))), 4)
    except Exception:
        results["energy"] = 0.0

    # Embeddings
    audio_16k = es.MonoLoader(filename=req.file_path, sampleRate=16000, resampleQuality=4)()
    effnet_path = os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb")

    try:
        embeddings = es.TensorflowPredictEffnetDiscogs(
            graphFilename=effnet_path, output="PartitionedCall:1"
        )(audio_16k)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        results.update({k: 0.0 for k in [
            "danceability", "mood_happy", "mood_sad",
            "mood_relaxed", "mood_aggressive", "mood_party"
        ]})
        return {"file_path": req.file_path, "title": title, "artist": artist, "album": album, **results}

    # Mood classifiers
    mood_models = {
        "mood_happy": "mood_happy-discogs-effnet-1.pb",
        "mood_sad": "mood_sad-discogs-effnet-1.pb",
        "mood_relaxed": "mood_relaxed-discogs-effnet-1.pb",
        "mood_aggressive": "mood_aggressive-discogs-effnet-1.pb",
        "mood_party": "mood_party-discogs-effnet-1.pb",
    }

    for key, model_file in mood_models.items():
        model_path = os.path.join(MODELS_DIR, model_file)
        try:
            if not os.path.exists(model_path):
                results[key] = 0.0
                continue
            preds = es.TensorflowPredict2D(
                graphFilename=model_path, output="model/Softmax"
            )(embeddings)
            results[key] = round(float(np.mean(preds[:, 1])), 4)
        except Exception as e:
            logger.warning(f"{key} failed: {e}")
            results[key] = 0.0

    # Danceability
    dance_path = os.path.join(MODELS_DIR, "danceability-discogs-effnet-1.pb")
    try:
        if os.path.exists(dance_path):
            preds = es.TensorflowPredict2D(
                graphFilename=dance_path, output="model/Softmax"
            )(embeddings)
            results["danceability"] = round(float(np.mean(preds[:, 1])), 4)
        else:
            results["danceability"] = 0.0
    except Exception as e:
        logger.warning(f"Danceability failed: {e}")
        results["danceability"] = 0.0

    return {"file_path": req.file_path, "title": title, "artist": artist, "album": album, **results}
