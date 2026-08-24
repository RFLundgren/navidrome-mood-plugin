"""Mood Analyzer Service — standalone HTTP API for audio mood analysis.

Uses essentia-tensorflow with Discogs-EffNet embeddings to extract:
- Mood scores: happy, sad, relaxed, aggressive, party
- Danceability, BPM, energy

Genre/BPM context boosts and optional Last.fm crowd-sourced tag integration
are used to improve classification accuracy.

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import concurrent.futures
import json
import logging
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request

import mutagen
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy essentia/TensorFlow warnings that flood logs
logging.getLogger("essentia").setLevel(logging.ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Mood Analyzer", version="1.3.0")

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
ANALYSIS_DURATION = 120.0  # seconds — cap audio loaded per track for predictable analysis time

# /api/analysis/file must never read outside of this directory (see
# _resolve_music_path below). Resolved once at import time so every later
# comparison uses the same, symlink-free baseline.
MUSIC_DIR = os.path.realpath(os.environ.get("MUSIC_DIR", "/music"))

# Reject local files above this size before decoding — a legitimate track
# is a few tens of MB at most; this just keeps a mistaken/huge file from
# ballooning memory during decode.
MAX_ANALYSIS_FILE_SIZE_MB = int(os.environ.get("MAX_ANALYSIS_FILE_SIZE_MB", "200"))
MAX_ANALYSIS_FILE_SIZE_BYTES = MAX_ANALYSIS_FILE_SIZE_MB * 1024 * 1024

# Wall-clock safety net for /api/analysis/file, mirroring the timeout
# /api/analysis/url already gets from its ffmpeg subprocess call. Note this
# bounds how long the *request* waits, not the underlying native decode —
# essentia calls run in a worker thread and Python cannot forcibly kill a
# thread, so a pathological input can still burn CPU in the background
# after the client gets its 504. MUSIC_DIR containment + the size cap above
# are what actually keep untrusted input out of that code path.
FILE_ANALYSIS_TIMEOUT_SECONDS = float(os.environ.get("FILE_ANALYSIS_TIMEOUT_SECONDS", "90"))

_es = None
_models = {}


def _load_essentia():
    global _es
    if _es is None:
        import essentia.standard as es
        _es = es
    return _es


def _get_model(key, model_file, output_layer, is_effnet=False):
    """Load and cache an Essentia-TensorFlow model."""
    global _models
    if key not in _models:
        es = _load_essentia()
        model_path = os.path.join(MODELS_DIR, model_file)
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None
        try:
            if is_effnet:
                _models[key] = es.TensorflowPredictEffnetDiscogs(
                    graphFilename=model_path, output=output_layer
                )
            else:
                _models[key] = es.TensorflowPredict2D(
                    graphFilename=model_path, output=output_layer
                )
            logger.info(f"Loaded model: {key}")
        except Exception as e:
            logger.error(f"Failed to load model {key}: {e}")
            return None
    return _models.get(key)


# ── Genre/BPM Context Boosts ─────────────────────────────────

GENRE_BOOSTS = {
    # High-energy dance genres: boost danceability + party
    "drum": {"danceability": 0.35, "mood_party": 0.15, "mood_aggressive": 0.10},
    "jungle": {"danceability": 0.35, "mood_party": 0.15, "mood_aggressive": 0.10},
    "dnb": {"danceability": 0.35, "mood_party": 0.15, "mood_aggressive": 0.10},
    "dance": {"danceability": 0.20, "mood_party": 0.10},
    "house": {"danceability": 0.20, "mood_party": 0.10},
    "techno": {"danceability": 0.25, "mood_party": 0.15, "mood_aggressive": 0.10},
    "trance": {"danceability": 0.20, "mood_party": 0.10, "mood_happy": 0.10},
    "edm": {"danceability": 0.20, "mood_party": 0.10},
    "electronic": {"danceability": 0.10, "mood_party": 0.05},
    "disco": {"danceability": 0.20, "mood_party": 0.15, "mood_happy": 0.10},
    "funk": {"danceability": 0.15, "mood_party": 0.10, "mood_happy": 0.05},
    # Aggressive genres
    "metal": {"mood_aggressive": 0.25, "mood_relaxed": -0.15},
    "hardcore": {"mood_aggressive": 0.25, "danceability": 0.15},
    "punk": {"mood_aggressive": 0.15, "mood_party": 0.05},
    "industrial": {"mood_aggressive": 0.20},
    # Chill genres
    "ambient": {"mood_relaxed": 0.20, "mood_aggressive": -0.10},
    "downtempo": {"mood_relaxed": 0.15, "danceability": -0.10},
    "chillout": {"mood_relaxed": 0.20, "mood_party": -0.10},
    "lounge": {"mood_relaxed": 0.15},
    "easy listening": {"mood_relaxed": 0.20, "mood_happy": 0.10},
    "new age": {"mood_relaxed": 0.20},
    # Sad/melancholy
    "blues": {"mood_sad": 0.10},
    "emo": {"mood_sad": 0.15, "mood_aggressive": 0.05},
    # Happy/upbeat
    "reggae": {"mood_happy": 0.10, "mood_relaxed": 0.10},
    "ska": {"mood_happy": 0.15, "danceability": 0.10},
    "pop": {"mood_happy": 0.05, "danceability": 0.05},
    # R&B/Soul
    "r&b": {"mood_relaxed": 0.05, "mood_party": 0.05},
    "soul": {"mood_relaxed": 0.05, "mood_happy": 0.05},
}

BPM_DANCEABILITY_BOOST = {
    (140, 180): 0.20,  # DnB, breakbeat, fast house
    (180, 250): 0.15,  # Hardcore, speedcore
    (90, 110): 0.05,   # Hip-hop, trip-hop
}


# ── Last.fm Tag Boosts ────────────────────────────────────────

LASTFM_API_BASE = "https://ws.audioscrobbler.com/2.0/"

# Crowd-sourced tag keywords → score adjustments.
# Each keyword is applied at most once even if multiple tags match.
# Total per-field delta is capped at ±0.20 to limit Last.fm influence.
LASTFM_TAG_BOOSTS = {
    # Mood/feel tags
    "chill":      {"mood_relaxed": 0.12, "mood_aggressive": -0.05},
    "chillout":   {"mood_relaxed": 0.12, "mood_aggressive": -0.05},
    "relax":      {"mood_relaxed": 0.10},        # relax / relaxing / relaxed
    "calm":       {"mood_relaxed": 0.10},
    "mellow":     {"mood_relaxed": 0.08},
    "peaceful":   {"mood_relaxed": 0.08},
    "sleep":      {"mood_relaxed": 0.08, "mood_aggressive": -0.05},
    "acoustic":   {"mood_relaxed": 0.08},
    "ambient":    {"mood_relaxed": 0.10, "mood_aggressive": -0.08},
    "atmospheric":{"mood_relaxed": 0.08},
    "folk":       {"mood_relaxed": 0.08},
    "classical":  {"mood_relaxed": 0.10},
    "jazz":       {"mood_relaxed": 0.06},
    "happy":      {"mood_happy": 0.12},
    "feel good":  {"mood_happy": 0.12},
    "uplift":     {"mood_happy": 0.10},          # uplifting / uplifted
    "joyful":     {"mood_happy": 0.08},
    "upbeat":     {"mood_happy": 0.08, "danceability": 0.05},
    "sad":        {"mood_sad": 0.12, "mood_happy": -0.05},
    "melanchol":  {"mood_sad": 0.12},            # melancholy / melancholic
    "depress":    {"mood_sad": 0.08},
    "heartbreak": {"mood_sad": 0.08},
    "aggressive": {"mood_aggressive": 0.12},
    "angry":      {"mood_aggressive": 0.10},
    "brutal":     {"mood_aggressive": 0.15, "mood_relaxed": -0.10},
    "heavy":      {"mood_aggressive": 0.10, "mood_relaxed": -0.08},
    "party":      {"mood_party": 0.12, "danceability": 0.08},
    "danc":       {"danceability": 0.12, "mood_party": 0.08},   # dance / danceable
    "club":       {"danceability": 0.08, "mood_party": 0.08},
    "energetic":  {"danceability": 0.08, "mood_party": 0.05},
    # Genre tags reinforcing GENRE_BOOSTS for tracks missing local genre metadata
    "metal":      {"mood_aggressive": 0.12, "mood_relaxed": -0.10},
    "thrash":     {"mood_aggressive": 0.15, "mood_relaxed": -0.10},
    "death":      {"mood_aggressive": 0.15, "mood_relaxed": -0.10},  # death metal
    "hardcore":   {"mood_aggressive": 0.12, "mood_relaxed": -0.08},
    "punk":       {"mood_aggressive": 0.08},
    "grunge":     {"mood_aggressive": 0.06, "mood_sad": 0.05},
    "industrial": {"mood_aggressive": 0.10},
    "blues":      {"mood_sad": 0.08, "mood_relaxed": 0.05},
}


def _get_lastfm_tags(artist: str, title: str, api_key: str) -> list:
    """Fetch top tags for a track from Last.fm. Returns lowercased tag names."""
    if not api_key or not artist or not title:
        logger.info(f"Last.fm skipped — key={'set' if api_key else 'MISSING'} artist={artist!r} title={title!r}")
        return []
    try:
        params = urllib.parse.urlencode({
            "method": "track.getTopTags",
            "artist": artist,
            "track": title,
            "api_key": api_key,
            "format": "json",
            "autocorrect": 1,
        })
        req = urllib.request.Request(
            f"{LASTFM_API_BASE}?{params}",
            headers={"User-Agent": "navidrome-mood-analyzer/1.3.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        if "error" in data:
            return []
        tags = data.get("toptags", {}).get("tag", [])
        if isinstance(tags, dict):  # single-tag edge case from Last.fm API
            tags = [tags]
        return [t["name"].lower() for t in tags[:10] if isinstance(t, dict) and t.get("name")]
    except Exception as e:
        logger.warning(f"Last.fm lookup failed for {artist!r} - {title!r}: {e}")
        return []


def _apply_lastfm_boosts(scores: dict, tags: list, weight: float = 1.0) -> dict:
    """Apply Last.fm tag-based score adjustments. Each keyword matched at most once."""
    if not tags or weight == 0.0:
        return scores
    adjusted = dict(scores)
    field_delta = {}

    for keyword, boosts in LASTFM_TAG_BOOSTS.items():
        if any(keyword in tag for tag in tags):
            for field, delta in boosts.items():
                field_delta[field] = field_delta.get(field, 0.0) + delta * weight

    max_delta = 0.20
    for field, delta in field_delta.items():
        if field in adjusted:
            capped = max(-max_delta, min(max_delta, delta))
            adjusted[field] = round(max(0.0, min(1.0, adjusted[field] + capped)), 4)

    return adjusted


def _apply_context_boosts(scores, genre, artist, bpm, weight=1.0):
    """Adjust raw essentia scores based on genre/artist/BPM context."""
    adjusted = dict(scores)
    genre_lower = (genre or "").lower()

    for keyword, boosts in GENRE_BOOSTS.items():
        if keyword in genre_lower:
            for score_key, boost in boosts.items():
                if score_key in adjusted:
                    adjusted[score_key] = adjusted[score_key] + boost * weight

    if bpm and bpm > 0:
        effective_bpm = bpm
        if 80 <= bpm <= 95 and any(k in genre_lower for k in ("drum", "jungle", "dnb", "bass")):
            effective_bpm = bpm * 2

        for (lo, hi), boost in BPM_DANCEABILITY_BOOST.items():
            if lo <= effective_bpm <= hi:
                adjusted["danceability"] = adjusted.get("danceability", 0) + boost
                break

    for key in adjusted:
        if isinstance(adjusted[key], float):
            adjusted[key] = round(max(0.0, min(1.0, adjusted[key])), 4)

    return adjusted


# ── API ───────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    file_path: str
    lastfm_api_key: str = ""
    genre_weight: float = 1.0
    lastfm_weight: float = 1.0

class AnalyzeUrlRequest(BaseModel):
    url: str
    lastfm_api_key: str = ""
    artist: str = ""
    title: str = ""
    genre_weight: float = 1.0
    lastfm_weight: float = 1.0


@app.get("/health")
def health():
    try:
        _load_essentia()
        effnet = os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb")
        return {"status": "ok", "models_available": os.path.exists(effnet)}
    except ImportError as e:
        return {"status": "error", "message": f"essentia not installed: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}


def _analyze_path(file_path: str, api_key: str = "", genre_weight: float = 1.0, lastfm_weight: float = 1.0) -> dict:
    es = _load_essentia()
    results = {}

    # Read metadata
    title, artist, album, genre = "", "", "", ""
    try:
        tags = mutagen.File(file_path)
        if tags:
            title = str((tags.get("\xa9nam") or tags.get("title") or [""])[0])
            artist = str((tags.get("\xa9ART") or tags.get("artist") or [""])[0])
            album = str((tags.get("\xa9alb") or tags.get("album") or [""])[0])
            genre = str((tags.get("\xa9gen") or tags.get("genre") or [""])[0])
    except Exception:
        pass
    if not title:
        title = os.path.splitext(os.path.basename(file_path))[0]

    # BPM
    try:
        audio_44k = es.MonoLoader(filename=file_path, sampleRate=44100)()
        max_samples = int(ANALYSIS_DURATION * 44100)
        if len(audio_44k) > max_samples:
            audio_44k = audio_44k[:max_samples]
        bpm = es.RhythmExtractor2013(method="multifeature")(audio_44k)[0]
        results["bpm"] = round(float(bpm), 1)
    except Exception as e:
        logger.warning(f"BPM failed for {file_path}: {e}")
        results["bpm"] = 0.0

    # Energy (RMS)
    try:
        results["energy"] = round(float(np.sqrt(np.mean(audio_44k ** 2))), 4)
    except Exception:
        results["energy"] = 0.0

    # Embeddings
    audio_16k = es.MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
    max_samples_16k = int(ANALYSIS_DURATION * 16000)
    if len(audio_16k) > max_samples_16k:
        audio_16k = audio_16k[:max_samples_16k]

    effnet_model = _get_model("effnet", "discogs-effnet-bs64-1.pb", "PartitionedCall:1", is_effnet=True)
    if not effnet_model:
        logger.error("Effnet model not available")
        return {"error": "Internal model error"}

    try:
        embeddings = effnet_model(audio_16k)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        results.update({k: 0.0 for k in [
            "danceability", "mood_happy", "mood_sad",
            "mood_relaxed", "mood_aggressive", "mood_party"
        ]})
        results = _apply_context_boosts(results, genre, artist, results.get("bpm", 0), weight=genre_weight)
        return {"file_path": file_path, "title": title, "artist": artist,
                "album": album, "genre": genre, **results}

    # Mood classifiers
    mood_models = {
        "mood_happy": "mood_happy-discogs-effnet-1.pb",
        "mood_sad": "mood_sad-discogs-effnet-1.pb",
        "mood_relaxed": "mood_relaxed-discogs-effnet-1.pb",
        "mood_aggressive": "mood_aggressive-discogs-effnet-1.pb",
        "mood_party": "mood_party-discogs-effnet-1.pb",
    }

    for key, model_file in mood_models.items():
        try:
            model = _get_model(key, model_file, "model/Softmax")
            if not model:
                results[key] = 0.0
                continue
            preds = model(embeddings)
            results[key] = round(float(np.mean(preds[:, 1])), 4)
        except Exception as e:
            logger.warning(f"{key} failed: {e}")
            results[key] = 0.0

    # Danceability
    try:
        dance_model = _get_model("danceability", "danceability-discogs-effnet-1.pb", "model/Softmax")
        if dance_model:
            preds = dance_model(embeddings)
            results["danceability"] = round(float(np.mean(preds[:, 1])), 4)
        else:
            results["danceability"] = 0.0
    except Exception as e:
        logger.warning(f"Danceability failed: {e}")
        results["danceability"] = 0.0

    # Apply genre/BPM context boosts
    results = _apply_context_boosts(results, genre, artist, results.get("bpm", 0), weight=genre_weight)

    # Apply Last.fm crowd-sourced tag boosts
    lastfm_tags = _get_lastfm_tags(artist, title, api_key)
    if lastfm_tags:
        logger.info(f"Last.fm tags for {artist!r} - {title!r}: {lastfm_tags}")
        results = _apply_lastfm_boosts(results, lastfm_tags, weight=lastfm_weight)

    return {"file_path": file_path, "title": title, "artist": artist,
            "album": album, "genre": genre, **results}


def _resolve_music_path(file_path: str) -> str:
    """Resolve file_path and enforce that it stays inside MUSIC_DIR.

    file_path may be absolute or relative to MUSIC_DIR either way; symlinks
    are fully resolved via realpath() *before* the containment check, so a
    symlink that lives inside MUSIC_DIR but points outside of it is caught
    too. Raises HTTPException for anything that isn't a regular file inside
    MUSIC_DIR, or that exceeds MAX_ANALYSIS_FILE_SIZE_BYTES.
    """
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")

    candidate = file_path if os.path.isabs(file_path) else os.path.join(MUSIC_DIR, file_path)
    resolved = os.path.realpath(candidate)

    if os.path.commonpath([resolved, MUSIC_DIR]) != MUSIC_DIR:
        logger.warning(f"Rejected file_path outside MUSIC_DIR: {file_path!r} -> {resolved!r}")
        raise HTTPException(status_code=403, detail="file_path must be inside the music library")

    # isfile() follows symlinks (already resolved above) and is False for
    # directories, device files, pipes, sockets, etc. — this alone blocks
    # things like /dev/zero or /dev/null even if they somehow resolved
    # inside MUSIC_DIR.
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if os.path.getsize(resolved) > MAX_ANALYSIS_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_ANALYSIS_FILE_SIZE_MB} MB analysis limit",
        )

    return resolved


@app.post("/api/analysis/file")
def analyze_file(req: AnalyzeRequest):
    resolved_path = _resolve_music_path(req.file_path)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_analyze_path, resolved_path, req.lastfm_api_key, req.genre_weight, req.lastfm_weight)
    try:
        return future.result(timeout=FILE_ANALYSIS_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        logger.error(f"Analysis timed out after {FILE_ANALYSIS_TIMEOUT_SECONDS}s for {resolved_path}")
        raise HTTPException(status_code=504, detail="Analysis timed out")
    finally:
        # wait=False: don't block this response on a hung worker thread.
        pool.shutdown(wait=False)


@app.post("/api/analysis/url")
def analyze_url(req: AnalyzeUrlRequest):
    """Fetch audio from a URL and analyze it.

    Uses ffmpeg to stream only the first ANALYSIS_DURATION seconds directly
    from the URL, converting to a small mono WAV (~7 MB) regardless of the
    source format or bitrate. This avoids downloading multi-hundred-MB FLAC
    files in full before analysis can begin.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-t", str(ANALYSIS_DURATION),
                "-i", req.url,
                "-ar", "44100",
                "-ac", "1",
                "-sample_fmt", "s16",
                tmp_path,
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace").strip()
            logger.error(f"ffmpeg failed (rc={result.returncode}):\n{err}")
            last_line = err.splitlines()[-1] if err else "ffmpeg failed"
            raise Exception(last_line)

        result = _analyze_path(tmp_path, genre_weight=req.genre_weight)
        # Use plugin-supplied artist/title for Last.fm — temp WAV metadata is unreliable
        logger.info(f"URL analysis request — key={'set' if req.lastfm_api_key else 'MISSING'} artist={req.artist!r} title={req.title!r}")
        if req.lastfm_api_key and req.artist and req.title:
            lastfm_tags = _get_lastfm_tags(req.artist, req.title, req.lastfm_api_key)
            if lastfm_tags:
                logger.info(f"Last.fm tags for {req.artist!r} - {req.title!r}: {lastfm_tags}")
                result = _apply_lastfm_boosts(result, lastfm_tags, weight=req.lastfm_weight)
        return result
    except Exception as e:
        logger.error(f"URL analysis error ({type(e).__name__}): {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch or analyze URL: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
