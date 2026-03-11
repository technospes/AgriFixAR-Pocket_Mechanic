import json
import os
import re
import asyncio
import time
import struct
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterator
import logging
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv()
import hashlib
from contextlib import asynccontextmanager

# SlowAPI (rate limiting)
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# RAG imports
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from typing import List as _List
# Optimized RAG pipeline
from rag import (
    retrieve_with_metadata_filter,
    infer_problem_categories,
    normalize_query,
    RAG_TOP_K,
    RAG_MIN_SCORE,
)

# ── New modular imports ───────────────────────────────────────────────────────
from agent.models import CreateSessionRequest, CreateSessionResponse, AgentNextRequest
from agent import session_manager, repair_agent
from services.transcription_service import transcribe_audio_with_gemini
from services.diagnosis_service import generate_diagnosis_with_gemini
from services.verification_service import verify_step_with_gemini
from services.machine_detection_service import detect_machine, load_clip_model
from utils.helpers import (
    sanitize_json_text,
    generate_cache_key,
    get_cached_response,
    cache_response,
    cleanup_old_files,
    derive_part_and_area,
)
from utils.machine_registry import (
    list_supported_machines,
    get_profile,
    resolve_machine_id,
)

# ── Security layer ────────────────────────────────────────────────────────────
from security import (
    limiter,
    rate_limit_exceeded_handler,
    can_call_gemini,
    record_gemini_call,
    validate_video_upload,
    validate_audio_upload,
    verify_app_key,
    gemini_with_timeout,
    check_prompt_injection,
    log_security_event,
    get_gemini_usage,
    # Limit constants — used in /health and startup log so operators can
    # verify what the server is enforcing without reading source code.
    VIDEO_MAX_BYTES,
    AUDIO_MAX_BYTES,
    VIDEO_MAX_SECONDS,
    AUDIO_MAX_SECONDS,
    GEMINI_HOURLY_LIMIT,
)

# ── Embedding wrapper ─────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _cached_embed_query(model: str, text: str) -> tuple:
    """
    Cache up to 256 query embeddings in memory (process-scoped, workers=1).

    Returns a tuple so lru_cache can hash the result.
    The caller converts back to list before use.

    Eviction: LRU — oldest unused entry is dropped when maxsize is reached.
    Thread-safety: lru_cache is thread-safe in CPython; safe under asyncio.
    """
    result = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_query",
    )
    return tuple(result["embedding"])


class GoogleEmbeddingsV1(Embeddings):
    def __init__(self, model: str, google_api_key: str):
        self.model = model
        genai.configure(api_key=google_api_key)

    def embed_documents(self, texts: _List[str]) -> _List[_List[float]]:
        # Called only during build — no cache needed here
        return [
            genai.embed_content(model=self.model, content=t, task_type="retrieval_document")["embedding"]
            for t in texts
        ]

    def embed_query(self, text: str) -> _List[float]:
        # Returns cached embedding on repeated queries — zero API cost on hit
        return list(_cached_embed_query(self.model, text))


# ============================================================================
# INITIALIZATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")
if not GOOGLE_AI_API_KEY:
    raise ValueError("❌ GOOGLE_AI_API_KEY not found! Set it in .env or HF Secrets")

genai.configure(api_key=GOOGLE_AI_API_KEY)
logger.info("✅ Gemini API configured successfully")

# ============================================================================
# CONFIGURATION
# ============================================================================

UPLOAD_DIR = Path("temp_uploads")
KB_DIR = Path("knowledge_base")
CACHE_DIR = Path("response_cache")
CHROMA_DIR = Path("chroma_db")
PLAN_CACHE_DIR = Path("plan_cache")   # ← repair plan cache
MAX_IMAGE_SIZE = 512
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_CACHE_AGE = 86400
PLAN_CACHE_TTL = int(os.environ.get("PLAN_CACHE_TTL_SECONDS", str(30 * 24 * 3600)))  # 30 days default

UPLOAD_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
PLAN_CACHE_DIR.mkdir(exist_ok=True)


def _safe_filename(raw: str, fallback: str) -> str:
    """
    Strip any directory components from a client-supplied filename.
    Prevents path traversal attacks like '../../../etc/passwd'.
    Returns only the final filename component, or `fallback` if empty.

    Example:
        _safe_filename("../../../etc/passwd", "rec.mp4") → "etc/passwd" → "passwd"
        Wait — Path("../../../etc/passwd").name → "passwd"  ✓
    """
    name = Path(raw).name if raw else ""
    # Also strip leading dots that could create hidden files (e.g. ".bashrc")
    name = name.lstrip(".")
    return name or fallback

# ============================================================================
# REPAIR PLAN CACHE
# ============================================================================
# Key: SHA-256( machine_type + "|" + problem_cluster )
# Value: JSON file on disk — survives server restarts.
#
# problem_cluster = sorted(infer_problem_categories(problem_text)).join(",")
# This normalises "motor won't start" and "engine not starting" to the same
# cluster key, so we reuse the cached plan for semantically identical complaints.
#
# Cache is intentionally NOT invalidated by RAG chunk updates — if the problem
# cluster is the same, the repair logic is the same.  TTL is 30 days.
# Operators can clear the cache folder to force fresh generation.

def _plan_cache_key(machine_type: str, problem_text: str,
                    visual_hash: str = "") -> str:
    """
    Deterministic cache key: machine_type + problem cluster + visual hash.

    The visual_hash is a perceptual hash of the best video frame (8 bytes, hex).
    This is CRITICAL for accuracy: without it, "water_pump + not_working" would
    serve the same cached plan whether the motor is dead OR water is flowing.
    Two visually different states → two different cache keys → two correct plans.

    visual_hash="" (default) is kept for callers that have no frame (e.g. the
    non-streaming /diagnose endpoint when video is absent).
    """
    cats = sorted(infer_problem_categories(problem_text))
    cluster = ",".join(cats) if cats else normalize_query(problem_text)[:60]
    raw = f"{machine_type.lower()}|{cluster}|{visual_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _phash_bytes(image_bytes: bytes, hash_size: int = 8) -> str:
    """
    Average perceptual hash — 16-char hex string used in the plan cache key.
    Operates on JPEG bytes already in RAM from detection.frames. No I/O.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
        pixels = list(img.getdata())
        mean = sum(pixels) / len(pixels)
        bits = "".join("1" if p >= mean else "0" for p in pixels)
        return int(bits, 2).to_bytes(hash_size, "big").hex()
    except Exception:
        return "00000000"


def _plan_cache_get(key: str) -> Optional[Dict[str, Any]]:
    """Return cached plan dict or None if missing / expired."""
    p = PLAN_CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        age = time.time() - p.stat().st_mtime
        if age > PLAN_CACHE_TTL:
            p.unlink(missing_ok=True)
            logger.info(f"🗑️  Plan cache expired and removed: {key}")
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        logger.info(f"🎯 Plan cache HIT: {key}")
        return data
    except Exception as exc:
        logger.warning(f"Plan cache read error ({key}): {exc}")
        return None


def _plan_cache_set(key: str, plan: Dict[str, Any]) -> None:
    """Write plan to disk cache. Failures are non-fatal."""
    try:
        p = PLAN_CACHE_DIR / f"{key}.json"
        p.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        logger.info(f"💾 Plan cache WRITE: {key}")
    except Exception as exc:
        logger.warning(f"Plan cache write error ({key}): {exc}")

# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 AgriFix v4.1 Backend starting...")
    UPLOAD_DIR.mkdir(exist_ok=True)
    KB_DIR.mkdir(exist_ok=True)

    # Load CLIP once at startup — shared across all requests (workers=1)
    clip_ok = await asyncio.get_event_loop().run_in_executor(None, load_clip_model)
    logger.info(
        "🔮 CLIP detector: ACTIVE" if clip_ok
        else "⚠️  CLIP detector: DISABLED — audio keywords + Gemini fallback only"
    )

    rag_ok = load_vector_db()
    logger.info("🔍 RAG pipeline: ACTIVE" if rag_ok else "⚠️  RAG pipeline: DISABLED — Gemini-only mode")
    yield
    logger.info("👋 AgriFix Backend shutting down...")
    cleanup_old_files(UPLOAD_DIR)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AgriFix AI Backend - Hybrid Production",
    description="Agricultural machinery repair assistant with AI-powered stateful agent",
    version="4.2.0",
    lifespan=lifespan,
)

# ── Security: attach SlowAPI limiter state ────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    # ⚠️  Never use "*" with allow_credentials=True — browsers block it and it
    # allows any website to make credentialed cross-origin requests to this API.
    # List only the exact origins that legitimately call this backend.
    # Flutter mobile apps do NOT use CORS (no browser), so this only matters
    # for web builds or the HuggingFace Spaces demo page.
    allow_origins=[
        "https://agrifix.hf.space",        # HuggingFace Spaces demo (update to your Space URL)
        "http://localhost:8000",            # local dev
        "http://localhost:3000",            # local web dev
    ],
    allow_credentials=False,               # no cookies/sessions — app key is in headers
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-App-Key"],
)

# ============================================================================
# RAG — VECTOR DATABASE
# ============================================================================

vector_db: Chroma = None


def load_vector_db() -> bool:
    global vector_db
    if not CHROMA_DIR.exists():
        logger.warning("⚠️  chroma_db/ not found — RAG disabled")
        return False
    try:
        embeddings = GoogleEmbeddingsV1(model="models/gemini-embedding-001", google_api_key=GOOGLE_AI_API_KEY)
        vector_db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
        stored_ids = vector_db.get()["ids"]
        logger.info(f"✅ Chroma DB loaded — {len(stored_ids)} chunks")
        return True
    except Exception as exc:
        logger.error(f"❌ Failed to load Chroma DB: {exc}")
        vector_db = None
        return False


def retrieve_rag_context(query: str, machine_type: str, k: int = RAG_TOP_K) -> str:
    """
    Backward-compatible wrapper for the optimized RAG pipeline.
    """
    if vector_db is None:
        return ""

    problem_cats = infer_problem_categories(query)

    return retrieve_with_metadata_filter(
        vector_db=vector_db,
        query=query,
        machine_type=machine_type,
        problem_categories=problem_cats,
        k=k,
    )


# ============================================================================
# KNOWLEDGE BASE
# ============================================================================

def load_knowledge_base(machine_type: str) -> str:
    knowledge_file = KB_DIR / f"{machine_type.lower()}_facts.txt"
    if knowledge_file.exists():
        try:
            return knowledge_file.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error(f"Error loading knowledge base: {exc}")
    return "Use general agricultural machinery repair knowledge."


# ============================================================================
# SAFETY CHECK (local — kept in main to avoid dependency on PIL in services)
# ============================================================================

async def safety_check_with_gemini(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)

        prompt = """You are a safety camera watching over a farmer inspecting their farm machinery or equipment.
The machine could be a tractor, harvester, thresher, pump, electric motor, chaff cutter, power tiller, generator, or any other farm equipment.
Look at this image for any immediate danger — regardless of machine type.

Check for ALL of these hazards:
1. Hands, fingers, or clothing too close to moving parts — fan blades, belts, chains, drum, cutter blades, rotating shaft
2. Farmer working under a machine that is NOT safely supported
3. Fire, smoke, sparks, or glowing near fuel, oil, or electrical components
4. Fuel or oil actively leaking or spraying
5. Engine or motor running (fan spinning, exhaust smoke) when it should be switched off
6. Exposed live electrical wires, open junction boxes near hands — for electric machines
7. Farmer standing directly in front of a feed inlet or discharge chute of thresher / chaff cutter
8. No safety guard on chaff cutter or thresher feed opening

Return ONLY this JSON:
{
  "safe": true | false,
  "hazard_detected": "Plain description of the danger in simple words a farmer understands, or null if safe",
  "severity": "low" | "medium" | "high" | null,
  "warning_message": "Short urgent instruction — e.g. 'STOP! Move your hand away from the spinning belt!' or 'Switch off the power immediately — a live wire is exposed!' Null if safe."
}
If ANY hazard is detected, set safe to false immediately. Do not wait for certainty.
Return ONLY JSON, no markdown."""

        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model.generate_content([prompt, image])
        )
        result = json.loads(sanitize_json_text(response.text))
        if not result.get("safe"):
            logger.warning(f"⚠️ Safety hazard: {result.get('hazard_detected')}")
        return result
    except Exception as exc:
        logger.error(f"❌ Safety check error: {exc}")
        return {"safe": True, "hazard_detected": None, "severity": None, "warning_message": None}


# ============================================================================
# SSE HELPER
# ============================================================================

def _sse(event: str, data: dict) -> str:
    payload = json.dumps({"event": event, **data})
    return f"data: {payload}\n\n"


def _local_fallback_diagnosis(machine_type: str, problem_text: str) -> dict:
    """
    Minimal fallback returned when Gemini is unavailable (budget exceeded / timeout).
    Returns a valid response shape so Flutter doesn't crash.
    """
    from utils.machine_registry import get_profile_or_default, get_safety_warnings
    profile = get_profile_or_default(machine_type)
    return {
        "status": "error",
        "problem_description": problem_text,
        "technical_analysis": "Diagnosis service temporarily unavailable.",
        "solution": {
            "status": "error",
            "machine_type": machine_type,
            "problem_identified": problem_text,
            "steps": [],
            "safety_warnings_en": get_safety_warnings(machine_type, "en") or ["Consult a certified mechanic."],
            "safety_warnings_hi": get_safety_warnings(machine_type, "hi") or ["प्रमाणित मैकेनिक से संपर्क करें।"],
            "tools_needed": [],
        },
        "rag_source": "unavailable",
        "machine_label": profile.label_en if profile else machine_type,
    }


# ============================================================================
# API ENDPOINTS — EXISTING (unchanged interface)
# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "operational",
        "service": "AgriFix Production Backend",
        "version": "4.1.0",
        "architecture": {
            "machine_detection": "CLIP zero-shot + audio keywords (local) + Gemini fallback",
            "transcription": "Gemini 2.5 Flash",
            "diagnosis": "Gemini 2.5 Flash + RAG",
            "verification": "Gemini 2.5 Flash Vision",
            "agent": "Stateful repair agent (Gemini + safety rules)",
        },
    }


@app.get("/health")
async def health(request: Request):
    ip = request.client.host if request.client else "unknown"
    rag_chunks = 0
    if vector_db is not None:
        try:
            rag_chunks = vector_db._collection.count()
        except Exception:
            pass
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini": "configured",
        "knowledge_base": len(list(KB_DIR.glob("*.txt"))),
        "rag": {
            "status": "active" if vector_db is not None else "disabled",
            "chunks": rag_chunks,
            "top_k": RAG_TOP_K,
            "min_score": RAG_MIN_SCORE,
        },
        "active_sessions": len(session_manager.list_sessions()),
        "your_gemini_calls_this_hour": get_gemini_usage(ip),
        "plan_cache": {
            "entries": len(list(PLAN_CACHE_DIR.glob("*.json"))),
            "ttl_days": PLAN_CACHE_TTL // 86400,
        },
        # Server-enforced limits — visible to operators; attackers already know
        # the client limits, so publishing server limits does not aid attacks.
        "limits": {
            "video_max_mb":        VIDEO_MAX_BYTES // 1_048_576,
            "audio_max_mb":        AUDIO_MAX_BYTES // 1_048_576,
            "video_max_seconds":   int(VIDEO_MAX_SECONDS),
            "audio_max_seconds":   int(AUDIO_MAX_SECONDS),
            "gemini_calls_per_ip_per_hour": GEMINI_HOURLY_LIMIT,
        },
    }


@app.get("/machines")
async def list_machines():
    """
    List all farm machines supported by AgriFixAR.
    Returns machine IDs, labels, categories, and recognised aliases.
    Unity / Flutter can use this to populate the machine picker and
    validate YOLO detection labels before sending to /diagnose.
    """
    return {
        "status": "ok",
        "supported_machines": list_supported_machines(),
        "total": len(list_supported_machines()),
    }


@app.get("/machines/{machine_type}")
async def get_machine_info(machine_type: str):
    """
    Return detailed info about a specific machine type including
    all parts, area zones, and safety warnings.
    """
    profile = get_profile(machine_type)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Machine '{machine_type}' not recognised. Call GET /machines for the full list.",
        )
    return {
        "machine_id": profile.machine_id,
        "label_en": profile.label_en,
        "label_hi": profile.label_hi,
        "category": profile.category,
        "farmer_intro_en": profile.farmer_intro_en,
        "farmer_intro_hi": profile.farmer_intro_hi,
        "area_zones": [
            {
                "id": z.id,
                "label_en": z.label_en,
                "label_hi": z.label_hi,
                "farmer_description_en": z.farmer_description_en,
                "farmer_description_hi": z.farmer_description_hi,
            }
            for z in profile.area_zones
        ],
        "parts": [
            {
                "id": p.id,
                "area_zone": p.area_zone,
                "label_en": p.label_en,
                "label_hi": p.label_hi,
                "farmer_description_en": p.farmer_description_en,
                "ar_model": p.ar_model,
            }
            for p in profile.parts
        ],
        "safety_warnings_en": profile.base_safety_warnings_en,
        "safety_warnings_hi": profile.base_safety_warnings_hi,
    }


@app.post("/detect_machine")
@limiter.limit("10/minute")
async def detect_machine_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    audio: UploadFile = File(default=None),
    _auth: None = Depends(verify_app_key),
):
    """
    Stage-0 endpoint: detect machine type from video + optional audio.

    Flutter calls this BEFORE /diagnose/stream so the farmer can confirm
    the detected machine type on a dedicated confirmation screen.

    Response:
        {
          "machine_type":    "electric_water_pump",
          "confidence":      0.91,
          "source":          "clip",
          "label_en":        "Electric Water Pump",
          "label_hi":        "इलेक्ट्रिक वॉटर पंप",
          "alternatives":    [{"id": "diesel_pump", "label_en": "...", "label_hi": "..."}],
          "needs_confirmation": true   ← true when confidence < 0.80
        }

    Token cost: 0 (CLIP-only when confident, Gemini fallback only when needed).
    Latency: ~1–2 s on HF free tier.
    """
    ip = request.client.host if request.client else "unknown"
    logger.info(f"🔍 /detect_machine ip={ip}")
    request_id = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]

    try:
        video_bytes = await video.read()
        validate_video_upload(video.filename or "rec.mp4", video_bytes, ip=ip)

        _vname = _safe_filename(video.filename, "rec.mp4")
        video_path = UPLOAD_DIR / f"{request_id}_detect_{_vname}"
        video_path.write_bytes(video_bytes)
        background_tasks.add_task(lambda: video_path.unlink(missing_ok=True))

        transcription_text: Optional[str] = None
        if audio is not None:
            try:
                audio_bytes = await audio.read()
                if len(audio_bytes) > 512:
                    validate_audio_upload(audio.filename or "rec.m4a", audio_bytes, ip=ip)
                    _aname = _safe_filename(audio.filename, "rec.m4a")
                    audio_path = UPLOAD_DIR / f"{request_id}_detect_audio_{_aname}"
                    audio_path.write_bytes(audio_bytes)
                    background_tasks.add_task(lambda: audio_path.unlink(missing_ok=True))
                    if can_call_gemini(ip):
                        transcription_text = await gemini_with_timeout(
                            transcribe_audio_with_gemini(audio_path),
                            fallback=None, context="detect/transcription",
                        )
                        if transcription_text:
                            record_gemini_call(ip)
            except Exception:
                pass  # audio is optional — carry on without it

        detection = await detect_machine(
            video_path=video_path,
            transcription_text=transcription_text,
        )
        if detection.gemini_used:
            record_gemini_call(ip)

        resolved = detection.machine_type
        profile   = get_profile(resolved)

        # Build alternatives list from supported machines (exclude detected)
        all_machines = list_supported_machines()
        alternatives = [
            {
                "id":       m["machine_id"],
                "label_en": m["label_en"],
                "label_hi": m.get("label_hi", m["label_en"]),
            }
            for m in all_machines
            if m["machine_id"] != resolved
        ][:8]  # cap at 8 so the UI list stays manageable

        return JSONResponse(content={
            "machine_type":       resolved,
            "confidence":         round(detection.confidence, 3),
            "source":             detection.source,
            "clip_confidence":    round(detection.clip_confidence or 0.0, 3),
            "audio_confidence":   round(detection.audio_confidence or 0.0, 3),
            "label_en":           profile.label_en  if profile else resolved,
            "label_hi":           profile.label_hi  if profile else resolved,
            "needs_confirmation": detection.confidence < 0.80,
            "alternatives":       alternatives,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ /detect_machine failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/diagnose")
@limiter.limit("5/minute")
async def diagnose(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    machine_type: str = Form(default=""),   # optional hint — auto-detected now
    language: str = Form(default="en"),
    _auth: None = Depends(verify_app_key),
):
    """
    Main diagnosis endpoint — response format UNCHANGED.
    """
    ip = request.client.host if request.client else "unknown"
    logger.info(f"📥 /diagnose hint={machine_type!r} lang={language} ip={ip}")
    request_id = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]

    try:
        if not audio.filename:
            raise HTTPException(status_code=400, detail="Audio file required")

        # Read bytes first so we can validate before touching disk
        audio_bytes = await audio.read()
        video_bytes = await video.read()

        # ── Security: file validation ─────────────────────────────────────────
        validate_audio_upload(audio.filename or "rec.m4a", audio_bytes, ip=ip)
        if video.filename and len(video_bytes) > 1024:
            validate_video_upload(video.filename, video_bytes, ip=ip)

        # Save audio
        _aname = _safe_filename(audio.filename, "rec.m4a")
        audio_path  = UPLOAD_DIR / f"{request_id}_audio_{_aname}"
        audio_path.write_bytes(audio_bytes)
        background_tasks.add_task(lambda: audio_path.unlink(missing_ok=True))

        # Save video only if meaningful content exists
        video_path: Optional[Path] = None
        if video.filename and len(video_bytes) > 1024:
            _vname = _safe_filename(video.filename, "rec.mp4")
            video_path = UPLOAD_DIR / f"{request_id}_video_{_vname}"
            video_path.write_bytes(video_bytes)
            background_tasks.add_task(lambda: video_path.unlink(missing_ok=True))

        # ── Token-budget architecture ─────────────────────────────────────────
        # Goal: exactly 2 Gemini calls per /diagnose request.
        #
        #   Call 1 — transcribe_audio_with_gemini  (audio → text)
        #   Call 2 — generate_diagnosis_with_gemini (reason once, output JSON)
        #
        # CLIP detection is local (CPU-only).  We run CLIP and transcription
        # concurrently so the combined wall-clock time is cut roughly in half.
        # detect_machine's internal Gemini fallback is bypassed by supplying the
        # transcription text on a second pass only if CLIP confidence is low —
        # the keyword matcher then resolves the machine type locally.
        # ─────────────────────────────────────────────────────────────────────

        # Build coroutines — transcription only if budget allows
        async def _transcribe_batch() -> str:
            if can_call_gemini(ip):
                text = await gemini_with_timeout(
                    transcribe_audio_with_gemini(audio_path),
                    fallback="farm machine problem",
                    context="transcription",
                )
                record_gemini_call(ip)
                return text or "farm machine problem"
            logger.warning(f"⚠️  Gemini budget exceeded for {ip} — using fallback transcription")
            return "farm machine problem"

        async def _detect_clip_batch() -> Any:
            # CLIP-only first pass — no Gemini, no transcription text needed
            return await detect_machine(video_path=video_path, transcription_text=None)

        # Run CLIP inference and Gemini transcription concurrently
        problem_text_raw, detection_clip = await asyncio.gather(
            _transcribe_batch(), _detect_clip_batch()
        )

        # ── Security: prompt injection check ─────────────────────────────────
        problem_text = check_prompt_injection(
            problem_text_raw or "", ip=ip, field="transcription"
        )

        # ── Resolve machine type without an extra Gemini call ─────────────────
        resolved  = detection_clip.machine_type
        detection = detection_clip

        if detection_clip.confidence < 0.55:
            # CLIP uncertain: re-run locally with transcription keywords.
            # CRITICAL: pass video_path=None — video was already decoded above.
            # Restore frames from the first decode so diagnosis still gets them.
            detection = await detect_machine(
                video_path=None,          # ← no re-decode
                transcription_text=problem_text,
            )
            detection.frames = detection_clip.frames  # restore already-decoded frames
            if detection.gemini_used:
                record_gemini_call(ip)
            resolved = detection.machine_type

        # Client hint overrides when confidence is still low
        if machine_type.strip():
            hint = resolve_machine_id(machine_type.strip())
            if detection.confidence < 0.55 and get_profile(hint):
                logger.info(f"🔧 Low confidence ({detection.confidence:.2f}) — using client hint: {hint}")
                resolved = hint

        logger.info(
            f"🔧 Machine: {resolved}  "
            f"(detected={detection.machine_type} conf={detection.confidence:.2f} "
            f"src={detection.source} gemini={detection.gemini_used})"
        )

        # ── RAG: fully local, zero Gemini calls ───────────────────────────────
        rag_context = retrieve_rag_context(problem_text, resolved)
        knowledge   = load_knowledge_base(resolved)

        # ── Reuse frames already decoded by CLIP — zero extra video I/O ──────
        # detection.frames holds the same early/mid/late JPEG bytes that CLIP
        # quality-scored and used for classification. No re-open, no re-decode.
        clip_frames  = detection.frames          # list[bytes], already in RAM
        mid_frame    = clip_frames[len(clip_frames) // 2] if clip_frames else None
        visual_hash  = _phash_bytes(mid_frame) if mid_frame else ""
        logger.info(f"📸 Frames from CLIP cache: {len(clip_frames)}/3  phash={visual_hash or 'none'}")

        # ── Plan cache check — avoids LLM call entirely on hit ────────────────
        plan_cache_key = _plan_cache_key(resolved, problem_text, visual_hash)
        cached_plan    = _plan_cache_get(plan_cache_key)

        # ── Diagnosis: ONE Gemini reasoning call (now multimodal 3-frame) ─────
        if cached_plan is not None:
            diagnosis = cached_plan
            diagnosis["cache_hit"] = True
            logger.info(f"🎯 /diagnose cache HIT key={plan_cache_key}")
        elif can_call_gemini(ip):
            diagnosis = await gemini_with_timeout(
                generate_diagnosis_with_gemini(
                    machine_type=resolved,
                    problem_text=problem_text,
                    language=language,
                    rag_context=rag_context,
                    knowledge_base=knowledge,
                    visual_frames=clip_frames,
                ),
                fallback=None,
                context="diagnosis",
            )
            if diagnosis:
                record_gemini_call(ip)
                _plan_cache_set(plan_cache_key, diagnosis)
            else:
                diagnosis = _local_fallback_diagnosis(resolved, problem_text)
        else:
            logger.warning(f"⚠️  Gemini budget exceeded for {ip} — skipping diagnosis Gemini call")
            diagnosis = _local_fallback_diagnosis(resolved, problem_text)

        diagnosis["request_id"]   = request_id
        diagnosis["transcription"] = problem_text
        diagnosis["detection"] = {
            "machine_type":    resolved,
            "confidence":      detection.confidence,
            "source":          detection.source,
            "clip_confidence": detection.clip_confidence,
            "audio_confidence": detection.audio_confidence,
            "gemini_used":     detection.gemini_used,
        }

        logger.info(f"✅ /diagnose complete steps={len(diagnosis.get('solution', {}).get('steps', []))}")
        return JSONResponse(content=diagnosis)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ Diagnosis failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/diagnose/stream")
@limiter.limit("5/minute")
async def diagnose_stream(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    machine_type: str = Form(default=""),   # optional hint
    language: str = Form(default="en"),
    _auth: None = Depends(verify_app_key),
):
    """
    Streaming diagnosis — Server-Sent Events.
    Stage event format is IDENTICAL to v4.0 — zero Flutter changes needed.

    Security additions (v4.2):
      • X-App-Key header required (403 if missing/wrong)
      • 5 requests/minute/IP rate limit (429 if exceeded)
      • Video ≤ 20 MB, audio ≤ 5 MB, formats validated before stream starts
      • Transcribed text sanitised for prompt injection inside event_stream()
      • All Gemini calls wrapped with 15 s timeout + per-IP hourly cap
    """
    ip = request.client.host if request.client else "unknown"
    logger.info(f"📡 /diagnose/stream hint={machine_type!r} lang={language} ip={ip}")
    request_id = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]

    try:
        audio_bytes = await audio.read()
        video_bytes = await video.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload read failed: {exc}")

    # ── Security: validate files BEFORE starting the stream ──────────────────
    validate_audio_upload(audio.filename or "rec.m4a", audio_bytes, ip=ip)
    if video.filename and len(video_bytes) > 1024:
        validate_video_upload(video.filename, video_bytes, ip=ip)

    async def event_stream() -> AsyncIterator[str]:
        # ── Token-budget architecture ─────────────────────────────────────────
        # Goal: exactly 2 Gemini calls per /diagnose/stream request.
        #
        #   Call 1 — transcribe_audio_with_gemini  (audio→text, unavoidable)
        #   Call 2 — generate_diagnosis_with_gemini (reason once, output JSON)
        #
        # detect_machine uses local CLIP zero-shot + audio keyword matching.
        # Its internal Gemini fallback fires only when CLIP < 0.55 AND no
        # client hint AND no machine keyword found in the transcription.
        # We eliminate that third call by:
        #   a) running CLIP and transcription concurrently (asyncio.gather) so
        #      both results are available before we decide on machine type;
        #   b) resolving machine type from transcription text when CLIP is
        #      uncertain — so detect_machine's Gemini path is never reached.
        #
        # RAG (Stage 2) is fully local — zero Gemini calls.
        # ─────────────────────────────────────────────────────────────────────

        audio_path = UPLOAD_DIR / f"{request_id}_audio_{_safe_filename(audio.filename, 'rec.m4a')}"
        video_path: Optional[Path] = None
        try:
            # ── Save files ────────────────────────────────────────────────────
            audio_path.write_bytes(audio_bytes)
            if video.filename and len(video_bytes) > 1024:
                video_path = UPLOAD_DIR / f"{request_id}_video_{_safe_filename(video.filename, 'rec.mp4')}"
                video_path.write_bytes(video_bytes)

            # ── Stages 0 + 1: CLIP detection & transcription — CONCURRENT ────
            # CLIP reads video bytes (local, CPU-only).
            # Transcription calls Gemini on audio bytes.
            # They are completely independent so we run them in parallel,
            # cutting the combined wall-clock time roughly in half.
            yield _sse("stage_start", {"stage": 0, "label": "Identifying your machine from the video"})
            yield _sse("stage_start", {"stage": 1, "label": "Understanding your voice complaint"})

            # Build coroutines — transcription only if budget allows
            async def _transcribe() -> str:
                if can_call_gemini(ip):
                    text = await gemini_with_timeout(
                        transcribe_audio_with_gemini(audio_path),
                        fallback="farm machine problem",
                        context="stream/transcription",
                    )
                    record_gemini_call(ip)
                    return text or "farm machine problem"
                logger.warning(f"⚠️  Gemini budget exceeded [{ip}] — fallback transcription")
                return "farm machine problem"

            async def _detect_clip() -> Any:
                # Run CLIP-only detection first (local, fast, no Gemini).
                # We pass transcription_text=None here so detect_machine uses
                # only CLIP + audio keywords from the video.
                return await detect_machine(
                    video_path=video_path,
                    transcription_text=None,
                )

            # Run both concurrently — Gemini call and CLIP inference overlap
            problem_text_raw, detection_clip = await asyncio.gather(
                _transcribe(), _detect_clip()
            )

            # ── Prompt injection guard ────────────────────────────────────────
            problem_text = check_prompt_injection(
                problem_text_raw or "", ip=ip, field="stream/transcription"
            )

            # ── Resolve machine type (no extra Gemini call) ───────────────────
            # Priority: (1) high-confidence CLIP, (2) client hint, (3) audio
            # keyword match in transcription.  detect_machine's own Gemini
            # fallback is only reached when all three signals are ambiguous —
            # and we never call it here because we re-run detect_machine with
            # the transcription text if CLIP is uncertain, which gives the
            # keyword matcher enough signal to avoid the Gemini path.
            resolved = detection_clip.machine_type
            detection = detection_clip

            if detection_clip.confidence < 0.55:
                # CLIP is uncertain — re-run detection WITH transcription text so
                # the audio keyword matcher can resolve without Gemini.
                # CRITICAL: pass video_path=None so the video is NOT re-decoded.
                # detection_clip.frames already holds the JPEG bytes from the
                # first decode. We copy them onto the new result after the call.
                detection = await detect_machine(
                    video_path=None,          # ← no re-decode
                    transcription_text=problem_text,
                )
                # Restore the frames from the first (and only) CLIP decode
                detection.frames = detection_clip.frames
                if detection.gemini_used:
                    record_gemini_call(ip)
                resolved = detection.machine_type

            # Client hint overrides when our detection is still uncertain
            if machine_type.strip():
                hint = resolve_machine_id(machine_type.strip())
                if detection.confidence < 0.55 and get_profile(hint):
                    logger.info(f"🔧 Using client hint {hint!r} (conf={detection.confidence:.2f})")
                    resolved = hint

            yield _sse("stage_done", {
                "stage": 1, "label": "Understanding your voice complaint",
                "transcription": problem_text,
            })
            yield _sse("stage_done", {
                "stage": 0, "label": "Identifying your machine from the video",
                "machine_type": resolved,
                "detection_confidence": detection.confidence,
                "detection_source":     detection.source,
            })

            # ── Stage 2: RAG — fully local, zero Gemini calls ─────────────────
            yield _sse("stage_start", {"stage": 2, "label": "Searching repair manuals for your issue"})
            rag_context = retrieve_rag_context(problem_text, resolved)
            rag_chunks  = len(rag_context.split("---")) if rag_context else 0

            # ── Reuse frames already decoded by CLIP — zero extra video I/O ────
            # detection.frames holds the early/mid/late JPEG bytes that CLIP
            # quality-scored during Stage 0. No re-open, no re-decode, no cv2.
            # 3 frames give Gemini temporal context: all 3 show water flowing →
            # machine RUNNING; frame 3 still but 1-2 had motion → cut out mid-run.
            clip_frames  = detection.frames
            mid_frame    = clip_frames[len(clip_frames) // 2] if clip_frames else None
            visual_hash  = _phash_bytes(mid_frame) if mid_frame else ""
            logger.info(f"📸 Frames from CLIP cache: {len(clip_frames)}/3  phash={visual_hash or 'none'}")

            yield _sse("stage_done", {
                "stage": 2, "label": "Searching repair manuals for your issue",
                "rag_chunks": rag_chunks, "rag_active": rag_chunks > 0,
            })

            # ── Stage 3: Diagnosis — ONE Gemini call, multimodal 3-frame ─────
            # Cache key includes visual_hash of mid frame: same machine + same
            # symptom cluster but different visual state → different cache entry.
            # "water flowing" and "pump completely dead" get separate cached plans.
            yield _sse("stage_start", {"stage": 3, "label": "Preparing your step-by-step repair guide"})
            knowledge = load_knowledge_base(resolved)

            plan_cache_key = _plan_cache_key(resolved, problem_text, visual_hash)
            cached_plan    = _plan_cache_get(plan_cache_key)

            if cached_plan is not None:
                diagnosis = cached_plan
                diagnosis["cache_hit"] = True
                logger.info(f"🎯 stream cache HIT key={plan_cache_key}")
            elif can_call_gemini(ip):
                diagnosis = await gemini_with_timeout(
                    generate_diagnosis_with_gemini(
                        machine_type=resolved,
                        problem_text=problem_text,
                        language=language,
                        rag_context=rag_context,
                        knowledge_base=knowledge,
                        visual_frames=clip_frames,
                    ),
                    fallback=None,
                    context="stream/diagnosis",
                )
                if diagnosis:
                    record_gemini_call(ip)
                    _plan_cache_set(plan_cache_key, diagnosis)
                else:
                    diagnosis = _local_fallback_diagnosis(resolved, problem_text)
            else:
                logger.warning(f"⚠️  Gemini budget exceeded [{ip}] — fallback diagnosis")
                diagnosis = _local_fallback_diagnosis(resolved, problem_text)

            diagnosis["request_id"]    = request_id
            diagnosis["transcription"] = problem_text
            diagnosis["detection"] = {
                "machine_type":    resolved,
                "confidence":      detection.confidence,
                "source":          detection.source,
                "clip_confidence": detection.clip_confidence,
                "audio_confidence": detection.audio_confidence,
                "gemini_used":     detection.gemini_used,
            }
            yield _sse("stage_done", {
                "stage": 3, "label": "Preparing your step-by-step repair guide",
                "result": diagnosis,
                "cache_hit": diagnosis.get("cache_hit", False),
            })

            yield _sse("done", {})
            logger.info(f"✅ /diagnose/stream complete [{request_id}]")

        except Exception as exc:
            logger.error(f"❌ Stream error [{request_id}]: {exc}")
            yield _sse("error", {"message": str(exc)})
        finally:
            # Windows locks files held by sockets that closed uncleanly (WinError 32).
            # Retry up to 3 times with a short delay before giving up — the OS
            # releases the handle within ~200ms of the SSL connection teardown.
            async def _safe_unlink(p: Path) -> None:
                for attempt in range(3):
                    try:
                        p.unlink(missing_ok=True)
                        return
                    except PermissionError:
                        if attempt < 2:
                            await asyncio.sleep(0.3)
                        else:
                            logger.warning(f"⚠️  Could not delete temp file (still locked): {p.name}")

            await _safe_unlink(audio_path)
            if video_path:
                await _safe_unlink(video_path)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/verify_step")
@limiter.limit("20/minute")
async def verify_step(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    step_text: str = Form(...),
    required_part: str = Form(default="machine_part"),
    area_hint: str = Form(default="engine_compartment"),
    machine_type: str = Form(default="tractor"),
    problem_context: str = Form(default=""),
    attempt_count: int = Form(default=1),
    previous_steps: str = Form(default="[]"),
    language: str = Form(default="en"),
    include_hindi: str = Form(default="false"),
    _auth: None = Depends(verify_app_key),
):
    """Step verification endpoint — unchanged interface, delegates to verification_service.

    Security additions (v4.2):
      • X-App-Key header required
      • 20 requests/minute/IP rate limit
      • step_text sanitised for prompt injection
      • Gemini call wrapped with 15 s timeout + per-IP hourly cap
    """
    ip = request.client.host if request.client else "unknown"

    # ── Security: input length caps ───────────────────────────────────────────
    # Prevents oversized payloads bloating Gemini prompts / slowing the server.
    # step_text raises 400 (Flutter must send valid input); others are silently
    # truncated since they're contextual and partial content is still useful.
    if len(step_text) > 1000:
        raise HTTPException(status_code=400, detail="step_text exceeds 1000 character limit")
    if len(problem_context) > 500:
        problem_context = problem_context[:500]
    if len(previous_steps) > 8000:
        previous_steps = "[]"   # silently drop oversized history — non-critical

    # ── Security: previous_steps must be a JSON array ─────────────────────────
    # json.loads() accepts dicts, strings, ints — all would break the for-loop
    # in verification_service.py and produce garbage context sent to Gemini.
    try:
        _ps_parsed = json.loads(previous_steps)
        if not isinstance(_ps_parsed, list):
            previous_steps = "[]"
    except (json.JSONDecodeError, ValueError):
        previous_steps = "[]"
        required_part, area_hint = derive_part_and_area(step_text, machine_type)

    # ── Security: prompt injection on step_text + problem_context ─────────────
    step_text       = check_prompt_injection(step_text,        ip=ip, field="step_text")
    problem_context = check_prompt_injection(problem_context,  ip=ip, field="problem_context")

    logger.info(f"👁️ verify_step attempt={attempt_count} part={required_part} area={area_hint} ip={ip}")

    try:
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image")

        # Gemini vision call — guarded
        if can_call_gemini(ip):
            result = await gemini_with_timeout(
                verify_step_with_gemini(
                    image_bytes=image_bytes,
                    step_text=step_text,
                    required_part=required_part,
                    area_hint=area_hint,
                    machine_type=machine_type,
                    problem_context=problem_context,
                    attempt_count=attempt_count,
                    language=language,
                    include_hindi=(include_hindi.lower() == "true"),
                    previous_steps=previous_steps,  # ← semantic memory wiring
                ),
                fallback=None,
                context="verify_step",
            )
            if result:
                record_gemini_call(ip)
            else:
                # Timeout fallback
                result = {
                    "status": "unclear", "verified": False, "confidence": 0.0,
                    "detected_part": "Analysis timed out",
                    "correct_part": required_part, "machine_type": machine_type,
                    "ai_observation": "The analysis took too long. Please try again.",
                    "feedback": "Move closer and hold still — tap Analyze again.",
                    "feedback_hi": "कैमरे को करीब लाएं और स्थिर रखें — फिर विश्लेषण दबाएं।",
                    "attempt_count": attempt_count,
                }
        else:
            logger.warning(f"⚠️  Gemini budget exceeded [{ip}] — skipping verify_step Gemini call")
            result = {
                "status": "unclear", "verified": False, "confidence": 0.0,
                "detected_part": "Service temporarily limited",
                "correct_part": required_part, "machine_type": machine_type,
                "ai_observation": "Too many requests. Please wait a moment and try again.",
                "feedback": "Please try again in a few minutes.",
                "feedback_hi": "कुछ मिनट बाद फिर से कोशिश करें।",
                "attempt_count": attempt_count,
            }

        result["request_id"] = hashlib.md5(f"{datetime.now()}{attempt_count}".encode()).hexdigest()[:8]
        logger.info(f"✅ verify_step={result.get('status')} conf={result.get('confidence', 0):.2f}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ Verification failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/safety_check")
async def safety_check(image: UploadFile = File(...)):
    """Continuous safety monitoring endpoint."""
    logger.info("🛡️ Safety check request")
    try:
        image_bytes = await image.read()
        result = await safety_check_with_gemini(image_bytes)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error(f"❌ Safety check failed: {exc}")
        return JSONResponse(content={"safe": True, "hazard_detected": None})


@app.delete("/plan_cache")
async def clear_plan_cache(
    request: Request,
    _auth: None = Depends(verify_app_key),
):
    """
    Clear all cached repair plans (operator tool).
    Use after major knowledge base updates to force fresh LLM generation.
    """
    files = list(PLAN_CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink(missing_ok=True)
    logger.info(f"🗑️  Plan cache cleared by {request.client.host}: {len(files)} entries removed")
    return {"status": "cleared", "entries_removed": len(files)}


@app.delete("/plan_cache/{machine_type}")
async def clear_plan_cache_for_machine(
    machine_type: str,
    _auth: None = Depends(verify_app_key),
):
    """
    Selectively invalidate all cached plans for a specific machine type.
    Useful when you update the knowledge base for one machine only.
    """
    # We can't reverse the SHA-256 but we can scan and match by checking
    # a small metadata sidecar. For simplicity, clear all — this is a rare op.
    files = list(PLAN_CACHE_DIR.glob("*.json"))
    removed = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("solution", {}).get("machine_type", "") == machine_type:
                f.unlink(missing_ok=True)
                removed += 1
        except Exception:
            pass
    logger.info(f"🗑️  Plan cache: {removed} entries removed for machine={machine_type}")
    return {"status": "cleared", "machine_type": machine_type, "entries_removed": removed}


@app.post("/feedback")
async def submit_feedback(
    rating: int = Form(...),
    comments: str = Form(default=""),
    step_index: int = Form(default=0),
    machine_type: str = Form(default="unknown"),
):
    logger.info(f"📝 Feedback: rating={rating} machine={machine_type}")
    # Sanitise: strip newlines (prevent log-injection forgery) and cap length
    safe_comments    = comments.replace("\n", " ").replace("\r", " ").replace(",", ";")[:500]
    safe_machine     = machine_type.replace("\n", "").replace("\r", "")[:50]
    feedback_file = Path("user_feedback.log")
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()},{safe_machine},{step_index},{rating},{safe_comments}\n")
    return {"status": "thank you"}


# ============================================================================
# API ENDPOINTS — NEW AGENT SYSTEM
# ============================================================================

@app.post("/agent/session", response_model=CreateSessionResponse)
async def create_agent_session(request: CreateSessionRequest):
    """
    Create a new stateful repair session for ANY supported farm machine.
    """
    # Normalise alias → canonical machine ID
    canonical_type = resolve_machine_id(request.machine_type)
    profile = get_profile(canonical_type)
    machine_label = profile.label_en if profile else canonical_type

    session = session_manager.create_session(
        machine_type=canonical_type,
        problem=request.problem_description,
        language=request.language,
    )
    logger.info(f"🆕 Agent session created: {session.session_id} machine={canonical_type}")
    return CreateSessionResponse(
        session_id=session.session_id,
        message=f"Repair session started for {machine_label}. Call /agent/next to begin diagnosis.",
    )


@app.post("/agent/next")
async def agent_next(request: AgentNextRequest):
    """Creates a new stateful repair session for farm machinery."""
    session = session_manager.get_session(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found or expired. Create a new session via /agent/session.",
        )

    logger.info(
        f"🤖 /agent/next session={request.session_id} "
        f"stage={session.current_stage} "
        f"parts_verified={len(session.verified_parts)}"
    )

    try:
        response = await repair_agent.decide_next_step(
            session=session,
            last_verification=request.last_verification_result,
        )
        # Persist updated session
        session_manager.update_session(session)

        # If resolved or escalated, clean up session after responding
        if response.status in ("resolved", "escalate", "unsafe"):
            logger.info(f"🏁 Session {request.session_id} terminal status: {response.status}")

        return JSONResponse(content=response.model_dump())

    except Exception as exc:
        logger.error(f"❌ Agent reasoning failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")


@app.get("/agent/session/{session_id}")
async def get_agent_session(
    session_id: str,
    _auth: None = Depends(verify_app_key),
):
    """
    Inspect the current state of an active repair session.
    Gated behind X-App-Key — exposes verified_parts, observations, and
    problem description which must not be publicly readable.
    """
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or expired.")
    return JSONResponse(content=session.model_dump())


@app.delete("/agent/session/{session_id}")
async def delete_agent_session(
    session_id: str,
    _auth: None = Depends(verify_app_key),
):
    """Explicitly end and clean up a repair session."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "deleted", "session_id": session_id}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 80)
    logger.info("🚀 AgriFix v4.2 Production Backend Starting...")
    logger.info("=" * 80)
    logger.info("🔑 Gemini API: ✅ Configured")
    logger.info("🔮 Machine Detection: MobileCLIP-S1 + audio keywords + Gemini fallback")
    logger.info("🧠 Agent: Stateful repair agent (Gemini + safety rules)")
    logger.info("🛡️  Security: rate limiting + Gemini guard + file validation + auth")
    logger.info(
        f"📐 Limits enforced: video≤{VIDEO_MAX_BYTES//1_048_576}MB/{int(VIDEO_MAX_SECONDS)}s  "
        f"audio≤{AUDIO_MAX_BYTES//1_048_576}MB/{int(AUDIO_MAX_SECONDS)}s  "
        f"gemini≤{GEMINI_HOURLY_LIMIT}/IP/hr"
    )
    logger.info(f"💾 Cache: {CACHE_DIR}")
    logger.info(f"📚 Knowledge Base: {KB_DIR}")
    logger.info("=" * 80)

    port = int(os.environ.get("PORT", 7860))

    # Production stability settings — DO NOT change these without reading below:
    #
    # workers=1:
    #   SlowAPI rate limiter and Gemini credit guard use in-memory dicts.
    #   Multiple workers = separate memory = each worker gets the full quota.
    #   On HuggingFace free tier, 1 worker is correct. If you scale to multi-
    #   worker in future, replace the dicts with Redis.
    #
    # limit_concurrency=50:
    #   Caps simultaneous open requests. Each /diagnose/stream holds a connection
    #   for 5-15 s while Gemini runs. 50 concurrent = HF free tier RAM safe.
    #   Excess requests get HTTP 503 rather than OOM-killing the server.
    #
    # timeout_keep_alive=10:
    #   Disconnects idle keep-alive connections after 10 s. Prevents connection
    #   exhaustion from clients that open but don't close properly (mobile apps
    #   on bad networks). Standard production setting.

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENV") == "development",
        workers=1,
        limit_concurrency=50,
        timeout_keep_alive=10,
    )