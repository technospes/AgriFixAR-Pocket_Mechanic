import json
import os
import re
import asyncio
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
class GoogleEmbeddingsV1(Embeddings):
    def __init__(self, model: str, google_api_key: str):
        self.model = model
        genai.configure(api_key=google_api_key)

    def embed_documents(self, texts: _List[str]) -> _List[_List[float]]:
        return [
            genai.embed_content(model=self.model, content=t, task_type="retrieval_document")["embedding"]
            for t in texts
        ]

    def embed_query(self, text: str) -> _List[float]:
        return genai.embed_content(model=self.model, content=text, task_type="retrieval_query")["embedding"]


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
RAG_TOP_K = 5
RAG_MIN_SCORE = 0.25
MAX_IMAGE_SIZE = 512
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_CACHE_AGE = 86400

UPLOAD_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        count = vector_db._collection.count()
        logger.info(f"✅ Chroma DB loaded — {count} chunks")
        return True
    except Exception as exc:
        logger.error(f"❌ Failed to load Chroma DB: {exc}")
        vector_db = None
        return False


def retrieve_rag_context(query: str, machine_type: str, k: int = RAG_TOP_K) -> str:
    if vector_db is None:
        return ""
    try:
        enriched_query = f"{machine_type} {query}"
        results = vector_db.similarity_search_with_relevance_scores(enriched_query, k=k)
        good_chunks = [(doc, score) for doc, score in results if score >= RAG_MIN_SCORE]
        if not good_chunks:
            return ""
        parts = []
        for doc, score in good_chunks:
            source = doc.metadata.get("source_file", "manual")
            machine = doc.metadata.get("machine_type", machine_type)
            parts.append(f"[Source: {source} | Machine: {machine} | Relevance: {score:.2f}]\n{doc.page_content.strip()}")
        logger.info(f"📚 RAG: {len(good_chunks)}/{k} chunks injected")
        return "\n\n---\n\n".join(parts)
    except Exception as exc:
        logger.error(f"❌ RAG retrieval failed: {exc}")
        return ""


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

    Security additions (v4.2):
      • X-App-Key header required (403 if missing/wrong)
      • 5 requests/minute/IP rate limit (429 if exceeded)
      • Video ≤ 20 MB, audio ≤ 5 MB, formats validated (400 if violated)
      • Transcribed text sanitised for prompt injection
      • Gemini calls wrapped with 15 s timeout + per-IP hourly cap
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
        audio_path  = UPLOAD_DIR / f"{request_id}_audio_{audio.filename}"
        audio_path.write_bytes(audio_bytes)
        background_tasks.add_task(lambda: audio_path.unlink(missing_ok=True))

        # Save video only if meaningful content exists
        video_path: Optional[Path] = None
        if video.filename and len(video_bytes) > 1024:
            video_path = UPLOAD_DIR / f"{request_id}_video_{video.filename}"
            video_path.write_bytes(video_bytes)
            background_tasks.add_task(lambda: video_path.unlink(missing_ok=True))

        # Transcribe audio — Gemini call, guarded
        if can_call_gemini(ip):
            problem_text = await gemini_with_timeout(
                transcribe_audio_with_gemini(audio_path),
                fallback="farm machine problem",
                context="transcription",
            )
            record_gemini_call(ip)
        else:
            problem_text = "farm machine problem"
            logger.warning(f"⚠️  Gemini budget exceeded for {ip} — using fallback transcription")

        # ── Security: prompt injection check on transcribed text ──────────────
        problem_text = check_prompt_injection(problem_text or "", ip=ip, field="transcription")

        # Detect machine (local CLIP + audio; Gemini only if budget allows)
        detection = await detect_machine(
            video_path=video_path,
            transcription_text=problem_text,
        )
        # Record Gemini usage if detection used it
        if detection.gemini_used:
            record_gemini_call(ip)

        # Resolve final machine_type
        resolved = detection.machine_type
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

        rag_context = retrieve_rag_context(problem_text, resolved)
        knowledge   = load_knowledge_base(resolved)

        # Diagnosis — Gemini call, guarded
        if can_call_gemini(ip):
            diagnosis = await gemini_with_timeout(
                generate_diagnosis_with_gemini(
                    machine_type=resolved,
                    problem_text=problem_text,
                    language=language,
                    rag_context=rag_context,
                    knowledge_base=knowledge,
                ),
                fallback=None,
                context="diagnosis",
            )
            if diagnosis:
                record_gemini_call(ip)
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
        audio_path = UPLOAD_DIR / f"{request_id}_audio_{audio.filename or 'rec.m4a'}"
        video_path: Optional[Path] = None
        try:
            yield _sse("stage_start", {"stage": 0, "label": "Analyzing machinery type"})

            # Save files
            audio_path.write_bytes(audio_bytes)
            if video.filename and len(video_bytes) > 1024:
                video_path = UPLOAD_DIR / f"{request_id}_video_{video.filename or 'rec.mp4'}"
                video_path.write_bytes(video_bytes)

            # Stage 1 — Transcription (Gemini, guarded)
            yield _sse("stage_start", {"stage": 1, "label": "Transcribing audio complaint"})
            if can_call_gemini(ip):
                problem_text = await gemini_with_timeout(
                    transcribe_audio_with_gemini(audio_path),
                    fallback="farm machine problem",
                    context="stream/transcription",
                )
                record_gemini_call(ip)
            else:
                problem_text = "farm machine problem"
                logger.warning(f"⚠️  Gemini budget exceeded [{ip}] — fallback transcription")

            # ── Security: prompt injection check ─────────────────────────────
            problem_text = check_prompt_injection(
                problem_text or "", ip=ip, field="stream/transcription"
            )

            yield _sse("stage_done", {
                "stage": 1, "label": "Transcribing audio complaint",
                "transcription": problem_text,
            })

            # Machine detection (local + optional Gemini)
            detection = await detect_machine(
                video_path=video_path,
                transcription_text=problem_text,
            )
            if detection.gemini_used:
                record_gemini_call(ip)

            # Resolve machine_type
            resolved = detection.machine_type
            if machine_type.strip():
                hint = resolve_machine_id(machine_type.strip())
                if detection.confidence < 0.55 and get_profile(hint):
                    resolved = hint

            yield _sse("stage_done", {
                "stage": 0, "label": "Analyzing machinery type",
                "machine_type": resolved,
                "detection_confidence": detection.confidence,
                "detection_source":     detection.source,
            })

            # Stage 2 — RAG (local, no Gemini)
            yield _sse("stage_start", {"stage": 2, "label": "Querying repair manuals"})
            rag_context = retrieve_rag_context(problem_text, resolved)
            rag_chunks  = len(rag_context.split("---")) if rag_context else 0
            yield _sse("stage_done", {
                "stage": 2, "label": "Querying repair manuals",
                "rag_chunks": rag_chunks, "rag_active": rag_chunks > 0,
            })

            # Stage 3 — Diagnosis (Gemini, guarded)
            yield _sse("stage_start", {"stage": 3, "label": "Generating step-by-step guide"})
            knowledge = load_knowledge_base(resolved)

            if can_call_gemini(ip):
                diagnosis = await gemini_with_timeout(
                    generate_diagnosis_with_gemini(
                        machine_type=resolved,
                        problem_text=problem_text,
                        language=language,
                        rag_context=rag_context,
                        knowledge_base=knowledge,
                    ),
                    fallback=None,
                    context="stream/diagnosis",
                )
                if diagnosis:
                    record_gemini_call(ip)
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
                "stage": 3, "label": "Generating step-by-step guide",
                "result": diagnosis,
            })

            yield _sse("done", {})
            logger.info(f"✅ /diagnose/stream complete [{request_id}]")

        except Exception as exc:
            logger.error(f"❌ Stream error [{request_id}]: {exc}")
            yield _sse("error", {"message": str(exc)})
        finally:
            audio_path.unlink(missing_ok=True)
            if video_path:
                video_path.unlink(missing_ok=True)

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

    if required_part == "machine_part":
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


@app.post("/feedback")
async def submit_feedback(
    rating: int = Form(...),
    comments: str = Form(default=""),
    step_index: int = Form(default=0),
    machine_type: str = Form(default="unknown"),
):
    logger.info(f"📝 Feedback: rating={rating} machine={machine_type}")
    feedback_file = Path("user_feedback.log")
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()},{machine_type},{step_index},{rating},{comments}\n")
    return {"status": "thank you"}


# ============================================================================
# API ENDPOINTS — NEW AGENT SYSTEM
# ============================================================================

@app.post("/agent/session", response_model=CreateSessionResponse)
async def create_agent_session(request: CreateSessionRequest):
    """
    Create a new stateful repair session for ANY supported farm machine.

    Call this once when the farmer describes their problem (after YOLO identifies the machine).
    Returns a session_id to pass to every subsequent /agent/next call.

    Supported machine_type values: tractor, harvester, thresher, submersible_pump,
    water_pump, electric_motor, power_tiller, rotavator, chaff_cutter, generator,
    diesel_engine — or any recognised alias (call GET /machines for full list).

    Body:
      {
        "machine_type": "thresher",
        "problem_description": "The threshing drum jammed with wheat crop",
        "language": "en"
      }
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
    """
    Get the next diagnostic step for an active repair session.

    The client calls this after each /verify_step, passing the full
    verification result. The agent reasons about the session state and
    returns ONE safe, logical next step.

    Body:
      {
        "session_id": "<uuid>",
        "last_verification_result": { ...full /verify_step response... }
      }

    Returns the same next_step structure that the Flutter AR Guide already
    understands from the original /diagnose endpoint.
    """
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
async def get_agent_session(session_id: str):
    """
    Inspect the current state of an active repair session.
    Useful for debugging and client-side state recovery.
    """
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or expired.")
    return JSONResponse(content=session.model_dump())


@app.delete("/agent/session/{session_id}")
async def delete_agent_session(session_id: str):
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

    port = int(os.environ.get("PORT", 7680))

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