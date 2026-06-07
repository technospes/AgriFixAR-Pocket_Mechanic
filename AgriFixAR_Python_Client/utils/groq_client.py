"""
utils/groq_client.py
Shared Groq client — single source of truth for all LLM text-generation calls.

All text-generation modules (diagnosis_service, multihop_diagnosis, query_router,
repair_agent, etc.) must import from here. Never instantiate Groq() elsewhere.

VISION MODEL NOTE:
  Visual inference (visual_gate.py, locate_part_service.py, verification_service.py)
  uses llama-3.2-11b-vision-preview and will be migrated in a SEPARATE future task.
  Do NOT use this client for vision calls yet.
"""  # MIGRATED: Gemini → Groq

from __future__ import annotations
import os
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()
from groq import Groq  # MIGRATED: Gemini → Groq

logger = logging.getLogger(__name__)

# ── API key ────────────────────────────────────────────────────────────────────
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # MIGRATED: Gemini → Groq
if not _GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing — set this environment variable before starting the server.")  # MIGRATED: Gemini → Groq

# ── Shared client instance ─────────────────────────────────────────────────────
groq_client = Groq(api_key=_GROQ_API_KEY)  # MIGRATED: Gemini → Groq

# ── Model identifiers ──────────────────────────────────────────────────────────
TEXT_MODEL = "llama-3.3-70b-versatile"  # MIGRATED: Gemini → Groq
TEXT_MODEL_FALLBACK = "llama-3.1-8b-instant"  # FAILOVER: primary → fallback

# FUTURE: vision model reserved for a separate migration task
# VISION_MODEL = "llama-3.2-11b-vision-preview"  # MIGRATED: Gemini → Groq (future)

# ── Model registry (preparation for future multi-level failover) ──────────────
AVAILABLE_MODELS = [  # FAILOVER: primary → fallback
    TEXT_MODEL,
    TEXT_MODEL_FALLBACK,
]

# ── Default generation config for JSON-producing calls ───────────────────────
# temperature=0.1  → near-deterministic for grounded repair reasoning
# max_tokens=2048  → sufficient for diagnosis JSON + reasoning CoT
JSON_CONFIG = {  # MIGRATED: Gemini → Groq
    "temperature": 0.1,
    "max_tokens": 2048,
}

# ── Lighter config for short-output calls (router, multihop) ──────────────────
SHORT_CONFIG = {  # MIGRATED: Gemini → Groq
    "temperature": 0.1,
    "max_tokens": 400,
}

# ── Recoverable error signals (trigger failover) ──────────────────────────────
_RECOVERABLE_ERRORS = (  # FAILOVER: primary → fallback
    "429",
    "rate limit",
    "rate_limit_exceeded",
    "quota",
    "tokens per day",
    "tpd",
    "tpm",
    "rpm",
    "too many requests",
    "capacity",
    "overloaded",
    "service unavailable",
)

# ── Non-recoverable error signals (raise immediately, no failover) ─────────────
_NON_RECOVERABLE_ERRORS = (  # FAILOVER: primary → fallback
    "authentication",
    "invalid api key",
    "permission denied",
    "invalid request",
    "malformed",
    "bad request",
)


def groq_chat_completion(  # FAILOVER: primary → fallback
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> Any:
    """
    FAILOVER: primary → fallback
    Centralized Groq completion wrapper with automatic primary→fallback failover.

    Flow:
        1. Try TEXT_MODEL (llama-3.3-70b-versatile).
        2. On recoverable failure (rate limit, quota, capacity), try TEXT_MODEL_FALLBACK
           (llama-3.1-8b-instant) with optionally reduced max_tokens.
        3. On non-recoverable failure (auth, bad request), raise immediately.
        4. If fallback also fails, raise with both error messages logged.

    Uses the module-level groq_client — never creates a new client per call.

    Returns:
        The raw Groq completion response object (same as groq_client.chat.completions.create).
    """
    primary_error: Optional[Exception] = None  # FAILOVER: primary → fallback

    # ── Attempt primary model ─────────────────────────────────────────────────
    try:
        response = groq_client.chat.completions.create(  # FAILOVER: primary → fallback
            model=TEXT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.debug(  # FAILOVER: primary → fallback
            "Groq primary model succeeded | model=%s",
            TEXT_MODEL,
        )
        return response

    except Exception as e:  # FAILOVER: primary → fallback
        err_str = str(e).lower()

        # Non-recoverable — raise immediately, no failover
        if any(kw in err_str for kw in _NON_RECOVERABLE_ERRORS):  # FAILOVER: primary → fallback
            logger.error(  # FAILOVER: primary → fallback
                "Groq non-recoverable error | model=%s error=%s",
                TEXT_MODEL, e,
            )
            raise

        # Recoverable — fall through to fallback
        if any(kw in err_str for kw in _RECOVERABLE_ERRORS):  # FAILOVER: primary → fallback
            logger.warning(  # FAILOVER: primary → fallback
                "Groq primary failed; attempting fallback | error=%s",
                e,
            )
            primary_error = e
        else:
            # Unknown error category — also attempt fallback but log as warning
            logger.warning(  # FAILOVER: primary → fallback
                "Groq primary failed with unknown error; attempting fallback | error=%s",
                e,
            )
            primary_error = e

    # ── Token-saving fallback: reduce max_tokens when daily/per-minute limit hit ─
    fallback_max_tokens = max_tokens  # FAILOVER: primary → fallback
    if primary_error is not None:
        primary_err_str = str(primary_error).lower()
        if "tokens per day" in primary_err_str or "tpd" in primary_err_str:  # FAILOVER: primary → fallback
            fallback_max_tokens = min(max_tokens, 1200)  # FAILOVER: primary → fallback
            logger.info(  # FAILOVER: primary → fallback
                "FAILOVER: TPD limit hit — capping fallback max_tokens to %d",
                fallback_max_tokens,
            )

    # ── Attempt fallback model ────────────────────────────────────────────────
    try:
        response = groq_client.chat.completions.create(  # FAILOVER: primary → fallback
            model=TEXT_MODEL_FALLBACK,
            messages=messages,
            temperature=temperature,
            max_tokens=fallback_max_tokens,
        )
        logger.info(  # FAILOVER: primary → fallback
            "Groq fallback model used successfully | model=%s",
            TEXT_MODEL_FALLBACK,
        )
        return response

    except Exception as fallback_error:  # FAILOVER: primary → fallback
        logger.error(  # FAILOVER: primary → fallback
            "Groq fallback failed | primary=%s fallback=%s",
            primary_error,
            fallback_error,
        )
        raise fallback_error  # FAILOVER: primary → fallback


logger.info(
    "Groq client initialised | text_model=%s fallback_model=%s",  # FAILOVER: primary → fallback
    TEXT_MODEL,
    TEXT_MODEL_FALLBACK,
)