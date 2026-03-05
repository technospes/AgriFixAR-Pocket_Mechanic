"""
agent/session_manager.py
Thread-safe in-memory store for active repair sessions.

Design notes:
  - Single process (Uvicorn workers=1 on HuggingFace) → plain dict is safe.
  - For multi-worker deployments replace with Redis / external store.
  - Sessions expire after SESSION_TTL_SECONDS of inactivity.
"""

from __future__ import annotations
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

from agent.models import RepairSession

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
SESSION_TTL_SECONDS = 3600  # 1 hour inactivity → session dropped

# ── Store ────────────────────────────────────────────────────────────────────
_sessions: Dict[str, RepairSession] = {}
_last_access: Dict[str, datetime] = {}


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def create_session(machine_type: str, problem: str, language: str = "en") -> RepairSession:
    """Create a brand-new repair session and return it."""
    session_id = str(uuid.uuid4())
    session = RepairSession(
        session_id=session_id,
        machine_type=machine_type,
        problem=problem,
        language=language,
    )
    _sessions[session_id] = session
    _last_access[session_id] = datetime.utcnow()
    logger.info(f"🆕 Session created: {session_id}  machine={machine_type}")
    return session


def get_session(session_id: str) -> Optional[RepairSession]:
    """Return session or None if not found / expired."""
    _evict_expired()
    session = _sessions.get(session_id)
    if session:
        _last_access[session_id] = datetime.utcnow()
    return session


def update_session(session: RepairSession) -> None:
    """Persist updated session back to the store."""
    _sessions[session.session_id] = session
    _last_access[session.session_id] = datetime.utcnow()
    logger.debug(f"💾 Session updated: {session.session_id}  stage={session.current_stage}")


def delete_session(session_id: str) -> bool:
    """Explicitly remove a session (e.g. after resolution)."""
    existed = session_id in _sessions
    _sessions.pop(session_id, None)
    _last_access.pop(session_id, None)
    if existed:
        logger.info(f"🗑️  Session deleted: {session_id}")
    return existed


def list_sessions() -> list[str]:
    """Return all active session IDs (debug / admin use)."""
    _evict_expired()
    return list(_sessions.keys())


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _evict_expired() -> None:
    """Remove sessions that have been idle beyond TTL."""
    cutoff = datetime.utcnow() - timedelta(seconds=SESSION_TTL_SECONDS)
    expired = [sid for sid, ts in _last_access.items() if ts < cutoff]
    for sid in expired:
        _sessions.pop(sid, None)
        _last_access.pop(sid, None)
        logger.info(f"⏰ Session expired + evicted: {sid}")
