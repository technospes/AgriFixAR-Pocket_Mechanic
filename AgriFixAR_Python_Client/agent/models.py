"""
agent/models.py
Pydantic data structures for the AgriFix repair agent system.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Literal
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Session state stored in memory
# ─────────────────────────────────────────────

class RepairSession(BaseModel):
    session_id: str
    machine_type: str
    problem: str
    verified_parts: Dict[str, Literal["ok", "damaged", "unclear"]] = Field(default_factory=dict)
    diagnostic_path: List[str] = Field(default_factory=list)
    current_stage: int = 0
    attempt_count: int = 0
    last_verification: Optional[Dict] = None
    language: str = "en"


# ─────────────────────────────────────────────
# /agent/next  —  request / response
# ─────────────────────────────────────────────

class AgentNextRequest(BaseModel):
    session_id: str
    last_verification_result: Dict  # full JSON blob from /verify_step


class NextStepDetail(BaseModel):
    text: str
    text_en: str
    text_hi: str
    visual_cue: str
    ar_model: str
    required_part: str
    area_hint: str
    safety_warning: Optional[str] = None


class UpdatedMemory(BaseModel):
    verified_parts: Dict[str, str]
    diagnostic_path: List[str]


class AgentNextResponse(BaseModel):
    status: Literal["continue", "resolved", "escalate", "unsafe"]
    reasoning_summary: str
    next_step: NextStepDetail
    updated_memory: UpdatedMemory


# ─────────────────────────────────────────────
# /agent/session  —  session creation
# ─────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    machine_type: str
    problem_description: str
    language: str = "en"


class CreateSessionResponse(BaseModel):
    session_id: str
    message: str
