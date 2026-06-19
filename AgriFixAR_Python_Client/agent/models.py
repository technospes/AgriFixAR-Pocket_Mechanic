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
    verified_observations: Dict[str, str] = Field(default_factory=dict)
    # ↑ Gemini's actual visual finding per part, e.g.:
    #   {"battery_terminal": "white powder visible on both clamps — corrosion confirmed"}
    diagnostic_path: List[str] = Field(default_factory=list)
    generated_steps: List[str] = Field(default_factory=list)
    # ↑ Step IDs + part from the diagnosis plan, e.g. ["s1:battery_terminal:visual", ...]
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
    # ── Core instruction (bilingual) ─────────────────────────────────────────
    text: str
    text_en: str
    text_hi: str

    # ── AR / visual anchoring ────────────────────────────────────────────────
    visual_cue: str
    ar_model: str
    required_part: str
    area_hint: str

    # ── Safety ──────────────────────────────────────────────────────────────
    safety_warning: Optional[str] = None

    # ── Structured repair output fields ──────────────────────────────────────
    # What the farmer should observe when the step succeeds (physical)
    expected_result: str = ""
    expected_result_hi: str = ""

    # Most likely cause if the step fails + single corrective action
    if_failed: str = ""
    if_failed_hi: str = ""

    # Concrete condition that means STOP and call a mechanic
    escalate_if: str = ""
    escalate_if_hi: str = ""

    # Tool from the machine-specific allowed list, or null for visual-only steps
    required_tool: Optional[str] = None


class UpdatedMemory(BaseModel):
    verified_parts: Dict[
        str,
        Literal["ok", "damaged", "unclear"]
    ]
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
    diagnosis_steps: List[Dict] = Field(default_factory=list)
    # ↑ Optional: pass the steps array from /diagnose response so the agent
    #   knows exactly which parts the plan covers and in what order.


class CreateSessionResponse(BaseModel):
    session_id: str
    message: str