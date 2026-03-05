from __future__ import annotations
import asyncio
import json
import logging
import re

import google.generativeai as genai

from agent.models import RepairSession, AgentNextResponse, NextStepDetail, UpdatedMemory
from agent import safety_rules
from utils.machine_registry import (
    get_profile_or_default,
    get_allowed_area_ids,
    get_compact_parts_list,
    get_compact_diagnostic_hint,
    get_compact_safety_keywords,
    get_critical_parts,
    get_fuel_system_parts,
    is_electric_machine,
)

logger = logging.getLogger(__name__)
_GEMINI_MODEL = "models/gemini-2.5-flash"


# ── Master Agent Prompt ───────────────────────────────────────────────────────
# Design rules applied:
#  • System role: 2 lines (was 9 bullets ~148 tok → ~20 tok)
#  • No diagnostic_context repeat (was ~93 tok)
#  • No machine label/intro repeat (was ~42 tok)
#  • Logic rules → 1-line machine-specific triage (was 8 items ~192 tok → ~15 tok)
#  • Safety keywords → compact (was full sentences ~59 tok → ~15 tok)
#  • All accuracy-critical data kept: session memory, verified parts,
#    last verification, safety context, area hints, output schema
_MASTER_AGENT_PROMPT = """\
You are a stateful farm machinery diagnostic agent. Decide ONE safe next step.
Rules: never re-check verified-OK parts; unclear→retry same step; unsafe→stop immediately.

MACHINE: {machine_type} | STAGE: {current_stage} | ATTEMPTS: {attempt_count}
TRIAGE ORDER: {triage_hint}
SAFETY KEYWORDS: {safety_kw}

PROBLEM: {problem_description}

VERIFIED PARTS:
{verified_parts_json}

LAST CAMERA RESULT:
{last_verification_json}

SAFETY CONTEXT:
{safety_context}

ALLOWED area_hint: {allowed_area_hints}
KNOWN PARTS: {known_parts}

Return ONLY this JSON:
{{
  "status": "continue" | "resolved" | "escalate" | "unsafe",
  "reasoning_summary": "<2-3 sentences — what was found and why this next step>",
  "next_step": {{
    "text": "<primary language>",
    "text_en": "<English — colour/shape/position/landmark>",
    "text_hi": "<Hindi — simple village language>",
    "visual_cue": "<snake_case_part_id>",
    "ar_model": "<part.obj>",
    "required_part": "<snake_case_part_id>",
    "area_hint": "<one of allowed values above>",
    "safety_warning": "<one sentence or null>"
  }},
  "updated_memory": {{
    "verified_parts": {{"<part>": "ok|damaged|unclear"}},
    "diagnostic_path": ["<step_label>"]
  }}
}}"""


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

async def decide_next_step(
    session: RepairSession,
    last_verification: dict,
) -> AgentNextResponse:
    """Core agent reasoning — machine-aware, token-optimised."""

    _apply_verification(session, last_verification)

    forced = safety_rules.pre_check(session)
    if forced:
        logger.info(f"🛡️  Safety pre-check forced response [{session.machine_type}]")
        return forced

    profile        = get_profile_or_default(session.machine_type)
    allowed_areas  = " | ".join(get_allowed_area_ids(session.machine_type))
    known_parts    = get_compact_parts_list(session.machine_type)
    triage_hint    = get_compact_diagnostic_hint(session.machine_type)
    safety_kw      = get_compact_safety_keywords(session.machine_type)
    safety_context = _build_safety_context(session)

    prompt = _MASTER_AGENT_PROMPT.format(
        machine_type         = session.machine_type,
        current_stage        = session.current_stage,
        attempt_count        = session.attempt_count,
        triage_hint          = triage_hint,
        safety_kw            = safety_kw,
        problem_description  = session.problem,
        verified_parts_json  = json.dumps(session.verified_parts, indent=2),
        last_verification_json = json.dumps(last_verification, indent=2),
        safety_context       = safety_context,
        allowed_area_hints   = allowed_areas,
        known_parts          = known_parts,
    )

    raw = await _call_gemini(prompt)
    response = _parse_response(raw, session.machine_type)
    response = safety_rules.post_check(response, session)

    session.verified_parts.update(response.updated_memory.verified_parts)
    for step in response.updated_memory.diagnostic_path:
        if step not in session.diagnostic_path:
            session.diagnostic_path.append(step)
    session.current_stage  += 1
    session.attempt_count  += 1
    session.last_verification = last_verification

    logger.info(
        f"🤖 Agent [{session.machine_type}] stage={session.current_stage} "
        f"status={response.status} part={response.next_step.required_part}"
    )
    return response


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _apply_verification(session: RepairSession, verification: dict) -> None:
    """Merge /verify_step result into session.verified_parts."""
    part = (
        verification.get("required_part")
        or verification.get("correct_part")
        or verification.get("part_detected")
    )
    if not part or part in ("none", "machine_part", "unknown"):
        return
    status = verification.get("status", "unclear")
    conf   = float(verification.get("confidence", 0))

    if status in ("pass", "verified") and conf >= 0.6:
        session.verified_parts[part] = "ok"
        logger.info(f"✅ [{session.machine_type}] OK: {part} (conf={conf:.2f})")
    elif status == "fail" and conf >= 0.6:
        session.verified_parts[part] = "damaged"
        logger.warning(f"⚠️  [{session.machine_type}] DAMAGED: {part} (conf={conf:.2f})")
    else:
        session.verified_parts[part] = "unclear"
        logger.info(f"❓ [{session.machine_type}] unclear: {part} (conf={conf:.2f})")


def _build_safety_context(session: RepairSession) -> str:
    """Compact safety summary — accuracy-critical, kept in full."""
    machine_type = session.machine_type
    fuel_parts   = set(get_fuel_system_parts(machine_type))
    is_electric  = is_electric_machine(machine_type)

    damaged = [p for p, s in session.verified_parts.items() if s == "damaged"]
    ok      = [p for p, s in session.verified_parts.items() if s == "ok"]
    lines   = []

    if damaged:
        lines.append(f"DAMAGED: {', '.join(damaged)}")
        fuel_dmg = set(damaged) & fuel_parts
        if fuel_dmg:
            lines.append(f"FUEL_LEAK({', '.join(fuel_dmg)}): block ignition/crank steps.")
    if ok:
        lines.append(f"SKIP(already_ok): {', '.join(ok)}")
    if is_electric:
        lines.append("ELECTRIC: power_off required before every step.")
    if not lines:
        lines.append("No parts verified yet — start with safest external check.")
    return "\n".join(lines)


async def _call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel(_GEMINI_MODEL)
    return (await asyncio.get_event_loop().run_in_executor(
        None, lambda: model.generate_content(prompt)
    )).text


def _parse_response(raw: str, machine_type: str) -> AgentNextResponse:
    text = raw.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*",     "", text)
    text = re.sub(r"\s*```$",     "", text)
    text = re.sub(r",\s*}",       "}", text)
    text = re.sub(r",\s*]",       "]", text)

    allowed = get_allowed_area_ids(machine_type)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"❌ [{machine_type}] Invalid JSON: {exc}\n{text[:400]}")
        return _fallback_response(machine_type, f"JSON parse error: {exc}")

    try:
        ns   = data["next_step"]
        um   = data.get("updated_memory", {})
        area = ns.get("area_hint", "")
        if area not in allowed:
            logger.warning(f"⚠️  [{machine_type}] Invalid area_hint '{area}' → correcting")
            ns["area_hint"] = allowed[0] if allowed else "engine_compartment"

        return AgentNextResponse(
            status             = data.get("status", "continue"),
            reasoning_summary  = data.get("reasoning_summary", ""),
            next_step = NextStepDetail(
                text           = ns.get("text", ""),
                text_en        = ns.get("text_en", ""),
                text_hi        = ns.get("text_hi", ""),
                visual_cue     = ns.get("visual_cue", "unknown"),
                ar_model       = ns.get("ar_model", "part.obj"),
                required_part  = ns.get("required_part", "unknown"),
                area_hint      = ns["area_hint"],
                safety_warning = ns.get("safety_warning"),
            ),
            updated_memory = UpdatedMemory(
                verified_parts  = um.get("verified_parts", {}),
                diagnostic_path = um.get("diagnostic_path", []),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.error(f"❌ [{machine_type}] Schema error: {exc}")
        return _fallback_response(machine_type, f"Schema error: {exc}")


def _fallback_response(machine_type: str, reason: str) -> AgentNextResponse:
    allowed = get_allowed_area_ids(machine_type)
    return AgentNextResponse(
        status = "escalate",
        reasoning_summary = f"Agent error [{machine_type}]: {reason}",
        next_step = NextStepDetail(
            text="Unable to determine next step. Please consult a mechanic.",
            text_en="Unable to determine next step. Consult a certified mechanic.",
            text_hi="अगला कदम निर्धारित नहीं हो सका। प्रमाणित मैकेनिक से संपर्क करें।",
            visual_cue="none", ar_model="none.obj", required_part="none",
            area_hint=allowed[0] if allowed else "engine_compartment",
            safety_warning="Stop repairs and seek professional assistance.",
        ),
        updated_memory=UpdatedMemory(verified_parts={}, diagnostic_path=["agent_error"]),
    )