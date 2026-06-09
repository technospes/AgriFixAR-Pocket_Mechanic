from __future__ import annotations
import asyncio
import json
import logging
import re

# MIGRATED: Gemini → Groq — google.generativeai removed
from utils.groq_client import groq_client, TEXT_MODEL, JSON_CONFIG  # MIGRATED: Gemini → Groq
from utils.json_repair import repair_json

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
# _GEMINI_MODEL removed — TEXT_MODEL from groq_client used instead  # MIGRATED: Gemini → Groq

# ── Machine-specific tool registry ────────────────────────────────────────────
# Prevents hallucinated tools. Only tools confirmed available on-farm per
# machine category are allowed. LLM must pick from this list or set null.
_MACHINE_TOOLS: dict[str, list[str]] = {
    "tractor":          ["multimeter", "spanner_set", "screwdriver_flat",
                         "screwdriver_phillips", "pliers", "wrench_adjustable",
                         "funnel", "clean_cloth", "torch_light"],
    "harvester":        ["spanner_set", "wrench_adjustable", "screwdriver_flat",
                         "pliers", "clean_cloth", "torch_light", "grease_gun"],
    "thresher":         ["spanner_set", "screwdriver_flat", "pliers",
                         "wrench_adjustable", "clean_cloth", "torch_light"],
    "submersible_pump": ["multimeter", "insulated_screwdriver",
                         "rubber_gloves", "torch_light", "pliers"],
    "water_pump":       ["pliers", "spanner_set", "screwdriver_flat",
                         "clean_cloth", "torch_light", "funnel"],
    "electric_motor":   ["multimeter", "insulated_screwdriver",
                         "rubber_gloves", "torch_light", "pliers"],
    "power_tiller":     ["spanner_set", "screwdriver_flat", "pliers",
                         "wrench_adjustable", "clean_cloth", "torch_light"],
    "chaff_cutter":     ["spanner_set", "screwdriver_flat", "pliers",
                         "clean_cloth", "torch_light"],
    "diesel_engine":    ["multimeter", "spanner_set", "screwdriver_flat",
                         "pliers", "wrench_adjustable", "funnel",
                         "clean_cloth", "torch_light"],
    "rotavator":        ["spanner_set", "screwdriver_flat", "pliers",
                         "wrench_adjustable", "clean_cloth", "torch_light"],
    "generator":        ["multimeter", "insulated_screwdriver",
                         "rubber_gloves", "pliers", "torch_light"],
}
_DEFAULT_TOOLS = ["spanner_set", "screwdriver_flat", "pliers", "clean_cloth", "torch_light"]


def _allowed_tools(machine_type: str) -> list[str]:
    return _MACHINE_TOOLS.get(machine_type, _DEFAULT_TOOLS)


def _tools_prompt_block(machine_type: str) -> str:
    tools = _allowed_tools(machine_type)
    return f"ALLOWED TOOLS (choose required_tool from this list or null): {', '.join(tools)}"


# ── Master Agent Prompt ───────────────────────────────────────────────────────
_MASTER_AGENT_PROMPT = """\
You are a stateful farm machinery diagnostic agent. Decide ONE safe next step.
Rules: never re-check verified-OK parts; unclear→retry same step; unsafe→stop immediately.
Step text rule: text_en must be 3–4 sentences — WHERE part is (colour+shape+landmark), \
WHAT to do with hands, WHAT to see/hear/feel when done. No jargon. Farmer has zero training.

MACHINE: {machine_type} | STAGE: {current_stage} | ATTEMPTS: {attempt_count}
TRIAGE ORDER: {triage_hint}
SAFETY KEYWORDS: {safety_kw}
{tools_block}

PROBLEM: {problem_description}

VERIFIED PARTS (pass/fail/unclear):
{verified_parts_json}

WHAT THE CAMERA ACTUALLY SAW (Gemini's visual findings per part):
{visual_observations}

DIAGNOSIS PLAN — STEPS ALREADY GENERATED (do not repeat or skip any):
{generated_steps_hint}

LAST CAMERA RESULT:
{last_verification_json}

SAFETY CONTEXT:
{safety_context}

ALLOWED area_hint: {allowed_area_hints}
KNOWN PARTS: {known_parts}

STRUCTURED STEP FORMAT — fill every field with real machine-specific content:
  • expected_result: what the farmer SEES/HEARS/FEELS when this step goes right.
    Be physical — "oil drips out" not "check is complete".
  • if_failed: the SINGLE most likely cause of step failure + one corrective action.
    Must differ from this step's instruction.
  • escalate_if: the condition that means STOP and call a mechanic.
    Must be a concrete observable (not "problem persists").
  • safety_warning: ONE sentence specific to this step's hazard, or null if no hazard.
  • required_tool: ONE tool from the ALLOWED TOOLS list that is genuinely needed,
    or null if the step is visual/observation-only.
    NEVER invent tools not on the allowed list.

Return ONLY this JSON:
{{
  "status": "continue" | "resolved" | "escalate" | "unsafe",
  "reasoning_summary": "<2-3 sentences — what was found and why this next step>",
  "next_step": {{
    "text": "<copy of text_en>",
    "text_en": "<3–4 sentences: WHERE part is + colour/shape/landmark | WHAT hands do | WHAT to see/hear when correct>",
    "text_hi": "<same 3–4 sentences in simple village Hindi — NOT formal>",
    "visual_cue": "<snake_case_part_id>",
    "ar_model": "<part.obj>",
    "required_part": "<snake_case_part_id>",
    "area_hint": "<one of allowed values above>",
    "safety_warning": "<one plain sentence or null>",
    "expected_result": "<what farmer sees/hears/feels when step succeeds — physical>",
    "expected_result_hi": "<same in simple Hindi>",
    "if_failed": "<most likely cause of failure + one corrective action>",
    "if_failed_hi": "<same in simple Hindi>",
    "escalate_if": "<concrete observable condition that means call a mechanic>",
    "escalate_if_hi": "<same in simple Hindi>",
    "required_tool": "<one tool from ALLOWED list or null>"
  }},
  "updated_memory": {{
    "verified_parts": {{"<part>": "ok|damaged|unclear"}},
    "diagnostic_path": ["<step_label>"]
  }}
}}"""


def _format_observations(session: RepairSession) -> str:
    if not session.verified_observations:
        return "None yet."
    return "\n".join(
        f"  {part}: {obs}"
        for part, obs in session.verified_observations.items()
    )


def _format_generated_steps(session: RepairSession) -> str:
    if not session.generated_steps:
        return "Not linked to a diagnosis plan."
    done = set(session.verified_parts.keys())
    lines = []
    for entry in session.generated_steps:
        parts = entry.split(":")
        step_id   = parts[0] if len(parts) > 0 else "?"
        part_id   = parts[1] if len(parts) > 1 else "?"
        step_type = parts[2] if len(parts) > 2 else "?"
        status = "✓ done" if part_id in done else "→ pending"
        lines.append(f"  {step_id}: {part_id} ({step_type}) [{status}]")
    return "\n".join(lines)


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
    tools_block    = _tools_prompt_block(session.machine_type)

    prompt = _MASTER_AGENT_PROMPT.format(
        machine_type           = session.machine_type,
        current_stage          = session.current_stage,
        attempt_count          = session.attempt_count,
        triage_hint            = triage_hint,
        safety_kw              = safety_kw,
        tools_block            = tools_block,
        problem_description    = session.problem,
        verified_parts_json    = json.dumps(session.verified_parts, indent=2),
        last_verification_json = json.dumps(last_verification, indent=2),
        safety_context         = safety_context,
        allowed_area_hints     = allowed_areas,
        known_parts            = known_parts,
        visual_observations    = _format_observations(session),
        generated_steps_hint   = _format_generated_steps(session),
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
        f"status={response.status} part={response.next_step.required_part} "
        f"tool={response.next_step.required_tool}"
    )
    return response


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _apply_verification(session: RepairSession, verification: dict) -> None:
    """Merge /verify_step result into session.verified_parts and verified_observations."""
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

    obs = (verification.get("ai_observation") or "").strip()
    if obs and obs.lower() not in ("", "none", "null"):
        session.verified_observations[part] = obs
        logger.info(f"👁️  [{session.machine_type}] observation stored: {part} → {obs[:80]}")


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


async def _call_gemini(prompt: str) -> str:  # MIGRATED: Gemini → Groq (function name preserved per RULE 4)
    response = await asyncio.to_thread(  # MIGRATED: Gemini → Groq
        lambda: groq_client.chat.completions.create(  # MIGRATED: Gemini → Groq
            model=TEXT_MODEL,  # MIGRATED: Gemini → Groq
            messages=[{"role": "user", "content": prompt}],  # MIGRATED: Gemini → Groq
            **JSON_CONFIG,  # MIGRATED: Gemini → Groq
        )
    )
    return response.choices[0].message.content  # MIGRATED: Gemini → Groq


def _validate_tool(tool: str | None, machine_type: str) -> str | None:
    """Reject any tool not on the allowed list for this machine. Returns None if invalid."""
    if not tool:
        return None
    allowed = _allowed_tools(machine_type)
    tool_clean = tool.lower().strip().replace(" ", "_")
    if tool_clean in allowed:
        return tool_clean
    # Partial-match fallback (e.g. "adjustable wrench" → "wrench_adjustable")
    for a in allowed:
        if tool_clean in a or a in tool_clean:
            logger.info(f"🔧 [{machine_type}] tool fuzzy-match: '{tool}' → '{a}'")
            return a
    logger.warning(f"⚠️  [{machine_type}] Hallucinated tool rejected: '{tool}'")
    return None


def _parse_response(raw: str, machine_type: str) -> AgentNextResponse:
    allowed = get_allowed_area_ids(machine_type)

    try:
        data = repair_json(raw)
    except json.JSONDecodeError as exc:
        logger.error(f"❌ [{machine_type}] Invalid JSON: {exc}\n{raw[:400]}")
        return _fallback_response(machine_type, f"JSON parse error: {exc}")

    try:
        ns   = data["next_step"]
        um   = data.get("updated_memory", {})
        area = ns.get("area_hint", "")
        if area not in allowed:
            logger.warning(f"⚠️  [{machine_type}] Invalid area_hint '{area}' → correcting")
            ns["area_hint"] = allowed[0] if allowed else "engine_compartment"

        text_en = ns.get("text_en", "")
        if len(text_en.split()) < 25:
            logger.warning(
                f"⚠️  [{machine_type}] Agent text_en only {len(text_en.split())} words — "
                "expected 3–4 guided sentences"
            )
        if not ns.get("text") and text_en:
            ns["text"] = text_en

        # ── Validate required_tool against allowed list ───────────────────
        raw_tool     = ns.get("required_tool")
        validated_tool = _validate_tool(raw_tool, machine_type)

        # ── Warn on missing structured fields ─────────────────────────────
        for field in ("expected_result", "if_failed", "escalate_if"):
            if not ns.get(field):
                logger.warning(f"⚠️  [{machine_type}] Missing structured field: {field}")

        return AgentNextResponse(
            status             = data.get("status", "continue"),
            reasoning_summary  = data.get("reasoning_summary", ""),
            next_step = NextStepDetail(
                text             = ns.get("text", ""),
                text_en          = ns.get("text_en", ""),
                text_hi          = ns.get("text_hi", ""),
                visual_cue       = ns.get("visual_cue", "unknown"),
                ar_model         = ns.get("ar_model", "part.obj"),
                required_part    = ns.get("required_part", "unknown"),
                area_hint        = ns["area_hint"],
                safety_warning   = ns.get("safety_warning"),
                expected_result    = ns.get("expected_result", ""),
                expected_result_hi = ns.get("expected_result_hi", ""),
                if_failed          = ns.get("if_failed", ""),
                if_failed_hi       = ns.get("if_failed_hi", ""),
                escalate_if        = ns.get("escalate_if", ""),
                escalate_if_hi     = ns.get("escalate_if_hi", ""),
                required_tool      = validated_tool,
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
            expected_result="N/A",
            expected_result_hi="N/A",
            if_failed="Contact a certified mechanic.",
            if_failed_hi="प्रमाणित मैकेनिक से संपर्क करें।",
            escalate_if="Immediately — agent could not generate a safe step.",
            escalate_if_hi="तुरंत — एजेंट सुरक्षित कदम नहीं बना सका।",
            required_tool=None,
        ),
        updated_memory=UpdatedMemory(verified_parts={}, diagnostic_path=["agent_error"]),
    )