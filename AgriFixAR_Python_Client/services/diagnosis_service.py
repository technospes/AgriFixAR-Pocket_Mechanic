from __future__ import annotations
import asyncio
import hashlib
import io
import json
import logging
import re
from typing import List, Optional, TypedDict
from enum import StrEnum
import os
# MIGRATED: Gemini → Groq — google.generativeai removed
from utils.groq_client import groq_client, TEXT_MODEL, JSON_CONFIG, groq_chat_completion  # MIGRATED: Gemini → Groq  # FAILOVER: primary → fallback
from utils.json_repair import repair_json
from PIL import Image

from rag import retrieve_with_confidence, RAG_WEAK_THRESHOLD
from query_router import route_query, load_machine_registry
from utils.helpers import generate_cache_key, get_cached_response, cache_response
from utils.machine_registry import (
    get_profile_or_default, get_allowed_area_ids, get_compact_parts_list,
    get_compact_safety_keywords, is_electric_machine, is_tractor_attachment,
)
from services.safety_guards import (
    run_text_hazard_checks,
    _guard_electric_hazard,
    _guard_dangerous_workaround,
    _guard_emergency_hazard,
)
# FIX 3: Multi-hop diagnostic chain — subsystem classification + root cause ranking
from multihop_diagnosis import run_diagnostic_chain

# FIX 5: Procedure validator — safety step injection before response is returned
from procedure_validator import validate_procedure

# FIX 6: Adaptive clarification loop — ask targeted questions before escalating
# clarification_loop imports removed — orchestrator handles clarification exclusively

# FIX 6: Verification gate — verify machine/component before final repair
from services.verification_service import verify_step_with_gemini

# Repair-plan structural validation — single source of truth, shared with
# repair_agent.py. See agent/validation.py for the rationale.
from agent.validation import InvalidRepairPlan, validate_repair_plan_steps


logger = logging.getLogger(__name__)
# _GEMINI_MODEL removed — TEXT_MODEL from groq_client used instead  # MIGRATED: Gemini → Groq
_RAG_STRICT_THRESHOLD = 0.40

# ── Escalation response schema ────────────────────────────────────────────────

class RagSource(StrEnum):
    """Canonical rag_source values. One typo breaks analytics — use these."""
    PRE_CALL            = "pre_call_guard"
    ERROR               = "error"
    NO_CONTEXT          = "no_context"
    EMPTY_RAG_MAINT     = "guard5_empty_rag_maintenance"
    UNKNOWN_ATTACHMENT  = "guard_unknown_attachment"
    ELECTRIC_HAZARD     = "pre_guard_electric_shock_injury"
    WORKAROUND          = "pre_guard_workaround"
    EMERGENCY_HAZARD    = "emergency_hazard_guard"
    ESCALATION          = "escalation"


class EscalationResponse(TypedDict, total=False):
    """Canonical shape of every escalation response. Flutter parses this."""
    status: str
    problem_description: str
    technical_analysis: str
    solution: dict
    rag_source: str
    machine_label: str
    # Optional: unsafe scene
    unsafe_scene_suspected: bool
    unsafe_scene_message: str
    # Optional: emergency hazard metadata
    hazard_category: str
    hazard_level: str
    emergency_stop_required: bool
    requires_power_isolation: bool
    requires_medical_attention: bool
    requires_fire_response: bool
    requires_evacuation: bool
    # Internal markers
    cache_hit: bool
    _informational_override: bool
    _reassurance_override: bool
    _verification_issues: list

# ── FIX 6: Verification gate thresholds ──────────────────────────────────────
# Below this confidence, verification fails and we return "Need verification"
_VERIFICATION_CONFIDENCE_THRESHOLD = 0.60
# Only run the verification gate when risk_level is HIGH or CRITICAL
_VERIFICATION_RISK_LEVELS = {"HIGH", "CRITICAL"}

_GROUNDING_RULE = """\
GROUNDING RULE (non-negotiable): The retrieved evidence below is your PRIMARY
knowledge source. Reason over these chunks first. Your DIAGNOSIS — which part
is at fault, what fixes it, and any spec/interval/measurement — MUST be derived
exclusively from the KNOWLEDGE BASE CONTEXT below. Never invent a cause, fix,
spec, or interval that isn't in the context.

Use your general mechanical knowledge ONLY to:
• Connect evidence across chunks
• Explain terminology in farmer-friendly language
• Describe what named parts look like and where they are located
• Fill obvious procedural gaps (e.g., "tighten the bolt" when the manual says
  "secure the component")

Never contradict the retrieved evidence. If the evidence is insufficient to
diagnose the actual fault, state: "This is not covered in the manual — consult
a certified mechanic." """

_WEAK_CONTEXT_HEADER = """\
 RAG CONTEXT QUALITY: LOW (top relevance score < 0.40)
The retrieved manual excerpts have LOW relevance to this query.
STRICT GROUNDING MODE IS ACTIVE:
  • Only state what the chunks below explicitly say.
  • Do NOT speculate, generalise, or use outside knowledge to fill gaps.
  • If a specific fault or procedure is not covered, say so and escalate.
"""

_STRONG_CONTEXT_HEADER = """\
RAG CONTEXT QUALITY: STRONG (top relevance score ≥ 0.40)
Manual excerpts are highly relevant. Use them as the primary source.
"""

_NO_CONTEXT_ESCALATION_EN = "I was unable to find relevant information in the technical manual for this specific situation. Please consult a certified mechanic or your nearest Mahindra service centre for a safe diagnosis."
_NO_CONTEXT_ESCALATION_HI = "इस समस्या के लिए हमारे तकनीकी मैनुअल में कोई जानकारी नहीं मिली। कृपया प्रमाणित मैकेनिक या नजदीकी महिंद्रा सर्विस सेंटर से सुरक्षित निदान करवाएं।"

def _has_visual_context_text(problem_text: str) -> bool:
    return "visual context:" in (problem_text or "").lower()

def _extract_visual_snippet(problem_text: str) -> str:
    match = re.search(r'visual context:\s*(.+?)(?:audio:|$)', problem_text or "", re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

_SPEC_FIDELITY_RULE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 7 — FLUID SUBSTITUTION & SPEC FIDELITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the query involves using a non-OEM fluid (cooking oil, mustard oil,
engine oil in place of hydraulic oil, petrol in a diesel engine, etc.):
  1. REJECT the substitute explicitly and state the exact damage it causes.
  2. COPY the exact OEM specification (grade, viscosity, standard) verbatim
     from the matched chunk — do NOT paraphrase or approximate it.
  3. Include this verbatim specification in both technical_analysis AND in
     a dedicated step.

If the query involves suction failure, loss of prime, or priming on a
water pump:
  1. The foot valve / suction pipe check is MANDATORY — include it as a
     named step even if the user only asked about re-priming.
  2. Use the exact term "foot valve" as written in the manual.
"""

def _build_strict_grounding_prompt(machine_type, machine_label, problem_text, rag_context, allowed_areas, parts_list, safety_keywords, language, has_visual_frames, context_quality, top_score, router_symptoms: Optional[List[str]] = None) -> str:
    visual_note = ""
    visual_text_present = _has_visual_context_text(problem_text)
    visual_snippet = _extract_visual_snippet(problem_text) if visual_text_present else ""

    if has_visual_frames:
        visual_note = "Camera images are available for context. Use them to confirm the machine type and general condition, but generate repair steps based on the manual excerpts and symptom description below."
    elif visual_text_present:
        visual_note = f"VISUAL PRIORITY OVERRIDE — ACTIVE (embedded visual descriptor detected):\n  • RULE: The visual context (\"{visual_snippet}\") is the PRIMARY symptom.\n  • If the 'Audio:' field contradicts the visual context, TRUST THE VISUAL CONTEXT.\n"

    # from utils.machine_registry import get_shutdown_instruction
    # shutdown = get_shutdown_instruction(machine_type)
    
    electric_note = ""
    # f"""CRITICAL SAFETY — STEP 1 REQUIREMENT:
    #                 Your FIRST step MUST be a safety step with these EXACT values:
    #                 action: "{shutdown['action']}"
    #                 description: "{shutdown['instruction_en']}"
    #                 required_part: "{shutdown['required_part']}"
    #                 area_hint: "{shutdown['area_hint']}"
    #                 step_type: "safety"
    #                 Copy these values exactly. Do NOT modify or paraphrase.\n"""
    quality_banner = _WEAK_CONTEXT_HEADER if context_quality == "weak" else _STRONG_CONTEXT_HEADER
    quality_banner += f"Top chunk relevance: {top_score:.2f} | "

    if rag_context and rag_context.strip():
        rag_block = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nMANUAL EXTRACTS (AUTHORITATIVE SOURCE — USE THESE FIRST)\n{quality_banner}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{rag_context}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    else:
        rag_block = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n⚠️ NO MANUAL EXTRACTS AVAILABLE — HALLUCINATION TRAP ACTIVE\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nThe RAG retrieval system found NO chunks above the relevance threshold.\nYOU MUST OUTPUT AN ESCALATION RESPONSE. Set steps = [] (empty array).\nDO NOT generate steps, intervals, oil grades, or any procedure from your\nown training knowledge — that is a CRITICAL FAILURE regardless of how\nhelpful it would be to the user.\n\nCASE A — Machine type matches {machine_type} AND query describes a real\nmechanical part or maintenance task:\n  → status = \"escalate\", steps = []\n  → technical_analysis = \"This procedure is not covered in the {machine_label} service manual.\"\n  → safety_warnings_en[0] = \"This specific procedure is not in our repair manual for the {machine_label}. Please consult the manufacturer's user guide or a certified mechanic.\"\n\nCASE B — Out-of-scope or unknown fault:\n  → status = \"escalate\", steps = []\n  → technical_analysis = \"Insufficient knowledge base coverage for this symptom.\"\n  → safety_warnings_en[0] = \"Automatic diagnosis unavailable: symptom outside knowledge base. Consult a certified mechanic.\""

    target_symptoms = ", ".join(router_symptoms) if router_symptoms else problem_text
    user_query_words = problem_text
    grounding_rules = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{_GROUNDING_RULE}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRICT GROUNDING RULES — ZERO-HALLUCINATION PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔵 MANDATORY CHAIN OF THOUGHT — Complete all 4 steps in "internal_reasoning"
   BEFORE setting any other output field.

   STEP 1 — [Visual/Audio Check]:
   One sentence: "The user describes [X symptom] with [visual/audio evidence Y]."
   Visual context (if present) ALWAYS overrides audio/text description.

   STEP 2 — RAG CHUNK SEARCH:
    The user said: "{user_query_words}"
    Review each chunk. A chunk matches if:
    • Its PROBLEM field contains the SAME mechanical symptom OR describes
      a fault in the SAME component/system, even if the exact wording differs
      (e.g. "clutch is stuck" matches "Clutch fails to work" / "Clutch cable
      damage" — these are all clutch malfunctions; do NOT require literal
      word-for-word symptom equality).
    • Prefer chunks where MULTIPLE user symptoms or the SAME component appear.
    • Only state "no chunk covers this symptom" if NO chunk mentions the same
      component or system at all.
    List all matching chunk IDs with their PROBLEM fields and explain why each
    matches.
   
   The user's extracted symptoms are: {target_symptoms}
   
   After listing ALL chunks, rank them by SYMPTOM OVERLAP:
   • A chunk matches if its PROBLEM field contains the SAME mechanical symptom.
   • Prefer chunks where MULTIPLE user symptoms appear in the SAME chunk.
   • The chunk with the highest symptom overlap is your PRIMARY DIAGNOSIS SOURCE.
   • State which chunk best matches the symptoms. If NO chunk has any symptom overlap, state "no chunk covers this symptom".
   
   ⛔ YOU MUST INCLUDE THE CHUNK CONTENT. "A safety chunk applies" = GROUNDING FAILURE.

   STEP 3 — [Action Decision]:
   State exactly one of:
     • "ESCALATE — Universal safety hazard (Chunk [ID]) overrides all other rules."
     • "ESCALATE — Dangerous workaround detected (Chunk [ID])."
     • "ESCALATE — No chunk PROBLEM field mentions any component related to this symptom."
     • "ESCALATE — Routine maintenance, no active fault (Chunk [ID])."
     • "SUCCESS/REASSURE — Chunk [ID] explicitly confirms symptom is normal."
     • "SUCCESS/FIRE_EXT — Chunk 3b informational query, ESCALATE_IF says DO NOT escalate."
     • "DIAGNOSE — Chunk [ID] covers the same component/system as the symptom
       (component-class match per RULE 5); use its PROBLEM/STEPS as the closest
       applicable fix."

   STEP 4 — [Faithfulness Verification]:
   Check only the DIAGNOSTIC claims in your draft — which part is at fault,
   what fixes it, any spec/interval/part number. Is each one explicitly
   written in the matched chunk? If not, delete or soften it.
   
   Do NOT apply this check to general descriptive language (how to locate
   a part using visible landmarks, what it looks like, what healthy vs.
   faulty looks like). That is general mechanical knowledge, not a
   diagnostic claim, and is REQUIRED by the FARMER INSTRUCTION STANDARD
   below even when the manual itself doesn't spell it out.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — UNIVERSAL SAFETY MASTER (ABSOLUTE HIGHEST PRIORITY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If STEP 2 identifies ANY Universal_Safety_Master chunk whose ESCALATE_IF
field contains an active trigger condition matching the problem:
  → status = "escalate"
  → COPY-PASTE MANDATE: Copy the chunk's STEPS field CHARACTER-FOR-CHARACTER
    into safety_warnings_en[0]. Zero paraphrasing allowed.
  → technical_analysis = the chunk's PROBLEM field verbatim.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — REASSURANCE (Normal Operation / Readings)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Apply ONLY when:
  ✅ A chunk EXPLICITLY states this specific symptom or reading is normal.
  ✅ NO Universal Safety chunk's ESCALATE_IF has been triggered.

  If these are met:
    → status = "success"
    → Provide 1-2 steps explaining why it is normal based on the chunk.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 — FIRE EXTINGUISHER QUERY (Informational — No Active Fire)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the matched chunk's ESCALATE_IF says "DO NOT escalate":
  → status = "success"
  → COPY-PASTE MANDATE: Copy that chunk's STEPS verbatim.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 4 — ROUTINE MAINTENANCE REDIRECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the query is a maintenance schedule with no active fault:
  → status = "escalate"
  → You MUST populate "solution.steps" with the maintenance procedure from the chunk. Do NOT leave steps empty.

RULE 5 — HALLUCINATION TRAP (Strict Failsafe)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If STEP 2 result is "no chunk covers this symptom":
  → status = "escalate", steps = []
  → DO NOT generate steps, estimates, or procedures from training knowledge.

If the chunks cover the SAME SYSTEM or COMPONENT CLASS as the user's problem, PROCEED TO DIAGNOSE. Match at the COMPONENT/SYSTEM level, taking the chunk with the highest symptom overlap as your single source of truth.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 6 — COPY-PASTE MANDATE (Universal — No Exceptions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When any rule instructs you to copy a STEPS field:
  1. Locate the exact STEPS string in the Manual Extracts above.
  2. Copy it CHARACTER-FOR-CHARACTER into the target output field.
{_SPEC_FIDELITY_RULE}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 8 — MISSING / REMOVED PART (Visual Context Override)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the visual context explicitly states that a part is MISSING, REMOVED, or ABSENT:
  → Treat this as a confirmed fault — proceed to DIAGNOSE.
  → Warn about the consequences verbatim in technical_analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 9 — FALSE POSITIVE GUARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the problem mentions "spark plug" cleaning or inspection:
  → Do NOT escalate for electrical hazard. Proceed to DIAGNOSE.
"""

    output_format = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — JSON ONLY, NO MARKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL SCHEMA RULES:
  1. "internal_reasoning" MUST be the first key and MUST contain all 4 CoT steps.
  2. "solution" MUST always be present.
  3. status="escalate" → "solution.steps" MUST be [] (empty array, not null) EXCEPT for Rule 4 (Maintenance) where you must provide steps.
  4. status="success"  → "solution.steps" MUST contain at least 1 step object.
  5. "required_part" MUST be a snake_case ID from: {parts_list}
  6. "area_hint" MUST be one of: {allowed_areas}

🌾 FARMER INSTRUCTION STANDARD:

  Your job is to TEACH the farmer, not merely list repair steps.
  The farmer should feel more confident after reading each step, not overwhelmed.

   ═══ USER MENTAL MODEL ═══
  Assume the operator:
    • Has never repaired this machine before
    • Cannot identify components by name
    • Does not know mechanical terminology
    • May be anxious about making mistakes

  Write instructions that increase confidence while remaining
  technically accurate.

  "action":      A short imperative (e.g. "Switch OFF the pump", "Inspect the capacitor").
                 This is the repair action — what the farmer must do.

  "description": The teaching. This should reduce uncertainty for a first-time
                 operator. When relevant, include:
                   • How to locate the component using visible landmarks
                     (e.g. "where the power cable enters", "beside the fan cover",
                     "underneath the fuel tank") — prefer landmarks over left/right
                   • How to visually identify the component
                   • What healthy looks like and what failure looks like
                   • How to perform the action safely
                   • How the operator can confirm the expected outcome
                     using sight, sound, or touch (only when safe).
                     Prefer observations over measurements.
                     Example: "The humming noise should stop."
                     Example: "Water should begin flowing steadily."
                 Include only information that helps perform the current step.
                 Do not force every step to contain every category.
    "step_type": One of: "safety" | "inspection" | "repair" | "verification".
                   Classify each step:
                   - "safety": ONLY for mid-repair safety warnings when a NEW hazard emerges that wasn't covered by the initial safety gate. Do NOT generate safety steps for power-off/shutdown at the start — the app's SafetyGate already handles this before AR begins. Only use "safety" if a step creates a NEW risk (e.g., "Warning: the part you just removed exposes live terminals").
                   - "inspection": Camera-verifiable visual check ONLY. The farmer points their phone at a specific visible part. FORBIDDEN: measurement tools (tape, gauge), exact numbers (40-45mm), tool-based adjustments. REQUIRED: specific snake_case required_part, visible landmark description, what healthy vs faulty looks like. If the manual specifies a measurement, convert it to a visual observation using body-part comparisons (finger-width, thumb-length).
                   - "repair": Replace, tighten, remove, install, adjust a part
                   - "verification": Restore power, start machine, confirm operation

  ═══ COMPONENT INTRODUCTION ═══
  Introduce a component only if the operator must interact with it.
  If a component is only mentioned for context, do not interrupt the
  procedure with a long explanation.

  The first time the operator must interact with a component:
    1. Describe only the characteristics that are actually visible
       without disassembly. Do not describe hidden internal components
       as though they can be seen.
    2. Then give its technical name.

  When describing locations, use this priority:
    1. Permanent external landmarks (power cable entry, fan cover,
       fuel tank, radiator, belt pulley)
    2. Nearby visible components
    3. Clock-face directions (e.g. "at the 2 o'clock position
       relative to the fan cover")
  Avoid left/right unless orientation is already established.

  Example:
    "Look for the small rectangular plastic box attached to the outside
     of the motor where the electrical cable enters. This is called the
     capacitor box."

  After a component has been introduced, you may simply call it by name.
  Do not repeat the full identification unless another similar component
  could reasonably be confused with it.

  ═══ STEP DETAIL BY TYPE ═══
  Safety steps (e.g. "Switch OFF the main breaker"):
    Be concise and unambiguous. Do not add unnecessary explanation.

  Inspection steps (CAMERA-VERIFIABLE ONLY):
    Design EVERY inspection step so the farmer can point their phone
    camera at a specific part and the AI vision system can verify it.
    - Include visual identification landmarks, what healthy looks like,
      and what damage/fault looks like.
    - NEVER ask the farmer to measure anything with a tool (tape, gauge,
      multimeter). If the manual specifies a measurement, convert it to
      a visual observation: "about two finger-widths" not "40-45mm".
    - Every inspection step MUST have a specific required_part the
      camera can point at. Use snake_case from the allowed parts list.
    - The camera IS the verification tool — use it for everything visual.

  Repair/replacement steps:
    Include physical actions, tool placement, expected outcome.
    After describing what to do, add a CAMERA VERIFICATION sub-step:
    "After completing, point camera at [part] to verify it's correctly
    installed/tightened/replaced."

  Verification steps (e.g. "Start the pump and check for leaks"):
    Include what to observe and how to confirm success.
    When possible, make this camera-verifiable: "Point camera at [part]
    and confirm [expected visual outcome]."

  ═══ SAFETY PREREQUISITES ═══
  Every repair step must include every safety prerequisite immediately
  required for that step. Do not assume the operator remembers a warning
  from several steps earlier.

  Example: If a step involves touching electrical components, repeat
  "Ensure the main power is OFF" even if mentioned in Step 1.

  ═══ NEVER TELL THE FARMER TO READ THE MANUAL ═══
  The farmer may be semi-literate or unable to read English/Hindi text.
  Your job is to BE the manual — translate everything into simple,
  actionable instructions the farmer can follow immediately.

  BANNED PHRASES — never include these in any step's action or description:
  • "Refer to the manual" / "Read the user guide"
  • "Check the manufacturer's instructions" / "See service manual"
  • "Consult a certified mechanic" / "Bring a technician"
  • "Contact Mahindra service centre" / "Visit the dealership"
  • Any measurement with units: "40-45mm", "2cm", "0.5 inches", "measure with tape"
  • Any tool-based measurement: "use a measuring tape", "check with multimeter", "use a feeler gauge"

  If the repair truly requires a professional (e.g., internal engine work,
  electrical hazards beyond simple checks), escalate the ENTIRE diagnosis
  with status="escalate" and steps=[]. Do NOT generate helpful steps and
  then end with "if problem persists, consult a mechanic."

  ═══ LANGUAGE RULES ═══
  Words like "inspect", "check", and "verify" are acceptable only when
  immediately followed by specific guidance on what to look for.

  GOOD: "Inspect the capacitor for bulging, oil leakage, or burn marks."
  BAD:  "Inspect the capacitor."

  GOOD: "Check whether the terminal screws are loose or burnt."
  BAD:  "Check the wiring."

  Use cautious language ("typically", "usually", "commonly") only when
  relying on general engineering knowledge. When information comes directly
  from the manual, state it confidently. Do not overuse hedging words.

  ═══ QUALITY STANDARD ═══
  Write only enough information to complete the current step safely.
  Do not explain future steps early.
  Do not explain previous steps again.
  Each step should solve only one immediate problem.

  ═══ LAST STEP MUST BE ACTIONABLE ═══
  The final step MUST be a verification check the farmer can actually do:
  "Start the engine and check that the clutch engages smoothly"
  NOT: "If the problem persists, consult a mechanic"
  NOT: "Refer to the manual for further guidance"
  
  The escalation condition already exists in the escalate_if field.
  Do not repeat it as a step. Every step must teach the farmer something
  they can do right now.

  ═══ WORKED EXAMPLE ═══
  Manual chunk says only: "Clutch — Clutch cable damage."

  BAD (manual-literal, what you must NOT do):
    "action": "Check the clutch cable for damage"
    "description": ""

  GOOD (same diagnosis, taught clearly — this is what's required):
    "action": "Check the clutch cable for fraying or stretching"
    "description": "The clutch cable runs from the clutch pedal to the
    clutch fork near the gearbox housing, usually visible as a thick wire
    cable in a metal sheath. Follow it from the pedal toward the engine.
    A damaged cable looks frayed, kinked, or stretched, and the clutch
    pedal will feel loose or won't fully disengage the clutch. The fix —
    cable replacement — comes from the manual; the description of what
    the cable looks like and where it runs is general mechanical knowledge
    used here only to help you find it." 

    ═══ CONSISTENCY RULE ═══
    Apply the teaching standard above to EVERY step — including the last one.
    The final step must be as detailed as the first. "Check for other issues"
    without landmarks or visual guidance is NOT acceptable. If the manual
    mentions "welding points breaking off, fork pin bent, spring failure",
    describe WHERE to look for each and WHAT each looks like when damaged.

  The goal is safe completion, not comprehensive education.
"""

    prompt = f"""
You are an experienced agricultural mechanic explaining repairs to a
first-time {machine_label} operator. Your job is to safely guide the
operator through the repair using the manufacturer's manual.
Reduce uncertainty. Never assume the operator already knows machinery
or component names.
{visual_note}
""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBLEM DESCRIPTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Machine: {machine_label}
Symptoms: {problem_text}

{rag_block}

{grounding_rules}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAFETY WARNINGS & LANGUAGE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Include "⚠️ ESCALATE_IF:" entries verbatim in safety_warnings_en.
- No jargon. Translate technical concepts to plain English.

{output_format}
""".strip()
    return prompt

# ── Unified escalation response factory ───────────────────────────────────────
# Single source of truth for every escalation response dict. All pre-LLM guards,
# fallback paths, and no-context handlers produce responses through this function.
#
# Reserved keys (status, solution, problem_description, technical_analysis,
# safety_warnings) are protected — metadata cannot overwrite them.

import copy as _copy

_BASE_ESCALATION: dict = {
    "status": "escalate",
    "solution": {
        "steps": [],
        "tools_needed": [],
    },
}

_RESERVED_KEYS: frozenset = frozenset({
    "status", "solution", "problem_description", "technical_analysis",
    "safety_warnings_en", "safety_warnings_hi", "rag_source", "machine_label",
})


def _build_escalation_response(
    machine_type: str,
    machine_label: str,
    problem_text: str,
    technical_analysis: str,
    safety_warnings_en: List[str],
    safety_warnings_hi: List[str],
    *,
    rag_source: RagSource = RagSource.ESCALATION,
    problem_identified: str = "",
    metadata: Optional[dict] = None,
) -> dict:
    """
    Produce a canonical escalation response dict.

    Args:
        machine_type:        Canonical machine ID (e.g. "electric_motor").
        machine_label:       Human-readable label (e.g. "Electric Motor").
        problem_text:        Original farmer query text.
        technical_analysis:  One-paragraph explanation of why escalated.
        safety_warnings_en:  List of English safety messages.
        safety_warnings_hi:  List of Hindi safety messages.
        rag_source:          Canonical RagSource enum value.
        problem_identified:  Short fault label (defaults to technical_analysis).
        metadata:            Optional dict of extra fields merged into response.
                             Reserved keys are silently dropped.

    Returns:
        EscalationResponse-compatible dict.
    """
    result = _copy.deepcopy(_BASE_ESCALATION)
    result.update({
        "problem_description": problem_text,
        "technical_analysis": technical_analysis,
        "solution": {
            **result["solution"],
            "machine_type": machine_type,
            "problem_identified": problem_identified or technical_analysis,
            "safety_warnings_en": list(safety_warnings_en),
            "safety_warnings_hi": list(safety_warnings_hi),
        },
        "rag_source": rag_source.value,
        "machine_label": machine_label,
    })

    # Merge caller-supplied metadata, protecting reserved keys
    if metadata:
        for key, value in metadata.items():
            if key not in _RESERVED_KEYS:
                result[key] = value

    return result

def _should_hard_escalate(problem_text: str, rag_context: str) -> Optional[str]:
    if not problem_text or not problem_text.strip(): return "__empty_input__"
    return None

_MAINTENANCE_PATTERNS = [
    re.compile(r'\boil\s+change\b', re.IGNORECASE),
    re.compile(r'\boil\s+(?:change\s+)?interval\b', re.IGNORECASE),
    re.compile(r'\bfilter\s+(?:change|replace|interval)\b', re.IGNORECASE),
    re.compile(r'\bgrease\s+(?:schedule|interval|nipple|point)\b', re.IGNORECASE),
    re.compile(r'\bservice\s+interval\b', re.IGNORECASE),
]

def _guard5_empty_rag_maintenance(problem_text: str, rag_context: str, machine_type: str, machine_label: str) -> Optional[dict]:
    if rag_context and rag_context.strip(): return None
    if not any(p.search(problem_text or "") for p in _MAINTENANCE_PATTERNS): return None
    logger.warning(f"🛡️ GUARD 5 PRE-LLM: Empty RAG + maintenance query [{machine_type}] — escalating")
    return _build_escalation_response(
        machine_type=machine_type,
        machine_label=machine_label,
        problem_text=problem_text,
        technical_analysis=f"This maintenance procedure is not covered in the {machine_label} service manual.",
        safety_warnings_en=[f"This specific procedure is not in our repair manual for the {machine_label}. Please consult the manufacturer's user guide or a certified mechanic."],
        safety_warnings_hi=[f"यह रखरखाव की जानकारी हमारे {machine_label} मैनुअल में नहीं है। कृपया निर्माता की गाइड देखें या प्रमाणित मैकेनिक से पूछें।"],
        rag_source=RagSource.EMPTY_RAG_MAINT,
        problem_identified="Routine maintenance schedule query — not in knowledge base.",
    )


# ─── PRE-LLM GUARDS ─────────────────────────────────────────────────────────

_UNKNOWN_ATTACHMENT_PATTERNS = [
    re.compile(r'\bunknown\s+attachment\b', re.IGNORECASE),
    re.compile(r'\bcannot\s+identify.*\battachment\b', re.IGNORECASE),
    re.compile(r'\blaser\s+land\s+level', re.IGNORECASE),
    re.compile(r'\bunrecognized\s+(?:attachment|implement|tool)\b', re.IGNORECASE),
]

def _guard_unknown_attachment(problem_text: str, machine_type: str, machine_label: str) -> Optional[dict]:
    if not any(p.search(problem_text or "") for p in _UNKNOWN_ATTACHMENT_PATTERNS): return None
    logger.warning(f"🛡️ GUARD UNKNOWN_ATTACHMENT: Unidentified implement [{machine_type}]")
    return _build_escalation_response(
        machine_type=machine_type,
        machine_label=machine_label,
        problem_text=problem_text,
        technical_analysis="The attachment or implement cannot be identified. This falls outside the knowledge base coverage for this machine.",
        safety_warnings_en=["This attachment is not covered in our repair manual. Consult the manufacturer."],
        safety_warnings_hi=["यह उपकरण हमारे मैनुअल में नहीं है।"],
        rag_source=RagSource.UNKNOWN_ATTACHMENT,
        problem_identified="Unknown implement — insufficient knowledge base coverage.",
    )

# ─── NON-DESTRUCTIVE POST-LLM GUARDS ────────────────────────────────────────

_FIRE_EXT_PROBLEM_KEYWORDS = ["fire extinguisher", "agni shamak", "agnisamak", "fire cylinder"]
_ACTIVE_FIRE_KEYWORDS      = ["aag lagi", "jal raha", "flames", "burning", "smoke coming", "dhuan aa raha", "active fire", "sparks"]

def _guard1_fire_extinguisher(diagnosis, problem_text, rag_context, machine_type, machine_label):
    prob_lower = (problem_text or "").lower()
    if not any(kw in prob_lower for kw in _FIRE_EXT_PROBLEM_KEYWORDS): return None
    if any(kw in prob_lower.replace("fire extinguisher", "").replace("fire cylinder", "") for kw in _ACTIVE_FIRE_KEYWORDS): return None
    logger.info("🛡️ GUARD 1 NON-DESTRUCTIVE OVERRIDE: Fire extinguisher informational")
    diagnosis["status"] = "success"
    diagnosis["_informational_override"] = True
    diagnosis["technical_analysis"] = "User asks how to use a mounted fire extinguisher. No active fire. " + diagnosis.get("technical_analysis", "")
    return diagnosis

_NORMAL_GAUGE_PATTERNS = [
    re.compile(r'gauge\s+(?:is\s+)?pointing\s+(?:to|at|exactly|in\s+the)', re.IGNORECASE),
    re.compile(r'dipstick\s+(?:showing|between|level\s+between)', re.IGNORECASE),
    re.compile(r'(?:in\s+the\s+)?(?:green|normal)\s+zone', re.IGNORECASE),
]
_UNSAFE_KW_FOR_REASSURANCE = ["fire", "aag", "smoke", "dhuan", "leak", "tapak", "overheat", "garam", "sparks", "injury", "hurt", "chot", "bleeding"]

def _guard3_reassurance_visual_gauge(diagnosis, problem_text, machine_type, machine_label):
    if not _has_visual_context_text(problem_text): return None
    if not any(p.search(problem_text or "") for p in _NORMAL_GAUGE_PATTERNS): return None
    if any(kw in (problem_text or "").lower() for kw in _UNSAFE_KW_FOR_REASSURANCE): return None
    logger.info("🛡️ GUARD 3 NON-DESTRUCTIVE OVERRIDE: Normal visual gauge")
    diagnosis["status"] = "success"
    diagnosis["_reassurance_override"] = True
    return diagnosis

def _deduplicate_steps(steps: List[dict]) -> List[dict]:
    if not steps: return []
    action_verbs_re = re.compile(
        r'\b(check|inspect|look|clean|tighten|remove|replace|install|adjust|push|pull|turn|scan|tap|press|fill|drain|test|measure|hold)\b',
        re.IGNORECASE
    )
    seen, unique = set(), []
    for step in steps:
        req_part = step.get("required_part")
        if not req_part:
            unique.append(step)
            continue
        text_en = step.get("text_en") or ""
        m = action_verbs_re.search(text_en)
        action = m.group(1).lower() if m else ""
        vis_cue = step.get("visual_cue") or ""
        area_hint = step.get("area_hint") or ""
        sig = f"{req_part}:{vis_cue}:{area_hint}:{action}"
        if sig not in seen:
            seen.add(sig)
            unique.append(step)
    return unique

def _derive_unsafe_scene(problem_text: str, technical_analysis: str, has_any_visual: bool) -> tuple:
    pt = problem_text if problem_text else ""
    ta = technical_analysis if technical_analysis else ""
    combined = (pt + " " + ta).lower()
    clean_combined = combined.replace("fire extinguisher", "").replace("fire cylinder", "")
    injury_kw        = ["hurt", "chot", "cut", "kat gaya", "injured", "bleeding", "electric shock", "current laga"]
    active_hazard_kw = ["fire", "aag", "sparks flying", "smoke billowing", "oil spray", "fuel spray", "aag lagi", "jal raha", "sparks"]
    moving_near_kw   = ["belt moving", "ghoom raha", "spinning near", "rotating near", "hand near", "caught in"]
    if any(kw in clean_combined for kw in injury_kw):
        return True, "Injury/Shock risk detected. Stop immediately and ensure area is safe."
    if any(kw in clean_combined for kw in active_hazard_kw):
        return True, "Active hazard suspected (fire/spark/spray). Do not proceed until resolved."
    if any(kw in clean_combined for kw in moving_near_kw):
        return True, "Moving parts near inspection area. Ensure all rotation has stopped before inspection."
    if "dangerous workaround" in clean_combined:
        return True, "Dangerous mechanical workaround detected. Do not operate machine."
    return False, ""

def _fallback_escalation_response(machine_type, machine_label, problem_text, error_reason):
    return _build_escalation_response(
        machine_type=machine_type,
        machine_label=machine_label,
        problem_text=problem_text,
        technical_analysis=f"Unable to diagnose: {error_reason}",
        safety_warnings_en=["Automatic diagnosis unavailable.", "Consult a certified technician."],
        safety_warnings_hi=["स्वचालित निदान उपलब्ध नहीं है।", "प्रमाणित तकनीशियन से संपर्क करें।"],
        rag_source=RagSource.ERROR,
        problem_identified=problem_text,
    )


# ── FIX 6: Verification Gate ──────────────────────────────────────────────────

def _extract_risk_level_from_rag(rag_context: str) -> str:
    """Extract the highest risk level seen across chunks in the RAG context."""
    if not rag_context:
        return "LOW"
    # Scan all Risk: fields — take the highest
    risks = re.findall(r'\|\s*Risk:\s*(\w+)', rag_context)
    order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
    if not risks:
        return "LOW"
    return max(risks, key=lambda r: order.get(r.upper(), 0)).upper()


async def _run_verification_gate(
    machine_type: str,
    problem_text: str,
    rag_context: str,
    risk_level: str,
    visual_frames: Optional[List[bytes]],
    language: str,
) -> Optional[dict]:
    """
    FIX 6: Verification gate — verify machine type and component existence
    before committing to the final repair steps.

    Runs only when:
      • risk_level is HIGH or CRITICAL
      • at least one visual frame is available
      • the diagnosis is not already escalated

    Returns None if verification passes or is skipped.
    Returns a "need_verification" dict if verification fails (< threshold confidence).
    """
    if risk_level not in _VERIFICATION_RISK_LEVELS:
        logger.debug("Verification gate SKIP: risk_level=%s", risk_level)
        return None

    if not visual_frames:
        logger.debug("Verification gate SKIP: no visual frames available")
        return None

    # Use the first frame for verification
    frame_bytes = visual_frames[0]

    # Extract the first step's required_part from the RAG context as the target
    # (best proxy for "what we're about to repair")
    required_part_match = re.search(r'"required_part"\s*:\s*"([^"]+)"', rag_context)
    required_part = required_part_match.group(1) if required_part_match else "main_component"

    # Extract area hint
    area_hint_match = re.search(r'"area_hint"\s*:\s*"([^"]+)"', rag_context)
    area_hint = area_hint_match.group(1) if area_hint_match else "engine_compartment"

    logger.info(
        "FIX 6: Running verification gate for %s | part=%s | risk=%s",
        machine_type, required_part, risk_level,
    )

    try:
        verification = await verify_step_with_gemini(
            image_bytes     = frame_bytes,
            step_text       = f"Verify this is the correct {machine_type} showing the {required_part}",
            required_part   = required_part,
            area_hint       = area_hint,
            machine_type    = machine_type,
            problem_context = problem_text,
            attempt_count   = 1,
            language        = language,
        )

        v_status = verification.get("status", "unclear")
        v_conf   = float(verification.get("confidence", 0.0))

        if v_status == "unsafe":
            logger.warning("FIX 6: Verification found UNSAFE condition — blocking repair")
            return {
                "status": "need_verification",
                "verification_failed": True,
                "verification_reason": "unsafe_condition_detected",
                "message_en": "An unsafe condition was detected in the camera image. Do not proceed with repairs. Ensure the area is safe first.",
                "message_hi": "कैमरे में असुरक्षित स्थिति दिखी। मरम्मत शुरू न करें। पहले क्षेत्र को सुरक्षित करें।",
                "ai_observation": verification.get("ai_observation", ""),
                "steps": [],
                "machine_type": machine_type,
            }

        if v_conf < _VERIFICATION_CONFIDENCE_THRESHOLD:
            logger.warning(
                "FIX 6: Verification confidence too low (%.2f < %.2f) — need verification",
                v_conf, _VERIFICATION_CONFIDENCE_THRESHOLD,
            )
            return {
                "status": "need_verification",
                "verification_failed": True,
                "verification_reason": "low_confidence",
                "confidence": round(v_conf, 3),
                "message_en": (
                    "I cannot confidently verify the machine type and component from the camera image. "
                    "Please ensure you are pointing the camera at the correct machine and component "
                    "before proceeding with repairs. Need verification before continuing."
                ),
                "message_hi": (
                    "मैं कैमरे की छवि से मशीन और पुर्जे की पुष्टि नहीं कर सका। "
                    "कृपया सुनिश्चित करें कि कैमरा सही मशीन और पुर्जे पर है। "
                    "आगे बढ़ने से पहले सत्यापन आवश्यक है।"
                ),
                "ai_observation": verification.get("ai_observation", ""),
                "steps": [],
                "machine_type": machine_type,
            }

        logger.info(
            "FIX 6: Verification PASSED: part=%s conf=%.2f status=%s",
            required_part, v_conf, v_status,
        )
        return None  # Verification passed — proceed normally

    except Exception as exc:
        logger.warning("FIX 6: Verification gate error (non-fatal): %s", exc)
        return None  # On error, allow through (don't block farmer)


# MIGRATED: Groq format normalization
def _normalize_status(diagnosis: dict) -> dict:
    """
    Normalize Groq/Gemini status to stable top-level lowercase.  # MIGRATED: Groq format normalization

    Supports:
      top-level status
      nested solution.status
      uppercase variants

    Maps:
      diagnose -> success
    """  # MIGRATED: Groq format normalization
    if not isinstance(diagnosis, dict):  # MIGRATED: Groq format normalization
        return diagnosis  # MIGRATED: Groq format normalization

    status = diagnosis.get("status")  # MIGRATED: Groq format normalization

    if not status:  # MIGRATED: Groq format normalization
        solution = diagnosis.get("solution", {})  # MIGRATED: Groq format normalization
        if isinstance(solution, dict):  # MIGRATED: Groq format normalization
            status = solution.get("status")  # MIGRATED: Groq format normalization

    status = str(status or "").strip().lower()  # MIGRATED: Groq format normalization

    if status == "diagnose":  # MIGRATED: Groq format normalization
        status = "success"  # MIGRATED: Groq format normalization

    if status:  # MIGRATED: Groq format normalization
        diagnosis["status"] = status  # MIGRATED: Groq format normalization

    return diagnosis  # MIGRATED: Groq format normalization

def _contains_actionable_guidance(diagnosis: dict) -> bool:
    """True if the _post_process_diagnosis contains concrete repair guidance (not just text)."""
    solution = diagnosis.get("solution", {})
    return bool(
        solution.get("steps")
        or solution.get("tools_needed")
        or solution.get("parts")
        or solution.get("verification_steps")
    )

async def _post_process_diagnosis(
    diagnosis: dict,
    machine_type: str,
    machine_label: str,
    problem_text: str,
    rag_context: str,
    visual_frames: Optional[List[bytes]],
    context_quality: str,
) -> dict:

    # MIGRATED: Groq format normalization — normalize before any routing logic
    diagnosis = _normalize_status(diagnosis)  # MIGRATED: Groq format normalization

    # 1. NON-DESTRUCTIVE ROUTING OVERRIDES (Safety Injectors)
    g1 = _guard1_fire_extinguisher(diagnosis, problem_text, rag_context, machine_type, machine_label)
    g3 = _guard3_reassurance_visual_gauge(diagnosis, problem_text, machine_type, machine_label)

    if g1: diagnosis = g1
    if g3: diagnosis = g3

    # 2. Schema integrity
    if "solution" not in diagnosis or not isinstance(diagnosis.get("solution"), dict):
        diagnosis["solution"] = {}

    if diagnosis.get("status") == "escalate" and not diagnosis.get("rag_source"):
        diagnosis["rag_source"] = "escalation"

    # 3. Normalize step field names + deduplication
    # The LLM (especially fallback models) may return steps with non-standard
    # field names like "description"/"warning" instead of "text_en"/"safety_warning".
    # Normalize them so Flutter always sees a consistent schema.
    steps = diagnosis["solution"].get("steps", [])
    _normalized_steps = []
    for _s in steps:
        if not isinstance(_s, dict):
            continue
        _ns = dict(_s)

        # Build text_en from action + description without destroying original fields.
        # action = short imperative (good for title/speech), description = explanation.
        # Concatenate both so the farmer sees the full instruction.
        _action = _ns.get("action", "").strip()
        _desc = _ns.get("description", "").strip()

        if "text_en" not in _ns:
            # Format snake_case action for display
            _display = _action.replace("_", " ") if "_" in _action and " " not in _action else _action
            if _display and _desc:
                _ns["text_en"] = f"{_display}\n{_desc}"
            elif _display:
                _ns["text_en"] = _display
            elif _desc:
                _ns["text_en"] = _desc
            elif _action:
                _ns["text_en"] = _action
            elif _desc:
                _ns["text_en"] = _desc

        # Map other common LLM field name variations to canonical names
        if "description_hi" in _ns and "text_hi" not in _ns:
            _ns["text_hi"] = _ns.get("description_hi", "")
        if "warning" in _ns and "safety_warning" not in _ns:
            _ns["safety_warning"] = _ns.get("warning", "")
        if "part" in _ns and "required_part" not in _ns:
            _ns["required_part"] = _ns.get("part", "")
        # Infer step_type if LLM didn't provide it — diagnosis owns this classification
        if "step_type" not in _ns:
            if not _ns.get("required_part") or _ns.get("required_part") in ("none", "unknown", ""):
                _ns["step_type"] = "safety"
            else:
                _ns["step_type"] = "inspection"
        if "area_hint" not in _ns or not _ns.get("area_hint"):
            _ns["area_hint"] = "engine_compartment"

        _normalized_steps.append(_ns)
    # DEBUG: Verify filter will run
    logger.info(f"🔍 DEBUG: About to filter {len(_normalized_steps)} steps")
    for i, s in enumerate(_normalized_steps):
        logger.info(f"🔍   Step {i}: type={s.get('step_type')} part={s.get('required_part')}")

    # ── Filter safety steps handled by Flutter SafetyGate ─────────────────
    # This runs AFTER the normalization loop, filtering the complete list.
    _SHUTDOWN_PARTS = {"ignition_key", "power_switch", "mcb", "circuit_breaker", 
                       "main_isolator", "battery_terminal", "fuel_valve", "machine_part"}
    
    _steps_before = len(_normalized_steps)
    _normalized_steps = [
        s for s in _normalized_steps
        if not (
            s.get("step_type") == "safety"
            and s.get("required_part", "") in _SHUTDOWN_PARTS
        )
    ]
    if len(_normalized_steps) < _steps_before:
        logger.info(f"🛡️  Removed %d safety step(s) — Flutter SafetyGate handles pre-AR safety",
            _steps_before - len(_normalized_steps))
    
    steps = _normalized_steps
    diagnosis["solution"]["steps"] = _deduplicate_steps(steps)
    if diagnosis.get("status") != "escalate":
        try:
            raw_steps  = diagnosis["solution"].get("steps", [])
            risk_level = diagnosis.get("risk_level") or "MEDIUM"
            tools_list = diagnosis["solution"].get("tools_needed", [])
            validation = validate_procedure(
                steps        = raw_steps,
                machine_type = machine_type,
                risk_level   = risk_level,
                tools_list   = tools_list,
                safe_inject  = False,
            )

            # Always apply safe_steps — even in Path A this is the correct list
            # (auto-injected shutdown step is now Flutter-compatible per FIX A2).
            diagnosis["solution"]["steps"] = validation.safe_steps

            # Log all issues unconditionally — not gated by passed flag.
            logger.info("FIX 5: %s", validation.summary())
            for issue in validation.issues:
                level = logging.WARNING if issue.severity in ("CRITICAL", "HIGH") else logging.DEBUG
                logger.log(level, "  [%s] %s: %s", issue.severity, issue.rule, issue.detail[:80])

            # Path B: unresolvable CRITICALs — block the response.
            # This is the enforcement gate that was previously missing.
            unresolvable_criticals = [
                i for i in validation.issues
                if i.severity == "CRITICAL" and not i.auto_fix
            ]
            if unresolvable_criticals:
                logger.error(
                    "FIX 5: BLOCKING dangerous steps for %s — %d unresolvable CRITICAL issue(s): %s",
                    machine_type,
                    len(unresolvable_criticals),
                    [i.rule for i in unresolvable_criticals],
                )
                diagnosis["status"]  = "escalate"
                diagnosis["solution"]["steps"] = []
                critical_detail_msgs = [i.detail for i in unresolvable_criticals]
                existing_en = diagnosis["solution"].get("safety_warnings_en", [])
                diagnosis["solution"]["safety_warnings_en"] = critical_detail_msgs + existing_en
                # Preserve machine_label so Flutter can render the escalation card correctly.
                diagnosis.setdefault("machine_label", machine_label)

            # Path B also: surface HIGH non-fixable issues as warnings (no blocking,
            # but they must be visible — not just in logs).
            elif not validation.passed:
                # HIGH issues only (CRITICALs handled above, WARNINGs don't block)
                high_msgs = [
                    i.detail for i in validation.issues
                    if i.severity == "HIGH" and not i.auto_fix
                ]
                if high_msgs:
                    existing_en = diagnosis["solution"].get("safety_warnings_en", [])
                    diagnosis["solution"]["safety_warnings_en"] = high_msgs + existing_en

        except Exception as exc:
            logger.warning("FIX 5: Procedure validation failed (non-fatal): %s", exc)
    # ────────────────────────────────────────────────────────────────────────

    # ── Step ID assignment — SINGLE SOURCE OF TRUTH ─────────────────────────
    # This runs once, here, after every mutation that can add/remove/reorder
    # steps (dedup + procedure_validator's safe_steps injection) has already
    # happened. Every downstream consumer — Flutter's SolutionData, the
    # agent's RepairPlan/RepairPlanStep — must trust diagnosis["solution"]
    # ["steps"][i]["step_id"] as final. No fallback generation, no silent
    # patching anywhere further down the pipeline (session creation, the
    # agent's step-advancement logic, etc.) — if it's missing there, that's
    # a bug in THIS function, not something to paper over downstream.
    _final_steps = diagnosis["solution"].get("steps", [])
    for _idx, _step in enumerate(_final_steps):
        if isinstance(_step, dict):
            _step["step_id"] = f"s{_idx + 1}"

    # Structural validation happens immediately after generation — not
    # thousands of lines later — so an invalid plan can never exist beyond
    # this point. If this raises, it means THIS function has a bug (the
    # loop above is the only thing that sets step_id, so blank/duplicate
    # ids here would mean the assignment itself is broken). This is a
    # backend defect, not a mechanical fault — it must propagate as
    # InvalidRepairPlan, NOT get wrapped in an escalation card telling the
    # farmer to "consult a certified mechanic." Nothing is wrong with their
    # equipment; something is wrong with this service. The caller of
    # generate_diagnosis_with_gemini() (main.py) must catch InvalidRepairPlan
    # separately from normal escalation handling and return a generic
    # service-error response.
    validate_repair_plan_steps(_final_steps, context=f"machine={machine_type}")
    # ────────────────────────────────────────────────────────────────────────

    # ── C7: Weak-context grounding enforcement ─────────────────────────────
    # The LLM is instructed to escalate when context is weak, but that's a soft
    # prompt instruction. This is the code-level backstop: if retrieval quality
    # is weak AND the model returned actionable guidance anyway, force escalation
    # through the canonical factory. No in-place mutation — single return path.
    if context_quality == "weak" and _contains_actionable_guidance(diagnosis):
        logger.warning(
            "C7: Weak context with actionable guidance (status='%s', steps=%d) — "
            "forcing escalation via factory",
            diagnosis.get("status"),
            len(diagnosis.get("solution", {}).get("steps", [])),
        )
        return _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis=(
                "The repair manual does not contain enough information to "
                "safely guide this repair. Consult a certified mechanic."
            ),
            safety_warnings_en=[
                "The repair manual does not contain enough information to "
                "safely guide this repair. Consult a certified mechanic."
            ],
            safety_warnings_hi=[
                "मरम्मत मैनुअल में इस समस्या के लिए पर्याप्त जानकारी नहीं है। "
                "कृपया प्रमाणित मैकेनिक से संपर्क करें।"
            ],
            rag_source=RagSource.NO_CONTEXT,
            problem_identified="Insufficient knowledge base coverage for safe repair guidance.",
        )

    # 4. Unsafe-scene flag
    if diagnosis.get("_reassurance_override") or diagnosis.get("_informational_override") or diagnosis.get("_electric_hazard_override"):
        pass
    else:
        technical_analysis = diagnosis.get("technical_analysis") or ""
        unsafe, unsafe_msg = _derive_unsafe_scene(
            problem_text or "", technical_analysis,
            bool(visual_frames) or _has_visual_context_text(problem_text or "")
        )
        _no_real_hazard_sources = {"no_context", "escalation", "guard_unknown_attachment", None}
        _hazard_keywords = ["fire", "aag", "shock", "bijli", "current laga", "blood", "bleeding", "hurt", "chot", "sparks", "jal raha", "aag lagi"]
        prob_lower = (problem_text or "").lower()
        if (unsafe and not diagnosis.get("_electric_hazard_override") and diagnosis.get("rag_source") in _no_real_hazard_sources and not any(kw in prob_lower for kw in _hazard_keywords)):
            unsafe, unsafe_msg = False, ""
        if "unsafe_scene_suspected" not in diagnosis:
            diagnosis["unsafe_scene_suspected"] = unsafe
            if unsafe:
                diagnosis["unsafe_scene_message"] = unsafe_msg

    # 5. Metadata
    if not diagnosis.get("rag_source"):
        diagnosis["rag_source"] = "RAG+Gemini" if rag_context else "Gemini-only"
    diagnosis["machine_label"] = machine_label

    diagnosis = _enrich_industrial_response(diagnosis, rag_context, problem_text)

    return diagnosis


# ─────────────────────────────────────────────────────────────────────────────
# INDUSTRIAL RESPONSE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_industrial_response(
    diagnosis: dict,
    rag_context: str,
    problem_text: str,
) -> dict:
    sol = diagnosis.get("solution", {})

    if not diagnosis.get("confidence_label"):
        if rag_context:
            m = re.search(r"\[Retrieval Confidence:\s*(\w+)\]", rag_context)
            diagnosis["confidence_label"] = m.group(1) if m else "MEDIUM"
        else:
            diagnosis["confidence_label"] = "INSUFFICIENT"

    if not sol.get("verification_steps"):
        steps = sol.get("steps", [])
        if steps and diagnosis.get("status") not in ("escalate", "unsafe"):
            last_step = steps[-1] if steps else {}
            part = last_step.get("part", last_step.get("visual_cue", "the repaired part"))
            sol["verification_steps"] = [
                f"After completing repairs, run the machine for 2–3 minutes at idle.",
                f"Check {part} for leaks, unusual noise, or heat.",
                f"Verify normal operation: expected output should return to baseline.",
                "If the original symptom persists, escalate to a certified mechanic.",
            ]

    if not sol.get("escalation_conditions"):
        escalation_lines = re.findall(r"ESCALATE_IF:\n(.+?)(?=\n[A-Z]|\Z)", rag_context or "", re.DOTALL)
        if escalation_lines:
            sol["escalation_conditions"] = [l.strip() for l in escalation_lines[:3] if l.strip()]
        else:
            sol["escalation_conditions"] = [
                "Fault persists after following all repair steps.",
                "Any step involving electrical disassembly or internal component replacement.",
                "Machine shows signs of structural damage or unusual fluid loss.",
            ]

    if not sol.get("preventive_maintenance"):
        problem_lower = problem_text.lower()
        if any(kw in problem_lower for kw in ["bearing", "seal", "oil", "grease", "lubric"]):
            tip = "Check and replenish lubricant every 250 operating hours or as per manual schedule."
        elif any(kw in problem_lower for kw in ["belt", "drive", "coupling"]):
            tip = "Inspect drive belts for wear and correct tension every 100 hours."
        elif any(kw in problem_lower for kw in ["filter", "air", "fuel"]):
            tip = "Replace fuel and air filters at the manufacturer's recommended interval."
        elif any(kw in problem_lower for kw in ["battery", "electric", "wiring"]):
            tip = "Clean battery terminals monthly and check wiring insulation for cracks."
        else:
            tip = "Follow the manufacturer's periodic maintenance schedule to prevent recurrence."
        sol["preventive_maintenance"] = [tip]

    steps = sol.get("steps", [])
    issues: List[str] = []
    if not steps and diagnosis.get("status") == "success":
        issues.append("WARNING: status=success but steps list is empty.")
    if diagnosis.get("status") == "success" and steps:
        if not any(s.get("required_part") or s.get("text_en") for s in steps):
            issues.append("WARNING: steps present but missing required_part/text_en fields.")
    if not sol.get("safety_warnings_en"):   # covers None and []
        sol["safety_warnings_en"] = ["Follow all standard safety precautions."]
    if issues:
        logger.warning("Diagnosis verification issues: %s", issues)
        diagnosis["_verification_issues"] = issues

    diagnosis["solution"] = sol
    return diagnosis


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def generate_diagnosis_with_gemini(
    machine_type: str,
    problem_text: str,
    language: str = "en",
    rag_context: str = "",
    knowledge_base: str = "",
    visual_frames: Optional[List[bytes]] = None,
    router_symptoms: Optional[List[str]] = None,
    router_confidence: float = 1.0,
    clarification_round: int = 0,
    vector_db=None,
    top_score: float = 0.0,
) -> dict:
    logger.info(
        f"🧠 Diagnosis v11.0: machine={machine_type}, "
        f"problem='{problem_text[:60]}...', "
        f"rag={'YES' if rag_context else 'NO'}, "
        f"visual_text={'YES' if _has_visual_context_text(problem_text) else 'NO'}"
    )

    visual_hash = ""
    if visual_frames:
        mid_frame = visual_frames[len(visual_frames) // 2]
        visual_hash = hashlib.md5(mid_frame).hexdigest()[:8]
    profile       = get_profile_or_default(machine_type)
    machine_label = profile.label_en

    if top_score <= 0.0 and (rag_context or knowledge_base):
        score_match = re.search(r'Relevance:\s*([\d.]+)', rag_context or knowledge_base)
        if score_match:
            top_score = float(score_match.group(1))
        else:
            top_score = 0.60
    context_quality = "strong" if top_score >= _RAG_STRICT_THRESHOLD else "weak"

    cache_key = generate_cache_key("diag_v302", machine_type, f"{problem_text}|vhash:{visual_hash}", language)
    cached = get_cached_response(cache_key)
    if cached:
        try:
            processed = await _post_process_diagnosis(
                cached, machine_type, machine_label, problem_text,
                rag_context or knowledge_base, visual_frames,
                context_quality=context_quality,
            )
            processed["cache_hit"] = True
            return processed
        except InvalidRepairPlan as exc:
            # The cached plan re-validated as broken (e.g. cached before
            # this validation existed). This is a backend defect, not a
            # mechanical fault — do NOT wrap it in an escalation card.
            # The raw cached entry was itself never re-validated on write,
            # so it's not necessarily safe either; surface this loudly and
            # let it propagate so the API layer can return a service error
            # instead of silently serving a plan we know is broken.
            logger.error(
                "❌ Cached repair plan failed validation for machine=%s: %s",
                machine_type, exc,
            )
            raise
        except Exception as exc:
            logger.warning("Cache post-processing failed, returning raw cached plan: %s", exc)
            cached["cache_hit"] = True
            return cached
    allowed_areas = " | ".join(get_allowed_area_ids(machine_type))
    parts_list    = get_compact_parts_list(machine_type)
    safety_kw     = get_compact_safety_keywords(machine_type)

    effective_rag = rag_context or knowledge_base

    trigger = _should_hard_escalate(problem_text, effective_rag)
    if trigger:
        result = _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis=f"Pre-call guard triggered: '{trigger}'. No input to process.",
            safety_warnings_en=["Automatic diagnosis unavailable: no input received."],
            safety_warnings_hi=["स्वचालित निदान उपलब्ध नहीं है: कोई इनपुट प्राप्त नहीं हुआ।"],
            rag_source=RagSource.PRE_CALL,
            problem_identified="Empty or invalid input.",
        )
        cache_response(cache_key, result)
        return result

    g5 = _guard5_empty_rag_maintenance(problem_text, effective_rag, machine_type, machine_label)
    if g5 is not None:
        cache_response(cache_key, g5)
        return g5

    g_unknown = _guard_unknown_attachment(problem_text, machine_type, machine_label)
    if g_unknown is not None:
        cache_response(cache_key, g_unknown)
        return g_unknown

    g_pre_electric = _guard_electric_hazard(problem_text, machine_type)
    if g_pre_electric is not None:
        result = _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis=g_pre_electric["technical_analysis"],
            safety_warnings_en=g_pre_electric["safety_warnings_en"],
            safety_warnings_hi=g_pre_electric["safety_warnings_hi"],
            rag_source=RagSource.ELECTRIC_HAZARD,
            problem_identified=g_pre_electric["technical_analysis"],
            metadata={
                "unsafe_scene_suspected": True,
                "unsafe_scene_message": "Electric hazard detected. Ensure power is cut before contact.",
            },
        )
        cache_response(cache_key, result)
        return result

    g_pre_workaround = _guard_dangerous_workaround(problem_text)
    if g_pre_workaround is not None:
        result = _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis=g_pre_workaround["technical_analysis"],
            safety_warnings_en=g_pre_workaround["safety_warnings_en"],
            safety_warnings_hi=g_pre_workaround["safety_warnings_hi"],
            rag_source=RagSource.WORKAROUND,
            problem_identified=g_pre_workaround["technical_analysis"],
            metadata={
                "unsafe_scene_suspected": True,
                "unsafe_scene_message": "Dangerous mechanical workaround detected. Do not operate machine.",
            },
        )
        cache_response(cache_key, result)
        return result
        
    g_emergency = _guard_emergency_hazard(problem_text)
    if g_emergency is not None:
        result = _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis=g_emergency["technical_analysis"],
            safety_warnings_en=g_emergency["safety_warnings_en"],
            safety_warnings_hi=g_emergency["safety_warnings_hi"],
            rag_source=RagSource.EMERGENCY_HAZARD,
            problem_identified=g_emergency["technical_analysis"],
        )
        cache_response(cache_key, result)
        return result

    # FIX 4: Config-driven threshold — skip multihop when primary retrieval is strong.
    # Saves 2 Gemini calls per request for high-confidence queries.
    _MULTIHOP_MIN_SCORE: float = float(os.environ.get("AGRIFIX_MULTIHOP_MIN_SCORE", "0.60"))
    
    symptoms = router_symptoms or []
    n_chunks_estimate = effective_rag.count("[Source:") if effective_rag else 0
    
    if top_score > _MULTIHOP_MIN_SCORE:
        logger.info(
            "FIX 4: Multihop skipped — score=%.3f > threshold=%.2f (fast-path)",
            top_score, _MULTIHOP_MIN_SCORE,
        )
        chain = None
    else:
        chain = await run_diagnostic_chain(
            machine_type       = machine_type,
            symptoms           = symptoms,
            rag_context        = effective_rag,
            router_confidence  = router_confidence,
            n_rag_chunks       = n_chunks_estimate,
        )
        if chain and chain.enriched_query and vector_db is not None and chain.chain_ok is not False:
            try:
                from rag import retrieve_with_confidence as _retrieve
                rag_v2, score_v2, n_v2 = _retrieve(vector_db, chain.enriched_query, machine_type)
                if score_v2 > top_score and rag_v2:
                    logger.info(
                        "FIX 3: Multi-hop re-retrieval improved score %.3f → %.3f (%d chunks)",
                        top_score, score_v2, n_v2,
                    )
                    effective_rag = rag_v2
                    top_score     = score_v2
                    context_quality = "strong" if top_score >= _RAG_STRICT_THRESHOLD else "weak"
            except Exception as exc:
                logger.warning("FIX 3: Multi-hop re-retrieval failed: %s", exc)
    # ────────────────────────────────────────────────────────────────────────

    if not effective_rag:
        logger.warning(f"⚠️ No RAG context for [{machine_type}] — returning safe no-context escalation")
        result = _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis="No relevant manual excerpts found for this query.",
            safety_warnings_en=[_NO_CONTEXT_ESCALATION_EN],
            safety_warnings_hi=[_NO_CONTEXT_ESCALATION_HI],
            rag_source=RagSource.NO_CONTEXT,
            problem_identified=problem_text,
        )
        cache_response(cache_key, result)
        return result

    # ── FIX 6: Pre-LLM Verification Gate ─────────────────────────────────────
    # Verify machine type and component before committing to repair steps.
    # Only fires for HIGH/CRITICAL risk when visual frames are available.
    risk_level_from_rag = _extract_risk_level_from_rag(effective_rag)
    verification_block = await _run_verification_gate(
        machine_type   = machine_type,
        problem_text   = problem_text,
        rag_context    = effective_rag,
        risk_level     = risk_level_from_rag,
        visual_frames  = visual_frames,
        language       = language,
    )
    if verification_block is not None:
        logger.warning("FIX 6: Verification gate BLOCKED repair: %s", verification_block.get("verification_reason"))
        cache_response(cache_key, verification_block)
        return verification_block
    # ────────────────────────────────────────────────────────────────────────

    prompt = _build_strict_grounding_prompt(
        machine_type      = machine_type,
        machine_label     = machine_label,
        problem_text      = problem_text,
        rag_context       = effective_rag,
        allowed_areas     = allowed_areas,
        parts_list        = parts_list,
        safety_keywords   = safety_kw,
        language          = language,
        has_visual_frames = bool(visual_frames),
        context_quality   = context_quality,
        top_score         = top_score,
        router_symptoms   = symptoms,
    )
    try:
        response = await asyncio.to_thread(
            lambda: groq_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **JSON_CONFIG,
            )
        )
        raw_text = response.choices[0].message.content
        diagnosis = repair_json(raw_text)
        if not isinstance(diagnosis, dict):
            raise ValueError(
                f"LLM returned non-dict JSON ({type(diagnosis).__name__}): "
                f"{str(diagnosis)[:80]}"
            )

        diagnosis = await _post_process_diagnosis(
            diagnosis, machine_type, machine_label, problem_text, effective_rag, visual_frames,
            context_quality=context_quality,
        )
        cache_response(cache_key, diagnosis)
        return diagnosis

    except InvalidRepairPlan as exc:
        # A structurally broken repair plan is a backend defect, not a
        # mechanical fault with the farmer's machine. Do NOT catch this
        # below and wrap it in _fallback_escalation_response() — that
        # would tell the farmer to "consult a certified mechanic" when
        # the actual problem is in this service. Let it propagate; the
        # API layer (main.py) must catch InvalidRepairPlan separately
        # and return a generic service-error response (e.g. HTTP 500).
        logger.error("❌ Invalid repair plan for machine=%s: %s", machine_type, exc)
        raise
    except Exception as exc:
        logger.error("❌ Diagnosis generation failed: %s", exc)
        return _fallback_escalation_response(machine_type, machine_label, problem_text, str(exc))