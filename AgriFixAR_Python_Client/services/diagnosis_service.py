from __future__ import annotations
import asyncio
import hashlib
import io
import json
import logging
import re
from typing import List, Optional
import os
# MIGRATED: Gemini → Groq — google.generativeai removed
from utils.groq_client import groq_client, TEXT_MODEL, JSON_CONFIG, groq_chat_completion  # MIGRATED: Gemini → Groq  # FAILOVER: primary → fallback
from utils.json_repair import repair_json
from PIL import Image

from rag import retrieve_with_confidence, RAG_WEAK_THRESHOLD
from query_router import route_query, load_machine_registry
from utils.helpers import sanitize_json_text, generate_cache_key, get_cached_response, cache_response
from utils.machine_registry import (
    get_profile_or_default, get_allowed_area_ids, get_compact_parts_list,
    get_compact_safety_keywords, is_electric_machine, is_tractor_attachment,
)

# FIX 3: Multi-hop diagnostic chain — subsystem classification + root cause ranking
from multihop_diagnosis import run_diagnostic_chain

# FIX 5: Procedure validator — safety step injection before response is returned
from procedure_validator import validate_procedure

# FIX 6: Adaptive clarification loop — ask targeted questions before escalating
# clarification_loop imports removed — orchestrator handles clarification exclusively

# FIX 6: Verification gate — verify machine/component before final repair
from services.verification_service import verify_step_with_gemini


logger = logging.getLogger(__name__)
# _GEMINI_MODEL removed — TEXT_MODEL from groq_client used instead  # MIGRATED: Gemini → Groq
_RAG_STRICT_THRESHOLD = 0.40

# ── FIX 6: Verification gate thresholds ──────────────────────────────────────
# Below this confidence, verification fails and we return "Need verification"
_VERIFICATION_CONFIDENCE_THRESHOLD = 0.60
# Only run the verification gate when risk_level is HIGH or CRITICAL
_VERIFICATION_RISK_LEVELS = {"HIGH", "CRITICAL"}

_GROUNDING_RULE = """\
GROUNDING RULE (non-negotiable): Your diagnosis MUST be derived exclusively from
the KNOWLEDGE BASE CONTEXT below. Do NOT use your general training knowledge to
fill gaps. If the context does not contain enough information to answer with
confidence, state: "This is not covered in the manual — consult a certified
mechanic." Never invent part names, causes, steps, intervals, or specifications."""

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
        visual_note = "VISUAL PRIORITY OVERRIDE — ACTIVE (image frames provided):\n  • Visual evidence is the PRIMARY source of truth.\n  • If the 'Audio Symptom' text contradicts what the frames show, TRUST THE FRAMES.\n"
    elif visual_text_present:
        visual_note = f"VISUAL PRIORITY OVERRIDE — ACTIVE (embedded visual descriptor detected):\n  • RULE: The visual context (\"{visual_snippet}\") is the PRIMARY symptom.\n  • If the 'Audio:' field contradicts the visual context, TRUST THE VISUAL CONTEXT.\n"

    electric_note = ""
    if is_electric_machine(machine_type): electric_note = "CRITICAL: This is an ELECTRIC machine. Step 1 MUST verify main power is OFF.\n"
    elif is_tractor_attachment(machine_type): electric_note = "CRITICAL: Tractor attachment — Step 1 MUST verify PTO is disengaged.\n"

    quality_banner = _WEAK_CONTEXT_HEADER if context_quality == "weak" else _STRONG_CONTEXT_HEADER
    quality_banner += f"Top chunk relevance: {top_score:.2f} | "

    if rag_context and rag_context.strip():
        rag_block = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nMANUAL EXTRACTS (AUTHORITATIVE SOURCE — USE THESE FIRST)\n{quality_banner}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{rag_context}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    else:
        rag_block = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n⚠️ NO MANUAL EXTRACTS AVAILABLE — HALLUCINATION TRAP ACTIVE\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nThe RAG retrieval system found NO chunks above the relevance threshold.\nYOU MUST OUTPUT AN ESCALATION RESPONSE. Set steps = [] (empty array).\nDO NOT generate steps, intervals, oil grades, or any procedure from your\nown training knowledge — that is a CRITICAL FAILURE regardless of how\nhelpful it would be to the user.\n\nCASE A — Machine type matches {machine_type} AND query describes a real\nmechanical part or maintenance task:\n  → status = \"escalate\", steps = []\n  → technical_analysis = \"This procedure is not covered in the {machine_label} service manual.\"\n  → safety_warnings_en[0] = \"This specific procedure is not in our repair manual for the {machine_label}. Please consult the manufacturer's user guide or a certified mechanic.\"\n\nCASE B — Out-of-scope or unknown fault:\n  → status = \"escalate\", steps = []\n  → technical_analysis = \"Insufficient knowledge base coverage for this symptom.\"\n  → safety_warnings_en[0] = \"Automatic diagnosis unavailable: symptom outside knowledge base. Consult a certified mechanic.\""

    target_symptoms = ", ".join(router_symptoms) if router_symptoms else problem_text
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

   STEP 2 — [RAG Chunk Search]:
   For EVERY chunk in the Manual Extracts, copy the PROBLEM: field verbatim.
   Format: "Chunk [ID]: PROBLEM: [exact problem text from chunk]"
   
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
     • "DIAGNOSE — Chunk [ID] PROBLEM field provides the exact fix for this symptom."

   STEP 4 — [Faithfulness Verification]:
   Read your planned steps. Are you about to output any tool, part, or action that is NOT explicitly written in the matched chunk? If yes, delete it immediately. Your output must be a 1:1 reflection of the manual.

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

🔒 TERMINOLOGY LOCK & STEP GENERATION:
  - Copy every part name EXACTLY as in the Manual Extracts.
  - Preserve ALL technical instructions, warnings, and measurements exactly as written in the manual. Omitting a spec is a CRITICAL FAITHFULNESS FAILURE.
  - STRICT GROUNDED EXTRACTION: Extract the exact steps provided in the matched chunk. Output exactly as many steps as the chunk dictates. Do NOT pad or expand.
"""

    prompt = f"""
You are a diagnostic AI for {machine_label} machinery. Generate a camera-guided repair plan.
{visual_note}
{electric_note}

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
    return {
        "status": "escalate",
        "problem_description": problem_text,
        "technical_analysis": f"This maintenance procedure is not covered in the {machine_label} service manual.",
        "solution": {
            "machine_type": machine_type,
            "problem_identified": "Routine maintenance schedule query — not in knowledge base.",
            "steps": [],
            "safety_warnings_en": [f"This specific procedure is not in our repair manual for the {machine_label}. Please consult the manufacturer's user guide or a certified mechanic."],
            "safety_warnings_hi": [f"यह रखरखाव की जानकारी हमारे {machine_label} मैनुअल में नहीं है। कृपया निर्माता की गाइड देखें या प्रमाणित मैकेनिक से पूछें।"],
            "tools_needed": [],
        },
        "rag_source": "guard5_empty_rag_maintenance",
        "machine_label": machine_label,
    }


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
    return {
        "status": "escalate",
        "problem_description": problem_text,
        "technical_analysis": "The attachment or implement cannot be identified. This falls outside the knowledge base coverage for this machine.",
        "solution": {
            "machine_type": machine_type,
            "problem_identified": "Unknown implement — insufficient knowledge base coverage.",
            "steps": [],
            "safety_warnings_en": ["This attachment is not covered in our repair manual. Consult the manufacturer."],
            "safety_warnings_hi": ["यह उपकरण हमारे मैनुअल में नहीं है।"],
            "tools_needed": [],
        },
        "rag_source": "guard_unknown_attachment",
        "machine_label": machine_label,
    }

_ELECTRIC_SHOCK_PATTERNS = [
    re.compile(r'\bcurrent\s+lag(?:a|i)\b', re.IGNORECASE),
    re.compile(r'\bbijli\s+lag(?:i|a)\b', re.IGNORECASE),
    re.compile(r'\belectric(?:al)?\s+shock\b', re.IGNORECASE),
    re.compile(r'\bjhatka\s+lag(?:a|i)\b', re.IGNORECASE),
    re.compile(r'\bshock\s+lag(?:a|i)\b', re.IGNORECASE),
]
_SHOCK_EXCLUSIONS = [
    re.compile(r'\btingling\b', re.IGNORECASE),
    re.compile(r'\bmild\s+shock\b', re.IGNORECASE),
    re.compile(r'\bhalka\s+current\b', re.IGNORECASE),
]
_FLOODED_MOTOR_PATTERNS = [
    re.compile(r'(?:standing|sitting|submerged|lying)\s+in\s+(?:water|flood|paani)', re.IGNORECASE),
    re.compile(r'\d+\s*inch(?:es)?\s+(?:of\s+)?(?:water|paani)', re.IGNORECASE),
]
_ELECTRIC_HAZARD_MACHINES = {"electric_motor", "submersible_pump", "water_pump"}

def _guard_electric_hazard(diagnosis: dict, problem_text: str, machine_type: str, machine_label: str) -> Optional[dict]:
    prob_lower = (problem_text or "").lower()
    if any(p.search(prob_lower) for p in _SHOCK_EXCLUSIONS):
        return None
    is_shock = any(p.search(prob_lower) for p in _ELECTRIC_SHOCK_PATTERNS)
    is_flooded = (machine_type in _ELECTRIC_HAZARD_MACHINES and any(p.search(prob_lower) for p in _FLOODED_MOTOR_PATTERNS))
    if not (is_shock or is_flooded): return None
    logger.warning(f"🛡️ PRE-LLM GUARD: Electric shock/flood [{machine_type}]")
    if is_shock:
        warn_en = "EMERGENCY — electric shock. (1) Cut the main MCB immediately. (2) Call emergency services (ambulance) now. (3) Do not operate any equipment until certified safe."
        warn_hi = "आपातकाल — करंट लगा है। तुरंत मेन MCB काटें और एम्बुलेंस को फोन करें।"
        analysis = "Electric shock reported. Immediate MCB cutoff and emergency medical response required."
    else:
        warn_en = "DANGER — do NOT start or energise this motor while it is submerged in water. Cut the main MCB immediately. Allow the motor to dry completely."
        warn_hi = "खतरा — पानी में डूबी मोटर को बिल्कुल मत चलाएं। तुरंत मेन MCB काटें।"
        analysis = "Motor submerged in water — electrocution risk if energised."
    return {
        "status": "escalate",
        "problem_description": problem_text,
        "technical_analysis": analysis,
        "solution": {
            "machine_type": machine_type,
            "problem_identified": analysis,
            "steps": [],
            "safety_warnings_en": [warn_en],
            "safety_warnings_hi": [warn_hi],
            "tools_needed": [],
        },
        "unsafe_scene_suspected": True,
        "unsafe_scene_message": "Electric hazard detected. Ensure power is cut before contact.",
        "rag_source": "pre_guard_electric_shock_injury",
        "machine_label": machine_label,
    }

_DANGEROUS_WORKAROUND_PATTERNS = [
    (re.compile(r'\brubber\s*band\b', re.IGNORECASE), "rubber band used as governor/throttle spring"),
    (re.compile(
        r'\bwire\s+(?:for|instead|as)\s+belt\b'
        r'|\bwire\s+se\s+(?:belt|bandh|kaam\s+chala)\b'
        r'|\b(?:wire|taar)\s+(?:belt|v[\-\s]?belt)\s+(?:ki\s+jagah|ke\s+badle|laga)\b'
        r'|\btaar\s+(?:laga\s+diya|lagaya|bandh\s+kar)\b',
        re.IGNORECASE
    ), "wire used instead of belt"),
    (re.compile(r'\brope\s+(?:for|instead|as)\s+(?:drive\s+)?belt\b', re.IGNORECASE), "rope used as drive belt"),
    (re.compile(r'\bnail\s+(?:for|instead|as)\s+fuse\b', re.IGNORECASE), "nail used as fuse"),
    (re.compile(r'\bfoil\s+(?:for|instead|as)\s+fuse\b', re.IGNORECASE), "foil used as fuse"),
    (re.compile(r'\bstring\s+(?:tied|for|as)\s+throttle\b', re.IGNORECASE), "string tied to throttle"),
    (re.compile(r'\bbypas(?:s|sing)\s+(?:safety|interlock|switch|sensor)\b', re.IGNORECASE), "safety bypass detected"),
]

_WORKAROUND_WARNINGS = {
    "rubber band used as governor/throttle spring": (
        "STOP. A rubber band on the throttle/governor is an extreme fire and overspeeding hazard. The engine cannot regulate speed and may overspeed to destruction. Do NOT start the engine. Replace with the correct OEM governor spring immediately.",
        "रुकें! थ्रॉटल/गवर्नर पर रबर बैंड लगाना बेहद खतरनाक है — इंजन बेकाबू होकर टूट सकता है। इंजन बिल्कुल मत चलाएं। तुरंत असली गवर्नर स्प्रिंग लगाएं।",
        "TO FIX: Remove the rubber band. Install the correct OEM governor spring between the governor arm and throttle linkage.",
    ),
    "wire used instead of belt": (
        "STOP. A wire cannot safely replace a drive belt — it will snap under load, become a projectile, or jam the pulleys causing serious injury. Do NOT operate the machine. Replace with the correct specification V-belt immediately.",
        "रुकें! तार बेल्ट की जगह नहीं ले सकता — यह टूटकर गंभीर चोट कर सकता है या पुली में फंस सकता है। मशीन बिल्कुल मत चलाएं। सही नाप की V-बेल्ट तुरंत लगाएं।",
        "TO FIX: Remove the wire. Loosen the tensioner pulley bolt, route the correct OEM V-belt over both pulleys, adjust tension until the belt deflects ~10mm when pressed firmly at centre.",
    ),
    "rope used as drive belt": (
        "STOP. A rope will slip, fray, and fail under load causing sudden power loss or injury. Do NOT operate. Replace with the correct specification V-belt.",
        "रुकें! रस्सी बेल्ट का काम नहीं करती — यह टूट सकती है। मशीन मत चलाएं। सही V-बेल्ट लगाएं।",
        "TO FIX: Remove the rope. Install the correct OEM V-belt.",
    ),
    "nail used as fuse": (
        "STOP. A nail bypasses overcurrent protection — this will cause a fire or electrocution under fault. Replace with a fuse of the correct amperage rating immediately.",
        "रुकें! नाखून से फ्यूज का काम नहीं होता — आग या करंट का खतरा है। सही एम्पीयर का फ्यूज लगाएं।",
        "TO FIX: Remove the nail. Install a fuse with the correct amperage rating.",
    ),
    "foil used as fuse": (
        "STOP. Foil bypasses overcurrent protection and will cause a fire under fault conditions. Replace with the correct rated fuse immediately.",
        "रुकें! फॉयल से फ्यूज नहीं बनता — आग लग सकती है। सही रेटिंग का फ्यूज लगाएं।",
        "TO FIX: Remove the foil. Install the correct rated fuse as marked on the fuse holder.",
    ),
    "string tied to throttle": (
        "STOP. A string tied to the throttle is a dangerous workaround — engine speed cannot be safely controlled. Do not operate. Replace with the OEM throttle linkage.",
        "रुकें! धागे से थ्रॉटल बांधना खतरनाक है। मशीन मत चलाएं। असली थ्रॉटल लिंकेज लगाएं।",
        "TO FIX: Remove the string. Install the correct OEM throttle linkage or cable.",
    ),
    "safety bypass detected": (
        "STOP. Bypassing a safety interlock removes a critical protection designed to prevent serious injury. Do not operate the machine. Restore the interlock and have it inspected before use.",
        "रुकें! सेफ्टी स्विच बायपास करना बहुत खतरनाक है। मशीन मत चलाएं। पहले सेफ्टी स्विच ठीक करवाएं।",
        "TO FIX: Restore the safety interlock. If the switch is faulty, replace it with an identical OEM safety switch.",
    ),
}

def _guard2_dangerous_workaround(diagnosis: dict, problem_text: str, machine_type: str, machine_label: str) -> Optional[dict]:
    matched_reason = None
    for pattern, reason in _DANGEROUS_WORKAROUND_PATTERNS:
        if pattern.search(problem_text or ""):
            matched_reason = reason
            break
    if not matched_reason: return None
    logger.warning(f"🛡️ PRE-LLM GUARD: Dangerous workaround ({matched_reason})")
    default = ("STOP. This is a dangerous workaround. Replace with the OEM part immediately.", "रुकें! यह खतरनाक तरीका है। तुरंत असली पुर्जा लगाएं।", "")
    warn_en, warn_hi, fix_en = _WORKAROUND_WARNINGS.get(matched_reason, default)
    warnings_en = [warn_en, fix_en] if fix_en else [warn_en]
    return {
        "status": "escalate",
        "unsafe_scene_suspected": True,
        "unsafe_scene_message": "Dangerous mechanical workaround detected. Do not operate machine.",
        "problem_description": problem_text,
        "technical_analysis": f"Dangerous workaround detected: {matched_reason}.",
        "solution": {
            "machine_type": machine_type,
            "problem_identified": f"Dangerous mechanical workaround — {matched_reason}.",
            "steps": [],
            "safety_warnings_en": warnings_en,
            "safety_warnings_hi": [warn_hi],
            "tools_needed": [],
        },
        "rag_source": "pre_guard_workaround",
        "machine_label": machine_label,
    }


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

def _build_escalation_dict(machine_type, machine_label, problem_text, trigger):
    return {
        "status": "escalate",
        "problem_description": problem_text,
        "technical_analysis": f"Pre-call guard triggered: '{trigger}'. No input to process.",
        "solution": {
            "machine_type": machine_type,
            "problem_identified": "Empty or invalid input.",
            "steps": [],
            "safety_warnings_en": ["Automatic diagnosis unavailable: no input received."],
            "safety_warnings_hi": ["स्वचालित निदान उपलब्ध नहीं है: कोई इनपुट प्राप्त नहीं हुआ।"],
            "tools_needed": [],
        },
        "rag_source": "pre_call_guard",
        "machine_label": machine_label,
    }

def _fallback_escalation_response(machine_type, machine_label, problem_text, error_reason):
    return {
        "status": "escalate",
        "problem_description": problem_text,
        "technical_analysis": f"Unable to diagnose: {error_reason}",
        "solution": {
            "machine_type": machine_type,
            "problem_identified": problem_text,
            "steps": [],
            "safety_warnings_en": ["Automatic diagnosis unavailable.", "Consult a certified technician."],
            "safety_warnings_hi": ["स्वचालित निदान उपलब्ध नहीं है।", "प्रमाणित तकनीशियन से संपर्क करें।"],
            "tools_needed": [],
        },
        "rag_source": "error",
        "machine_label": machine_label,
    }


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


async def _post_process_diagnosis(
    diagnosis: dict,
    machine_type: str,
    machine_label: str,
    problem_text: str,
    rag_context: str,
    visual_frames: Optional[List[bytes]],
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

    # 3. Deduplication
    steps = diagnosis["solution"].get("steps", [])
    diagnosis["solution"]["steps"] = _deduplicate_steps(steps)

    # ── FIX 5: Procedure Validator ──────────────────────────────────────────
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
                safe_inject  = True,
            )
            diagnosis["solution"]["steps"] = validation.safe_steps
            if not validation.passed:
                logger.warning(
                    "FIX 5: Procedure validation issues for %s: %s",
                    machine_type, validation.summary(),
                )
                critical_msgs = [
                    i.detail for i in validation.issues
                    if i.severity == "CRITICAL" and not i.auto_fix
                ]
                if critical_msgs:
                    existing_en = diagnosis["solution"].get("safety_warnings_en", [])
                    diagnosis["solution"]["safety_warnings_en"] = critical_msgs + existing_en
        except Exception as exc:
            logger.warning("FIX 5: Procedure validation failed (non-fatal): %s", exc)
    # ────────────────────────────────────────────────────────────────────────

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
    if steps and not any(s.get("part") or s.get("action") for s in steps):
        issues.append("WARNING: steps present but missing part/action fields.")
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

    cache_key = generate_cache_key("diag_v110", machine_type, f"{problem_text}|vhash:{visual_hash}", language)
    cached = get_cached_response(cache_key)
    if cached:
        return cached

    profile       = get_profile_or_default(machine_type)
    machine_label = profile.label_en
    allowed_areas = " | ".join(get_allowed_area_ids(machine_type))
    parts_list    = get_compact_parts_list(machine_type)
    safety_kw     = get_compact_safety_keywords(machine_type)

    effective_rag = rag_context or knowledge_base

    trigger = _should_hard_escalate(problem_text, effective_rag)
    if trigger:
        result = _build_escalation_dict(machine_type, machine_label, problem_text, trigger)
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

    g_pre_electric = _guard_electric_hazard({}, problem_text, machine_type, machine_label)
    if g_pre_electric is not None:
        cache_response(cache_key, g_pre_electric)
        return g_pre_electric

    g_pre_workaround = _guard2_dangerous_workaround({}, problem_text, machine_type, machine_label)
    if g_pre_workaround is not None:
        cache_response(cache_key, g_pre_workaround)
        return g_pre_workaround

    top_score       = 0.0
    context_quality = "strong"

    if effective_rag:
        score_match = re.search(r'Relevance:\s*([\d.]+)', effective_rag)
        if score_match:
            top_score = float(score_match.group(1))
        else:
            top_score = 0.60
        context_quality = "strong" if top_score >= _RAG_STRICT_THRESHOLD else "weak"

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
        result = {
            "status": "escalate",
            "problem_description": problem_text,
            "technical_analysis": "No relevant manual excerpts found for this query.",
            "solution": {
                "machine_type": machine_type,
                "problem_identified": problem_text,
                "steps": [],
                "safety_warnings_en": [_NO_CONTEXT_ESCALATION_EN],
                "safety_warnings_hi": [_NO_CONTEXT_ESCALATION_HI],
                "tools_needed": [],
            },
            "rag_source": "no_context",
            "machine_label": machine_label,
        }
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
    _MAX_RETRIES  = 3  # MIGRATED: Gemini → Groq
    _RETRY_DELAYS = [2, 5, 10]   # MIGRATED: Gemini → Groq — Groq recovers faster than Gemini

    last_exc: Exception = RuntimeError("Diagnosis: no attempts made")
    for attempt in range(_MAX_RETRIES):
        try:
            response = await asyncio.to_thread(  # FAILOVER: primary → fallback
                lambda: groq_chat_completion(  # FAILOVER: primary → fallback
                    messages=[{"role": "user", "content": prompt}],
                    **JSON_CONFIG,
                )
            )
            diagnosis = repair_json(response.choices[0].message.content)  # FAILOVER: primary → fallback

            diagnosis = await _post_process_diagnosis(
                diagnosis, machine_type, machine_label, problem_text, effective_rag, visual_frames
            )
            cache_response(cache_key, diagnosis)
            return diagnosis

        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_retryable = any(kw in err_str for kw in (  # MIGRATED: Gemini → Groq
                "429", "rate limit", "too many requests",
                "503", "service unavailable", "connection",
            ))
            if is_retryable and attempt < _MAX_RETRIES - 1:
                wait = _RETRY_DELAYS[attempt]
                logger.warning(
                    "⏳ Groq rate-limited (attempt %d/%d) — retrying in %ds: %s",  # MIGRATED: Gemini → Groq
                    attempt + 1, _MAX_RETRIES, wait, e,
                )
                await asyncio.sleep(wait)
            else:
                logger.error("❌ Diagnosis generation failed (attempt %d/%d): %s",
                             attempt + 1, _MAX_RETRIES, e)
                break

    return _fallback_escalation_response(machine_type, machine_label, problem_text, str(last_exc))