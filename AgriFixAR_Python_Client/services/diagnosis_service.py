"""
diagnosis_service.py — Production Diagnosis Service v6.0
AgriFix Multimodal Diagnostic System

CHANGES FROM v5.4 → v6.0:
  - STRIPPED BLOAT: Removed all Python regex guards for out-of-scope, fire, 
    injury, and maintenance.
  - SIMPLIFIED PRE-CALL GUARD: Now only handles catastrophic API failures.
  - UNIVERSAL SAFETY MASTER: Edge case routing and escalation is now handled
    entirely by RAG context via Universal_Safety_Master.txt chunks and new
    grounding rules in the prompt.
"""

from __future__ import annotations
import asyncio
import hashlib
import io
import json
import logging
import re
from typing import List, Optional

import google.generativeai as genai
from PIL import Image

from utils.helpers import (
    sanitize_json_text,
    generate_cache_key,
    get_cached_response,
    cache_response,
)
from utils.machine_registry import (
    get_profile_or_default,
    get_allowed_area_ids,
    get_compact_parts_list,
    get_compact_safety_keywords,
    is_electric_machine,
    is_tractor_attachment,
)

logger = logging.getLogger(__name__)
_GEMINI_MODEL = "models/gemini-2.5-flash"


def _has_visual_context_text(problem_text: str) -> bool:
    return "visual context:" in problem_text.lower()

def _extract_visual_snippet(problem_text: str) -> str:
    match = re.search(r'visual context:\s*(.+?)(?:audio:|$)', problem_text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

def _build_strict_grounding_prompt(
    machine_type: str,
    machine_label: str,
    problem_text: str,
    rag_context: str,
    allowed_areas: str,
    parts_list: str,
    safety_keywords: str,
    language: str,
    has_visual_frames: bool,
) -> str:

    visual_note = ""
    visual_text_present = _has_visual_context_text(problem_text)
    visual_snippet = _extract_visual_snippet(problem_text) if visual_text_present else ""

    if has_visual_frames:
        visual_note = f"""
VISUAL PRIORITY OVERRIDE — ACTIVE (image frames provided):
  • Visual evidence is the PRIMARY source of truth.
  • If the "Audio Symptom" text contradicts what the frames show, TRUST THE FRAMES.
"""
    elif visual_text_present:
        visual_note = f"""
VISUAL PRIORITY OVERRIDE — ACTIVE (embedded visual descriptor detected):
  • RULE: The visual context ("{visual_snippet}") is the PRIMARY symptom.
  • If the "Audio:" field contradicts the visual context, TRUST THE VISUAL CONTEXT.
"""

    electric_note = ""
    if is_electric_machine(machine_type):
        electric_note = "CRITICAL: This is an ELECTRIC machine. Step 1 MUST verify main power is OFF via the main breaker.\n"
    elif is_tractor_attachment(machine_type):
        electric_note = "CRITICAL: This is a tractor attachment. Step 1 MUST verify PTO is disengaged and implement is lowered.\n"

    if rag_context and rag_context.strip():
        rag_block = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANUAL EXTRACTS (AUTHORITATIVE SOURCE — USE THESE FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{rag_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    else:
        rag_block = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ NO MANUAL EXTRACTS AVAILABLE — HALLUCINATION TRAP ACTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The RAG retrieval system found NO chunks above the relevance threshold for
this symptom.

YOU MUST OUTPUT AN ESCALATION RESPONSE — DO NOT ATTEMPT TO DIAGNOSE.
Set steps = [] (empty array).

CHOOSE the correct escalation tone based on the query:

CASE A — Query is a legitimate repair topic for a {machine_label} BUT the
specific procedure is not in our service manual (e.g. routine cleaning, 
lubrication schedules, consumable replacement):
  → status = "escalate"
  → technical_analysis = "This procedure is not covered in the {machine_label} service manual."
  → safety_warnings_en[0] = "This specific procedure is not in our repair manual for the {machine_label}. Please consult the manufacturer's user guide or a certified mechanic for routine maintenance tasks like this."
  → safety_warnings_hi[0] = (same in Hindi)

CASE B — Query is completely outside repair scope (wrong machine type,
impossible symptom, no recognisable fault description):
  → status = "escalate"
  → technical_analysis = "Insufficient knowledge base coverage for this symptom."
  → safety_warnings_en[0] = "Automatic diagnosis unavailable: machine or symptom outside knowledge base. Consult a certified mechanic."

Apply CASE A when the machine type matches ({machine_type}) and the query describes
a real mechanical part or maintenance task.
Apply CASE B for anything else (unknown machine, nonsensical symptom).
"""

    hindi_note = "Output ALL text_hi fields in simple village Hindi.\n"

    grounding_rules = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT GROUNDING RULES — ZERO-HALLUCINATION PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔵 MANDATORY CHAIN OF THOUGHT — YOU MUST COMPLETE ALL 3 STEPS BEFORE
   SETTING ANY OUTPUT FIELD. WRITE THIS IN "internal_reasoning".

   STEP 1 — [Visual/Audio Check]:
   In one sentence, state exactly what is happening:
   "The user describes [X symptom] with [visual/audio evidence Y]."
   If visual context is present, it overrides audio/text description.

   STEP 2 — [RAG Chunk Search]:
   Scan EVERY chunk in the Manual Extracts above.
   State the result as one of these exact formats:
     • "Matched: [Source file] Chunk [ID] — ESCALATE_IF triggered: [quote the trigger text]"
     • "Matched: [Source file] Chunk [ID] — Diagnostic content applies, no ESCALATE_IF."
     • "No chunk match found for this symptom."
   YOU MUST NAME THE CHUNK ID. Vague statements like "a safety chunk applies"
   are FORBIDDEN and constitute a grounding failure.

   STEP 3 — [Action Decision]:
   State your routing decision as exactly one of:
     • "ESCALATE — Universal safety hazard (Chunk [ID]) overrides all other rules."
     • "ESCALATE — Hallucination Trap: no chunk covers this specific fault."
     • "DIAGNOSE — Chunk [ID] provides specific fault steps, no safety override."
     • "REASSURE — Chunk [ID] (e.g. Chunk 16) explicitly confirms reading is normal."

   ⛔ DO NOT set status, steps, or any other field until all 3 steps are written.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — UNIVERSAL SAFETY MASTER (ABSOLUTE HIGHEST PRIORITY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If STEP 2 identifies ANY Universal_Safety_Master chunk whose ESCALATE_IF
condition matches the problem description:

  → status = "escalate", steps = []
  → COPY-PASTE MANDATE (see Rule 6 below): Copy the matched chunk's STEPS
    field CHARACTER-FOR-CHARACTER into safety_warnings_en[0].
  → COPY-PASTE MANDATE: Copy the matched chunk's STEPS_HI field
    CHARACTER-FOR-CHARACTER into safety_warnings_hi[0].
  → technical_analysis = the matched chunk's PROBLEM field verbatim.

  ⛔ ABSOLUTE: This rule fires BEFORE Reassurance (Rule 2). If a Universal
  Safety chunk matches AND a Reassurance chunk matches for the same query,
  the Universal Safety chunk ALWAYS wins. There is no exception.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — REASSURANCE (Normal Readings Only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONLY apply this rule when ALL of the following are true:
  ✅ STEP 2 identified a chunk (e.g. Chunk 16) whose content EXPLICITLY
     states this reading or condition is within normal operating range.
  ✅ NO Universal Safety Master chunk's ESCALATE_IF has been triggered.

If both conditions are met:
  → status = "success", generate 1-2 confirmation steps from chunk content.
  → COPY-PASTE MANDATE: Any warning or instruction in the chunk STEPS field
    must be copied verbatim, not paraphrased.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 — FIRE EXTINGUISHER (Informational — No Active Fire)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If STEP 2 matches Chunk 3b AND no active fire is present:
  → status = "success"
  → COPY-PASTE MANDATE: Copy Chunk 3b STEPS verbatim into the step text.
  → Do NOT escalate this informational query.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 4 — ROUTINE MAINTENANCE REDIRECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If STEP 2 identifies the query as a maintenance schedule (Chunk 4 pattern)
with no active fault present:
  → status = "escalate", steps = []
  → COPY-PASTE MANDATE: Copy the matched chunk's STEPS verbatim into
    safety_warnings_en[0].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 5 — HALLUCINATION TRAP (Strict Failsafe)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If STEP 2 result is "No chunk match found":
  → status = "escalate", steps = []
  → DO NOT generate diagnostic steps from your training knowledge.
  → DO NOT guess causes, suggest probable faults, or attempt to help.
  → You have exactly TWO valid responses based on the query type:

  CASE A — Machine type is {machine_type} AND query describes a real part
  or mechanical symptom, but no chunk covers the specific procedure:
    → technical_analysis = "This procedure is not covered in the {machine_label} service manual."
    → safety_warnings_en[0] = "This specific procedure is not in our repair manual for the {machine_label}. Please consult the manufacturer's user guide or a certified mechanic."

  CASE B — Machine type is unknown, query is out-of-scope, or symptom is
  not recognizable as a mechanical fault:
    → technical_analysis = "Insufficient knowledge base coverage for this symptom."
    → safety_warnings_en[0] = "Automatic diagnosis unavailable: machine or symptom outside knowledge base. Consult a certified mechanic."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 6 — COPY-PASTE MANDATE (Universal — No Exceptions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⛔ CRITICAL FAILURE DEFINITION: Summarizing, shortening, paraphrasing,
or reordering ANY text from a chunk's STEPS or STEPS_HI field is a
CRITICAL FAILURE that will cause this system to injure or kill a farmer.

When ANY rule above instructs you to copy a STEPS field:
  1. Locate the exact STEPS string in the Manual Extracts.
  2. Copy it character-for-character into the target output field.
  3. Do NOT remove sentences. Do NOT combine sentences. Do NOT omit
     measurements, tool names, or action verbs.
  4. EXAMPLES OF CRITICAL FAILURES:
       ✗ Chunk says "Apply firm pressure to the wound with a clean cloth"
         → you output "apply pressure to wound"                [KILL RISK]
       ✗ Chunk says "pull the pin, aim at the BASE of the flame"
         → you output "use the extinguisher on the fire"       [KILL RISK]
       ✗ Chunk says "seek emergency surgical medical attention"
         → you output "go to a doctor"                         [KILL RISK]

CONFLICT RESOLUTION:
  • Active hazard chunk (fire, injury, shock, entanglement) vs. machine chunk
    → Active hazard chunk wins unconditionally.
  • Informational chunk (Chunk 3b, Chunk 16) vs. escalation trigger
    → Only apply informational chunk if NO active hazard exists.
"""

    output_format = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — JSON ONLY, NO MARKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL SCHEMA RULES:
  1. "internal_reasoning" MUST be the first key and MUST contain all 3
     CoT steps labeled [Visual/Audio Check], [RAG Chunk Search], [Action].
  2. "solution" MUST always be present — even on escalations.
  3. status="escalate" → "solution.steps" MUST be [] (empty array, not null).
  4. status="success"  → "solution.steps" MUST contain at least 1 step object.
  5. Every step's "text_en" MUST be 3-4 sentences:
       Sentence 1: WHERE the part is (colour + shape + landmark).
       Sentence 2: WHAT to do with your hands.
       Sentence 3-4: WHAT to see/hear/feel when done correctly.
  6. "required_part" MUST be a snake_case ID from: {parts_list}
  7. "area_hint" MUST be one of: {allowed_areas}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔒 TERMINOLOGY LOCK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Copy every part name EXACTLY as it appears in the Manual Extracts.
  ✗ Manual says "foot valve"  → you output "bottom filter"   [CRITICAL FAIL]
  ✗ Manual says "clevis pin"  → you output "lock pin"        [CRITICAL FAIL]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESCALATE EXAMPLE — Active Injury (Chunk 1)
⚠️ This example is for ONE scenario. For ALL other hazards, copy the
   matching chunk's STEPS field verbatim — do NOT reuse injury text.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "internal_reasoning": "[Visual/Audio Check]: User describes 'haat cut gaya' — a visible cut/bleeding injury. [RAG Chunk Search]: Matched Universal_Safety_Master Chunk 1 — ESCALATE_IF triggered: 'Any mention of injury, blood, cut, or physical harm to a person near the machine.' [Action]: ESCALATE — Universal safety hazard (Chunk 1) overrides all other rules.",
  "status": "escalate",
  "problem_description": "...",
  "technical_analysis": "User or bystander has a visible injury, cut, or bleeding from machine contact.",
  "solution": {{
    "machine_type": "{machine_type}",
    "problem_identified": "Active injury detected — first aid required immediately.",
    "steps": [],
    "safety_warnings_en": [
      "STOP all repair activity immediately. Move the injured person away from the machine. Apply firm pressure to the wound with a clean cloth. Call emergency services or transport to the nearest medical facility without delay. Do NOT operate the machine while injured. The machine must NOT be restarted until the guard is repaired and the area is safe. Working with an open injury near moving blades is a life-threatening secondary hazard."
    ],
    "safety_warnings_hi": [
      "तुरंत मरम्मत बंद करें। घायल व्यक्ति को मशीन से दूर ले जाएं। साफ कपड़े से घाव पर दबाव डालें। तुरंत नजदीकी अस्पताल जाएं। मशीन का गार्ड ठीक होने तक इसे न चलाएं।"
    ],
    "tools_needed": []
  }}
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUCCESS EXAMPLE — Fault with RAG chunk match
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "internal_reasoning": "[Visual/Audio Check]: User reports engine not starting with no cranking sound on {machine_label}. [RAG Chunk Search]: Matched {machine_label} manual Chunk 3 — Diagnostic content applies, battery and starter fault pattern. No ESCALATE_IF triggered. [Action]: DIAGNOSE — Chunk 3 provides specific fault steps, no safety override.",
  "status": "success",
  "problem_description": "...",
  "technical_analysis": "Based on Manual Extracts Chunk 3: ...",
  "solution": {{
    "machine_type": "{machine_type}",
    "problem_identified": "Short summary of identified fault",
    "steps": [
      {{
        "text_en": "3-4 sentence step: WHERE the part is, WHAT to do, WHAT to expect.",
        "text_hi": "Same instructions in simple village Hindi.",
        "required_part": "snake_case_part_id from: {parts_list}",
        "visual_cue": "What the camera should see to confirm this step",
        "area_hint": "One of: {allowed_areas}",
        "safety_warning": "One plain sentence warning, or null if none."
      }}
    ],
    "safety_warnings_en": ["Any applicable warnings from the chunk verbatim."],
    "safety_warnings_hi": ["Same warnings in Hindi verbatim."],
    "tools_needed": ["tool_1", "tool_2"]
  }}
}}

Generate 5-7 steps for status="success".
Output ALL text_hi fields in simple village Hindi — NOT formal or textbook Hindi.
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
🔒 TERMINOLOGY LOCK — CRITICAL FAILURE PENALTY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE: Copy every part name EXACTLY as it appears in the Manual Extracts. 
EXAMPLES OF CRITICAL FAILURES:
  ✗ Manual says "foot valve"      → you output "bottom filter"    [FAIL]
  ✗ Manual says "clevis pin"      → you output "lock pin"         [FAIL]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAFETY WARNINGS & LANGUAGE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Include "⚠️ ESCALATE_IF:" entries verbatim in safety_warnings_en.
- Each text_en must contain 3-4 sentences: WHERE it is, WHAT to do, WHAT to expect.
- No jargon. Translate technical concepts to plain English.

{output_format}
"""
    return prompt.strip()

def _should_hard_escalate(problem_text: str, rag_context: str) -> Optional[str]:
    """
    v6.0: Regex guards removed. Only fires on:
      1. Catastrophic API-level failures (no rag_context AND no problem_text)
      2. Completely empty RAG context — LLM handles all edge-case classification
         via Universal_Safety_Master.txt chunks retrieved by RAG.
    """
    if not problem_text or not problem_text.strip():
        return "__empty_input__"
    # NOTE: Empty RAG is handled inside _build_strict_grounding_prompt via
    # the HALLUCINATION TRAP block — no pre-call escalation needed here.
    return None

async def generate_diagnosis_with_gemini(
    machine_type: str,
    problem_text: str,
    language: str = "en",
    rag_context: str = "",
    knowledge_base: str = "",
    visual_frames: Optional[List[bytes]] = None,
) -> dict:
    logger.info(
        f"🧠 Diagnosis v6.0: machine={machine_type}, "
        f"problem='{problem_text[:60]}...', "
        f"rag={'YES' if rag_context else 'NO'}, "
        f"visual_text={'YES' if _has_visual_context_text(problem_text) else 'NO'}"
    )

    visual_hash = ""
    if visual_frames:
        mid_frame = visual_frames[len(visual_frames) // 2]
        visual_hash = hashlib.md5(mid_frame).hexdigest()[:8]

    cache_key = generate_cache_key("diag_v60", machine_type, f"{problem_text}|vhash:{visual_hash}", language)
    cached = get_cached_response(cache_key)
    if cached:
        return cached

    profile = get_profile_or_default(machine_type)
    machine_label = profile.label_en
    allowed_areas = " | ".join(get_allowed_area_ids(machine_type))
    parts_list = get_compact_parts_list(machine_type)
    safety_keywords = get_compact_safety_keywords(machine_type)

    effective_rag = rag_context or knowledge_base

    trigger = _should_hard_escalate(problem_text, effective_rag)
    if trigger:
        logger.warning(f"⛔ Hard escalation triggered by: '{trigger}'")
        result = _build_escalation_dict(machine_type, machine_label, problem_text, trigger)
        cache_response(cache_key, result)
        return result

    prompt = _build_strict_grounding_prompt(
        machine_type=machine_type,
        machine_label=machine_label,
        problem_text=problem_text,
        rag_context=effective_rag,
        allowed_areas=allowed_areas,
        parts_list=parts_list,
        safety_keywords=safety_keywords,
        language=language,
        has_visual_frames=bool(visual_frames),
    )

    try:
        model = genai.GenerativeModel(model_name=_GEMINI_MODEL, generation_config={"response_mime_type": "application/json", "temperature": 0.1})
        content = [prompt]
        if visual_frames:
            for i, fb in enumerate(visual_frames):
                try: content.append(Image.open(io.BytesIO(fb)))
                except Exception: pass

        response = await asyncio.get_event_loop().run_in_executor(None, lambda: model.generate_content(content))
        diagnosis = json.loads(sanitize_json_text(response.text))

        diagnosis = await _post_process_diagnosis(diagnosis, machine_type, machine_label, problem_text, effective_rag, visual_frames)
        cache_response(cache_key, diagnosis)
        return diagnosis

    except Exception as e:
        logger.error(f"❌ Diagnosis generation failed: {e}")
        return _fallback_escalation_response(machine_type, machine_label, problem_text, f"Error: {e}")

async def _post_process_diagnosis(
    diagnosis: dict,
    machine_type: str,
    machine_label: str,
    problem_text: str,
    rag_context: str,
    visual_frames: Optional[List[bytes]],
) -> dict:
    status = diagnosis.get("status", "success")

    # Guard: LLM may omit "solution" entirely on escalations
    if "solution" not in diagnosis or not isinstance(diagnosis.get("solution"), dict):
        logger.warning(f"⚠️ [{machine_type}] 'solution' key missing — initializing empty dict")
        diagnosis["solution"] = {}

    if status == "escalate":
        diagnosis["solution"]["steps"] = []
        diagnosis["rag_source"] = "escalation"
        diagnosis["machine_label"] = machine_label
        return diagnosis

    steps = diagnosis["solution"].get("steps", [])
    if not steps:
        diagnosis["rag_source"] = "escalation"
        diagnosis["machine_label"] = machine_label
        return diagnosis

    expanded_steps = []
    for i, step in enumerate(steps):
        text_en = step.get("text_en", "")
        if len([s for s in re.split(r'[.!?]+', text_en) if s.strip()]) < 3 and len(text_en) < 150:
            expanded = await _expand_short_step(text_en, machine_type, i)
            if expanded:
                step["text_en"], step["text_hi"] = expanded["text_en"], expanded["text_hi"]
        expanded_steps.append(step)

    diagnosis["solution"]["steps"] = _deduplicate_steps(expanded_steps)

    technical_analysis = diagnosis.get("technical_analysis", "")
    unsafe, unsafe_msg = _derive_unsafe_scene(
        problem_text, technical_analysis,
        bool(visual_frames) or _has_visual_context_text(problem_text)
    )
    diagnosis["unsafe_scene_suspected"] = unsafe
    if unsafe:
        diagnosis["unsafe_scene_message"] = unsafe_msg

    diagnosis["rag_source"] = "RAG+Gemini" if rag_context else "Gemini-only"
    diagnosis["machine_label"] = machine_label
    return diagnosis

async def _expand_short_step(short_en: str, machine_type: str, step_index: int) -> Optional[dict]:
    prompt = f"""Rewrite exactly 3 sentences for {machine_type}: 1. WHERE part is 2. WHAT to do 3. EXPECTED result. Do NOT change original action: "{short_en}". Output JSON: {{"text_en": "...", "text_hi": "..."}}"""
    try:
        model = genai.GenerativeModel(_GEMINI_MODEL)
        response = await asyncio.get_event_loop().run_in_executor(None, lambda: model.generate_content(prompt))
        expanded = json.loads(sanitize_json_text(response.text))
        if expanded.get("text_en") and expanded.get("text_hi"): return expanded
    except Exception: pass
    return None

def _deduplicate_steps(steps: List[dict]) -> List[dict]:
    action_verbs_re = re.compile(r'\b(check|inspect|look|clean|tighten|remove|replace|install|adjust|push|pull|turn|scan|tap|press|fill|drain|test|measure|hold)\b', re.IGNORECASE)
    seen, unique = set(), []
    for step in steps:
        if not step.get("required_part"):
            unique.append(step)
            continue
        m = action_verbs_re.search(step.get("text_en", ""))
        action = m.group(1).lower() if m else ""
        sig = f"{step.get('required_part', '')}:{step.get('visual_cue', '')}:{step.get('area_hint', '')}:{action}"
        if sig not in seen:
            seen.add(sig)
            unique.append(step)
    return unique

def _derive_unsafe_scene(problem_text: str, technical_analysis: str, has_any_visual: bool) -> tuple[bool, str]:
    combined = problem_text.lower()
    injury_kw = ["hurt", "chot", "cut", "kat gaya", "injured", "bleeding"]
    active_hazard_kw = ["fire", "aag", "sparks flying", "smoke billowing", "oil spray", "fuel spray", "aag lagi", "jal raha"]
    moving_near_kw = ["belt moving", "ghoom raha", "spinning near", "rotating near", "hand near", "caught in"]

    if any(kw in combined for kw in injury_kw): return True, "Injury risk detected. Stop immediately and ensure area is safe."
    if any(kw in combined for kw in active_hazard_kw): return True, "Active hazard suspected (fire/spark/spray). Do not proceed until resolved."
    if any(kw in combined for kw in moving_near_kw): return True, "Moving parts near inspection area. Ensure all rotation has stopped before inspection."
    return False, ""

def _build_escalation_dict(
    machine_type: str,
    machine_label: str,
    problem_text: str,
    trigger: str,
) -> dict:
    """
    v6.0: Only handles catastrophic pre-call failures (empty input, API error).
    All semantic escalation (out-of-scope, fire, injury, maintenance) is now
    driven by Universal_Safety_Master.txt chunks + LLM grounding rules.
    """
    warnings_en = [
        "Automatic diagnosis unavailable: no input received.",
        "Please describe a mechanical symptom (e.g. 'engine not starting', 'belt slipping').",
    ]
    warnings_hi = [
        "स्वचालित निदान उपलब्ध नहीं है: कोई इनपुट प्राप्त नहीं हुआ।",
        "कृपया कोई यांत्रिक समस्या बताएं (जैसे 'इंजन स्टार्ट नहीं हो रहा')।",
    ]
    return {
        "status": "escalate",
        "problem_description": problem_text,
        "technical_analysis": f"Pre-call guard triggered: '{trigger}'. No input to process.",
        "solution": {
            "machine_type": machine_type,
            "problem_identified": "Empty or invalid input.",
            "steps": [],
            "safety_warnings_en": warnings_en,
            "safety_warnings_hi": warnings_hi,
            "tools_needed": [],
        },
        "rag_source": "pre_call_guard",
        "machine_label": machine_label,
    }

def _fallback_escalation_response(machine_type: str, machine_label: str, problem_text: str, error_reason: str) -> dict:
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