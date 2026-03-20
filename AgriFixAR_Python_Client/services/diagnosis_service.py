from __future__ import annotations
import asyncio
import json
import logging
import re
import google.generativeai as genai
from PIL import Image
from utils.helpers import sanitize_json_text, generate_cache_key, get_cached_response, cache_response
from utils.machine_registry import (
    get_profile_or_default,
    get_diagnostic_context,
    get_safety_warnings,
    get_farmer_intro,
    get_allowed_area_ids,
    get_compact_parts_list,
    is_electric_machine,
    is_tractor_attachment,
    get_compact_safety_keywords,
)

logger = logging.getLogger(__name__)
_GEMINI_MODEL = "models/gemini-2.5-flash"


# ── Step-writing rules ────────────────────────────────────────────────────────
# Goal: farmer with zero mechanical background understands every step alone.
# Token approach: one concrete BAD→GOOD example (highest compliance per token),
# then three numbered rules that cover the four things every step must contain.
# Total: ~110 tokens — vs old 85-tok block that produced 1-liner steps.
# The extra ~25 tokens recover themselves by eliminating confusion retries.
_LANG_RULES = """\
Farmer has ZERO mechanical training. Write steps they can follow completely alone.

RULES:
1. Every text_en: 3–4 sentences minimum. Must include: (a) WHERE the part is \
(colour + shape + what it sits next to), (b) WHAT to do with hands, \
(c) WHAT to see/hear/feel when done correctly.
2. No jargon. Replace: corrosion→"white powder on metal" | fraying→"wire looks fuzzy/split" \
| MCB_tripped→"the small switch on the box has popped upward" | \
belt_tension→"push the belt with one finger — it should bounce back, not sag" \
| cavitation→"rattling noise from the pump body".
3. text_hi: same 3–4 sentences in simple village Hindi — NOT formal/textbook Hindi.

BAD (1 line, abandoned): "Scan the battery terminal on the left side."
GOOD (guided, 3 sentences): "Open the big square metal cover on the left side of your \
tractor near the steering — you will find a large black box with two thick cables \
clamped to its top. Look closely at the silver clamps where the cables are bolted — \
if you see white or grey powder around them, that is the problem. Hold the camera \
steady so the clamps fill the white box, then tap Analyze."

5–7 steps. Simplest/safest first. area_hint must be one of the allowed values. \
safety_warning: one plain sentence or null."""


def _build_electric_note(machine_type: str) -> str:
    if is_electric_machine(machine_type):
        return "ELECTRIC MACHINE: Step 1 must verify main power is OFF. Never suggest touching live components.\n"
    if is_tractor_attachment(machine_type):
        return "TRACTOR ATTACHMENT: Step 1 must verify PTO disengaged+raised. Check shear_bolt early.\n"
    return ""


async def _expand_short_step(short_en: str, machine_type: str, step_index: int) -> dict | None:
    """
    Expand a short step into 3 sentences WITHOUT changing the repair action.
    Only adds WHERE the part is, WHAT the farmer does, and WHAT result to expect.
    Returns {"text_en": "...", "text_hi": "..."} or None on failure.

    Safety guarantee: the model is explicitly forbidden from changing the action.
    The original short_en is injected verbatim and the model must build around it,
    not replace it — so a "Check belt tension" can never become "Remove the belt".
    """
    prompt = f"""A repair guide step for a {machine_type} was written too briefly:

ORIGINAL STEP: "{short_en}"

Rewrite it as EXACTLY 3 sentences.

STRICT RULES:
• DO NOT change, remove, or reorder the repair action described above.
• Sentence 1 — WHERE the part is: describe its colour, shape, and what it sits next to in plain words a farmer understands.
• Sentence 2 — WHAT action the farmer must do with their hands (keep the EXACT same action as the original step).
• Sentence 3 — WHAT result they should see, hear, or feel when done correctly.
• No technical jargon. No new steps. No safety disclaimers.

Return ONLY this JSON (no markdown):
{{
  "text_en": "<3 sentences in plain English>",
  "text_hi": "<same 3 sentences in simple village Hindi — NOT formal Hindi>"
}}"""

    try:
        model = genai.GenerativeModel(_GEMINI_MODEL)
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model.generate_content(prompt)
        )
        expanded = json.loads(sanitize_json_text(response.text))
        if expanded.get("text_en") and expanded.get("text_hi"):
            logger.info(f"🔧 Step {step_index + 1} expanded safely ({len(short_en)} → {len(expanded['text_en'])} chars)")
            return expanded
    except Exception as exc:
        logger.warning(f"⚠️  Step {step_index + 1} expansion failed: {exc}")
    return None


async def generate_diagnosis_with_gemini(
    machine_type: str,
    problem_text: str,
    language: str = "en",
    rag_context: str = "",
    knowledge_base: str = "",
    visual_frames: list[bytes] | None = None,
) -> dict:
    """
    Generate a step-by-step camera repair guide for any supported farm machine.

    When `visual_frames` is supplied (1–3 JPEG bytes, early/mid/late from CLIP)
    the call is made as a multimodal Gemini request — all frames + text in one
    call, no extra API charge.

    Sending 3 frames instead of 1 gives Gemini temporal context:
      • If all 3 show water flowing → machine is RUNNING, diagnose mechanical fault
      • If frames 1-2 show motion but frame 3 is still → machine cut out mid-video
      • If all 3 are stopped → machine is dead, check power/electrical

    This directly fixes the transcription-mismatch problem (e.g. farmer says
    "motor jammed" but Whisper heard "not working" — Gemini sees flowing water
    and generates the correct mechanical-jam steps, not dead-power steps).
    """
    logger.info(f"🧠 Diagnosis: {machine_type} — {problem_text[:60]}..."
                + (f" [multimodal {len(visual_frames)}-frame]" if visual_frames else " [text-only]"))

    # visual_hash: phash of mid frame — ensures "pump dead" and "water flowing"
    # never share a cached answer even when machine_type + text are identical.
    # Folded into problem_text so generate_cache_key signature stays unchanged.
    import hashlib as _hashlib
    _mid = visual_frames[len(visual_frames) // 2] if visual_frames else None
    _visual_hash = _hashlib.md5(_mid).hexdigest()[:8] if _mid else ""
    _cache_problem = f"{problem_text}|vhash:{_visual_hash}"
    cache_key = generate_cache_key("diag", machine_type, _cache_problem, language)
    cached = get_cached_response(cache_key)
    if cached:
        logger.info(f"✅ Using cached diagnosis [visual_hash={_visual_hash or 'none'}]")
        return cached

    rag_source = "RAG+Gemini" if rag_context else "Gemini-only"
    profile      = get_profile_or_default(machine_type)
    machine_label = profile.label_en
    diag_ctx     = get_diagnostic_context(machine_type)
    allowed      = " | ".join(get_allowed_area_ids(machine_type))
    parts_list   = get_compact_parts_list(machine_type)
    safety_kw    = get_compact_safety_keywords(machine_type)
    electric_note = _build_electric_note(machine_type)

    # RAG / knowledge context — kept verbatim (accuracy-critical)
    if rag_context:
        ctx_block = f"MANUAL EXTRACTS (use these first):\n{rag_context}"
    elif knowledge_base:
        ctx_block = f"KNOWLEDGE BASE:\n{knowledge_base}"
    else:
        ctx_block = ""

    # Hindi instruction — single compact line
    hi_note = "Populate ALL text_hi fields in simple Hindi.\n" if language == "hi" else \
              "Populate text_hi fields in simple Hindi for every step.\n"

    # Safety warnings — EN only, compact format (saves ~59 tok vs EN+HI injection)
    safety_en = get_safety_warnings(machine_type, "en")
    safety_en_compact = "; ".join(safety_en)  # single line vs multiline list

    # ── Visual grounding block (only when frames are available) ──────────────
    # Placed at the TOP of the prompt before all other context — Gemini anchors
    # on visual truth first, then reads the (potentially noisy) transcript.
    # 3 frames (early/mid/late) give temporal context CLIP already verified.
    if visual_frames:
        n = len(visual_frames)
        frame_desc = "3 frames (early / mid / late)" if n == 3 else \
                     "2 frames (early / late)" if n == 2 else "1 frame (mid)"
        visual_block = f"""\
VISUAL EVIDENCE — {frame_desc} from the farmer's video are attached to this message.
CRITICAL INSTRUCTION: Study ALL attached images carefully BEFORE reading the transcript.

TEMPORAL REASONING (use all {n} frame(s) together):
  • All frames show machine RUNNING (water flowing / parts moving / lights on):
      → Complaint is about PERFORMANCE or NOISE — NOT about the machine being dead.
      → Do NOT generate steps to check power cables, breakers, or safety switches.
  • Frames 1–2 show motion but last frame is stopped:
      → Machine cut out mid-operation. Check thermal protection, overload, fuel.
  • All frames show machine OFF/STOPPED (no water, no motion, no lights):
      → Complaint is about FAILURE TO START — check power and electrical systems.
  • IF THE TRANSCRIPT CONTRADICTS WHAT YOU SEE, TRUST THE IMAGES.
      Example: transcript="not working", all frames show water flowing
      → Diagnose a RUNNING machine with a mechanical fault (jam/noise/low pressure).
      → NEVER generate dead-machine power-check steps in this case.

"""
    else:
        visual_block = ""

    prompt = f"""{visual_block}You are an expert farm machinery mechanic writing a camera-guided repair walkthrough.
{electric_note}
MACHINE: {machine_label} ({machine_type})
KNOWLEDGE: {diag_ctx}
SAFETY: {safety_kw}
{ctx_block}
PROBLEM (farmer's transcript — may be imprecise due to accent/noise): {problem_text}

{hi_note}{_LANG_RULES}

DIAGNOSTIC REASONING & ACCURACY (CRITICAL):
Before generating steps, analyze the problem and determine the most likely subsystem.
Classify the issue into one of these categories:
- electrical_starting
- fuel_supply
- engine_mechanical
- cooling_system
- transmission
- hydraulic_system
- sensor_or_electrical

Write this classification and your mechanical reasoning in the "technical_analysis" field.
Each step MUST directly help diagnose or fix the stated problem based on this reasoning.
Do NOT include generic maintenance checks UNLESS directly related to the symptom.
Prefer checking the most common mechanical failure points first. Order steps from easiest and safest to check → hardest.

STEP TYPES (CRITICAL RULE):
Choose the correct step_type for the physical reality of the task:
1. "visual": Farmer must point the camera at a visible part. Choose parts that can be seen with a phone camera. Do not use internal engine components that require disassembly.
2. "inspection": Physical check (e.g., pulling a belt, checking oil).
3. "action": A manual task (e.g., cleaning a filter, turning a wrench). Farmer just taps Done.
4. "observation": Listening/feeling (e.g., hearing a click, feeling heat).

WORKFLOW LIMITS & STRUCTURE (STRICT):
- Maximum 5 steps total.
- STEP ID RULE: Use sequential step IDs: s1, s2, s3, s4, s5. Do not skip numbers.
- Step 1 MUST ALWAYS be step_type "visual" to help the farmer locate the correct machine part.
- At most 2 "inspection" steps.
- At most 1 "observation" step.
- The remaining steps must be "visual" or "action".
- Ideal flow: visual (locate) -> inspection (check condition) -> action (fix) -> observation (confirm) -> action (finalise).

ROUTING & OPTIONS:
If step_type is "inspection" or "observation":
- "question_en" is REQUIRED.
- "options" MUST contain 2–4 items.
Use "next_step" with a step_id to jump, or set "next_step": null for linear progression.

Return ONLY this JSON (no markdown):
{{
  "status": "success",
  "problem_description": "<1 plain sentence>",
  "technical_analysis": "<State subsystem category here, then 2-3 sentences of mechanical reasoning>",
  "solution": {{
    "status": "ready",
    "machine_type": "{machine_type}",
    "problem_identified_en": "<clear short title in English>",
    "problem_identified_hi": "<clear short title in Hindi>",
    "steps": [
      {{
        "step_id": "s1",
        "step_type": "<visual | inspection | action | observation>",
        "text": "<copy of text_en>",
        "step_title_en": "<short action title>",
        "step_title_hi": "<same action title in simple Hindi>",
        "text_en": "<3–4 sentences: WHERE part is + WHAT to do>",
        "text_hi": "<same in simple village Hindi>",
        "visual_cue": "<snake_case_part_id or null>",
        "ar_model": "<part.obj or null>",
        "required_part": "<snake_case_part_id>",
        "area_hint": "<one of: {allowed}>",
        "safety_warning": "<one plain sentence or null>",
        "question_en": "<ONLY IF inspection/observation. e.g., 'What is the condition of the belt?'>",
        "question_hi": "<Question in Hindi>",
        "options": [
          {{
            "id": "a",
            "label_en": "Looks fine",
            "label_hi": "ठीक है",
            "next_step": null
          }}
        ]
      }}
    ],
    "safety_warnings_en": {json.dumps(safety_en)},
    "safety_warnings_hi": [],
    "tools_needed": ["<plain tool name>"]
  }}
}}
PARTS (use as required_part): {parts_list}"""

    try:
        import io as _io
        import base64 as _base64
        model = genai.GenerativeModel(_GEMINI_MODEL)

        # ── Build request: multimodal when frames are available ───────────────
        # All 1–3 CLIP frames are sent as inline_data parts BEFORE the text
        # prompt so Gemini anchors on visual truth first.
        # Cost: ~258 vision tokens per 512×512 frame × up to 3 frames = ~774
        # extra tokens per call (≈ $0.00009 at gemini-2.5-flash rates).
        # Everything — all frames + full text prompt — is ONE generate_content()
        # call. Zero extra API charges.
        if visual_frames:
            parts: list[dict] = []
            for i, frame_bytes in enumerate(visual_frames):
                frame_img = Image.open(_io.BytesIO(frame_bytes))
                frame_img.thumbnail((512, 512), Image.LANCZOS)
                buf = _io.BytesIO()
                frame_img.save(buf, format="JPEG", quality=85)
                label = ["early", "mid", "late"][i] if i < 3 else str(i)
                parts.append({"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": _base64.b64encode(buf.getvalue()).decode(),
                }})
                parts.append({"text": f"[Frame {i+1}/{len(visual_frames)} — {label}]"})
            parts.append({"text": prompt})
            content = [{"role": "user", "parts": parts}]
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: model.generate_content(content)
            )
            logger.info(f"🖼️  Diagnosis: MULTIMODAL call ({len(visual_frames)} CLIP frames + text)")
        else:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: model.generate_content(prompt)
            )
            logger.info("📝 Diagnosis: text-only call (no frames available)")
        json_text = sanitize_json_text(response.text)
        result = json.loads(json_text)

        if "solution" not in result or "steps" not in result["solution"]:
            raise ValueError("Missing solution/steps in response")

        # Deduplicate steps by required_part before any further processing.
        # Must run before expansion so we don't waste Gemini calls on steps
        # that are about to be dropped.
        result["solution"]["steps"] = _remove_duplicate_steps(result["solution"]["steps"])
        if len(result["solution"]["steps"]) < 3:
            raise ValueError(
                f"Only {len(result['solution']['steps'])} unique steps after deduplication — "
                "response quality too low; falling back"
            )

        # Post-process: validate area_hints, backfill safety_warnings_hi, sync text field
        allowed_list = get_allowed_area_ids(machine_type)
        safety_hi    = get_safety_warnings(machine_type, "hi")

        # Collect short steps for parallel expansion (one extra Gemini call per short step)
        expansion_tasks = []
        short_step_indices = []
        for i, step in enumerate(result["solution"]["steps"]):
            # Fix bad area_hint
            if step.get("area_hint") not in allowed_list:
                step["area_hint"] = allowed_list[0]
            # Keep legacy "text" field in sync with text_en
            if not step.get("text") and step.get("text_en"):
                step["text"] = step["text_en"]
            # Detect short steps — expand rather than just warn
            word_count = len(step.get("text_en", "").split())
            if word_count < 30:
                logger.warning(
                    f"⚠️  Step {i+1} text_en only {word_count} words — scheduling safe expansion"
                )
                expansion_tasks.append(_expand_short_step(step["text_en"], machine_type, i))
                short_step_indices.append(i)

        # Run all expansions in parallel to keep latency low
        if expansion_tasks:
            expansions = await asyncio.gather(*expansion_tasks, return_exceptions=True)
            for idx, expanded in zip(short_step_indices, expansions):
                if isinstance(expanded, dict) and expanded.get("text_en"):
                    step = result["solution"]["steps"][idx]
                    step["text_en"] = expanded["text_en"]
                    step["text_hi"] = expanded["text_hi"]
                    step["text"]    = expanded["text_en"]  # keep legacy field in sync
        # Fill Hindi safety warnings from registry (not generated — saves tokens)
        if not result["solution"].get("safety_warnings_hi"):
            result["solution"]["safety_warnings_hi"] = safety_hi

        result["rag_source"]    = rag_source
        result["machine_label"] = machine_label
        cache_response(cache_key, result)
        logger.info(f"✅ Diagnosis: {len(result['solution']['steps'])} steps [{rag_source}]")
        # ---------------------------------------------------------
        # 🟢 PURE DEBUG LOGGING (Zero impact on UI/UX)
        # ---------------------------------------------------------
        try:
            formatted_steps = json.dumps(
                result.get("solution", {}).get("steps", []), 
                indent=2, 
                ensure_ascii=False # Ensures Hindi characters print correctly in the terminal
            )
            logger.info(f"\n{'='*60}\n🤖 GENERATED STEPS LOG:\n{formatted_steps}\n{'='*60}")
        except Exception as log_exc:
            pass # If logging fails for any reason, fail silently. Never break the app!
        # ---------------------------------------------------------
        return result

    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(f"❌ Diagnosis parse error: {exc}")
    except Exception as exc:
        logger.error(f"❌ Diagnosis error: {exc}")

    return _fallback_diagnosis(machine_type, machine_label, problem_text,
                               get_safety_warnings(machine_type, "en"),
                               get_safety_warnings(machine_type, "hi"))


def _remove_duplicate_steps(steps: list[dict]) -> list[dict]:
    """
    Remove genuinely duplicate repair steps using a three-level signature.

    WHY three levels are needed
    ───────────────────────────
    Level 1 — required_part only (too aggressive):
        Drops "check battery voltage" AND "clean battery terminals" because
        both share required_part=battery_terminal. Legitimate revisits are lost.

    Level 2 — required_part + visual_cue + area_hint (better, your suggestion):
        Correctly keeps steps on the same part that point the camera at
        different components or different areas of the machine.
        Still misses cases where two steps share all three identifiers but
        perform genuinely different actions (check vs tighten vs clean).

    Level 3 — + action verb extracted from text_en (maximum accuracy):
        The first meaningful verb in text_en (check / clean / tighten /
        inspect / remove / replace …) is appended to the signature.
        Two steps are only considered duplicates when they share the same
        part, the same visual target, the same machine area, AND the same
        action verb — i.e. they are genuinely asking the farmer to do the
        same thing in the same place twice.

    Steps with no required_part (safety/intro steps with empty part IDs)
    are always kept — they cannot be deduplicated by part identity and
    dropping them silently would remove safety warnings.
    """
    # Common action verbs that appear at the start of repair step sentences.
    # Extracting the first match gives a stable, token-cheap action signal.
    _ACTION_VERBS = re.compile(
        r'\b(check|inspect|look|clean|tighten|loosen|remove|replace|install|adjust|'
        r'push|pull|turn|twist|open|close|start|stop|test|measure|look|hold|'
        r'scan|tap|press|fill|drain|spray|wipe|feel|listen|watch)\b',
        re.IGNORECASE,
    )

    def _action_verb(step: dict) -> str:
        """Return the first action verb found in text_en, or '' if none."""
        text = step.get("text_en") or step.get("text") or ""
        text = re.sub(r'^(now|next|then)\s*,?\s*', '', text)
        m = _ACTION_VERBS.search(text)
        return m.group(1).lower() if m else ""

    def _signature(step: dict) -> str:
        part      = step.get("required_part") or ""
        cue       = step.get("visual_cue")    or ""
        area      = step.get("area_hint")     or ""
        action    = _action_verb(step)
        return f"{part}:{cue}:{area}:{action}"

    seen_signatures: set[str] = set()
    unique_steps: list[dict]  = []

    for step in steps:
        part = step.get("required_part") or ""
        if not part:
            # No part ID → safety/intro step, always keep
            unique_steps.append(step)
            continue

        sig = _signature(step)
        if sig in seen_signatures:
            logger.warning(
                f"⚠️  Duplicate step removed — signature: {sig!r}"
            )
            continue

        seen_signatures.add(sig)
        unique_steps.append(step)

    return unique_steps


def _fallback_diagnosis(machine_type, machine_label, problem_text, safety_en, safety_hi):
    return {
        "status": "error",
        "problem_description": problem_text,
        "technical_analysis": f"Diagnosis generation failed for {machine_label}.",
        "solution": {
            "status": "error", "machine_type": machine_type,
            "problem_identified": problem_text, "steps": [],
            "safety_warnings_en": safety_en or ["Consult a certified mechanic."],
            "safety_warnings_hi": safety_hi or ["प्रमाणित मैकेनिक से संपर्क करें।"],
            "tools_needed": [],
        },
        "rag_source": "error",
        "machine_label": machine_label,
    }
