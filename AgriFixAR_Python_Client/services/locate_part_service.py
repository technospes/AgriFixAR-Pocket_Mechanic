"""
services/locate_part_service.py
AR Part Location — Gemini Vision returns normalised bbox.

Called by /locate_part endpoint (NEW) for the AR guidance loop.
This is SEPARATE from verify_step_with_gemini (which checks correctness).
This service answers ONE question: "Where is this part in the image?"

Token budget per call:
  Input:  ~493 tokens (image 258 + prompt ~235)
  Output: ~80  tokens (JSON only)
  Total:  ~573 tokens ≈ $0.00007

Anti-hallucination design:
  - 4 reject flags (boolean). If ANY is true → found=false → no arrow shown.
  - Gemini must CONFIRM the part before returning bbox.
  - confidence < 0.72 → caller treats as not found.
  - If Gemini contradicts itself (found=true + reject flag=true) →
    backend forces found=false before returning to Flutter.
"""

from __future__ import annotations
import asyncio
import json
import logging
import io
from typing import Optional

import google.generativeai as genai
from PIL import Image

import time as _time
from utils.helpers import sanitize_json_text
from utils.machine_registry import get_area_farmer_description, get_profile

# ── Server-side bbox cache ────────────────────────────────────────────────────
# Keyed by (machine_type, required_part, area_hint, ip).
# Stores the last valid detection result + timestamp.
# If age < _BBOX_CACHE_TTL_S, the cached result is returned without calling Gemini.
# This is the primary Gemini-call reducer: ~80-90% of locate requests
# arrive within 1.5 s of the previous identical request from the same user.
# TTL set to 0.8 s (not 1.5 s): at a 1 s polling interval this means only
# the IMMEDIATE next tick after a detection hits the cache. Any call arriving
# > 0.8 s later — e.g. after the farmer moves the phone — calls Gemini fresh.
# A 1.5 s TTL would cache across two ticks and return a stale bbox position
# when the camera has moved significantly between them.
_BBOX_CACHE_TTL_S: float = 0.8
_bbox_cache: dict = {}   # key → {result, ts, hits}

logger = logging.getLogger(__name__)

_GEMINI_MODEL    = "models/gemini-2.5-flash"
_CONF_THRESHOLD  = 0.82          # FIX 2: raised from 0.72 — LLM bboxes need higher bar
_MAX_IMAGE_PX    = 512           # Resize before sending — keeps tokens low
_MAX_IMAGE_DIM   = _MAX_IMAGE_PX

# ── Area-side hints — tell Gemini WHERE to look if not visible ────────────────
# Maps area_hint → short human direction.  Used in camera_guidance generation.
_AREA_DIRECTIONS: dict[str, str] = {
    "suction_side":     "the inlet/suction side of the pump",
    "discharge_side":   "the outlet/discharge side of the pump",
    "pump_body":        "the main body of the pump",
    "coupling_area":    "the coupling/shaft between motor and pump",
    "engine_bay":       "the engine compartment",
    "transmission":     "the gearbox/transmission area",
    "fuel_system":      "the fuel tank and filter area",
    "electrical_panel": "the electrical panel or control box",
    "top_cover":        "the top cover of the machine",
    "front_panel":      "the front face of the machine",
    "rear_panel":       "the rear/back of the machine",
    "left_side":        "the left side of the machine",
    "right_side":       "the right side of the machine",
    "underside":        "the bottom of the machine",
}


# ── Similar-part exclusion map ────────────────────────────────────────────
# When Gemini is asked to find part X, it must NOT accept these visually
# similar but functionally different parts. Adding a part here adds one
# sentence of context to the prompt — zero extra token cost at inference.
_SIMILAR_PARTS_MAP: dict[str, list[str]] = {
    "suction_pipe_joint":   ["discharge pipe joint", "pressure pipe", "outlet hose",
                             "return line", "random hose"],
    "priming_plug":         ["drain plug", "grease nipple", "breather cap",
                             "oil filler cap"],
    "foot_valve":           ["check valve", "pressure relief valve", "gate valve"],
    "capacitor":            ["relay", "contactor", "junction box", "motor terminal"],
    "fuel_filter":          ["oil filter", "air filter", "hydraulic filter"],
    "air_filter":           ["oil filter", "fuel filter", "pre-cleaner bowl"],
    "clutch_pedal":         ["brake pedal", "accelerator pedal", "foot rest"],
    "battery_terminal":     ["fuse holder", "relay socket", "wire connector"],
    "shear_bolt":           ["blade bolt", "gearbox bolt", "hex bolt"],
    "mcb_switch":           ["isolator switch", "contactor", "relay"],
}


async def locate_part_with_gemini(
    image_bytes: bytes,
    required_part: str,
    area_hint: str,
    machine_type: str,
    attempt_count: int = 1,
    language: str = "en",
    frame_id: int = 0,
    search_roi: tuple[float, float, float] | None = None,
) -> dict:
    """
    search_roi: (cx, cy, margin) — normalised coords of last known bbox centre.
    When provided, tells Gemini to focus its search in that region.
    This reduces false positives and speeds up reasoning on cluttered frames.
    Margin is half-width of the search window (e.g. 0.30 = search ±30% of image).
    """
    """
    Ask Gemini Vision to locate a specific part and return its bounding box.

    Returns a dict:
      found=True:
        {
          "found": true,
          "bbox": [cx, cy, w, h],   # normalised 0.0–1.0, cx/cy = centre
          "confidence": 0.85,
          "part_description": "round red plug on upper-left of pump body",
          "camera_guidance": null
        }

      found=False:
        {
          "found": false,
          "bbox": null,
          "confidence": 0.0,
          "part_description": null,
          "camera_guidance": "Turn to the back of the machine — priming plug is there"
        }
    """
    # ── Server-side bbox cache check ─────────────────────────────────────────
    _cache_key = f"{machine_type}|{required_part}|{area_hint}"
    _now_ts    = _time.monotonic()
    _cached    = _bbox_cache.get(_cache_key)
    if _cached and (_now_ts - _cached["ts"]) < _BBOX_CACHE_TTL_S:
        _cached["hits"] += 1
        _cached["result"]["frame_id"] = frame_id  # echo caller's frame_id
        logger.info(
            f"⚡ locate_part CACHE HIT key={_cache_key} "
            f"age={_now_ts - _cached['ts']:.2f}s hits={_cached['hits']} "
            f"conf={_cached['result']['confidence']} — skipped Gemini"
        )
        return _cached["result"]

    logger.info(
        f"🎯 locate_part: machine={machine_type} part={required_part} "
        f"area={area_hint} attempt={attempt_count}"
    )

    try:
        resized_bytes = await _resize_image(image_bytes)
        image = Image.open(io.BytesIO(resized_bytes))

        area_desc      = get_area_farmer_description(machine_type, area_hint, language)
        area_direction = _AREA_DIRECTIONS.get(area_hint, f"the {area_hint.replace('_', ' ')}")
        part_readable  = required_part.replace("_", " ")
        lang_note      = "camera_guidance in Hindi.\n" if language == "hi" else ""

        # ROI hint — narrows Gemini's search to the expected region
        roi_block = ""
        if search_roi is not None:
            roi_cx, roi_cy, roi_margin = search_roi
            roi_block = (
                f"\nROI HINT: The part was last seen near (cx={roi_cx:.2f}, cy={roi_cy:.2f}) "
                f"in normalised image coordinates. Focus your search within "
                f"±{roi_margin:.2f} of that position before searching elsewhere.\n"
            )

        # Build exclusion list — common visually similar parts to reject
        exclusions = _SIMILAR_PARTS_MAP.get(required_part, [])
        excl_block = (
            f'\nDO NOT identify any of these as the target: {", ".join(exclusions)}.\n'
            if exclusions else ""
        )

        prompt = f"""\
Machine: {machine_type}. Find ONLY: "{part_readable}" in {area_hint} ({area_desc}).
{lang_note}{roi_block}{excl_block}
IDENTITY CHECK — before setting found=true, confirm ALL of these:
  a. The visible part matches "{part_readable}" in shape, position and function.
  b. It is located in {area_hint} — not in a different section of the machine.
  c. At least 60% of the part is unobstructed and clearly visible.
{excl_block}
REJECT FLAGS — set true if any apply, and force found=false:
  part_not_visible     — cannot identify "{part_readable}" anywhere in frame
  wrong_area           — camera shows different machine section than {area_hint}
  image_too_blurry     — image out of focus, dark, or motion-blurred
  part_behind_machine  — part is likely on the other side/face of the machine
  wrong_part_detected  — a SIMILAR but DIFFERENT part is visible (not "{part_readable}")
  glare_detected       — metal glare / reflection covers >15% of the visible part area
  part_occluded        — part visibility < 60% (blocked by another component or hand)

BBOX RULES:
  • bbox = [cx, cy, w, h] normalised 0.0–1.0, cx/cy = centre of part.
  • bbox edges must not touch image boundary (all of cx±w/2, cy±h/2 must be in (0,1)).
  • part_visibility_pct: integer 0–100 estimating how much of the part is unobstructed.

CAMERA GUIDANCE — one short imperative (<12 words) for the farmer:
  • Wrong area  → "Point camera at {area_direction}"
  • Too close   → "Move phone back — part fills too much of frame"
  • Too far     → "Move closer to the {machine_type}"
  • Behind      → "Turn to the other side of the {machine_type}"
  • Glare       → "Tilt phone slightly to reduce reflection"
  • Occluded    → "Remove the obstruction and retry"

Return ONLY this JSON (no markdown, no preamble):
{{
  "found": true|false,
  "bbox": [cx, cy, w, h] or null,
  "confidence": 0.0-1.0,
  "part_description": "<colour, shape, exact location>",
  "camera_guidance": "<12 words max>",
  "part_visibility_pct": 0-100,
  "part_not_visible": true|false,
  "wrong_area": true|false,
  "image_too_blurry": true|false,
  "part_behind_machine": true|false,
  "wrong_part_detected": true|false,
  "glare_detected": true|false,
  "part_occluded": true|false
}}"""

        model    = genai.GenerativeModel(_GEMINI_MODEL)
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model.generate_content([prompt, image])
        )

        raw = json.loads(sanitize_json_text(response.text))

        # ── Anti-hallucination gate ───────────────────────────────────────────
        # If ANY reject flag is true, force found=false regardless of what
        # Gemini said in the "found" field. This prevents a hallucinated bbox
        # from ever reaching Flutter.
        reject_flags = [
            raw.get("part_not_visible",    False),
            raw.get("wrong_area",          False),
            raw.get("image_too_blurry",    False),
            raw.get("part_behind_machine", False),
            raw.get("wrong_part_detected", False),  # similar-but-wrong part
            raw.get("glare_detected",      False),  # metal glare covers part
            raw.get("part_occluded",       False),  # visibility < 60%
        ]
        any_rejected = any(reject_flags)

        # Visibility gate — reject even if flags clean, visibility too low
        vis_pct = int(raw.get("part_visibility_pct", 100))
        if vis_pct < 60 and raw.get("found", False):
            any_rejected = True
            logger.info(f"locate_part: rejected — visibility={vis_pct}% < 60%")

        raw_found = raw.get("found", False) and not any_rejected
        raw_conf  = float(raw.get("confidence", 0.0))

        # Second gate: confidence below threshold → not found
        if raw_conf < _CONF_THRESHOLD:
            raw_found = False

        bbox = None
        if raw_found:
            raw_bbox = raw.get("bbox")
            if (isinstance(raw_bbox, list) and len(raw_bbox) == 4
                    and all(isinstance(v, (int, float)) for v in raw_bbox)):
                cx, cy, w, h = (float(v) for v in raw_bbox)

                # FIX 5: Strict coordinate validation — all values strictly in (0,1)
                # LLMs sometimes return -0.1, 1.2, or 0.0 (edge) — all rejected.
                coords_ok = (0.0 < cx < 1.0 and 0.0 < cy < 1.0
                             and 0.0 < w < 1.0 and 0.0 < h < 1.0)

                # FIX 9: Size sanity — part must fill 3%–60% per axis,
                # and 2%–60% of total image area.
                area_frac = w * h
                # Area threshold lowered 0.02 → 0.009:
                # clutch_pedal w=0.10 h=0.18 area=0.018 was rejected.
                # Individual w/h >= 0.03 already prevent degenerate boxes.
                size_ok   = (0.03 <= w <= 0.60 and 0.03 <= h <= 0.60
                             and 0.009 <= area_frac <= 0.60)

                # Bbox must not spill outside image AND not touch the edge
                # (edge-touching bbox usually means part is cut off or machine
                # is partially out of frame → unreliable detection)
                _edge_margin = 0.01  # 1% buffer from edge
                bounds_ok = (
                    cx - w / 2 >= _edge_margin and
                    cx + w / 2 <= 1.0 - _edge_margin and
                    cy - h / 2 >= _edge_margin and
                    cy + h / 2 <= 1.0 - _edge_margin
                )

                if coords_ok and size_ok and bounds_ok:
                    bbox = [cx, cy, w, h]
                else:
                    raw_found = False
                    bbox = None
                    logger.warning(
                        f"⚠️  locate_part bbox rejected "
                        f"coords={coords_ok} size={size_ok} bounds={bounds_ok} "
                        f"cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}"
                    )

        # Build camera_guidance — always non-null
        guidance = raw.get("camera_guidance") or _default_guidance(
            required_part, area_hint, machine_type,
            raw.get("part_behind_machine", False),
            raw.get("wrong_area", False),
            raw.get("image_too_blurry", False),
            language,
        )

        result = {
            "found":              raw_found,
            "bbox":               bbox,
            "confidence":         round(raw_conf, 3) if raw_found else 0.0,
            "part_description":   raw.get("part_description") if raw_found else None,
            "camera_guidance":    guidance,
            "part_visibility_pct": vis_pct,
            "frame_id":           frame_id,  # echoed back so Flutter can discard stale
            "reject_flags": {
                "part_not_visible":    raw.get("part_not_visible",    False),
                "wrong_area":         raw.get("wrong_area",           False),
                "image_too_blurry":   raw.get("image_too_blurry",     False),
                "part_behind_machine":raw.get("part_behind_machine",  False),
                "wrong_part_detected":raw.get("wrong_part_detected",  False),
                "glare_detected":     raw.get("glare_detected",       False),
                "part_occluded":      raw.get("part_occluded",        False),
            },
        }

        flag_log = [k for k, v in result["reject_flags"].items() if v]
        logger.info(
            f"{'✅' if raw_found else '❌'} locate_part: found={raw_found} "
            f"conf={raw_conf:.2f} flags={flag_log or 'none'}"
        )
        # Cache successful detections so subsequent calls within 1.5 s skip Gemini
        if raw_found:
            _bbox_cache[_cache_key] = {"result": dict(result), "ts": _now_ts, "hits": 0}
            logger.debug(f"\U0001f4be locate_part cached key={_cache_key}")
        return result

    except json.JSONDecodeError as exc:
        logger.error(f"❌ locate_part JSON error: {exc}")
    except Exception as exc:
        logger.error(f"❌ locate_part error: {exc}")

    return _fallback_not_found(required_part, area_hint, machine_type, language)


def _default_guidance(
    part: str,
    area_hint: str,
    machine: str,
    behind: bool,
    wrong_area: bool,
    blurry: bool,
    lang: str,
) -> str:
    """Deterministic fallback guidance — never empty, never hallucinates."""
    part_r = part.replace("_", " ")
    area_r = _AREA_DIRECTIONS.get(area_hint, area_hint.replace("_", " "))

    if blurry:
        return ("कैमरा स्थिर रखें और करीब लाएं" if lang == "hi"
                else "Hold still and move closer — camera is out of focus")
    if behind:
        return (f"मशीन को पलटें — {part_r} पीछे की तरफ है" if lang == "hi"
                else f"Turn the {machine} around — {part_r} is on the other side")
    if wrong_area:
        return (f"कैमरा {area_r} की तरफ करें" if lang == "hi"
                else f"Point camera at {area_r}")
    return (f"{part_r} ढूंढने के लिए कैमरा {area_r} पर लाएं" if lang == "hi"
            else f"Move camera to {area_r} to find the {part_r}")


def _fallback_not_found(part: str, area: str, machine: str, lang: str) -> dict:
    return {
        "found":            False,
        "bbox":             None,
        "confidence":       0.0,
        "part_description": None,
        "camera_guidance":  _default_guidance(part, area, machine, False, False, False, lang),
        "reject_flags":     {"part_not_visible": True, "wrong_area": False,
                             "image_too_blurry": False, "part_behind_machine": False},
    }


async def _resize_image(image_bytes: bytes) -> bytes:
    """Resize to max 512px — keeps Gemini token cost at ~258 per image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width > _MAX_IMAGE_DIM or img.height > _MAX_IMAGE_DIM:
            img.thumbnail((_MAX_IMAGE_DIM, _MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
            return buf.getvalue()
        return image_bytes
    except Exception as exc:
        logger.warning(f"locate_part image resize failed: {exc}")
        return image_bytes