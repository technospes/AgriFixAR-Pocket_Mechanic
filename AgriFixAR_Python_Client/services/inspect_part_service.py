from __future__ import annotations
import logging
from typing import Optional
from utils.vision_client import vision_call

logger = logging.getLogger(__name__)

INSPECT_PROMPT = """You are inspecting a SINGLE specific part on agricultural machinery.
The part has ALREADY been verified as the correct component.
Your ONLY job: determine if this part shows VISIBLE DAMAGE.

Target: {required_part}
Area: {area_hint}
Machine: {machine_label}

INSPECTION RULES:
1. Look for: cracks, fraying, bulging, corrosion, burns, leaks, breaks, bends,
   missing pieces, loose connections, melted plastic, exposed wires, rust.
2. If damage is visible, describe it in simple farmer-friendly terms.
3. If you can see the part clearly but it looks normal, say "healthy".
4. If the part is partially hidden, dirty, or in shadow, say "unclear".
5. If the wrong thing is in frame, say "wrong_target".
6. Do NOT confuse normal wear with damage — be conservative.

Return ONLY valid JSON:
{{
  "outcome": "healthy",
  "severity": "none",
  "damage_description": "",
  "damage_description_hi": "",
  "repairability": "unknown",
  "observations": [],
  "safety_concern": false,
  "safety_note": ""
}}"""

async def inspect_part_service(
    image_bytes: bytes,
    machine_type: str,
    required_part: str,
    area_hint: str = "engine_compartment",
    language: str = "en",
) -> dict:
    """Inspect a verified part for damage. Returns InspectionResult."""
    
    from utils.machine_registry import get_profile_or_default
    profile = get_profile_or_default(machine_type)
    
    prompt = INSPECT_PROMPT.format(
        required_part=required_part.replace("_", " "),
        area_hint=area_hint.replace("_", " "),
        machine_label=profile.label_en,
    )
    
    try:
        response = await vision_call(
            image_bytes=image_bytes,
            prompt=prompt,
            temperature=0.2,
        )
        
        from utils.json_repair import repair_json
        data = repair_json(response)
        
        return {
            "outcome": data.get("outcome", "unclear"),
            "severity": data.get("severity", "none"),
            "damage_description": data.get("damage_description", ""),
            "damage_description_hi": data.get("damage_description_hi", ""),
            "repairability": data.get("repairability", "unknown"),
            "observations": data.get("observations", []),
            "safety_concern": data.get("safety_concern", False),
            "safety_note": data.get("safety_note", ""),
            "confidence": data.get("confidence", 0.5),
        }
        
    except Exception as exc:
        logger.error("Inspect part failed: %s", exc)
        return {
            "outcome": "unclear",
            "severity": "none",
            "damage_description": "",
            "damage_description_hi": "",
            "repairability": "unknown",
            "observations": [],
            "safety_concern": False,
            "safety_note": "",
            "confidence": 0.0,
        }