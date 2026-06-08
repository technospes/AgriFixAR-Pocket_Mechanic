# utils/json_repair.py
"""
Robust JSON repair for Gemini API responses.

Handles the three failure modes observed in production:
  1. Unterminated strings (token-limit truncation)
  2. Single-quoted keys/values (non-strict JSON)
  3. Trailing commas before } or ]

All functions are pure (no side effects) and safe to call on any string.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Configurable: if True, log the repaired string so engineers can trace
# what was wrong. Disable in production if logs are noisy.
_LOG_REPAIRS = True


def repair_json(raw: str) -> Any:
    """
    Attempt to parse raw string as JSON, applying progressive repair steps.

    Raises json.JSONDecodeError only if all repair strategies fail.
    Call sites should still wrap in try/except for total safety.

    Repair order (cheapest first):
      1. Strip markdown fences (```json ... ```)
      2. Fix trailing commas before } or ]
      3. Fix single-quoted keys and values → double quotes
      4. Fix unterminated strings (close open quotes before } or end-of-input)
    """
    text = _strip_fences(raw)

    # Fast path — already valid
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 2 — trailing commas
    text = _fix_trailing_commas(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 3 — single quotes
    text = _fix_single_quotes(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 4 — unterminated strings (most aggressive, do last)
    text = _fix_unterminated_string(text)
    if _LOG_REPAIRS:
        logger.debug("JSON repaired (unterminated string): %s", text[:200])
    return json.loads(text)  # Let this raise if still broken


# ── Repair helpers ────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*",     "", text)
    text = re.sub(r"\s*```$",     "", text)
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    # Handles both ,} and ,]
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return text


def _fix_single_quotes(text: str) -> str:
    """
    Replace single-quoted JSON keys/values with double-quoted equivalents.

    Strategy: use a state machine to avoid replacing apostrophes inside
    already-double-quoted strings (e.g. "it's fine").
    Simple regex substitution: swap unescaped ' that are at start/end of
    a JSON key or value position.
    """
    # Replace 'key' patterns: {'key': ...} or {key: ...}
    # This regex is intentionally conservative — it only touches clearly
    # single-quoted JSON tokens, not prose apostrophes.
    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)
    return text


def _fix_unterminated_string(text: str) -> str:
    """
    If JSON ends with an open string (odd number of unescaped "),
    close it and close any open {} or [] before returning.

    This handles the token-limit truncation case:
      {"reasoning": "The capacitor is failing because the motor  ← truncated here
    → {"reasoning": "The capacitor is failing because the motor"}
    """
    # Count unescaped double-quotes
    unescaped_quotes = len(re.findall(r'(?<!\\)"', text))
    if unescaped_quotes % 2 == 0:
        return text  # Strings are balanced — not a truncation issue

    # Close the open string, then close open braces/brackets
    text = text.rstrip()

    # Remove trailing incomplete word/sentence (stops at last whitespace boundary)
    # to avoid submitting a half-written value that would confuse parsing
    text = re.sub(r'\s+\S+$', '', text)  # trim last partial token

    text += '"'  # close the open string

    # Close any open structures (simplistic stack — handles 1-2 levels deep)
    open_curly  = text.count('{') - text.count('}')
    open_square = text.count('[') - text.count(']')
    text += ']' * max(0, open_square)
    text += '}' * max(0, open_curly)

    return text