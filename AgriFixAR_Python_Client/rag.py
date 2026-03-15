"""
rag.py — Optimized RAG pipeline for AgriFix
Drop this file next to main.py and replace the retrieve_rag_context()
call with retrieve_with_metadata_filter().

Features:
  - normalize_query()               — synonym normalization before embedding
  - retrieve_with_metadata_filter() — 3-pass filtered search
      Pass 1: machine_type AND ($or of all problem categories) — single DB call
      Pass 2: machine_type only
      Pass 3: global fallback
  - _rank_chunks()                  — composite score ranking (vector + structure boost - spec penalty)
  - _format_rag_context()           — minimal-token LLM context block
  - RAG_TOP_K = 4                   — keeps context tight
  - RAG_MIN_SCORE = 0.35            — rejects low-confidence chunks

Embedding cache lives in main.py — see main_patch.py STEP 6.
"""

import logging
import re
from typing import List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

RAG_TOP_K = 4          # Max chunks sent to Gemini
RAG_MIN_SCORE = 0.35   # Minimum cosine similarity — below this → rejected


# ── Synonym normalization ─────────────────────────────────────────────────────

# Maps colloquial / variant user words → canonical problem category keywords
# that match what the manuals use.  Extend freely.
SYNONYM_MAP = {
    # noise
    "sound":          "noise",
    "sounds":         "noise",
    "knocking":       "noise",
    "knock":          "noise",
    "banging":        "noise",
    "clicking":       "noise",
    "rattling":       "noise",
    "rattles":        "noise",
    "squealing":      "noise",
    "humming":        "noise",
    "grinding":       "noise",
    "vibrating":      "vibration",
    "shaking":        "vibration",
    "wobbling":       "vibration",
    # not starting
    "won't start":    "not starting",
    "wont start":     "not starting",
    "not starting":   "not starting",
    "dead":           "not starting",
    "no start":       "not starting",
    "doesn't start":  "not starting",
    "not working":    "not starting",
    # water / flow
    "no water":       "water flow",
    "low water":      "water flow",
    "water not coming": "water flow",
    "no flow":        "water flow",
    "low pressure":   "water flow",
    # overheating
    "too hot":        "overheating",
    "getting hot":    "overheating",
    "heating up":     "overheating",
    # leaking
    "dripping":       "leaking",
    "seeping":        "leaking",
    "oil leak":       "leaking oil",
    "fuel leak":      "leaking fuel",
    # smoke
    "fumes":          "smoke",
    "black smoke":    "smoke",
    "white smoke":    "smoke",
    # power
    "weak":           "power loss",
    "sluggish":       "power loss",
    "no power":       "power loss",
    # electrical
    "battery dead":   "electrical battery",
    "fuse blown":     "electrical fuse",
    "no electricity": "electrical",
    # fuel
    "out of fuel":    "fuel empty",
    "fuel problem":   "fuel",
    "diesel problem": "fuel diesel",
}


def normalize_query(query: str) -> str:
    """
    Replace colloquial / synonym words in the user query with canonical
    manual vocabulary before computing the embedding.

    Example:
        "pump makes knocking sound" → "pump makes noise noise"
        "motor won't start"         → "motor not starting"
    """
    normalized = query.lower()
    # Sort by length descending so multi-word phrases match before single words
    for phrase, replacement in sorted(SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        normalized = normalized.replace(phrase, replacement)
    return normalized


# ── Chunk ranking ─────────────────────────────────────────────────────────────

def _structure_boost(text: str) -> float:
    """
    Reward chunks that contain problem-diagnosis vocabulary.

    These are the chunks a repair technician actually needs — they describe
    what is wrong and what to do about it.  Pure spec chunks (bore, stroke,
    voltage ratings) don't mention these words and receive no boost.

    Weights are additive so a chunk with PROBLEM + CAUSE + FIX gets the
    full +0.65 boost, while a chunk with only FIX gets +0.20.

    Max possible return value: 0.65
    """
    t = text.lower()
    score = 0.0
    if "problem" in t:
        score += 0.25
    if "cause" in t:
        score += 0.20
    if "fix" in t or "what to do" in t:
        score += 0.20
    return score


def _spec_penalty(text: str) -> float:
    """
    Downweight chunks that look like specification tables.

    Spec chunks (cylinder bore, rated power, voltage ratings) are rarely
    useful for repair diagnosis.  We don't remove them from the result set
    entirely — they may still be the only available match on a global
    fallback — but we push them below actionable repair chunks.

    Returns -0.15 when any spec keyword is found, 0.0 otherwise.
    """
    t = text.lower()
    spec_keywords = [
        "specification", "rated power", "rpm", "bore", "stroke",
        "dimension", "weight", "capacity", "voltage",
    ]
    if any(k in t for k in spec_keywords):
        return -0.15
    return 0.0


def _rank_chunks(chunks_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """
    Re-order chunks using a composite score so the LLM always sees the most
    actionable repair content first.

    Formula
    -------
        final_score = vector_similarity * 0.75
                    + _structure_boost(text)   # up to +0.65
                    + _spec_penalty(text)       # 0 or -0.15

    The 0.75 weight on vector similarity preserves semantic relevance as the
    dominant signal while giving the structural boosts enough headroom to
    reorder chunks within a close similarity band.

    Example reordering
    ------------------
    Before (raw vector scores):
        0.81  Engine specification — bore 102 mm, stroke 120 mm
        0.79  PROBLEM: tractor not starting / CAUSE: battery / FIX: recharge

    After composite scoring:
        0.81 * 0.75 + 0     - 0.15 = 0.458   ← spec chunk
        0.79 * 0.75 + 0.65 + 0     = 1.2425  ← problem-fix chunk  ✓ ranks first

    Secondary sort: composite score descending (ties broken by higher score first).
    """
    def composite(item: Tuple[Document, float]) -> float:
        doc, vector_score = item
        return (
            vector_score * 0.75
            + _structure_boost(doc.page_content)
            + _spec_penalty(doc.page_content)
        )

    return sorted(chunks_with_scores, key=composite, reverse=True)


# ── Context formatter ─────────────────────────────────────────────────────────

def _format_rag_context(ranked_chunks: List[Tuple[Document, float]]) -> str:
    """
    Format retrieved chunks into a compact, LLM-friendly context block.

    Token budget: ~150–250 tokens for 4 cause_fix chunks.
    Format avoids embedding raw metadata blobs into the prompt.
    """
    parts = []
    for doc, score in ranked_chunks:
        source = doc.metadata.get("source_file", "manual")
        text = doc.page_content.strip()

        # Try to extract CAUSE and FIX lines for a cleaner presentation
        cause_match = re.search(
            r"(?:What'?s?\s+Happening|CAUSE|Possible\s+Cause|SYMPTOM)\s*[:\-]?\s*(.+?)(?=\n|$)",
            text, re.IGNORECASE,
        )
        fix_match = re.search(
            r"(?:What\s+to\s+Do|FIX|CORRECTIVE\s+ACTION|REPAIR)\s*[:\-]?\s*(.+?)(?=\n|$)",
            text, re.IGNORECASE,
        )

        if cause_match or fix_match:
            lines = [f"[Source: {source} | Relevance: {score:.2f}]"]
            if cause_match:
                lines.append(f"CAUSE:\n{cause_match.group(1).strip()}")
            if fix_match:
                lines.append(f"FIX:\n{fix_match.group(1).strip()}")
            parts.append("\n".join(lines))
        else:
            # Fallback: include raw text with header
            parts.append(f"[Source: {source} | Relevance: {score:.2f}]\n{text}")

    return "\n\n---\n\n".join(parts)


# ── 3-pass retrieval ──────────────────────────────────────────────────────────

def retrieve_with_metadata_filter(
    vector_db: Chroma,
    query: str,
    machine_type: str,
    problem_categories: Optional[List[str]] = None,
    k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
) -> str:
    """
    Three-pass metadata-filtered vector search.

    Pass 1 — machine_type AND problem_categories (narrowest, most accurate)
    Pass 2 — machine_type only                   (broader)
    Pass 3 — global search                        (fallback, no filter)

    Returns a formatted context string ready to inject into the Gemini prompt.
    Returns "" when RAG is unavailable or no chunks pass the relevance threshold.

    Parameters
    ----------
    vector_db          : Loaded Chroma instance
    query              : Raw user complaint text
    machine_type       : Detected machine type string, e.g. "pump"
    problem_categories : Optional list of detected problem categories, e.g. ["noise"]
    k                  : Maximum chunks to return (default RAG_TOP_K = 4)
    min_score          : Minimum similarity score (default RAG_MIN_SCORE = 0.35)
    """
    if vector_db is None:
        return ""

    # 1. Normalize the query using synonym mapping
    normalized_query = normalize_query(query)
    enriched_query = f"{machine_type} problem {normalized_query}"

    good_chunks: List[Tuple[Document, float]] = []

    # ── Pass 1: machine_type AND (category-1 OR category-2 OR …) ────────────────
    # Single Chroma call regardless of how many categories were detected.
    # Much faster than the old per-category loop which could issue N queries.
    if problem_categories:
        try:
            if len(problem_categories) == 1:
                # Simple form — no need for $or wrapper
                chroma_filter = {
                    "$and": [
                        {"machine_type":       {"$eq":       machine_type}},
                        {"problem_categories": {"$contains": problem_categories[0]}},
                    ]
                }
            else:
                # Multi-category: one query, all categories matched in parallel
                chroma_filter = {
                    "$and": [
                        {"machine_type": {"$eq": machine_type}},
                        {
                            "$or": [
                                {"problem_categories": {"$contains": cat}}
                                for cat in problem_categories
                            ]
                        },
                    ]
                }
            results = vector_db.similarity_search_with_relevance_scores(
                enriched_query, k=k, filter=chroma_filter
            )
            good_chunks = [(doc, sc) for doc, sc in results if sc >= min_score]
            if good_chunks:
                logger.info(
                    f"📚 RAG pass-1 ({machine_type} ∩ {problem_categories}): "
                    f"{len(good_chunks)} chunks above threshold"
                )
        except Exception as exc:
            logger.warning(f"RAG pass-1 filter failed: {exc}")

    # ── Pass 2: machine_type only ─────────────────────────────────────────────
    if not good_chunks:
        try:
            chroma_filter = {"machine_type": {"$eq": machine_type}}
            results = vector_db.similarity_search_with_relevance_scores(
                enriched_query, k=k, filter=chroma_filter
            )
            good_chunks = [(doc, sc) for doc, sc in results if sc >= min_score]
            if good_chunks:
                logger.info(
                    f"📚 RAG pass-2 ({machine_type} only): "
                    f"{len(good_chunks)} chunks above threshold"
                )
        except Exception as exc:
            logger.warning(f"RAG pass-2 filter failed: {exc}")

    # ── Pass 3: global fallback ───────────────────────────────────────────────
    if not good_chunks:
        try:
            results = vector_db.similarity_search_with_relevance_scores(
                enriched_query, k=k
            )
            good_chunks = [(doc, sc) for doc, sc in results if sc >= min_score]
            if good_chunks:
                logger.info(
                    f"📚 RAG pass-3 (global): {len(good_chunks)} chunks above threshold"
                )
            else:
                logger.info("📚 RAG: no chunks passed relevance threshold — skipping context injection")
                return ""
        except Exception as exc:
            logger.error(f"RAG pass-3 (global) failed: {exc}")
            return ""

    # ── Re-rank and format ────────────────────────────────────────────────────
    ranked = _rank_chunks(good_chunks)[:k]
    context = _format_rag_context(ranked)

    logger.info(
        f"📚 RAG: {len(ranked)}/{k} chunks injected | "
        f"types={[doc.metadata.get('section_type','?') for doc,_ in ranked]}"
    )
    return context


# ── Convenience: infer problem categories from query ─────────────────────────

# Keyword → category mapping mirrors build_knowledge.py extract_problem_keywords
_QUERY_CATEGORY_MAP = {

    "not_starting": [
        "not start","won't start","no start","dead","not working",
        "starter","crank","no crank","clicking",
        "jammed","jam","seized","locked","stuck","won't turn",
        "not spinning","not rotating","seized up","motor jam",
        "अटका","जाम","फंसा"
    ],

    "noise": [
        "noise","sound","knocking","clicking","grinding",
        "rattling","squealing","humming","whining"
    ],

    "leaking": [
        "leak","leaking","dripping","seeping",
        "oil leak","fuel leak","coolant leak","hydraulic leak"
    ],

    "overheating": [
        "overheat","hot","boiling","running hot",
        "temperature high"
    ],

    "vibration": [
        "vibrat","shaking","wobbl","unbalanced"
    ],

    "smoke": [
        "smoke","smoking","fumes","black smoke",
        "white smoke","blue smoke"
    ],

    "power_loss": [
        "low power","no power","weak","sluggish",
        "not pulling","power loss"
    ],

    "electrical": [
        "battery","starter","wiring","fuse",
        "relay","alternator","voltage"
    ],

    "fuel": [
        "fuel","diesel","petrol","injector",
        "carburetor","fuel pump","fuel filter"
    ],

    "water_flow": [
        "water","flow","pressure","discharge",
        "no water","low pressure","weak flow",
        "motor","pump motor","monoblock motor","submersible motor",
        "मोटर","पानी नहीं","पंप बंद"
    ],

    "cooling": [
        "radiator","coolant","cooling","fan",
        "thermostat","water pump"
    ],

    "hydraulic": [
        "hydraulic","lift","3 point",
        "hydraulic pump","cylinder"
    ],

    "transmission": [
        "gear","gearbox","transmission",
        "clutch","pto"
    ],
}

def infer_problem_categories(query: str) -> List[str]:
    """
    Lightweight keyword scan of the normalized query to produce
    a problem_categories list for Pass-1 metadata filtering.
    """

    normalized = normalize_query(query)
    found = []

    for category, terms in _QUERY_CATEGORY_MAP.items():
        if any(t in normalized for t in terms):
            found.append(category)

    return found[:2]