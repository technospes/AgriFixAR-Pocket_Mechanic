"""
rag.py — Production RAG Pipeline v7.0
AgriFix Multimodal Diagnostic System

CHANGES FROM v6.0 → v7.0:
  1. DELETED CATEGORY_KEYWORDS and infer_problem_categories.
     Brittle regex inference replaced by ChromaDB native semantic embeddings.
  2. SINGLE-PASS semantic search with universal injection.
     Filter: machine_type IN [machine_type, "universal"] — embeddings handle the rest.
  3. retrieve_context() wrapper preserved for backward compatibility.
"""

import logging
import re
from typing import List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RAG_TOP_K = 6
RAG_MIN_SCORE = 0.45

def normalize_query(query: str) -> str:
    normalized = query.lower().strip()
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def _structure_boost(text: str, metadata: dict) -> float:
    score = 0.0
    escalate_text = metadata.get("escalate_if", "").lower()
    if escalate_text and len(escalate_text) > 10:
        score += 0.90
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["warning", "danger", "critical", "fatal", "fire", "crush", "spark", "shock"]):
        score += 0.50
    if "cause" in text_lower or "symptom" in text_lower:
        score += 0.15
    if "fix" in text_lower or "repair" in text_lower or "steps" in text_lower:
        score += 0.15
    return min(score, 1.0)

def _rank_chunks(chunks_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    ranked = []
    for doc, vec_score in chunks_with_scores:
        boost = _structure_boost(doc.page_content, doc.metadata)
        composite = (vec_score * 0.70) + (boost * 0.30)
        ranked.append((doc, composite))
    ranked.sort(key=lambda x: -x[1])
    return ranked

def _format_rag_context(chunks: List[Tuple[Document, float]]) -> str:
    parts = []
    for doc, score in chunks:
        source   = doc.metadata.get("source_file", "manual")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        text     = doc.page_content.strip()
        problem  = doc.metadata.get("problem", "")
        escalate = doc.metadata.get("escalate_if", "")

        lines = [f"[Source: {source} | Chunk: {chunk_id} | Relevance: {score:.2f}]"]
        if escalate:
            lines.append(f"⚠️ ESCALATE_IF:\n{escalate}")
        if problem:
            lines.append(f"PROBLEM:\n{problem}")
        lines.append(f"DIAGNOSTIC CONTENT:\n{text}")
        parts.append("\n".join(lines))

    return "\n\n" + "=" * 70 + "\n\n".join(parts)

def retrieve_with_metadata_filter(
    vector_db: Chroma,
    query: str,
    machine_type: str,
    problem_categories: Optional[List[str]] = None,  # ignored in v7.0 — kept for API compat
    k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
) -> str:
    """
    v7.0: Single semantic pass.
    Filter = machine_type IN [machine_type, "universal"].
    ChromaDB embeddings handle symptom matching; no regex category inference needed.
    """
    if vector_db is None:
        logger.warning("Vector DB not available")
        return ""

    enriched_query = f"{normalize_query(query)} {machine_type}"

    # ── SINGLE PASS: semantic search scoped to machine + universal ────────
    chroma_filter = {"machine_type": {"$in": [machine_type, "universal"]}}

    try:
        logger.info(f"RAG v7.0 single-pass: machine={machine_type} + universal")
        results = vector_db.similarity_search_with_relevance_scores(
            enriched_query, k=k, filter=chroma_filter
        )
        good_chunks = [(doc, score) for doc, score in results if score >= min_score]
        logger.info(f"  → {len(good_chunks)} chunks above threshold {min_score}")
    except Exception as exc:
        logger.warning(f"RAG single-pass failed: {exc}. Falling back to unfiltered search.")
        good_chunks = []

    # ── FALLBACK: unfiltered semantic search ──────────────────────────────
    if len(good_chunks) < 2:
        try:
            logger.info("RAG fallback: unfiltered semantic search")
            results = vector_db.similarity_search_with_relevance_scores(
                enriched_query, k=k
            )
            good_chunks = [(doc, score) for doc, score in results if score >= min_score]
            logger.info(f"  → {len(good_chunks)} chunks above threshold {min_score}")
        except Exception as exc:
            logger.error(f"RAG fallback failed: {exc}")
            return ""

    if not good_chunks:
        logger.warning(
            f"No chunks above threshold {min_score} for "
            f"machine={machine_type}, query='{query[:50]}...'"
        )
        return ""

    ranked     = _rank_chunks(good_chunks)
    top_chunks = ranked[:5]

    logger.info(
        f"RAG retrieved {len(top_chunks)} chunks: "
        f"{[f'{s:.2f}' for _, s in top_chunks]}"
    )
    return _format_rag_context(top_chunks)

def retrieve_context(
    vector_db: Chroma,
    query: str,
    machine_type: str,
    k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
) -> str:
    """Backward-compatible wrapper. Category inference removed in v7.0."""
    return retrieve_with_metadata_filter(
        vector_db=vector_db,
        query=query,
        machine_type=machine_type,
        k=k,
        min_score=min_score,
    )