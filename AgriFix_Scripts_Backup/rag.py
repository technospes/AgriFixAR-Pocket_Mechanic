from __future__ import annotations
import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document

from crossencoder_reranker import _RERANKER
from mmr_dedup import mmr_deduplicate_from_query
from database_creation.metadata_schema import extract_escalate_if_from_content

logger = logging.getLogger(__name__)

# ── Retrieval hyper-parameters ────────────────────────────────────────────────
RAG_TOP_K           = 6
RAG_MIN_SCORE       = 0.15
_POST_RERANK_MIN_SCORE = 0.20
_CANDIDATE_K        = 30
_BM25_K1            = 1.5
_BM25_B             = 0.75
_W_VECTOR           = 0.40
_W_BM25             = 0.30
_W_METADATA         = 0.15
_W_RERANKER         = 0.15
_SAFETY_BOOST       = 0.15
_TAXONOMY_BOOST     = 0.30
_MAX_PER_PARENT     = 3

# ── Confidence thresholds — SINGLE SOURCE OF TRUTH ───────────────────────────
# FIX D: All confidence gates in the whole system use these two constants.
# Previously: db_lock=0.38, RAG_WEAK=0.55, clarification=0.60, post_rerank=0.20
# caused scores in the 0.39–0.54 band to pass the lock but skip clarification,
# producing low-quality LLM outputs.
#
# New unified ladder:
#   RETRIEVAL_LOCK  (0.38) — DB lock threshold (unchanged, keeps existing behaviour)
#   RETRIEVAL_WEAK  (0.50) — single weak-retrieval boundary used by:
#                            • clarification engine (was 0.60 — was too tight)
#                            • RAG classify_retrieval_result (was 0.55)
#                            • pipeline_orchestrator confidence fallback (was 0.55)
#   CONFIDENCE_HIGH / MEDIUM / LOW — for compute_confidence() labels
RETRIEVAL_LOCK   = 0.38    # db_lock hard floor
RAG_WEAK_THRESHOLD = 0.45  # FIX D: was 0.55; lowered to catch 0.39–0.54 band

CONFIDENCE_HIGH   = 0.75
CONFIDENCE_MEDIUM = 0.55
CONFIDENCE_LOW    = 0.35

# ─────────────────────────────────────────────────────────────────────────────
# Stop words
# ─────────────────────────────────────────────────────────────────────────────
_STOP_WORDS: Set[str] = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
    "from","is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall","can",
    "not","no","nor","so","yet","both","either","each","more","most","other",
    "some","such","than","that","this","these","those","it","its","my","your",
    "his","her","our","their","i","we","you","he","she","they","what","which",
    "who","when","where","how","if","as","up","out","about","into","through",
    "during","before","after","above","below","between","there","here","then",
    "any","all",
    "hai","hain","ka","ke","ki","ko","se","mein","par","aur","ek","yeh","woh",
    "kya","kab","kaise","nahi","nahin","bhi","toh","jo","jab","agar","lekin",
    "phir","ab","ya","hoga",
}

# ─────────────────────────────────────────────────────────────────────────────
# Failure taxonomy
# ─────────────────────────────────────────────────────────────────────────────
_FAILURE_TAXONOMY: Dict[str, List[str]] = {
    "electrical":     ["wiring","voltage","short","relay","fuse","mcb","capacitor",
                       "battery","alternator","motor winding","bijli","current"],
    "mechanical":     ["bearing","shaft","gear","coupling","belt","pulley","impeller",
                       "piston","crankshaft","camshaft","valve","spring"],
    "hydraulic":      ["hydraulic","cylinder","control valve","3-point","lift","hitch",
                       "oil pressure","pump pressure"],
    "thermal":        ["overheat","temperature","radiator","coolant","thermostat","garam"],
    "lubrication":    ["oil level","grease","lubricant","dry bearing","oil change","viscosity"],
    "cavitation":     ["cavitation","air lock","suction","prime","hawa","vacuum"],
    "corrosion":      ["corrosion","rust","oxidation","white powder","terminal"],
    "seal_failure":   ["mechanical seal","shaft seal","lip seal","packing","gland"],
    "blockage":       ["blockage","clog","choked","filter","strainer","jammed"],
}

def _query_taxonomy_tags(query_tokens: Set[str]) -> Set[str]:
    matched: Set[str] = set()
    q = " ".join(query_tokens)
    for cat, kws in _FAILURE_TAXONOMY.items():
        if any(kw in q for kw in kws):
            matched.add(cat)
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_query(query: str) -> str:
    normalized = query.lower().strip()
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def _tokenize(text: str) -> List[str]:
    raw = re.findall(r'\b\w+\b', text.lower())
    return [t for t in raw if len(t) > 1 and t not in _STOP_WORDS]

def _safe_meta_str(metadata: dict, key: str) -> str:
    val = metadata.get(key, "")
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val)


# ─────────────────────────────────────────────────────────────────────────────
# Parent-cap
# ─────────────────────────────────────────────────────────────────────────────

def cap_parent_chunks(
    results: List[Tuple[Document, float]],
    max_per_parent: int = _MAX_PER_PARENT,
) -> List[Tuple[Document, float]]:
    parent_counts: Dict[str, int] = {}
    filtered: List[Tuple[Document, float]] = []
    for doc, score in results:
        chunk_id = _safe_meta_str(doc.metadata, "chunk_id")
        parent = chunk_id.rsplit("_", 1)[0] if "_" in chunk_id else chunk_id
        count = parent_counts.get(parent, 0)
        if count < max_per_parent:
            parent_counts[parent] = count + 1
            filtered.append((doc, score))
        else:
            logger.debug("Parent cap: dropped chunk=%s (parent=%s, already=%d/%d)",
                chunk_id, parent, count, max_per_parent)
    if len(filtered) < len(results):
        logger.info("Parent cap: %d → %d chunks (max_per_parent=%d)",
            len(results), len(filtered), max_per_parent)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# BM25 scorer
# ─────────────────────────────────────────────────────────────────────────────

class _BM25Index:
    def __init__(self, docs: List[Tuple[Document, float]]):
        self.corpus: List[List[str]] = []
        self.doc_refs = docs
        for doc, _ in docs:
            self.corpus.append(_tokenize(doc.page_content))
        N = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(N, 1)
        self.df: Dict[str, int] = {}
        for doc_tokens in self.corpus:
            for tok in set(doc_tokens):
                self.df[tok] = self.df.get(tok, 0) + 1
        self.N = N

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc_tokens = self.corpus[doc_idx]
        dl = len(doc_tokens)
        tf = Counter(doc_tokens)
        score = 0.0
        for tok in query_tokens:
            if tok not in tf:
                continue
            n_q = self.df.get(tok, 0)
            idf = math.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1)
            tf_norm = (tf[tok] * (_BM25_K1 + 1)) / (
                tf[tok] + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(self.avgdl, 1))
            )
            score += idf * tf_norm
        return min(score / max(len(query_tokens) * 3, 1), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata match score
# ─────────────────────────────────────────────────────────────────────────────

def _metadata_match_score(
    meta: dict,
    machine_type: str,
    query_taxonomy: Set[str],
) -> float:
    score = 0.0
    chunk_machine = _safe_meta_str(meta, "machine_type").lower()
    if chunk_machine == machine_type:
        score += 0.50
    elif chunk_machine == "universal":
        score += 0.25

    chunk_taxonomy_raw = meta.get("failure_taxonomy", [])
    chunk_taxonomy = set(
        chunk_taxonomy_raw if isinstance(chunk_taxonomy_raw, list)
        else [chunk_taxonomy_raw]
    )
    overlap = len(query_taxonomy & chunk_taxonomy) / max(len(query_taxonomy), 1)
    score += overlap * _TAXONOMY_BOOST

    risk = _safe_meta_str(meta, "risk_level").upper()
    if risk in ("HIGH", "CRITICAL"):
        score += 0.10

    if _safe_meta_str(meta, "escalate_if"):
        score += 0.10

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Structure boost (heuristic fallback only)
# ─────────────────────────────────────────────────────────────────────────────

def _structure_boost(text: str, metadata: dict) -> float:
    score = 0.0
    escalate_text = _safe_meta_str(metadata, "escalate_if").lower()
    if escalate_text and len(escalate_text) > 10:
        score += 0.25
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["warning","danger","critical","fatal","fire","crush","spark","shock"]):
        score += 0.50
    if "cause" in text_lower or "symptom" in text_lower:
        score += 0.15
    if "fix" in text_lower or "repair" in text_lower or "steps" in text_lower:
        score += 0.15
    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence scorer
# ─────────────────────────────────────────────────────────────────────────────

def classify_retrieval_result(n_chunks: int, score: float) -> str:
    if n_chunks == 0:
        return "no_data"
    if score < RAG_WEAK_THRESHOLD:      # FIX D: unified threshold
        return "clarification_needed"
    return "success"

def compute_confidence(
    top_score: float,
    n_chunks: int,
    symptom_coverage: float,
    taxonomy_match: bool,
) -> Tuple[float, str]:
    base = min(top_score, 1.0)
    chunk_bonus = min(n_chunks / RAG_TOP_K, 1.0) * 0.10
    symptom_bonus = symptom_coverage * 0.15
    taxonomy_bonus = 0.10 if taxonomy_match else 0.0
    confidence = min(base + chunk_bonus + symptom_bonus + taxonomy_bonus, 1.0)
    if confidence >= CONFIDENCE_HIGH:
        label = "HIGH"
    elif confidence >= CONFIDENCE_MEDIUM:
        label = "MEDIUM"
    elif confidence >= CONFIDENCE_LOW:
        label = "LOW"
    else:
        label = "INSUFFICIENT"
    return round(confidence, 3), label


# ─────────────────────────────────────────────────────────────────────────────
# Multi-query candidate merger
# ─────────────────────────────────────────────────────────────────────────────

def _merge_candidates(
    candidate_lists: List[List[Tuple[Document, float]]],
) -> List[Tuple[Document, float]]:
    seen: Dict[str, Tuple[Document, float]] = {}
    for candidates in candidate_lists:
        for doc, score in candidates:
            chunk_id = _safe_meta_str(doc.metadata, "chunk_id")
            if not chunk_id or chunk_id == "unknown":
                chunk_id = doc.page_content[:100]
            if chunk_id not in seen or score > seen[chunk_id][1]:
                seen[chunk_id] = (doc, score)
    merged = list(seen.values())
    merged.sort(key=lambda t: -t[1])
    logger.info("Multi-query merge: %d candidate lists → %d unique chunks",
        len(candidate_lists), len(merged))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid re-ranker
# ─────────────────────────────────────────────────────────────────────────────

def _hybrid_rerank(
    query: str,
    candidates: List[Tuple[Document, float]],
    machine_type: str,
    top_n: int = RAG_TOP_K,
) -> Tuple[List[Tuple[Document, float]], float, str]:
    """
    FIX C: Return the raw (pre-normalisation) top score as the confidence signal.

    Previously, min-max normalisation always mapped the winner to 1.0, so the
    returned top_score was always 1.0 in single-candidate pools and meaningless
    as an absolute quality measure.  The caller (pipeline_orchestrator) then
    compared this 1.0 against the 0.50 weak threshold — it always passed, even
    when the actual best hybrid score was 0.35 (garbage retrieval).

    Fix: the tuple now returns raw_top as the confidence signal used by all
    downstream gates.  The normalised scores are still used internally to rank
    chunks for context formatting — we want the LLM to see the most relevant
    chunks first — but they are NOT returned as the accuracy indicator.
    """
    if not candidates:
        return [], 0.0, "INSUFFICIENT"

    norm_query    = normalize_query(query)
    query_tokens  = _tokenize(norm_query)
    query_tok_set = set(query_tokens)
    query_taxonomy = _query_taxonomy_tags(query_tok_set)

    bm25 = _BM25Index(candidates)
    reranker_scores = _RERANKER.score_batch_with_context(norm_query, candidates)

    scored: List[Tuple[Document, float]] = []
    raw_scores_for_confidence: List[float] = []

    for i, (doc, vec_score) in enumerate(candidates):
        logger.debug("Candidate %d: chunk_id=%s | vec=%.3f | content='%s...'",
        i, doc.metadata.get("chunk_id","?"), vec_score, doc.page_content[:60])
        meta    = doc.metadata or {}

        bm25_score     = bm25.score(query_tokens, i)
        meta_score     = _metadata_match_score(meta, machine_type, query_taxonomy)
        reranker_score = reranker_scores[i]

        hybrid = (
            _W_VECTOR   * min(vec_score, 1.0) +
            _W_BM25     * bm25_score           +
            _W_METADATA * meta_score            +
            _W_RERANKER * reranker_score
        )

        is_universal  = _safe_meta_str(meta, "machine_type").lower() == "universal"
        tags_text     = _safe_meta_str(meta, "tags").lower()
        escalate_text = _safe_meta_str(meta, "escalate_if").lower()
        safety_target = set(_tokenize(f"{tags_text} {escalate_text}"))
        if is_universal and (query_tok_set & safety_target):
            hybrid = min(hybrid + _SAFETY_BOOST, 1.0)
            logger.debug("Safety override: chunk_id=%s", meta.get("chunk_id", "?"))

        scored.append((doc, hybrid))
        raw_scores_for_confidence.append(hybrid)

    scored.sort(key=lambda t: -t[1])

    if not scored:
        return [], 0.0, "INSUFFICIENT"

    # FIX C: raw_top is the true quality signal — never normalised to 1.0
    raw_top = max(raw_scores_for_confidence)
    raw_avg = sum(raw_scores_for_confidence) / len(raw_scores_for_confidence)

    weak_pool = (raw_top < 0.25 and raw_avg < 0.10)
    if weak_pool:
        logger.info("Weak pool detected: raw_top=%.3f raw_avg=%.3f → confidence capped",
            raw_top, raw_avg)

    # Normalise only for in-list ranking (so LLM sees most-relevant chunk first)
    raw_scores = [s for _, s in scored]
    min_s, max_s = min(raw_scores), max(raw_scores)

    if max_s > min_s:
        normalized_scored = [
            (doc, (s - min_s) / (max_s - min_s))
            for doc, s in scored
        ]
    else:
        flat_val = raw_top
        normalized_scored = [(doc, flat_val) for doc, _ in scored]

    top = normalized_scored[:top_n]

    if not top:
        return [], 0.0, "INSUFFICIENT"

    # Confidence classification uses RAW score (FIX C)
    if weak_pool:
        confidence_val = min(raw_top, 0.35)
        confidence_label = "LOW"
    else:
        confidence_val = min(max(raw_top, 0.0), 1.0)
        if confidence_val >= CONFIDENCE_HIGH:
            confidence_label = "HIGH"
        elif confidence_val >= CONFIDENCE_MEDIUM:
            confidence_label = "MEDIUM"
        elif confidence_val >= CONFIDENCE_LOW:
            confidence_label = "LOW"
        else:
            confidence_label = "INSUFFICIENT"

    logger.info(
        "Hybrid rerank: %d→%d | raw_top=%.3f | conf=%s (%.3f) | weak_pool=%s",
        len(candidates), len(top), raw_top, confidence_label, confidence_val, weak_pool,
    )
    # FIX C: return raw_top as confidence signal, NOT the normalised top score
    return top, raw_top, confidence_label


# ─────────────────────────────────────────────────────────────────────────────
# RAG context formatter
# ─────────────────────────────────────────────────────────────────────────────

def _format_rag_context(
    chunks: List[Tuple[Document, float]],
    confidence_label: str = "MEDIUM",
) -> str:
    parts = []
    for doc, score in chunks:
        source   = _safe_meta_str(doc.metadata, "source_file") or "manual"
        chunk_id = _safe_meta_str(doc.metadata, "chunk_id")    or "unknown"
        text     = doc.page_content.strip()
        problem  = _safe_meta_str(doc.metadata, "problem")
        escalate = extract_escalate_if_from_content(doc.page_content)
        risk     = _safe_meta_str(doc.metadata, "risk_level")
        taxonomy = _safe_meta_str(doc.metadata, "failure_taxonomy")
        electric = doc.metadata.get("electrical_hazard", False)
        shutdown = doc.metadata.get("shutdown_required", False)

        lines = [
            f"[Source: {source} | Chunk: {chunk_id} | Relevance: {score:.2f}"
            f" | Risk: {risk} | Taxonomy: {taxonomy}]"
        ]
        if electric:
            lines.append("⚡ ELECTRICAL HAZARD — power OFF before any step.")
        if shutdown:
            lines.append("🔴 SHUTDOWN REQUIRED before proceeding.")
        if escalate:
            lines.append(f"⚠️ ESCALATE_IF:\n{escalate}")
        if problem:
            lines.append(f"PROBLEM:\n{problem}")
        lines.append(f"DIAGNOSTIC CONTENT:\n{text}")
        parts.append("\n".join(lines))

    header = f"[Retrieval Confidence: {confidence_label}]\n\n"
    return header + "\n\n" + "=" * 70 + "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# FIX F: Dynamic compatible machine types
# ─────────────────────────────────────────────────────────────────────────────

# Populated at startup by load_dynamic_compat_map() below.
# Falls back to the static table if not populated.
_DYNAMIC_COMPAT_MAP: Dict[str, List[str]] = {}

# Static fallback table (kept for cold-start / test environments)
_STATIC_COMPAT_MAP: Dict[str, List[str]] = {
    "electric_motor":    ["electric_motor","ac_motor","motor","universal"],
    "water_pump":        ["water_pump","pump","centrifugal_pump","universal"],
    "submersible_pump":  ["submersible_pump","ns_pump","borewell_pump","pump","universal"],
    "unknown":           ["universal"],
}

def load_dynamic_compat_map(vector_db: "Chroma") -> None:
    """
    FIX F: Build _compatible_machine_types map dynamically from the ChromaDB
    metadata so that any new machine added to the knowledge base is automatically
    supported without a code change.

    Call this once at startup after load_machine_registry(vector_db).

    Algorithm:
      1. Pull all machine_type values from ChromaDB metadata.
      2. For each machine_type, add itself + all machines that share a common
         word stem (e.g. "pump" links water_pump, submersible_pump, centrifugal_pump).
      3. Always append "universal" as the last fallback.
    """
    global _DYNAMIC_COMPAT_MAP
    try:
        collection = vector_db._collection
        result = collection.get(include=["metadatas"])
        all_machines: Set[str] = set()
        for meta in (result.get("metadatas") or []):
            mt = str(meta.get("machine_type", "")).lower().strip()
            if mt and mt not in ("", "unknown", "universal"):
                all_machines.add(mt)

        if not all_machines:
            logger.warning("load_dynamic_compat_map: no machine_type values found, using static table")
            return

        compat: Dict[str, List[str]] = {}
        for mt in all_machines:
            words = set(mt.replace("-", "_").split("_"))
            related = [mt]
            for other in all_machines:
                if other == mt:
                    continue
                other_words = set(other.replace("-", "_").split("_"))
                # Share at least one meaningful word (not "pump" alone — must be >3 chars)
                shared = {w for w in words & other_words if len(w) > 3}
                if shared:
                    related.append(other)
            related.append("universal")
            compat[mt] = list(dict.fromkeys(related))

        compat["unknown"] = ["universal"]
        _DYNAMIC_COMPAT_MAP = compat
        logger.info("load_dynamic_compat_map: built compat map for %d machines: %s",
            len(compat), list(compat.keys()))
    except Exception as exc:
        logger.warning("load_dynamic_compat_map failed (%s) — using static table", exc)


def _compatible_machine_types(machine_type: str) -> List[str]:
    """FIX F: Use dynamic map first, fall back to static table."""
    if _DYNAMIC_COMPAT_MAP:
        return _DYNAMIC_COMPAT_MAP.get(machine_type, [machine_type, "universal"])
    return _STATIC_COMPAT_MAP.get(machine_type, [machine_type, "universal"])


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB semantic retrieval
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_chroma_filter(vector_db: Chroma) -> None:
    try:
        sample = vector_db._collection.get(limit=5, include=["metadatas"])
        logger.info("ChromaDB metadata sample (first 5 chunks):")
        for i, meta in enumerate(sample.get("metadatas", [])):
            logger.info("  chunk[%d]: %s", i, meta)
    except Exception as exc:
        logger.error("diagnose_chroma_filter failed: %s", exc)


def _semantic_retrieve(
    vector_db: Chroma,
    query: str,
    machine_type: str,
    k: int = _CANDIDATE_K,
    min_score: float = RAG_MIN_SCORE,
) -> List[Tuple[Document, float]]:
    enriched_query = f"{normalize_query(query)} {machine_type}"
    allowed_types = _compatible_machine_types(machine_type)
    chroma_filter = {"machine_type": {"$in": allowed_types}}
    good: List[Tuple[Document, float]] = []

    try:
        results = vector_db.similarity_search_with_relevance_scores(
            enriched_query, k=k, filter=chroma_filter
        )
        good = [(doc, score) for doc, score in results if score >= min_score]
        logger.info("Semantic (filtered): %d chunks ≥ %.2f for machine=%s",
            len(good), min_score, machine_type)
    except Exception as exc:
        if "Error finding id" in str(exc) or "Internal error" in str(exc):
            logger.warning("Filtered search failed (metadata filter unstable) → fallback to unfiltered")
        else:
            logger.warning("Filtered search failed: %s → fallback to unfiltered", exc)

    if len(good) < 6:
        try:
            results = vector_db.similarity_search_with_relevance_scores(enriched_query, k=k)
            good = [(doc, score) for doc, score in results if score >= min_score]
            logger.info("Semantic (unfiltered fallback): %d chunks", len(good))
        except Exception as exc2:
            logger.error("Unfiltered search also failed: %s", exc2)

    return good


# ─────────────────────────────────────────────────────────────────────────────
# Multi-query retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve_multi_query(
    vector_db: Chroma,
    query_variants: List[str],
    machine_type: str,
    k_per_variant: int = 15,
    min_score: float = RAG_MIN_SCORE,
) -> List[Tuple[Document, float]]:
    candidate_lists: List[List[Tuple[Document, float]]] = []
    for variant in query_variants:
        if not variant or not variant.strip():
            continue
        candidates = _semantic_retrieve(vector_db, variant, machine_type,
            k=k_per_variant, min_score=min_score)
        if candidates:
            candidate_lists.append(candidates)
        logger.debug("Variant '%s': %d candidates", variant[:60], len(candidates))

    if not candidate_lists:
        return []

    merged = _merge_candidates(candidate_lists)
    return merged[:_CANDIDATE_K]


# ─────────────────────────────────────────────────────────────────────────────
# Adjacent chunk stitching
# ─────────────────────────────────────────────────────────────────────────────

def _stitch_adjacent_chunks(
    reranked: List[Tuple[Document, float]],
    vector_db: "Chroma",
    machine_type: str,
    query: str,
    max_siblings_per_chunk: int = 2,
    min_sibling_score: float = 0.10,
) -> List[Tuple[Document, float]]:
    """
    FIX E: After stitching, re-sort by score descending and cap total at
    RAG_TOP_K so the LLM always receives chunks in quality order and the
    context string does not grow unboundedly.

    Previous behaviour: siblings were appended AFTER reranked chunks, so the
    context formatter put them last regardless of quality.  If a sibling had
    score 0.82 but the anchor had 0.65, the LLM saw the anchor first — a
    quality ordering failure.
    """
    if not reranked or vector_db is None:
        return reranked

    selected_ids: set = {
        _safe_meta_str(doc.metadata, "chunk_id")
        for doc, _ in reranked
    }
    query_tokens = _tokenize(normalize_query(query))
    extra: List[Tuple[Document, float]] = []
    seen_parents: set = set()

    for anchor_doc, anchor_score in reranked:
        chunk_id = _safe_meta_str(anchor_doc.metadata, "chunk_id")
        if not chunk_id or "_" not in chunk_id:
            continue
        parent_id = chunk_id.rsplit("_", 1)[0]
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        try:
            results = vector_db._collection.get(
                where={"chunk_id": {"": f"^{parent_id}_"}},
                include=["documents", "metadatas"],
            )
        except Exception:
            try:
                results = vector_db._collection.get(include=["documents", "metadatas"])
            except Exception:
                continue

        sibling_docs = []
        for i, meta in enumerate(results.get("metadatas") or []):
            cid = str(meta.get("chunk_id", ""))
            if not cid.startswith(parent_id + "_"):
                continue
            if cid in selected_ids:
                continue
            doc_text = (results.get("documents") or [])[i] if i < len(results.get("documents") or []) else ""
            from langchain_core.documents import Document as _Doc
            sibling_docs.append(_Doc(page_content=doc_text, metadata=meta))

        if not sibling_docs:
            continue

        bm25_candidates = [(d, 0.0) for d in sibling_docs]
        bm25_idx = _BM25Index(bm25_candidates)
        scored_siblings = []
        for j, sib_doc in enumerate(sibling_docs):
            bm25_s = bm25_idx.score(query_tokens, j)
            capped = min(bm25_s, max(anchor_score - 0.01, min_sibling_score))
            scored_siblings.append((sib_doc, capped))

        scored_siblings.sort(key=lambda t: -t[1])
        for sib_doc, sib_score in scored_siblings[:max_siblings_per_chunk]:
            if sib_score >= min_sibling_score:
                sib_id = _safe_meta_str(sib_doc.metadata, "chunk_id")
                selected_ids.add(sib_id)
                extra.append((sib_doc, sib_score))

    if extra:
        logger.info("Sibling stitching: added %d adjacent chunks from %d parents",
            len(extra), len(seen_parents))

    combined = reranked + extra
    # FIX E: Re-sort by score so LLM receives chunks in quality order,
    # then cap at RAG_TOP_K to avoid context bloat.
    combined.sort(key=lambda t: -t[1])
    combined = combined[:RAG_TOP_K]
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Public API — retrieve_with_confidence
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_with_confidence(
    vector_db: Chroma,
    query: str,
    machine_type: str,
    k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
    query_variants: Optional[List[str]] = None,
) -> Tuple[str, float, int]:
    if vector_db is None:
        logger.warning("Vector DB not available")
        return "", 0.0, 0

    logger.info(
        "RAG v14.0 | machine=%s | query='%s%s' | variants=%d",
        machine_type, query[:60], "..." if len(query) > 60 else "",
        len(query_variants) if query_variants else 0,
    )

    all_variants = [query]
    if query_variants:
        for v in query_variants:
            if v and v.strip() and v.strip() != query.strip():
                all_variants.append(v.strip())

    # Phase 1 — multi-query semantic retrieval + merge
    if len(all_variants) > 1:
        candidates = _retrieve_multi_query(vector_db, all_variants, machine_type,
            k_per_variant=15, min_score=min_score)
    else:
        candidates = _semantic_retrieve(vector_db, query, machine_type,
            k=_CANDIDATE_K, min_score=min_score)

    if not candidates:
        logger.warning("No candidates above %.2f for machine=%s", min_score, machine_type)
        return "", 0.0, 0

    # Phase 2 — MMR deduplication
    candidates = mmr_deduplicate_from_query(candidates, query, top_n=_CANDIDATE_K)
    # Phase 3 — Parent cap
    # candidates = cap_parent_chunks(candidates, max_per_parent=_MAX_PER_PARENT)

    # Phase 4 — hybrid re-rank (returns raw score as confidence signal — FIX C)
    reranked, top_score, confidence_label = _hybrid_rerank(
        query, candidates, machine_type, top_n=k
    )

    if not reranked:
        return "", 0.0, 0

    # Post-rerank quality gate (uses normalised scores from reranked list)
    reranked = [(doc, s) for doc, s in reranked if s >= _POST_RERANK_MIN_SCORE]
    if not reranked:
        logger.info(
            "Post-rerank gate (%.2f): all chunks below threshold — no_data "
            "(machine=%s, query='%s')",
            _POST_RERANK_MIN_SCORE, machine_type, query[:60],
        )
        return "", 0.0, 0

    # Phase 5 — Adjacent chunk stitching (FIX E: re-sorts and caps)
    reranked = _stitch_adjacent_chunks(reranked, vector_db, machine_type, query,
        max_siblings_per_chunk=2)

    # ── Dynamic out-of-domain detection ──────────────────────────────────────
    _KNOWN_SYMPTOM_VOCAB: Set[str] = {
        "not","starting","start","started","starts","no","water","low","pressure",
        "vibration","vibrating","vibrates","noise","noisy","hot","heat","overheating",
        "smoke","smoking","spark","sparks","trip","tripping","trips",
        "hum","hums","humming","hummed",
        "dead","running","runs","pump","pumping","motor","engine","flow",
        "discharge","discharging","does","cannot",
    }

    raw_query_tokens = set(_tokenize(query))
    known_vocab = _KNOWN_SYMPTOM_VOCAB | set(
        machine_type.lower().replace("_", " ").split()
    )
    novel_nouns = raw_query_tokens - known_vocab

    if novel_nouns:
        retrieved_text_blob = " ".join(doc.page_content.lower() for doc, _ in reranked)
        matched_nouns = {
            n for n in novel_nouns
            if n in retrieved_text_blob or any(
                (len(n) >= 5 and word.startswith(n[:5])) or
                (len(n) >= 5 and n.startswith(word[:5]))
                for word in retrieved_text_blob.split()
            )
        }
        if not matched_nouns:
            logger.info(
                "Out-of-domain detected: novel nouns %s not found in any "
                "retrieved chunk — rejecting result (machine=%s, query='%s')",
                novel_nouns, machine_type, query[:80],
            )
            return "", 0.0, 0

    logger.info(
        "RAG v14.0 complete: %d chunks | raw_top=%.3f | conf=%s | %s",
        len(reranked), top_score, confidence_label,
        "WEAK" if top_score < RAG_WEAK_THRESHOLD else "STRONG",
    )

    context_str = _format_rag_context(reranked, confidence_label)
    return context_str, top_score, len(reranked)


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible thin wrappers
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_with_metadata_filter(
    vector_db: Chroma,
    query: str,
    machine_type: str,
    problem_categories: Optional[List[str]] = None,
    k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
) -> str:
    ctx, _, _ = retrieve_with_confidence(vector_db, query, machine_type, k, min_score)
    return ctx


def retrieve_context(
    vector_db: Chroma,
    query: str,
    machine_type: str,
    k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
) -> str:
    return retrieve_with_metadata_filter(vector_db, query, machine_type, k, min_score)