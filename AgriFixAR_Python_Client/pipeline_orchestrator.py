from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma

from query_router   import route_query, build_enriched_query, RouterOutput
from db_lock        import check_db_lock_from_rag, DbLockResult
from visual_gate    import run_visual_gate, extract_target_parts, get_camera_prompt, GateResult
from rag            import retrieve_with_confidence, RAG_WEAK_THRESHOLD, _STOP_WORDS
from clarification_loop import ClarificationEngine, _MAX_CLARIFICATION_ROUNDS, _CLARIFICATION_THRESHOLD

# FIX 3: Phase 0 OOD guard — rejects non-repair queries before retrieval
from ood_guard import check_ood

logger = logging.getLogger(__name__)

# Module-level clarification engine singleton (avoid re-init per request)
_clarification_engine = ClarificationEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Outcome of the full 4-phase pipeline.

    Attributes:
        phase_reached   — "ood_guard" | "router" | "db_lock" | "clarification" | "visual_gate" | "generation"
        blocked         — True if pipeline stopped before generation
        block_reason    — "ood_guard" | "db_lock" | "visual_gate_fail" | "clarification_needed" | "router_error"
        response        — Final JSON-safe dict to return to the client
        router          — RouterOutput from Phase 1
        lock            — DbLockResult from Phase 2 (None if not reached)
        gate            — GateResult from Phase 3 (None if not reached)
        rag_context     — Raw RAG context string (for downstream LLM calls)
        machine_type    — Resolved machine type
        language        — Language code
        rag_score       — Top RAG hybrid score (for confidence decisions)
        n_chunks        — Number of RAG chunks retrieved
    """
    phase_reached: str
    blocked: bool
    block_reason: str
    response: Dict[str, Any]
    router: RouterOutput
    lock: Optional[DbLockResult]
    gate: Optional[GateResult]
    rag_context: str
    machine_type: str
    language: str
    rag_score: float = 0.0
    n_chunks: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_full_pipeline(
    query: str,
    vector_db: Chroma,
    frame_bytes: Optional[bytes] = None,
    language: str = "en",
    machine_type_override: Optional[str] = None,
    clarification_round: int = 0,
    previous_clarification_answer: Optional[str] = None,
) -> PipelineResult:
    """
    Execute Phase 0 → Phase 1 → Phase 2 → (FIX 5: Confidence Fallback) → Phase 3 in sequence.

    Args:
        query                           Raw farmer input
        vector_db                       Initialised Chroma instance
        frame_bytes                     Best camera frame as JPEG bytes (None = text)
        language                        "en" | "hi"
        machine_type_override           Bypass Phase 1 router
        clarification_round             0 = first attempt (increments with each clarification)
        previous_clarification_answer   Farmer's answer to last clarification question.
                                        Appended to the query before retrieval.

    Returns PipelineResult. Check .blocked and .block_reason before proceeding.

    FIX 3 — Phase 0 OOD Guard:
        Rejects non-repair queries (price, brand, theft, etc.) before any retrieval.

    FIX 5 — Confidence Fallback:
        If top RAG score < RAG_WEAK_THRESHOLD (0.55) AND
        clarification rounds < _MAX_CLARIFICATION_ROUNDS (2):
            → return clarification question to farmer
            → farmer answers → caller retries with clarification_round+1
              and previous_clarification_answer set
        If all clarification rounds exhausted and still low confidence:
            → proceed to generation (LLM handles low-confidence grounding)
    """

    # ── Enrich query with clarification answer if available ──────────────────
    effective_query = query
    if previous_clarification_answer and previous_clarification_answer.strip():
        effective_query = f"{query} {previous_clarification_answer}".strip()
        logger.info(
            "Clarification answer appended: '%s' → enriched_query='%s...'",
            previous_clarification_answer[:60],
            effective_query[:80],
        )

    # ── Phase 0: OOD / Intent Guard (FIX 3) ──────────────────────────────────
    # Reject non-repair queries before any retrieval or LLM call.
    # Runs in <1ms — no network, no model inference.
    ood_result = check_ood(effective_query, machine_type_override or "")
    if ood_result.is_ood:
        logger.info(
            "Phase 0 OOD BLOCK: category=%s query='%s...'",
            ood_result.category, effective_query[:60],
        )
        return PipelineResult(
            phase_reached="ood_guard",
            blocked=True,
            block_reason="ood_guard",
            response={
                **ood_result.api_response(),
                "rag_score": 0.0,
                "machine_type": machine_type_override or "unknown",
            },
            router=RouterOutput(
                machine_type=machine_type_override or "unknown",
                symptoms=[],
                confidence=0.0,
                language=language,
                raw_query=effective_query,
                query_variants=[],
            ),
            lock=None,
            gate=None,
            rag_context="",
            machine_type=machine_type_override or "unknown",
            language=language,
            rag_score=0.0,
            n_chunks=0,
        )

    # ── Phase 1: Dynamic Query Router ────────────────────────────────────────
    if machine_type_override:
        router = RouterOutput(
            machine_type=machine_type_override,
            symptoms=[],
            confidence=1.0,
            language=language,
            raw_query=effective_query,
            query_variants=[],
        )
        logger.info("Phase 1 SKIPPED: machine_type override='%s'", machine_type_override)
    else:
        logger.info("Phase 1: routing query='%s...'", effective_query[:60])
        router = await route_query(effective_query)

        if not router.router_ok:
            logger.warning("Phase 1 FAILED: %s", router.error)

    enriched_query   = build_enriched_query(router)
    resolved_machine = router.machine_type

    logger.info(
        "Phase 1 complete: machine=%s conf=%.2f symptoms=%s variants=%d enriched='%s'",
        resolved_machine, router.confidence, router.symptoms,
        len(router.query_variants), enriched_query[:60],
    )

    # ── Phase 2: ChromaDB Retrieval + DB Lock ─────────────────────────────────
    logger.info("Phase 2: ChromaDB retrieval + lock check (multi-query=%d variants)",
                len(router.query_variants))

    # FIX 2: Pass query_variants for multi-query retrieval AND language for adaptive weights
    rag_result = retrieve_with_confidence(
        vector_db=vector_db,
        query=enriched_query,
        machine_type=resolved_machine,
        query_variants=router.query_variants,
        language=router.language,       # FIX 2: propagate detected language
    )
    context_str, score, n_chunks = rag_result

    lock = check_db_lock_from_rag(rag_result, machine_type=resolved_machine, query=enriched_query)

    # ── FIX 5: Strong Vagueness Pre-Check (Independent of chunks) ────────────
    # A single-word query like "Pump" might return chunks simply through 
    # frequency matching, but it lacks actionable diagnostic intent.
    query_tokens = query.lower().split()
    _content_tokens = [t for t in query_tokens if t not in _STOP_WORDS]
    is_vague = len(_content_tokens) < 3 and not router.symptoms

    if is_vague and resolved_machine != "unknown":
        # Only allow 1-2 word queries to bypass clarification IF they hit an incredibly
        # strong, unambiguous exact match in the DB (e.g. >= 0.80).
        if score < 0.80: 
            logger.info(
                "Vagueness override: tokens=%d symptoms=%s score=%.3f machine=%s → "
                "forcing clarification_needed",
                len(_content_tokens), router.symptoms, score, resolved_machine,
            )
            clarification = await _clarification_engine.get_clarification(
                machine_type  = resolved_machine,
                symptoms      = [],
                rag_context   = "",
                confidence    = score,
                round_number  = clarification_round,
            )
            if clarification.needs_clarification:
                clar_response = clarification.api_response()
                clar_response["clarification_round"] = clarification_round
                clar_response["rag_score"] = score
                clar_response["machine_type"] = resolved_machine
                return PipelineResult(
                    phase_reached="clarification",
                    blocked=True,
                    block_reason="clarification_needed",
                    response=clar_response,
                    router=router,
                    lock=lock,
                    gate=None,
                    rag_context="",
                    machine_type=resolved_machine,
                    language=language,
                    rag_score=score,
                    n_chunks=n_chunks,
                )

    if lock.locked:
        return PipelineResult(
            phase_reached="db_lock",
            blocked=True,
            block_reason="db_lock",
            response=lock.api_response(),
            router=router,
            lock=lock,
            gate=None,
            rag_context="",
            machine_type=resolved_machine,
            language=language,
            rag_score=score,
            n_chunks=n_chunks,
        )

    logger.info("Phase 2 PASSED: score=%.3f chunks=%d", score, n_chunks)

    # ── FIX 5: Confidence Fallback — clarification before escalation ─────────
    if (
        score < RAG_WEAK_THRESHOLD
        and clarification_round < _MAX_CLARIFICATION_ROUNDS
        and context_str
    ):
        logger.info(
            "FIX 5: Low confidence (%.3f < %.2f) at round=%d — triggering clarification",
            score, RAG_WEAK_THRESHOLD, clarification_round,
        )
        clarification = await _clarification_engine.get_clarification(
            machine_type  = resolved_machine,
            symptoms      = router.symptoms,
            rag_context   = context_str,
            confidence    = score,
            round_number  = clarification_round,
        )
        if clarification.needs_clarification:
            logger.info(
                "FIX 5: Clarification question: '%s'",
                clarification.question_en[:80],
            )
            clar_response = clarification.api_response()
            clar_response["clarification_round"] = clarification_round
            clar_response["rag_score"] = round(score, 3)
            clar_response["machine_type"] = resolved_machine

            return PipelineResult(
                phase_reached="clarification",
                blocked=True,
                block_reason="clarification_needed",
                response=clar_response,
                router=router,
                lock=lock,
                gate=None,
                rag_context="",
                machine_type=resolved_machine,
                language=language,
                rag_score=score,
                n_chunks=n_chunks,
            )

    # ── Phase 3: Visual Verification Gate ────────────────────────────────────
    target_parts = extract_target_parts(context_str)
    camera_prompt = get_camera_prompt(target_parts, language)

    logger.info(
        "Phase 3: visual gate | parts=%s | frame=%s | score=%.3f",
        target_parts, "yes" if frame_bytes else "no", score,
    )

    gate = await run_visual_gate(
        frame_bytes=frame_bytes,
        target_parts=target_parts,
        rag_chunk_text=context_str,
        machine_type=resolved_machine,
    )

    if not gate.gate_passed:
        logger.warning("Phase 3 BLOCKED: verdict=%s conf=%.2f", gate.verdict, gate.confidence)
        re_examine = gate.re_examine_response()
        re_examine["camera_prompt"] = camera_prompt
        return PipelineResult(
            phase_reached="visual_gate",
            blocked=True,
            block_reason="visual_gate_fail",
            response=re_examine,
            router=router,
            lock=lock,
            gate=gate,
            rag_context="",
            machine_type=resolved_machine,
            language=language,
            rag_score=score,
            n_chunks=n_chunks,
        )

    logger.info(
        "Phase 3 PASSED: part=%s fault='%s' conf=%.2f",
        gate.part_id, gate.fault_description[:60], gate.confidence,
    )

    # ── All phases passed — safe to generate repair steps ────────────────────
    return PipelineResult(
        phase_reached="generation",
        blocked=False,
        block_reason="",
        response={
            "status":             "ready_for_generation",
            "machine_type":       resolved_machine,
            "visual_observation": gate.fault_description,
            "confirmed_part":     gate.part_id,
            "gate_confidence":    round(gate.confidence, 3),
            "rag_score":          round(score, 3),
            "chunks_used":        n_chunks,
            "camera_prompt":      camera_prompt,
            "clarification_round": clarification_round,
        },
        router=router,
        lock=lock,
        gate=gate,
        rag_context=context_str,
        machine_type=resolved_machine,
        language=language,
        rag_score=score,
        n_chunks=n_chunks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight helpers
# ─────────────────────────────────────────────────────────────────────────────

async def resolve_machine_from_query(query: str) -> RouterOutput:
    """
    Thin wrapper around route_query() for use in /agent/session.
    Call this when the client hasn't provided an explicit machine_type.
    """
    return await route_query(query)


def build_clarification_retry_query(
    original_query: str,
    clarification_answer: str,
    router: RouterOutput,
) -> str:
    """
    Build an enriched query that incorporates the farmer's clarification answer.
    Called by the /diagnose handler before retrying the pipeline.

    Example:
        original_query = "pump start nahi"
        clarification_answer = "Motor hums but shaft not rotating"
        → "water pump not starting motor hums shaft not rotating capacitor"
    """
    base = build_enriched_query(router)
    combined = f"{base} {clarification_answer}".strip()
    logger.info("Clarification retry query: '%s'", combined[:80])
    return combined