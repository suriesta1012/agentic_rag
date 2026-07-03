"""
retrieval_planner.py
---------------------
Feature 5 — Retrieval Planner.

Deterministically decides *how much* to retrieve and *how hard* to look,
based on the classification output. Does NOT touch how retrieval works
internally — HybridRetriever, RRF, and the cross-encoder reranker are
untouched; this module only chooses their input parameters (top_k,
top_n, and whether to force the hybrid BM25+vector path instead of
letting HybridRetriever's own BM25-confidence shortcut decide).

  simple factual query   -> small top_k/top_n, BM25-first is fine
  moderate query          -> defaults from config.py
  comparison / multi-hop
  / complex query         -> larger top_k/top_n, force hybrid fusion
                             (a single BM25-confident sub-query shouldn't
                             short-circuit recall on a multi-part question)
"""

from config import (
    PLANNER_TOP_K_COMPLEX,
    PLANNER_TOP_K_MODERATE,
    PLANNER_TOP_K_SIMPLE,
    PLANNER_TOP_N_COMPLEX,
    PLANNER_TOP_N_MODERATE,
    PLANNER_TOP_N_SIMPLE,
)

from .schemas import ClassificationResult, Complexity, Intent, RetrievalPlan


def plan_retrieval(classification: ClassificationResult) -> RetrievalPlan:
    c = classification

    force_hybrid = c.intent in (Intent.COMPARISON, Intent.MULTI_HOP) or c.complexity == Complexity.COMPLEX

    if c.complexity == Complexity.SIMPLE:
        top_k, top_n = PLANNER_TOP_K_SIMPLE, PLANNER_TOP_N_SIMPLE
    elif c.complexity == Complexity.COMPLEX:
        top_k, top_n = PLANNER_TOP_K_COMPLEX, PLANNER_TOP_N_COMPLEX
    else:
        top_k, top_n = PLANNER_TOP_K_MODERATE, PLANNER_TOP_N_MODERATE

    return RetrievalPlan(
        top_k=top_k,
        top_n=top_n,
        force_hybrid=force_hybrid,
        metadata_filters=None,
        strategy="force_hybrid" if force_hybrid else "bm25_first",
    )
