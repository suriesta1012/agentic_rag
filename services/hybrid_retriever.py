"""
hybrid_retriever.py
-------------------
Retrieval strategy: BM25-first with lazy vector search.

Default flow
~~~~~~~~~~~~
1.  BM25 runs over all stored chunks (fast, in-process, no GPU needed).
2.  If BM25 returns enough high-confidence results (≥ bm25_confidence_k
    chunks whose normalised score ≥ bm25_min_score), we SKIP vector
    search entirely — saving embedding inference + ChromaDB round-trip.
3.  If BM25 results are thin or low-confidence, we ALSO run ChromaDB
    vector search and fuse both lists with Reciprocal Rank Fusion (RRF).

Why BM25-first?
~~~~~~~~~~~~~~~
Vector search requires a round-trip through the embedding model and
the ChromaDB index.  For keyword-heavy queries (exact product names,
codes, IDs, proper nouns) BM25 is often both faster AND more accurate.
The adaptive threshold gives us the speed win when it's safe to skip
vector search while preserving recall on semantic/paraphrase queries.

Config knobs (all in config.py / env)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  BM25_MIN_SCORE        — normalised score threshold (0-1) above which
                          a BM25 result is considered "confident"
  BM25_CONFIDENCE_K     — how many confident BM25 hits we need before
                          we skip vector search
  BM25_WEIGHT           — RRF weight for BM25 when fusion IS used
  VECTOR_WEIGHT         — RRF weight for vector search when fusion IS used
  RETRIEVAL_TOP_K       — candidates to return before reranking
"""

import math
from typing import List, Tuple

from langchain.schema import Document
from rank_bm25 import BM25Okapi

from config import (
    BM25_CONFIDENCE_K,
    BM25_MIN_SCORE,
    BM25_WEIGHT,
    RETRIEVAL_TOP_K,
    VECTOR_WEIGHT,
)
from core.observability import log_event, metrics, timed
from core.vectorstore import get_vectorstore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokeniser for BM25."""
    return text.lower().split()


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a document at 1-indexed *rank*."""
    return 1.0 / (k + rank)


def _normalize_bm25(scores: List[float]) -> List[float]:
    """
    Min-max normalise BM25 scores to [0, 1].
    If all scores are equal (including all-zero), return zeros.
    """
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    BM25-first hybrid retriever.

    Call `HybridRetriever.from_vectorstore()` to build from ChromaDB, or
    pass documents directly for testing.
    """

    def __init__(
        self,
        documents: List[Document],
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_min_score: float = BM25_MIN_SCORE,
        bm25_confidence_k: int = BM25_CONFIDENCE_K,
    ):
        self.documents = documents
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25_min_score = bm25_min_score
        self.bm25_confidence_k = bm25_confidence_k

        tokenised = [_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenised)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_vectorstore(
        cls,
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_min_score: float = BM25_MIN_SCORE,
        bm25_confidence_k: int = BM25_CONFIDENCE_K,
    ) -> "HybridRetriever":
        vs = get_vectorstore()
        raw = vs._collection.get(include=["documents", "metadatas"])

        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]

        if not docs:
            raise ValueError(
                "Vector store is empty — upload at least one document before "
                "building the hybrid retriever."
            )

        return cls(
            docs,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            bm25_min_score=bm25_min_score,
            bm25_confidence_k=bm25_confidence_k,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[Document]:
        """
        Return the top-k documents using BM25-first adaptive strategy.

        Emits structured log events and updates metrics counters.
        """
        fetch_k = min(k * 3, len(self.documents))

        # ── Step 1: BM25 ──────────────────────────────────────────────
        with timed("bm25_search", log=False):
            raw_scores = self.bm25.get_scores(_tokenize(query))
            norm_scores = _normalize_bm25(list(raw_scores))

            bm25_ranked: List[Tuple[int, float]] = sorted(
                enumerate(norm_scores), key=lambda x: x[1], reverse=True
            )[:fetch_k]

        confident_hits = [
            (idx, score) for idx, score in bm25_ranked
            if score >= self.bm25_min_score
        ]

        use_vector = len(confident_hits) < self.bm25_confidence_k

        # ── Step 2: Vector search (only if needed) ────────────────────
        vector_hits: List[Document] = []
        if use_vector:
            metrics.inc("vector_search_calls", method="query")
            with timed("vector_search", log=False):
                vs = get_vectorstore()
                vector_hits = vs.similarity_search(query, k=fetch_k)

        # ── Step 3: Build result set ──────────────────────────────────
        if not use_vector:
            # Pure BM25 path — annotate and return
            metrics.inc("bm25_only_retrievals")
            results: List[Document] = []
            for rank, (idx, norm_score) in enumerate(bm25_ranked[:k], start=1):
                doc = self.documents[idx]
                doc.metadata["bm25_score"] = round(norm_score, 6)
                doc.metadata["bm25_rank"] = rank
                doc.metadata["retrieval_mode"] = "bm25_only"
                results.append(doc)

            log_event(
                "retrieval_complete",
                mode="bm25_only",
                k=k,
                confident_hits=len(confident_hits),
                results=len(results),
            )
            return results

        # ── Step 4: RRF fusion ────────────────────────────────────────
        metrics.inc("hybrid_retrievals")

        bm25_rank_map: dict[str, int] = {}
        for rank, (idx, _) in enumerate(bm25_ranked, start=1):
            bm25_rank_map[self.documents[idx].page_content] = rank

        vector_rank_map: dict[str, int] = {}
        for rank, doc in enumerate(vector_hits, start=1):
            vector_rank_map[doc.page_content] = rank

        seen: dict[str, Document] = {}
        for idx, _ in bm25_ranked:
            d = self.documents[idx]
            seen[d.page_content] = d
        for d in vector_hits:
            seen[d.page_content] = d

        rrf_scores: dict[str, float] = {}
        for content in seen:
            b = self.bm25_weight * _rrf_score(bm25_rank_map.get(content, fetch_k + 1))
            v = self.vector_weight * _rrf_score(vector_rank_map.get(content, fetch_k + 1))
            rrf_scores[content] = b + v

        ranked = sorted(seen.values(), key=lambda d: rrf_scores[d.page_content], reverse=True)

        for doc in ranked[:k]:
            doc.metadata["rrf_score"] = round(rrf_scores[doc.page_content], 6)
            doc.metadata["bm25_rank"] = bm25_rank_map.get(doc.page_content)
            doc.metadata["vector_rank"] = vector_rank_map.get(doc.page_content)
            doc.metadata["retrieval_mode"] = "hybrid"

        log_event(
            "retrieval_complete",
            mode="hybrid",
            k=k,
            confident_bm25_hits=len(confident_hits),
            results=min(k, len(ranked)),
        )
        return ranked[:k]
