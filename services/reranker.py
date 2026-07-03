# reranker.py
"""
Cross-encoder reranker.

After the vector store returns RETRIEVAL_TOP_K candidates by cosine
similarity, the reranker scores every (query, chunk) pair more
precisely and returns only the top RERANK_TOP_N chunks.

Why bother?
  Bi-encoder embeddings (used in the vector store) compress a whole
  sentence into a single vector — fast but lossy. A cross-encoder
  reads the query and chunk *together*, giving much more accurate
  relevance scores at the cost of being slower. Running it only on
  the small candidate set keeps latency acceptable.
"""

from functools import lru_cache
from typing import List

from langchain.schema import Document
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RERANK_TOP_N


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    """Download / load the cross-encoder once and cache it."""
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, documents: List[Document], top_n: int = RERANK_TOP_N) -> List[Document]:
    """
    Re-score *documents* against *query* and return the top_n most relevant.

    Args:
        query:     The user's question.
        documents: Candidate documents from the vector store.
        top_n:     How many to keep (defaults to RERANK_TOP_N from config).

    Returns:
        A list of at most top_n Documents, sorted by descending relevance.
    """
    if not documents:
        return []

    encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = encoder.predict(pairs)

    # Attach score to each doc and sort descending
    scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:top_n]]

    # Optionally store the reranker score in metadata for debugging
    for score, doc in scored[:top_n]:
        doc.metadata["rerank_score"] = round(float(score), 4)

    return top_docs
