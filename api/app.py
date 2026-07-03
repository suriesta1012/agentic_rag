"""
Local RAG API  v4.0
-------------------
Endpoints
---------
GET  /              — status
GET  /health        — readiness probe
GET  /metrics       — Prometheus-compatible metrics
POST /upload        — ingest a PDF (incremental re-embedding + security scan)
POST /query         — BM25-first retrieval → optional vector → rerank → LLM
POST /eval          — evaluate a list of Q&A samples
POST /eval/auto     — auto-generate Q&A from the knowledge base and evaluate
DELETE /reset       — wipe the vector store + hash store

New in v4.0
-----------
  ① Incremental re-embedding: only changed chunks are re-embedded on re-upload.
  ② Security scan: uploaded docs and user queries are checked for injection
     patterns (regex) and semantic jailbreak/injection intent (embedding
     similarity — see core/semantic_guard.py).
  ③ Observability: structured JSON logs + Prometheus metrics on /metrics.
  ④ BM25-first retrieval: vector search is skipped when BM25 is already confident.
  ⑤ Rate limiting: per-client token-bucket limits, stricter on the
     expensive endpoints (/query, /upload, /eval*) — see core/rate_limiter.py.
"""

import os
import shutil
import tempfile
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from agentic.classifiers import classify
from agentic.pipeline import run_agentic_query
from agentic.router import decide
from config import (
    AGENTIC_ENABLED,
    MAX_PDF_SIZE_MB,
    RATE_LIMIT_ENABLED,
    RERANK_TOP_N,
    RETRIEVAL_TOP_K,
    SECURITY_SCAN_ENABLED,
)
from core.chunk_hash_store import (
    delete_hashes_for_source,
    find_changed_chunks,
    upsert_hashes,
)
from core.document_processor import load_and_chunk_pdf
from core.observability import log_event, metrics, timed
from core.rate_limiter import limiter
from core.security import SecurityViolation, sanitize_query, scan_chunks
from core.session_store import clear_session, get_history
from core.vectorstore import get_vectorstore
from evaluation.evaluator import EvalReport, EvalSample, Evaluator
from langchain.schema import Document
from services.hybrid_retriever import HybridRetriever
from services.llm import get_llm
from services.reranker import rerank

app = FastAPI(
    title="Local RAG System",
    description=(
        "Fully local RAG v5: agentic request layer (classify → route → "
        "rewrite/decompose → plan → retrieve → route model) on top of "
        "incremental re-embedding, prompt-injection guardrails, "
        "observability, and BM25-first adaptive retrieval."
    ),
    version="5.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to answer.")
    top_k: Optional[int] = Field(
        default=None, ge=1, le=100,
        description="Override the agentic retrieval planner's top_k. Leave unset to let it decide.",
    )
    top_n: Optional[int] = Field(
        default=None, ge=1, le=20,
        description="Override the agentic retrieval planner's top_n. Leave unset to let it decide.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Opaque conversation id. Enables follow-up detection and query rewriting.",
    )


class SourceChunk(BaseModel):
    page_content: str
    source: Optional[str] = None
    page: Optional[int] = None
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    retrieval_mode: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    retrieval_mode: str
    latency_ms: float
    sources: List[SourceChunk]
    # Agentic layer metadata (present when AGENTIC_ENABLED=1)
    router_action: Optional[str] = None
    router_reason: Optional[str] = None
    original_query: Optional[str] = None
    final_query: Optional[str] = None
    sub_queries: List[str] = Field(default_factory=list)
    model_used: Optional[str] = None
    model_tier: Optional[str] = None
    classification: Optional[dict] = None


class UploadResponse(BaseModel):
    filename: str
    total_chunks: int
    new_or_changed_chunks: int
    skipped_unchanged_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    vectorstore: str
    llm_model: str
    collection_count: int


class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ClassifyResponse(BaseModel):
    classification: dict
    router_action: str
    router_reason: str


class EvalSampleRequest(BaseModel):
    question: str = Field(..., min_length=1)
    expected_answer: Optional[str] = None


class EvalRequest(BaseModel):
    samples: List[EvalSampleRequest] = Field(..., min_items=1)
    save_results: bool = True


class EvalAutoRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=50)
    save_results: bool = True


class EvalResultResponse(BaseModel):
    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float]
    latency_ms: float
    overall_score: float
    sources: List[str]


class EvalReportResponse(BaseModel):
    sample_count: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: Optional[float]
    avg_latency_ms: float
    avg_overall_score: float
    results: List[EvalResultResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_to_source(doc: Document) -> SourceChunk:
    return SourceChunk(
        page_content=doc.page_content,
        source=doc.metadata.get("source"),
        page=doc.metadata.get("page"),
        rerank_score=doc.metadata.get("rerank_score"),
        rrf_score=doc.metadata.get("rrf_score"),
        bm25_rank=doc.metadata.get("bm25_rank"),
        vector_rank=doc.metadata.get("vector_rank"),
        retrieval_mode=doc.metadata.get("retrieval_mode"),
    )


def _report_to_response(report: EvalReport) -> EvalReportResponse:
    recall = report.avg_context_recall
    return EvalReportResponse(
        sample_count=len(report.results),
        avg_faithfulness=round(report.avg_faithfulness, 4),
        avg_answer_relevancy=round(report.avg_answer_relevancy, 4),
        avg_context_precision=round(report.avg_context_precision, 4),
        avg_context_recall=round(recall, 4) if recall == recall else None,
        avg_latency_ms=round(report.avg_latency_ms, 1),
        avg_overall_score=round(report.avg_overall, 4),
        results=[
            EvalResultResponse(
                question=r.question,
                answer=r.answer,
                faithfulness=round(r.faithfulness, 4),
                answer_relevancy=round(r.answer_relevancy, 4),
                context_precision=round(r.context_precision, 4),
                context_recall=round(r.context_recall, 4) if r.expected_answer else None,
                latency_ms=round(r.latency_ms, 1),
                overall_score=round(r.overall_score, 4),
                sources=r.sources,
            )
            for r in report.results
        ],
    )


# ---------------------------------------------------------------------------
# Middleware — per-request metrics
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    route = request.url.path
    method = request.method
    metrics.inc("http_requests_total", method=method, route=route,
                status=str(response.status_code))
    metrics.observe("http_request_duration_seconds", duration,
                    method=method, route=route)
    return response


# ---------------------------------------------------------------------------
# Middleware — rate limiting
# ---------------------------------------------------------------------------
# Added *after* the metrics middleware above: Starlette wraps middleware in
# the order they're registered, and the last one registered ends up
# outermost — i.e. it sees the request first. We want that here so a
# rate-limited request is rejected before it ever reaches routing, retrieval,
# or the LLM, instead of paying that cost and only then being counted.

# Endpoints that trigger embedding/LLM/PDF-parsing work get the stricter
# "expensive" budget; everything else (health checks, /classify, /metrics,
# session management) shares the more generous "default" budget.
_EXPENSIVE_PATH_PREFIXES = ("/query", "/upload", "/eval")


def _client_key(request: Request) -> str:
    """
    Identify the caller for rate-limiting purposes.

    Prefers X-Forwarded-For (first hop) when present, since this API is
    commonly run behind a reverse proxy where request.client.host would
    otherwise be the proxy's own address for every caller. Falls back to
    the direct connection address, then to a constant so an unknown/absent
    client still gets *a* bucket rather than bypassing the limiter.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    if not RATE_LIMIT_ENABLED:
        return await call_next(request)

    path = request.url.path
    tier = "expensive" if path.startswith(_EXPENSIVE_PATH_PREFIXES) else "default"
    client_key = _client_key(request)

    allowed, retry_after = limiter.check(client_key, tier=tier)
    if not allowed:
        metrics.inc("rate_limit_exceeded", tier=tier)
        log_event(
            "rate_limit_exceeded",
            level="warning",
            client=client_key,
            tier=tier,
            route=path,
            retry_after=retry_after,
        )
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    f"Rate limit exceeded for {tier} endpoints. "
                    f"Retry in {retry_after} seconds."
                )
            },
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Routes — info
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
async def root():
    return {"message": "Local RAG System v4.0 is running. See /docs for the API."}


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        return HealthResponse(
            status="ok",
            vectorstore="chromadb",
            llm_model=get_llm().model,
            collection_count=count,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}")


@app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
async def prometheus_metrics():
    """Prometheus-compatible text metrics."""
    return metrics.prometheus_text()


# ---------------------------------------------------------------------------
# Routes — agentic layer introspection
# ---------------------------------------------------------------------------

@app.post("/classify", response_model=ClassifyResponse, tags=["Agentic"])
async def classify_query(request: ClassifyRequest):
    """
    Run only the parallel classification + router stages (Feature 1 + 2)
    and return the metadata — no retrieval, no LLM generation. Useful for
    debugging routing decisions without paying for a full query.
    """
    history = get_history(request.session_id)
    classification = await classify(request.query, history)
    decision = decide(classification, has_history=bool(history))
    return ClassifyResponse(
        classification=classification.as_dict(),
        router_action=decision.action.value,
        router_reason=decision.reason,
    )


@app.delete("/session/{session_id}", tags=["Agentic"])
async def reset_session(session_id: str):
    """Clear a conversation's short-term history (used for follow-up detection)."""
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}


# ---------------------------------------------------------------------------
# Routes — ingestion  (Feature ① + ②)
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF.

    • Security scan rejects documents containing prompt-injection payloads.
    • Incremental hashing skips re-embedding unchanged chunks on re-upload.
    • Only new / modified chunks are written to ChromaDB.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # ── Size guard ────────────────────────────────────────────────────
    max_bytes = int(MAX_PDF_SIZE_MB * 1024 * 1024)
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)//1024//1024} MB). "
                   f"Max allowed: {MAX_PDF_SIZE_MB} MB.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # ── 1. Parse + chunk ──────────────────────────────────────────
        with timed("document_chunking", source=file.filename):
            chunks = load_and_chunk_pdf(tmp_path)

        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted.")

        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        # ── 2. Security scan (Feature ②) ──────────────────────────────
        if SECURITY_SCAN_ENABLED:
            try:
                with timed("security_scan", source=file.filename):
                    scan_chunks(chunks, source_filename=file.filename)
            except SecurityViolation as exc:
                metrics.inc("security_violations", reason="document_injection")
                log_event(
                    "security_violation",
                    level="warning",
                    source=file.filename,
                    detail=str(exc),
                )
                raise HTTPException(status_code=422, detail=str(exc))

        # ── 3. Incremental hashing (Feature ①) ───────────────────────
        new_chunk_pairs = [
            (chunk.metadata.get("chunk_index", i), chunk.page_content)
            for i, chunk in enumerate(chunks)
        ]

        changed_pairs, stale_ids = find_changed_chunks(file.filename, new_chunk_pairs)
        changed_indices = {idx for idx, _ in changed_pairs}

        chunks_to_embed = [
            chunk for chunk in chunks
            if chunk.metadata.get("chunk_index", -1) in changed_indices
               or chunk.metadata.get("chunk_index") is None
        ]

        skipped = len(chunks) - len(chunks_to_embed)

        log_event(
            "incremental_embed",
            source=file.filename,
            total=len(chunks),
            to_embed=len(chunks_to_embed),
            skipped=skipped,
            stale_ids=len(stale_ids),
        )

        # ── 4. Remove stale chunks from ChromaDB ─────────────────────
        vs = get_vectorstore()
        if stale_ids:
            # ChromaDB stores chunk_id in metadata; delete by where filter
            for stale_id in stale_ids:
                source, _, idx_str = stale_id.partition(":")
                try:
                    vs._collection.delete(
                        where={"$and": [
                            {"source": {"$eq": source}},
                            {"chunk_index": {"$eq": int(idx_str)}},
                        ]}
                    )
                except Exception:
                    pass  # best-effort; log but don't fail the upload

        # ── 5. Embed + store only changed chunks ─────────────────────
        if chunks_to_embed:
            with timed("embedding", source=file.filename, chunks=len(chunks_to_embed)):
                vs.add_documents(chunks_to_embed)

        # ── 6. Persist updated hashes ─────────────────────────────────
        upsert_hashes(file.filename, new_chunk_pairs)

        metrics.inc("documents_ingested")
        metrics.inc("chunks_embedded", amount=len(chunks_to_embed))

        return UploadResponse(
            filename=file.filename,
            total_chunks=len(chunks),
            new_or_changed_chunks=len(chunks_to_embed),
            skipped_unchanged_chunks=skipped,
            message=(
                f"Ingested '{file.filename}': "
                f"{len(chunks_to_embed)} chunk(s) embedded, "
                f"{skipped} unchanged chunk(s) skipped."
            ),
        )

    except (ValueError, HTTPException):
        raise
    except Exception as exc:
        log_event("ingestion_error", level="error", source=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Routes — query  (Feature ② + ③ + ④)
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base.

    With AGENTIC_ENABLED=1 (default), the request first passes through the
    agentic layer (see agentic/pipeline.py):
      1. Parallel classification (safety, intent, complexity, context
         dependency, query quality) — no retrieval yet.
      2. Conditional routing: reject / ask for clarification / answer
         directly (chitchat) / rewrite / decompose / straight to RAG.
      3. Retrieval planning picks top_k/top_n/strategy dynamically.
      4. Existing BM25-first hybrid retrieval, RRF, cross-encoder rerank
         (unchanged — see services/hybrid_retriever.py, reranker.py).
      5. Model router picks the small or large Ollama model.

    With AGENTIC_ENABLED=0, falls back to the original fixed pipeline:
    sanitise, hybrid retrieve, rerank, single LLM call.
    """
    if SECURITY_SCAN_ENABLED:
        try:
            query = sanitize_query(request.query)
        except SecurityViolation as exc:
            metrics.inc("security_violations", reason="query_injection")
            raise HTTPException(status_code=400, detail=str(exc))
    else:
        query = request.query

    vs = get_vectorstore()
    if vs._collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base is empty. Upload at least one document first.",
        )

    pipeline_start = time.perf_counter()

    if AGENTIC_ENABLED:
        result = await run_agentic_query(
            query,
            session_id=request.session_id,
            top_k_override=request.top_k,
            top_n_override=request.top_n,
        )
        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)
        metrics.observe("query_latency_ms", total_ms)
        metrics.inc("queries_total")

        log_event(
            "query_complete",
            query=query[:80],
            router_action=result.router_action.value if result.router_action else None,
            retrieval_mode=result.retrieval_mode,
            model_used=result.model_used,
            latency_ms=total_ms,
        )

        if result.rejected or result.needs_clarification:
            return QueryResponse(
                answer=result.message or "",
                retrieval_mode="rejected" if result.rejected else "clarification_needed",
                latency_ms=total_ms,
                sources=[],
                router_action=result.router_action.value if result.router_action else None,
                router_reason=result.router_reason,
                original_query=result.original_query,
                final_query=result.final_query,
            )

        return QueryResponse(
            answer=result.answer or "",
            retrieval_mode=result.retrieval_mode or "none",
            latency_ms=total_ms,
            sources=[_doc_to_source(doc) for doc in result.sources],
            router_action=result.router_action.value if result.router_action else None,
            router_reason=result.router_reason,
            original_query=result.original_query,
            final_query=result.final_query,
            sub_queries=result.sub_queries,
            model_used=result.model_used,
            model_tier=result.model_tier,
            classification=result.classification.as_dict() if result.classification else None,
        )

    # ── Legacy pipeline (AGENTIC_ENABLED=0): fixed retrieve, rerank, LLM ─
    legacy_top_k = request.top_k or RETRIEVAL_TOP_K
    legacy_top_n = request.top_n or RERANK_TOP_N

    try:
        retriever = HybridRetriever.from_vectorstore()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    with timed("retrieval", query=query[:80]):
        candidates: List[Document] = retriever.retrieve(query, k=legacy_top_k)

    if not candidates:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    retrieval_mode = candidates[0].metadata.get("retrieval_mode", "hybrid")

    with timed("reranking", candidates=len(candidates)):
        top_docs = rerank(query, candidates, top_n=legacy_top_n)

    context = "\n\n---\n\n".join(doc.page_content for doc in top_docs)
    prompt = (
        "Use only the context below to answer the question. "
        'If the answer is not in the context, say "I don\'t know."\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    with timed("llm_generation"):
        try:
            answer = get_llm().invoke(prompt)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"LLM error (is Ollama running?): {exc}"
            )

    total_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)
    metrics.observe("query_latency_ms", total_ms)
    metrics.inc("queries_total")

    log_event(
        "query_complete",
        query=query[:80],
        retrieval_mode=retrieval_mode,
        top_k=legacy_top_k,
        top_n=legacy_top_n,
        latency_ms=total_ms,
    )

    return QueryResponse(
        answer=answer,
        retrieval_mode=retrieval_mode,
        latency_ms=total_ms,
        sources=[_doc_to_source(doc) for doc in top_docs],
    )


# ---------------------------------------------------------------------------
# Routes — evaluation
# ---------------------------------------------------------------------------

@app.post("/eval", response_model=EvalReportResponse, tags=["Evaluation"])
async def evaluate(request: EvalRequest):
    vs = get_vectorstore()
    if vs._collection.count() == 0:
        raise HTTPException(status_code=404, detail="Knowledge base is empty.")

    samples = [
        EvalSample(question=s.question, expected_answer=s.expected_answer)
        for s in request.samples
    ]
    try:
        evaluator = Evaluator()
        report = evaluator.run(samples, save=request.save_results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")

    return _report_to_response(report)


@app.post("/eval/auto", response_model=EvalReportResponse, tags=["Evaluation"])
async def evaluate_auto(request: EvalAutoRequest):
    vs = get_vectorstore()
    if vs._collection.count() == 0:
        raise HTTPException(status_code=404, detail="Knowledge base is empty.")

    try:
        evaluator = Evaluator()
        report = evaluator.run_auto(count=request.count, save=request.save_results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auto-evaluation failed: {exc}")

    return _report_to_response(report)


# ---------------------------------------------------------------------------
# Routes — reset
# ---------------------------------------------------------------------------

@app.delete("/reset", tags=["Ingestion"])
async def reset_knowledge_base():
    """Wipe the vector store and the chunk-hash store."""
    try:
        vs = get_vectorstore()
        # Delete all docs for every source tracked in the hash store
        import sqlite3
        from pathlib import Path
        from config import CHROMA_PERSIST_DIR
        db_path = Path(CHROMA_PERSIST_DIR) / "chunk_hashes.db"
        if db_path.exists():
            con = sqlite3.connect(str(db_path))
            sources = [r[0] for r in con.execute("SELECT DISTINCT source FROM chunk_hashes").fetchall()]
            con.execute("DELETE FROM chunk_hashes")
            con.commit()
            con.close()

        vs._client.delete_collection(vs._collection.name)
        get_vectorstore.cache_clear()

        metrics.inc("resets_total")
        log_event("knowledge_base_reset")
        return {"message": "Knowledge base and hash store cleared successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
