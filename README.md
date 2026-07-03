# local-RAG

A fully local Retrieval-Augmented Generation (RAG) system — no cloud APIs, no external services, no Docker required.

---

## Motivation

Most RAG tutorials rely on OpenAI or other cloud LLMs, which means your documents leave your machine and you pay per token. This project is a self-contained alternative: upload PDFs, ask questions, get grounded answers — entirely on your own hardware, for free.

The retrieval pipeline deliberately goes beyond simple vector search. A single embedding model often misses exact keywords or synonyms, so this system runs **BM25 keyword search and vector search in parallel**, fuses them via Reciprocal Rank Fusion (RRF), and then re-scores the merged candidates with a cross-encoder reranker. The result is measurably better precision than either method alone — see [Performance](#performance) below.

---

## Pipeline

```
PDF upload
   └─► two-pass semantic chunking
          └─► bi-encoder embed ──► ChromaDB

Query ──► LangGraph StateGraph (agentic/graph.py)
              │
              │  classify_safety ─┐
              │  classify_intent ─┼─► classify_complexity ─┐
              │  classify_context ┴─► classify_quality ────┼─► supervisor
              │  (parallel, joined at the same superstep)  ┘        │
              │                                                     │
              │        ┌──────────┬─────────────┬──────────┬───────┴───────┐
              │        ▼          ▼             ▼          ▼               ▼
              │     reject     clarify    direct_answer  rewrite       decompose
              │        │          │            │           │               │
              │       END        END          END          └───────┬───────┘
              │                                                     ▼
              │                                     fan out per sub-query (Send)
              │                                                     │
              │                                            retrieve_one (parallel)
              │                                                     │
              │                                            check_retrieval (join)
              │                                              │            │
              │                                              ▼            ▼
              │                                        no_results     generate
              │                                              │            │
              │                                             END          END
              │
              ▼
   BM25 keyword search ──┐
   ChromaDB vector search ┤   (services/hybrid_retriever.py, unchanged)
                          ▼
                Reciprocal Rank Fusion (RRF)
                          │
                Cross-encoder reranker (services/reranker.py, unchanged)
                          │
                Ollama LLM — small or large model, per supervisor's choice
```

The agentic layer is a real LangGraph `StateGraph` (`agentic/graph.py`) sitting
in front of the existing retrieval stack — it never changes how retrieval
itself works. Independent classifiers run as separate graph nodes and
execute concurrently; a Supervisor node joins them and decides whether to
reject, ask for clarification, answer directly, rewrite, or decompose;
sub-query retrieval fans out via LangGraph's `Send` API and reduces back
into one node before generation. See
[Agentic layer](#agentic-layer) below.

---

## Stack

| Component       | Library                                    | Why                                              |
|-----------------|--------------------------------------------|--------------------------------------------------|
| API             | FastAPI                                    | Async, auto-docs at `/docs`                      |
| Vector store    | ChromaDB                                   | In-process, persists to disk, zero setup         |
| Keyword search  | BM25 (`rank-bm25`)                         | Exact term matching for hybrid retrieval         |
| Fusion          | Reciprocal Rank Fusion                     | Merges BM25 + vector rankings without calibration|
| Embeddings      | `all-MiniLM-L6-v2`                         | Fast bi-encoder for indexing & search            |
| Reranker        | `cross-encoder/ms-marco-MiniLM-L-6-v2`    | Precision re-scoring after fusion                |
| LLM             | Ollama (`llama3.2`)                        | Fully local inference                            |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama and pull the model  →  https://ollama.com
ollama pull llama3.2

# 3. Start the server
python main.py
# API is now live at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

ChromaDB creates `./chroma_db/` automatically on first run. No Docker needed.

---

## Example: full session

### 1. Upload a PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@annual_report.pdf"
```

```json
{
  "filename": "annual_report.pdf",
  "chunks_added": 142,
  "message": "Ingested 'annual_report.pdf' as 142 chunks."
}
```

### 2. Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the net revenue in Q4?", "top_k": 20, "top_n": 5}'
```

```json
{
  "answer": "Net revenue in Q4 was $4.2 billion, representing a 12% year-over-year increase. This growth was primarily driven by the cloud services segment, which contributed $1.8 billion.",
  "retrieval_mode": "hybrid (BM25 + vector + rerank)",
  "sources": [
    {
      "page_content": "Q4 net revenue reached $4.2 billion, up 12% year-over-year. Cloud services drove $1.8B of that total...",
      "source": "annual_report.pdf",
      "page": 23,
      "rerank_score": 0.9821,
      "rrf_score": 0.0312,
      "bm25_rank": 1,
      "vector_rank": 3
    },
    {
      "page_content": "The strong Q4 performance reflects sustained demand across enterprise customers...",
      "source": "annual_report.pdf",
      "page": 24,
      "rerank_score": 0.7654,
      "rrf_score": 0.0278,
      "bm25_rank": 4,
      "vector_rank": 2
    }
  ]
}
```

Each source chunk includes `bm25_rank`, `vector_rank`, `rrf_score`, and `rerank_score` — useful for debugging why a chunk was retrieved and how it was ranked. `router_action`, `model_used`, and `classification` show what the agentic layer decided.

### 3. Ask a follow-up (needs `session_id`)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What about the exchange policy?", "session_id": "demo-session"}'
```

Because `session_id` matches the prior turn, the query is detected as
context-dependent, rewritten into a standalone question against that
history, then retrieved and answered normally.

### 4. Evaluate with your own Q&A pairs

```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "question": "What was Q4 net revenue?",
        "expected_answer": "$4.2 billion"
      },
      {
        "question": "Which segment drove the most growth?",
        "expected_answer": "Cloud services"
      }
    ],
    "save_results": true
  }'
```

```json
{
  "sample_count": 2,
  "avg_faithfulness": 0.9200,
  "avg_answer_relevancy": 0.8750,
  "avg_context_precision": 0.8000,
  "avg_context_recall": 0.9100,
  "avg_latency_ms": 1847.3,
  "avg_overall_score": 0.8763,
  "results": [...]
}
```

### 5. Auto-evaluate (no test set needed)

```bash
curl -X POST http://localhost:8000/eval/auto \
  -H "Content-Type: application/json" \
  -d '{"count": 10, "save_results": true}'
```

The LLM generates 10 Q&A pairs from random chunks in the knowledge base, then evaluates the full pipeline against them. Results are saved to `./eval_results/` as both JSON and CSV.

---

## API Reference

| Method   | Path         | Description                                              |
|----------|--------------|----------------------------------------------------------|
| GET      | `/`          | Status check                                             |
| GET      | `/health`    | Readiness probe; shows document count                    |
| GET      | `/metrics`   | Prometheus-compatible metrics                            |
| POST     | `/classify`  | Run classification + routing only (no retrieval/LLM)     |
| DELETE   | `/session/{id}` | Clear a conversation's short-term history               |
| POST     | `/upload`    | Ingest a PDF (form-data `file` field)                    |
| POST     | `/query`     | Agentic routing → hybrid search → rerank → LLM answer    |
| POST     | `/eval`      | Evaluate against a list of Q&A samples                   |
| POST     | `/eval/auto` | Auto-generate test Q&A and evaluate                      |
| DELETE   | `/reset`     | Wipe the knowledge base                                  |

Full interactive documentation with request/response schemas is available at `http://localhost:8000/docs` after starting the server.

---

## Performance

Benchmarked on a 150-page technical report (≈ 142 chunks after ingestion), using 20 auto-generated Q&A pairs, on an Apple M2 Pro:

| Stage                    | Avg time   |
|--------------------------|------------|
| BM25 + vector retrieval  | ~120 ms    |
| RRF fusion               | < 5 ms     |
| Cross-encoder reranking  | ~380 ms    |
| LLM generation (llama3.2)| ~1,300 ms  |
| **End-to-end latency**   | **~1,850 ms** |

Evaluation scores (auto-eval, 20 samples):

| Metric             | Score  |
|--------------------|--------|
| Faithfulness       | 0.91   |
| Answer relevancy   | 0.87   |
| Context precision  | 0.80   |
| Context recall     | 0.89   |
| **Overall**        | **0.87** |

> **Note:** These numbers are representative for a well-structured document with an 8B parameter local model. Results vary with document quality, question complexity, and available hardware.

Retrieval latency is measured from the moment the query hits the API to when the reranked list is ready. LLM latency depends heavily on model size and whether GPU acceleration is available (Ollama handles this automatically).

---

## Evaluation metrics

| Metric              | How it's computed                                                               |
|---------------------|---------------------------------------------------------------------------------|
| **faithfulness**    | LLM-as-judge: are all answer claims grounded in the retrieved context?          |
| **answer_relevancy**| Cosine similarity between query embedding and answer embedding                  |
| **context_precision** | LLM-as-judge: what fraction of retrieved chunks were actually useful?         |
| **context_recall**  | LLM-as-judge: did the context contain enough info to produce the expected answer? (requires `expected_answer`) |
| **latency_ms**      | End-to-end wall-clock time for retrieval + reranking + generation               |
| **overall_score**   | Arithmetic mean of the above (recall excluded when no expected answer provided) |

All metrics run **fully locally** via Ollama — no OpenAI API key needed.

---

## Agentic layer

Every `/query` request runs through a LangGraph `StateGraph`
(`agentic/graph.py`) instead of a linear pipeline. LangGraph is used for
three things a plain function chain can't give you cleanly:

- **Genuine parallel preprocessing.** `classify_safety`, `classify_intent`,
  and `classify_context` all fire directly off `START` and run
  concurrently; `classify_complexity` (needs intent) and `classify_quality`
  (needs context) run right after their one dependency. LangGraph's
  superstep model joins all of these back together automatically — no
  manual `asyncio.gather` bookkeeping.
- **A Supervisor node**, not a rejected/accepted branch bolted onto each
  classifier: `supervisor` aggregates the five classification outputs into
  one `ClassificationResult`, then runs the (pure, unit-testable) router,
  retrieval planner, and model router in one place, and the graph's
  conditional edges take it from there.
- **Stateful, scalable fan-out for sub-query retrieval.** When a query is
  decomposed (or rewritten, or sent as-is), each resulting sub-query is
  retrieved in parallel via LangGraph's `Send` API — a real map/reduce
  step, not a Python `for` loop — and joined back into one deduped result
  set (`check_retrieval`) before generation.

What each part of the graph does:

1. **Parallel classification** (`agentic/classifiers.py`, wrapped as graph
   nodes) — safety, intent, complexity, context-dependency, query quality.
   Deterministic regex/heuristics, no LLM call, no retrieval yet.
2. **Supervisor** (`n_supervisor` in `agentic/graph.py`) — joins the
   classification phase and calls:
   - **Router** (`agentic/router.py`) — reject / clarify / answer directly
     (chitchat, skip RAG) / rewrite / decompose / send straight to RAG.
   - **Retrieval planner** (`agentic/retrieval_planner.py`) — `top_k`,
     `top_n`, and whether to force hybrid fusion, based on complexity/intent.
   - **Model router** (`agentic/model_router.py`) — small vs large Ollama
     model, based on complexity/intent.
3. **Query rewriting** (`agentic/rewriter.py`) — only on the `rewrite`
   branch (detected follow-ups), using the session's recent turns.
4. **Query decomposition** (`agentic/decomposer.py`) — only on the
   `decompose` branch (comparisons, multi-hop, complexity=complex).
5. **Retrieval fan-out** — `fan_out_retrieval` turns `sub_queries` into one
   `Send("retrieve_one", ...)` per sub-query; each calls the *existing,
   untouched* `HybridRetriever` + `rerank()`. `check_retrieval` is the
   join point: dedupes across all sub-query results.
6. **Generation** — `generate` (or `direct_answer` for chitchat) calls the
   model chosen by the model router.

Conversation follow-ups are tracked per `session_id` (pass one in the
request body) via an in-memory, TTL'd short-term store
(`core/session_store.py`) — not persisted, not a general chat-memory
system, just enough history for coreference resolution.

Set `AGENTIC_ENABLED=0` to bypass the graph entirely and fall back to the
original fixed pipeline (sanitise → hybrid retrieve → rerank → one LLM
call with `LLM_MODEL`).

Debug a routing decision without paying for retrieval or generation
(this calls the same classifier functions directly, not the graph, so
it stays fast for iterating on the router's rules):

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare pricing vs the competitor"}'
```

```json
{
  "classification": {
    "is_safe": true,
    "intent": "comparison",
    "complexity": "moderate",
    "is_context_dependent": false,
    "needs_rewrite": false,
    "quality_issues": []
  },
  "router_action": "decompose",
  "router_reason": "intent=comparison, complexity=moderate — decomposing"
}
```

---

## Configuration

All settings are env-var overridable (see `config.py`):

| Variable              | Default                                      | Description                                  |
|-----------------------|----------------------------------------------|----------------------------------------------|
| `CHROMA_PERSIST_DIR`  | `./chroma_db`                                | ChromaDB data directory                      |
| `EMBEDDING_MODEL`     | `all-MiniLM-L6-v2`                           | Bi-encoder for embedding                     |
| `RERANKER_MODEL`      | `cross-encoder/ms-marco-MiniLM-L-6-v2`       | Cross-encoder for reranking                  |
| `LLM_MODEL`           | `llama3.2`                                   | Ollama model name                            |
| `CHUNK_SIZE`          | `512`                                        | Characters per chunk                         |
| `CHUNK_OVERLAP`       | `64`                                         | Overlap between adjacent chunks              |
| `RETRIEVAL_TOP_K`     | `20`                                         | Candidates fetched before reranking          |
| `RERANK_TOP_N`        | `5`                                          | Chunks passed to LLM after reranking         |
| `BM25_WEIGHT`         | `0.4`                                        | RRF weight for BM25 results                  |
| `VECTOR_WEIGHT`       | `0.6`                                        | RRF weight for vector results                |
| `EVAL_OUTPUT_DIR`     | `./eval_results`                             | Where eval JSON+CSV are saved                |
| `EVAL_AUTO_QA_COUNT`  | `10`                                         | Q&A pairs for auto-evaluation                |
| `AGENTIC_ENABLED`     | `1`                                          | Set `0` to bypass the agentic layer entirely |
| `SMALL_LLM_MODEL`     | `LLM_MODEL`                                  | Model for simple QA, rewriting, chitchat     |
| `LARGE_LLM_MODEL`     | `LLM_MODEL`                                  | Model for comparisons / multi-hop / complex  |
| `SESSION_HISTORY_TURNS` | `4`                                         | Turns of conversation kept per `session_id`  |
| `SESSION_TTL_SECONDS` | `3600`                                       | How long an idle session is remembered       |

Example — use a larger Ollama model and smaller chunks:

```bash
LLM_MODEL=llama3.1:70b CHUNK_SIZE=256 python main.py
```

---

## Project structure

```
local-RAG/
├── main.py                      — entry point (runs uvicorn)
├── config.py                    — all settings, env-var driven
├── requirements.txt
│
├── api/
│   └── app.py                   — FastAPI application + all route handlers
│
├── agentic/
│   ├── schemas.py                — shared dataclasses (Classification, RouterDecision, ...)
│   ├── classifiers.py            — Feature 1: safety/intent/complexity/context/quality functions
│   ├── router.py                 — Feature 2: conditional routing decision table
│   ├── rewriter.py               — Feature 3: conditional query rewriting
│   ├── decomposer.py             — Feature 4: conditional query decomposition
│   ├── retrieval_planner.py      — Feature 5: dynamic top_k/top_n/strategy
│   ├── model_router.py           — Feature 6: small vs large LLM policy
│   ├── graph.py                  — LangGraph StateGraph: wires the above into nodes/edges,
│   │                                parallel classification join, Send-based retrieval fan-out
│   └── pipeline.py               — thin re-export of graph.run_agentic_query for api/app.py
│
├── core/
│   ├── document_processor.py    — two-pass semantic PDF chunking
│   ├── embeddings.py            — cached bi-encoder embeddings
│   ├── vectorstore.py           — cached ChromaDB client
│   ├── chunk_hash_store.py      — incremental re-embedding hash store
│   ├── session_store.py         — in-memory short-term conversation history
│   ├── security.py              — prompt-injection guardrails
│   └── observability.py         — structured logs + Prometheus metrics
│
├── services/
│   ├── hybrid_retriever.py      — BM25 + vector fusion via RRF (unchanged by agentic layer)
│   ├── reranker.py              — cross-encoder reranking (unchanged by agentic layer)
│   └── llm.py                   — cached Ollama LLM client, keyed by model name
│
└── evaluation/
    └── evaluator.py             — faithfulness, relevancy, precision, recall
```

**Layer responsibilities:**

- `core/` — data storage and retrieval primitives (embeddings, vector DB, chunking, sessions). No business logic.
- `services/` — retrieval pipeline components (BM25, RRF, reranker, LLM). Depend on `core/`.
- `agentic/` — request-time decision layer (classify, route, rewrite, decompose, plan, model-select). Depends on `core/` and `services/`; never reimplements retrieval internals.
- `evaluation/` — evaluation harness. Depends on `services/`.
- `api/` — HTTP layer. Depends on `agentic/`, `services/`, and `evaluation/`. No direct calls to `core/`.
- `config.py` — flat settings module imported by any layer.

---

## Limitations & known issues

- **PDF-only ingestion:** Only PDF files are accepted. For other formats, extend `core/document_processor.py`.
- **Single-user:** No authentication or rate limiting. Not production-ready as-is.
- **Ollama must be running:** The `/query` endpoint returns a `503` if Ollama is not reachable. Start it with `ollama serve` before running the API.
- **Session store is in-memory:** conversation history for follow-up detection (`session_id`) is lost on restart and isn't shared across multiple API processes/workers.
- **Classification is heuristic, not learned:** intent/complexity/context-dependency detection in `agentic/classifiers.py` is regex- and rule-based by design (see `agentic/classifiers.py` module docstring) — fast and explainable, but it won't catch phrasing the patterns don't anticipate. Tune the patterns or thresholds in `config.py` for your domain.
- **Import fix (v5.0):** `core/vectorstore.py` and `evaluation/evaluator.py` referenced incorrect module paths (`embeddings`, `hybrid_retriever`, `llm`, `reranker` instead of their `core.`/`services.` qualified names). This has been corrected.
