import os

# --- ChromaDB ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION",  "rag-documents")

# --- Embedding model ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Reranker model ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- LLM (via Ollama) ---
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

# --- Chunking ---
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",       512))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",    64))
RETRIEVAL_TOP_K  = int(os.getenv("RETRIEVAL_TOP_K",  20))
RERANK_TOP_N     = int(os.getenv("RERANK_TOP_N",     5))
# MIN_CHUNK_CHARS — trailing/orphan fragments shorter than this (e.g. a
# page-break leftover like "References" or a stray page number) get merged
# into the previous chunk instead of being kept as their own low-signal,
# retrieval-polluting chunk.
MIN_CHUNK_CHARS  = int(os.getenv("MIN_CHUNK_CHARS", 120))

# --- Hybrid retrieval (BM25 + optional vector) ---
BM25_WEIGHT   = float(os.getenv("BM25_WEIGHT",   0.4))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", 0.6))

# BM25-first adaptive thresholds (Feature 4)
# BM25_MIN_SCORE    — normalised BM25 score (0-1) a chunk must reach to be
#                     considered "confident" enough to skip vector search.
# BM25_CONFIDENCE_K — number of confident chunks required to skip vector search.
# Set BM25_CONFIDENCE_K=0 to ALWAYS run vector search (pure hybrid).
# Set BM25_MIN_SCORE=0.0  to NEVER skip vector search.
BM25_MIN_SCORE    = float(os.getenv("BM25_MIN_SCORE",    0.3))
BM25_CONFIDENCE_K = int(os.getenv("BM25_CONFIDENCE_K",   3))

# --- Security (Feature 2) ---
# MAX_PDF_SIZE_MB — reject uploads larger than this to limit DoS surface.
MAX_PDF_SIZE_MB   = float(os.getenv("MAX_PDF_SIZE_MB", 50.0))
# SECURITY_SCAN_ENABLED — set to "0" to disable (not recommended in prod).
SECURITY_SCAN_ENABLED = os.getenv("SECURITY_SCAN_ENABLED", "1") != "0"

# --- Semantic query-safety scan (Feature 1 extension) ---
# Catches paraphrased / obfuscated jailbreak & injection attempts that the
# regex layer in core/security.py misses (regex only catches known phrasings).
# Embeds the query and compares it against a bank of known-malicious example
# queries; flags it if cosine similarity to the nearest example clears the
# threshold. Reuses the same embedding model as retrieval, so no extra model
# download is required.
SEMANTIC_SAFETY_ENABLED   = os.getenv("SEMANTIC_SAFETY_ENABLED", "1") != "0"
SEMANTIC_SAFETY_THRESHOLD = float(os.getenv("SEMANTIC_SAFETY_THRESHOLD", 0.82))

# --- Rate limiting ---
# Token-bucket, per-client (IP, or X-Forwarded-For when behind a proxy).
# Two tiers: a cheap default for read-ish/info endpoints, and a stricter one
# for the expensive endpoints (LLM generation, PDF ingestion, evaluation).
RATE_LIMIT_ENABLED              = os.getenv("RATE_LIMIT_ENABLED", "1") != "0"
RATE_LIMIT_DEFAULT_PER_MINUTE   = int(os.getenv("RATE_LIMIT_DEFAULT_PER_MINUTE", 60))
RATE_LIMIT_EXPENSIVE_PER_MINUTE = int(os.getenv("RATE_LIMIT_EXPENSIVE_PER_MINUTE", 10))
RATE_LIMIT_BURST_MULTIPLIER     = float(os.getenv("RATE_LIMIT_BURST_MULTIPLIER", 1.5))

# --- Observability (Feature 3) ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- RAG evaluation ---
EVAL_OUTPUT_DIR   = os.getenv("EVAL_OUTPUT_DIR", "./eval_results")
EVAL_AUTO_QA_COUNT = int(os.getenv("EVAL_AUTO_QA_COUNT", 10))

# --- Agentic layer ---
# AGENTIC_ENABLED — set to "0" to bypass the whole agentic layer and fall
# back to the old "always hybrid-retrieve, always same model" behaviour.
AGENTIC_ENABLED = os.getenv("AGENTIC_ENABLED", "1") != "0"

# Model router (Feature 6). Both default to LLM_MODEL so the system works
# out of the box with a single pulled Ollama model; set LARGE_LLM_MODEL to
# a bigger model to actually get cost/latency tiering.
SMALL_LLM_MODEL = os.getenv("SMALL_LLM_MODEL", LLM_MODEL)
LARGE_LLM_MODEL = os.getenv("LARGE_LLM_MODEL", LLM_MODEL)

# Retrieval planner (Feature 5) — top_k / top_n bucketed by complexity.
PLANNER_TOP_K_SIMPLE     = int(os.getenv("PLANNER_TOP_K_SIMPLE",     10))
PLANNER_TOP_K_MODERATE   = int(os.getenv("PLANNER_TOP_K_MODERATE",   RETRIEVAL_TOP_K))
PLANNER_TOP_K_COMPLEX    = int(os.getenv("PLANNER_TOP_K_COMPLEX",    30))
PLANNER_TOP_N_SIMPLE     = int(os.getenv("PLANNER_TOP_N_SIMPLE",     3))
PLANNER_TOP_N_MODERATE   = int(os.getenv("PLANNER_TOP_N_MODERATE",   RERANK_TOP_N))
PLANNER_TOP_N_COMPLEX    = int(os.getenv("PLANNER_TOP_N_COMPLEX",    8))

# Classifier thresholds (Feature 1) — deterministic, no LLM calls.
COMPLEXITY_SIMPLE_MAX_WORDS   = int(os.getenv("COMPLEXITY_SIMPLE_MAX_WORDS",   8))
COMPLEXITY_COMPLEX_MIN_WORDS  = int(os.getenv("COMPLEXITY_COMPLEX_MIN_WORDS",  25))
QUERY_MIN_WORDS               = int(os.getenv("QUERY_MIN_WORDS",               2))

# Conversation / session (context-dependency detection + rewriting need
# short-term memory of the last few turns).
SESSION_HISTORY_TURNS = int(os.getenv("SESSION_HISTORY_TURNS", 4))
SESSION_TTL_SECONDS   = int(os.getenv("SESSION_TTL_SECONDS", 3600))
