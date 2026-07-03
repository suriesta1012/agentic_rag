"""
schemas.py
----------
Structured metadata types shared across the agentic layer.

Nothing in this file does any work — it's just the vocabulary that
classifiers.py, router.py, retrieval_planner.py, and model_router.py all
speak, so the pipeline can pass a single object between stages instead of
a pile of loose booleans.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Classification (Feature 1)
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    CHITCHAT      = "chitchat"       # greeting, thanks, small talk
    FACTUAL_QA    = "factual_qa"     # single-fact lookup
    SUMMARIZATION = "summarization"  # "summarize X", "list all Y"
    COMPARISON    = "comparison"     # "X vs Y", "difference between..."
    MULTI_HOP     = "multi_hop"      # chained / multi-part questions
    OUT_OF_SCOPE  = "out_of_scope"   # nothing to retrieve (e.g. "what's your name")


class Complexity(str, Enum):
    SIMPLE   = "simple"
    MODERATE = "moderate"
    COMPLEX  = "complex"


@dataclass
class ClassificationResult:
    # Safety
    is_safe: bool
    safety_reason: Optional[str] = None
    # Which layer(s) flagged it: "regex" and/or "semantic". Empty when safe.
    safety_flags: List[str] = field(default_factory=list)
    # Highest cosine similarity to the malicious-intent example bank (Feature 1
    # semantic scan). None when the semantic scan was skipped/disabled.
    semantic_similarity: Optional[float] = None

    # Intent
    intent: Intent = Intent.FACTUAL_QA

    # Complexity
    complexity: Complexity = Complexity.MODERATE
    word_count: int = 0

    # Context / conversation dependency
    is_context_dependent: bool = False
    coreference_terms: List[str] = field(default_factory=list)

    # Query quality
    needs_rewrite: bool = False
    quality_issues: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        d = {
            "is_safe": self.is_safe,
            "safety_reason": self.safety_reason,
            "safety_flags": self.safety_flags,
            "semantic_similarity": self.semantic_similarity,
            "intent": self.intent.value,
            "complexity": self.complexity.value,
            "word_count": self.word_count,
            "is_context_dependent": self.is_context_dependent,
            "coreference_terms": self.coreference_terms,
            "needs_rewrite": self.needs_rewrite,
            "quality_issues": self.quality_issues,
        }
        return d


# ---------------------------------------------------------------------------
# Routing (Feature 2)
# ---------------------------------------------------------------------------

class RouterAction(str, Enum):
    REJECT         = "reject"          # unsafe — refuse, no LLM call
    CLARIFY        = "clarify"         # too ambiguous to proceed — ask the user
    DIRECT_ANSWER  = "direct_answer"   # chitchat / out-of-scope — skip RAG entirely
    REWRITE        = "rewrite"         # rewrite query, then continue to RAG
    DECOMPOSE      = "decompose"       # split into sub-queries, then RAG each
    DIRECT_RAG     = "direct_rag"      # well-formed standalone query — RAG as-is


@dataclass
class RouterDecision:
    action: RouterAction
    reason: str


# ---------------------------------------------------------------------------
# Retrieval planning (Feature 5)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalPlan:
    top_k: int
    top_n: int
    force_hybrid: bool                     # skip BM25-only fast path
    metadata_filters: Optional[Dict] = None
    strategy: str = "bm25_first"           # "bm25_first" | "force_hybrid"


# ---------------------------------------------------------------------------
# Model routing (Feature 6)
# ---------------------------------------------------------------------------

@dataclass
class ModelChoice:
    model_name: str
    tier: str  # "small" | "large"


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------

@dataclass
class AgenticResult:
    """Everything api/app.py needs to build a response, plus debug metadata."""

    # Terminal outcome
    rejected: bool = False
    needs_clarification: bool = False
    message: Optional[str] = None          # set when rejected / needs_clarification
    answer: Optional[str] = None           # set on a normal answer

    # What the router decided and why
    router_action: Optional[RouterAction] = None
    router_reason: Optional[str] = None

    # Query transformations actually applied
    original_query: str = ""
    final_query: str = ""
    sub_queries: List[str] = field(default_factory=list)

    # Retrieval + model routing actually used
    retrieval_mode: Optional[str] = None
    model_used: Optional[str] = None
    model_tier: Optional[str] = None

    # Debug/observability
    classification: Optional[ClassificationResult] = None

    # Sources (langchain Document instances), kept generic to avoid a
    # hard import dependency on langchain in this module.
    sources: List = field(default_factory=list)
