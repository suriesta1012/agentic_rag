"""
classifiers.py
---------------
Feature 1 — Parallel Request Classification.

Five independent classifiers run concurrently, before any retrieval
happens, and produce one structured ClassificationResult:

  1. Safety classification         — reuses core/security.py's pattern scan
  2. Intent detection              — chitchat / factual / comparison / ...
  3. Complexity estimation         — simple / moderate / complex
  4. Context-dependency detection  — is this a follow-up needing history?
  5. Query quality check           — is a rewrite needed before retrieval?

Deliberately deterministic (regex + heuristics), not LLM calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per the agentic scope, "LLM-based routing when deterministic rules
suffice" is explicitly out of scope. Every classifier here is a cheap,
explainable function — no model inference, no added latency, no extra
Ollama round-trip before we even know if we need one. The LLM budget is
spent later, only where generation is actually required (rewriting,
decomposition, final answer).

Each classifier function is a small, independent, side-effect-free unit
that can be called from two places:
  - `classify()` below — a plain asyncio.gather composite, used by the
    `/classify` debug endpoint when you want metadata without spinning
    up the full graph.
  - `agentic/graph.py` — wraps each one as a LangGraph node so the
    *actual* parallel fan-out and Supervisor aggregation happens inside
    the StateGraph (see that module for the real production path used
    by `/query`).
"""

import asyncio
import re
from typing import List

from config import (
    COMPLEXITY_COMPLEX_MIN_WORDS,
    COMPLEXITY_SIMPLE_MAX_WORDS,
    QUERY_MIN_WORDS,
    SEMANTIC_SAFETY_ENABLED,
)
from core.security import scan_text
from core.semantic_guard import scan_query_semantic
from core.session_store import Turn

from .schemas import ClassificationResult, Complexity, Intent

# ---------------------------------------------------------------------------
# Pattern libraries
# ---------------------------------------------------------------------------

_CHITCHAT_RE = re.compile(
    r"(?i)^\s*(hi|hello|hey|hiya|yo|thanks|thank you|thx|ty|ok|okay|"
    r"cool|great|good morning|good evening|how are you|who are you|"
    r"what'?s up|bye|goodbye)\b"
    r"(\s+(there|guys|team|all|claude))?"
    r"\s*[!.,]*\s*$"
)

_OUT_OF_SCOPE_RE = re.compile(
    r"(?i)\b(what'?s your name|are you (a )?(human|robot|ai)|"
    r"tell me a joke|what model are you|who (made|built|created) you)\b"
)

_COMPARISON_RE = re.compile(
    r"(?i)\b(vs\.?|versus|compare|comparison|difference between|"
    r"better than|worse than|pros and cons|which is (better|best))\b"
)

_MULTI_HOP_RE = re.compile(
    r"(?i)\b(and then|after that|as well as|in addition to|"
    r"first .{0,40}\bthen\b|both .{0,40}\band\b)\b"
)

_SUMMARY_RE = re.compile(
    r"(?i)\b(summar(y|ize|ise)|list all|give me an overview|"
    r"key (points|takeaways)|tl;?dr)\b"
)

# Leading pronoun / demonstrative with no local antecedent => follow-up
_COREFERENCE_RE = re.compile(
    r"(?i)\b(it|this|that|these|those|they|them|he|she|his|her|"
    r"the (former|latter)|the (above|previous) (one|point|answer))\b"
)

_FOLLOWUP_STARTER_RE = re.compile(
    r"(?i)^\s*(and|what about|how about|also|ok(ay)? (but|and)|"
    r"why (is that|though)|and (why|how|what))\b"
)


def word_count(query: str) -> int:
    return len(query.split())


# ---------------------------------------------------------------------------
# Individual classifiers (sync — wrapped in to_thread by classify())
# ---------------------------------------------------------------------------

def classify_safety(query: str) -> tuple:
    """
    Two-layer safety check:
      1. Regex scan (core.security.scan_text)   — exact known phrasings.
      2. Semantic scan (core.semantic_guard)     — paraphrases/obfuscations
         of the same intent, caught via embedding similarity to a bank of
         known-malicious example queries.

    Either layer alone can fail the query. Returns
    (is_safe, reason, flags, semantic_similarity).
    """
    flags: List[str] = []
    reasons: List[str] = []
    semantic_similarity = None

    regex_result = scan_text(query)
    if not regex_result.passed:
        flags.append("regex")
        reasons.append("; ".join(regex_result.violations))

    if SEMANTIC_SAFETY_ENABLED:
        semantic_result = scan_query_semantic(query)
        semantic_similarity = semantic_result.max_similarity
        if semantic_result.is_suspicious:
            flags.append("semantic")
            reasons.append(
                f"semantic match to '{semantic_result.matched_category}' "
                f"pattern (similarity={semantic_result.max_similarity})"
            )

    is_safe = not flags
    reason = "; ".join(reasons) if reasons else None
    return is_safe, reason, flags, semantic_similarity


def classify_intent(query: str) -> Intent:
    if _CHITCHAT_RE.match(query.strip()):
        return Intent.CHITCHAT
    if _OUT_OF_SCOPE_RE.search(query):
        return Intent.OUT_OF_SCOPE
    if _COMPARISON_RE.search(query):
        return Intent.COMPARISON
    if _MULTI_HOP_RE.search(query) or query.count("?") > 1:
        return Intent.MULTI_HOP
    if _SUMMARY_RE.search(query):
        return Intent.SUMMARIZATION
    return Intent.FACTUAL_QA


def classify_complexity(query: str, intent: Intent) -> Complexity:
    n = word_count(query)
    if intent in (Intent.COMPARISON, Intent.MULTI_HOP):
        # Inherently multi-part — never "simple", regardless of length.
        return Complexity.COMPLEX if n >= COMPLEXITY_SIMPLE_MAX_WORDS else Complexity.MODERATE
    if n <= COMPLEXITY_SIMPLE_MAX_WORDS:
        return Complexity.SIMPLE
    if n >= COMPLEXITY_COMPLEX_MIN_WORDS:
        return Complexity.COMPLEX
    return Complexity.MODERATE


def classify_context_dependency(query: str, has_history: bool) -> tuple:
    if not has_history:
        return False, []
    terms = _COREFERENCE_RE.findall(query)
    starts_as_followup = bool(_FOLLOWUP_STARTER_RE.match(query.strip()))
    is_dependent = starts_as_followup or (bool(terms) and word_count(query) <= 12)
    flat_terms = sorted({t if isinstance(t, str) else t[0] for t in terms})
    return is_dependent, flat_terms


def classify_query_quality(query: str, is_context_dependent: bool) -> tuple:
    issues: List[str] = []
    n = word_count(query)

    if n < QUERY_MIN_WORDS:
        issues.append("too_short")
    if is_context_dependent:
        issues.append("unresolved_coreference")
    if not re.search(r"[a-zA-Z]", query):
        issues.append("no_lexical_content")

    needs_rewrite = "unresolved_coreference" in issues
    return needs_rewrite, issues


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def classify(query: str, history: List[Turn]) -> ClassificationResult:
    """
    Run all five classifiers concurrently and merge into one result.

    `history` may be empty — context-dependency then always resolves to
    False (nothing to depend on).
    """
    has_history = bool(history)

    safety_task = asyncio.to_thread(classify_safety, query)
    intent_task = asyncio.to_thread(classify_intent, query)

    is_safe, safety_reason, safety_flags, semantic_similarity = await safety_task
    intent = await intent_task

    complexity_task = asyncio.to_thread(classify_complexity, query, intent)
    context_task = asyncio.to_thread(classify_context_dependency, query, has_history)

    complexity, (is_context_dependent, coreference_terms) = await asyncio.gather(
        complexity_task, context_task
    )

    needs_rewrite, quality_issues = await asyncio.to_thread(
        classify_query_quality, query, is_context_dependent
    )

    return ClassificationResult(
        is_safe=is_safe,
        safety_reason=safety_reason,
        safety_flags=safety_flags,
        semantic_similarity=semantic_similarity,
        intent=intent,
        complexity=complexity,
        word_count=word_count(query),
        is_context_dependent=is_context_dependent,
        coreference_terms=coreference_terms,
        needs_rewrite=needs_rewrite,
        quality_issues=quality_issues,
    )
