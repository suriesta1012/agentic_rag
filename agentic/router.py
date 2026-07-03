"""
router.py
---------
Feature 2 — Conditional Router.

Pure decision table, no I/O, no LLM calls: given a ClassificationResult
(+ whether conversation history exists), decide the single next action.

Priority order matters — earlier checks short-circuit later ones:
  1. unsafe                                  -> REJECT
  2. context-dependent but no history to
     resolve it against                      -> CLARIFY
  3. chitchat / out-of-scope                 -> DIRECT_ANSWER (skip RAG)
  4. too little content to act on            -> CLARIFY
  5. needs rewrite (coreference, has history) -> REWRITE
  6. comparison / multi-hop / complex        -> DECOMPOSE
  7. otherwise                               -> DIRECT_RAG
"""

from .schemas import ClassificationResult, Complexity, Intent, RouterAction, RouterDecision


def decide(classification: ClassificationResult, has_history: bool) -> RouterDecision:
    c = classification

    if not c.is_safe:
        return RouterDecision(
            action=RouterAction.REJECT,
            reason=f"failed safety classification: {c.safety_reason}",
        )

    if c.is_context_dependent and not has_history:
        return RouterDecision(
            action=RouterAction.CLARIFY,
            reason="query refers to prior context, but no conversation history is available",
        )

    if c.intent in (Intent.CHITCHAT, Intent.OUT_OF_SCOPE):
        return RouterDecision(
            action=RouterAction.DIRECT_ANSWER,
            reason=f"intent={c.intent.value} — no retrieval needed",
        )

    if "too_short" in c.quality_issues or "no_lexical_content" in c.quality_issues:
        return RouterDecision(
            action=RouterAction.CLARIFY,
            reason="query is too short or has no usable content to retrieve against",
        )

    if c.needs_rewrite:
        return RouterDecision(
            action=RouterAction.REWRITE,
            reason="unresolved coreference — rewriting against conversation history",
        )

    if c.intent in (Intent.COMPARISON, Intent.MULTI_HOP) or c.complexity == Complexity.COMPLEX:
        return RouterDecision(
            action=RouterAction.DECOMPOSE,
            reason=f"intent={c.intent.value}, complexity={c.complexity.value} — decomposing",
        )

    return RouterDecision(
        action=RouterAction.DIRECT_RAG,
        reason="well-formed standalone query — sending directly to RAG",
    )
