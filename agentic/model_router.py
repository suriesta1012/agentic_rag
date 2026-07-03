"""
model_router.py
----------------
Feature 6 — Model Router.

Deterministic policy table (no LLM-based routing — the classification
metadata already tells us everything we need):

  Small LLM -> simple QA, rewriting, summaries, direct chitchat answers
  Large LLM -> comparisons, multi-hop, complex reasoning

This is purely a cost/latency optimisation knob. If SMALL_LLM_MODEL and
LARGE_LLM_MODEL are left at their defaults (both == LLM_MODEL), this is
a no-op — every query hits the same model, same as before the agentic
layer existed.
"""

from config import LARGE_LLM_MODEL, SMALL_LLM_MODEL

from .schemas import ClassificationResult, Complexity, Intent, ModelChoice

_LARGE_INTENTS = {Intent.COMPARISON, Intent.MULTI_HOP}


def choose_model(classification: ClassificationResult) -> ModelChoice:
    c = classification
    if c.intent in _LARGE_INTENTS or c.complexity == Complexity.COMPLEX:
        return ModelChoice(model_name=LARGE_LLM_MODEL, tier="large")
    return ModelChoice(model_name=SMALL_LLM_MODEL, tier="small")
