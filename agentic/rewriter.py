"""
rewriter.py
-----------
Feature 3 — Query Rewriting (conditional).

Only invoked by the pipeline when the router returns REWRITE — i.e. the
query has an unresolved coreference ("it", "that", a bare follow-up like
"what about pricing?") AND conversation history exists to resolve it
against. Well-formed standalone queries never reach this module.

Uses the SMALL model (rewriting is a light generation task per the model
router's policy — "Small LLM -> simple QA, rewrite, summaries").
"""

import asyncio
from typing import List

from config import SMALL_LLM_MODEL
from core.observability import log_event, timed
from core.session_store import Turn
from services.llm import get_llm

_PROMPT_TEMPLATE = """You rewrite follow-up questions into standalone questions.

Conversation so far:
{history}

Follow-up question: {query}

Rewrite the follow-up question as a single, standalone question that
makes sense with no prior context. Preserve the original intent exactly.
Do not answer the question. Output ONLY the rewritten question, nothing else.

Standalone question:"""


def _format_history(history: List[Turn]) -> str:
    lines = []
    for turn in history:
        lines.append(f"User: {turn.query}")
        lines.append(f"Assistant: {turn.answer}")
    return "\n".join(lines)


def _clean(rewritten: str, fallback: str) -> str:
    rewritten = rewritten.strip().strip('"').strip()
    # Guard against degenerate/empty completions from small local models.
    if not rewritten or len(rewritten.split()) < 2:
        return fallback
    return rewritten


async def rewrite_query(query: str, history: List[Turn]) -> str:
    """Resolve *query* against *history* into a standalone query."""
    if not history:
        return query

    prompt = _PROMPT_TEMPLATE.format(history=_format_history(history), query=query)
    llm = get_llm(SMALL_LLM_MODEL)

    with timed("query_rewrite", model=SMALL_LLM_MODEL):
        try:
            raw = await asyncio.to_thread(llm.invoke, prompt)
        except Exception as exc:
            log_event("query_rewrite_failed", level="warning", error=str(exc))
            return query

    rewritten = _clean(raw, fallback=query)
    log_event("query_rewritten", original=query[:80], rewritten=rewritten[:80])
    return rewritten
