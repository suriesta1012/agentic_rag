"""
decomposer.py
-------------
Feature 4 — Query Decomposition (conditional).

Only invoked by the pipeline when the router returns DECOMPOSE — i.e.
comparisons, multi-hop questions, or queries the complexity classifier
flagged as COMPLEX. Simple/moderate standalone questions never reach
this module (no "decomposing every query").

Uses the SMALL model — decomposition is a structured extraction task,
not deep reasoning; the LARGE model is reserved for actually answering
the resulting sub-questions (see model_router.py).
"""

import asyncio
import json
import re
from typing import List

from config import SMALL_LLM_MODEL
from core.observability import log_event, timed
from services.llm import get_llm

_PROMPT_TEMPLATE = """Break the following question into 2 to 4 independent,
self-contained sub-questions that together cover everything needed to
answer it. Each sub-question must be answerable on its own.

Question: {query}

Respond with ONLY a JSON array of strings, nothing else. Example:
["sub-question 1", "sub-question 2"]

JSON array:"""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_sub_queries(raw: str, fallback: str) -> List[str]:
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return [fallback]
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [fallback]

    sub_queries = [
        str(item).strip() for item in parsed
        if isinstance(item, str) and item.strip()
    ]
    if not sub_queries:
        return [fallback]
    return sub_queries[:4]


async def decompose_query(query: str) -> List[str]:
    """Split *query* into independently-retrievable sub-questions."""
    prompt = _PROMPT_TEMPLATE.format(query=query)
    llm = get_llm(SMALL_LLM_MODEL)

    with timed("query_decompose", model=SMALL_LLM_MODEL):
        try:
            raw = await asyncio.to_thread(llm.invoke, prompt)
        except Exception as exc:
            log_event("query_decompose_failed", level="warning", error=str(exc))
            return [query]

    sub_queries = _parse_sub_queries(raw, fallback=query)
    log_event("query_decomposed", original=query[:80], sub_queries=len(sub_queries))
    return sub_queries
