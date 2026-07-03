"""
pipeline.py
-----------
Thin re-export shim.

The agentic orchestration engine is now a LangGraph StateGraph — see
`agentic/graph.py` for the actual graph definition (nodes, edges,
Send-based fan-out for sub-query retrieval, etc).

This module exists only so external callers (api/app.py) don't need to
change their import path.
"""

from .graph import run_agentic_query  # noqa: F401
