"""
graph.py
--------
LangGraph implementation of the agentic layer.

This is the real orchestration engine behind `/query`: independent
classifiers run as separate graph nodes and execute concurrently
(LangGraph runs every node whose dependencies are satisfied within the
same superstep), a Supervisor node aggregates their outputs into one
routing + retrieval-plan + model-choice decision, and — for decomposed
or standalone queries alike — retrieval fans out per sub-query via the
Send API (map) and joins back into a single node (reduce) before
generation. Nothing about the underlying retrieval, fusion, or
reranking is reimplemented here; graph nodes just call the existing
`services/*` functions.

Graph shape
~~~~~~~~~~~
::

                    ┌─ classify_safety ─ safety_barrier ──┐
        START ──────┼─ classify_intent ─ classify_complexity ┤
                    └─ classify_context ─ classify_quality ──┘
                                                          │
                                                     supervisor
                                                          │
              ┌───────────┬────────────┬───────────┬─────┴─────┐
              ▼           ▼            ▼           ▼           ▼
           reject      clarify    direct_answer  rewrite   decompose
              │           │            │           │           │
             END         END          END          └─────┬─────┘  ◄── direct_rag_prepare
                                                           ▼
                                             fan_out_retrieval (Send × N)
                                                           │
                                                    retrieve_one (parallel)
                                                           │
                                                   check_retrieval (join)
                                                     │           │
                                                     ▼           ▼
                                                no_results     generate
                                                     │           │
                                                    END          END

Why a Supervisor instead of the router deciding directly?
The Supervisor node is where classification metadata, retrieval
planning, and model routing are aggregated into one place *after* the
parallel classification phase has joined — this is what lets Features
2, 5, and 6 (router, retrieval planner, model router) stay pure,
independently testable functions while the graph handles the actual
concurrency and state threading.
"""

import asyncio
import operator
from typing import Annotated, List, Optional, Tuple, TypedDict

from langchain.schema import Document
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from config import BM25_CONFIDENCE_K, LARGE_LLM_MODEL, SMALL_LLM_MODEL
from core.observability import log_event, timed
from core.session_store import Turn, append_turn, get_history
from services.hybrid_retriever import HybridRetriever
from services.llm import get_llm
from services.reranker import rerank

from .classifiers import (
    classify_complexity,
    classify_context_dependency,
    classify_intent,
    classify_query_quality,
    classify_safety,
    word_count,
)
from .decomposer import decompose_query
from .model_router import choose_model
from .retrieval_planner import plan_retrieval
from .rewriter import rewrite_query
from .router import decide
from .schemas import (
    AgenticResult,
    ClassificationResult,
    Complexity,
    Intent,
    ModelChoice,
    RetrievalPlan,
    RouterAction,
    RouterDecision,
)

# ---------------------------------------------------------------------------
# Prompts (identical to the pre-LangGraph pipeline)
# ---------------------------------------------------------------------------

_CHITCHAT_PROMPT = """You are a friendly assistant for a document Q&A system.
Respond briefly and naturally to this message. If it's a greeting or
small talk, respond in kind. If it's asking about you, briefly explain
you answer questions about the uploaded documents. Do not invent facts
about any documents.

Message: {query}

Response:"""

_RAG_PROMPT = """Use only the context below to answer the question. \
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""

_DECOMPOSED_PROMPT = """Answer the overall question using the sub-answers \
gathered below. Synthesize them into one coherent answer; do not just \
concatenate them. If the sub-answers don't cover something, say so.

Overall question: {query}

Sub-questions and findings:
{sub_answers}

Answer:"""


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    # Input
    query: str
    session_id: Optional[str]
    history: List[Turn]
    top_k_override: Optional[int]
    top_n_override: Optional[int]

    # Parallel classification outputs (Feature 1)
    # (is_safe, reason, flags ["regex", "semantic"], semantic_similarity)
    safety: Tuple[bool, Optional[str], List[str], Optional[float]]
    intent: Intent
    complexity: Complexity
    context: Tuple[bool, List[str]]
    quality: Tuple[bool, List[str]]

    # Supervisor aggregation (Features 2, 5, 6)
    classification: ClassificationResult
    decision: RouterDecision
    plan: RetrievalPlan
    model_choice: ModelChoice

    # Query transformation (Features 3, 4)
    working_query: str
    sub_queries: List[str]

    # Retrieval fan-out/fan-in — `docs` accumulates across every
    # parallel `retrieve_one` invocation via the `operator.add` reducer;
    # `final_docs` is the deduped result written once by check_retrieval.
    docs: Annotated[List[Document], operator.add]
    final_docs: List[Document]
    retrieval_mode: Optional[str]

    # Terminal outcome
    rejected: bool
    needs_clarification: bool
    message: Optional[str]
    answer: Optional[str]


# ---------------------------------------------------------------------------
# Classification nodes — genuinely independent work runs concurrently;
# complexity depends on intent and quality depends on context, so those
# two edges exist, but safety/intent/context all fire straight off START.
# ---------------------------------------------------------------------------

async def n_classify_safety(state: GraphState) -> dict:
    is_safe, reason, flags, semantic_similarity = await asyncio.to_thread(
        classify_safety, state["query"]
    )
    return {"safety": (is_safe, reason, flags, semantic_similarity)}


def n_safety_barrier(state: GraphState) -> dict:
    """
    No-op passthrough.

    LangGraph's superstep barrier only joins nodes that complete in the
    *same* superstep. classify_safety needs 1 hop from START;
    classify_complexity/classify_quality need 2 (they wait on
    classify_intent/classify_context first). Without this barrier,
    the supervisor would be scheduled a superstep early — as soon as
    safety alone arrives — and KeyError on the still-missing
    "complexity"/"quality" state. This node exists purely to add one
    hop so all three paths into `supervisor` arrive in lockstep.
    """
    return {}


async def n_classify_intent(state: GraphState) -> dict:
    intent = await asyncio.to_thread(classify_intent, state["query"])
    return {"intent": intent}


async def n_classify_complexity(state: GraphState) -> dict:
    complexity = await asyncio.to_thread(classify_complexity, state["query"], state["intent"])
    return {"complexity": complexity}


async def n_classify_context(state: GraphState) -> dict:
    is_dependent, terms = await asyncio.to_thread(
        classify_context_dependency, state["query"], bool(state.get("history"))
    )
    return {"context": (is_dependent, terms)}


async def n_classify_quality(state: GraphState) -> dict:
    is_context_dependent, _terms = state["context"]
    needs_rewrite, issues = await asyncio.to_thread(
        classify_query_quality, state["query"], is_context_dependent
    )
    return {"quality": (needs_rewrite, issues)}


# ---------------------------------------------------------------------------
# Supervisor — joins the classification phase, aggregates into a
# ClassificationResult, and runs the (pure, deterministic) router,
# retrieval planner, and model router in one place.
# ---------------------------------------------------------------------------

def n_supervisor(state: GraphState) -> dict:
    is_safe, safety_reason, safety_flags, semantic_similarity = state["safety"]
    is_context_dependent, coreference_terms = state["context"]
    needs_rewrite, quality_issues = state["quality"]

    classification = ClassificationResult(
        is_safe=is_safe,
        safety_reason=safety_reason,
        safety_flags=safety_flags,
        semantic_similarity=semantic_similarity,
        intent=state["intent"],
        complexity=state["complexity"],
        word_count=word_count(state["query"]),
        is_context_dependent=is_context_dependent,
        coreference_terms=coreference_terms,
        needs_rewrite=needs_rewrite,
        quality_issues=quality_issues,
    )
    decision = decide(classification, has_history=bool(state.get("history")))

    plan = plan_retrieval(classification)
    if state.get("top_k_override") is not None:
        plan.top_k = state["top_k_override"]
    if state.get("top_n_override") is not None:
        plan.top_n = state["top_n_override"]

    model_choice = choose_model(classification)

    log_event(
        "agentic_routed",
        query=state["query"][:80],
        action=decision.action.value,
        reason=decision.reason,
    )

    return {
        "classification": classification,
        "decision": decision,
        "plan": plan,
        "model_choice": model_choice,
    }


def route_from_supervisor(state: GraphState) -> str:
    return {
        RouterAction.REJECT: "reject",
        RouterAction.CLARIFY: "clarify",
        RouterAction.DIRECT_ANSWER: "direct_answer",
        RouterAction.REWRITE: "rewrite",
        RouterAction.DECOMPOSE: "decompose",
        RouterAction.DIRECT_RAG: "direct_rag_prepare",
    }[state["decision"].action]


# ---------------------------------------------------------------------------
# Terminal branches that skip retrieval entirely
# ---------------------------------------------------------------------------

def n_reject(state: GraphState) -> dict:
    return {"rejected": True, "message": "This request can't be processed: " + state["decision"].reason}


def n_clarify(state: GraphState) -> dict:
    return {"needs_clarification": True, "message": "Could you clarify your question? " + state["decision"].reason}


async def n_direct_answer(state: GraphState) -> dict:
    model = state["model_choice"]
    prompt = _CHITCHAT_PROMPT.format(query=state["query"])
    with timed("llm_generation", model=model.model_name, tier=model.tier, node="direct_answer"):
        answer = await asyncio.to_thread(get_llm(model.model_name).invoke, prompt)
    return {"answer": answer.strip() if isinstance(answer, str) else str(answer), "retrieval_mode": "none"}


# ---------------------------------------------------------------------------
# Query transformation (Features 3 + 4) — each sets `sub_queries`, which
# `fan_out_retrieval` below turns into a Send per sub-query.
# ---------------------------------------------------------------------------

async def n_rewrite(state: GraphState) -> dict:
    rewritten = await rewrite_query(state["query"], state.get("history", []))
    return {"working_query": rewritten, "sub_queries": [rewritten]}


async def n_decompose(state: GraphState) -> dict:
    sub_queries = await decompose_query(state["query"])
    return {"working_query": state["query"], "sub_queries": sub_queries}


def n_direct_rag_prepare(state: GraphState) -> dict:
    return {"working_query": state["query"], "sub_queries": [state["query"]]}


def fan_out_retrieval(state: GraphState) -> List[Send]:
    """Map: one `retrieve_one` task per sub-query, run in parallel."""
    plan = state["plan"]
    return [Send("retrieve_one", {"query": sq, "plan": plan}) for sq in state["sub_queries"]]


# ---------------------------------------------------------------------------
# Retrieval fan-out target — calls the *existing, untouched* hybrid
# retriever + cross-encoder reranker. Each parallel invocation only sees
# its own {"query", "plan"} slice (that's what Send passes as input);
# its returned `docs` list is concatenated into the shared state via the
# `operator.add` reducer declared on GraphState.docs.
# ---------------------------------------------------------------------------

def _retrieve_and_rerank(query: str, plan: RetrievalPlan) -> List[Document]:
    confidence_k = 0 if plan.force_hybrid else BM25_CONFIDENCE_K
    retriever = HybridRetriever.from_vectorstore(bm25_confidence_k=confidence_k)

    with timed("retrieval", query=query[:80], strategy=plan.strategy):
        candidates = retriever.retrieve(query, k=plan.top_k)
    with timed("reranking", candidates=len(candidates)):
        return rerank(query, candidates, top_n=plan.top_n)


async def n_retrieve_one(state: dict) -> dict:
    docs = await asyncio.to_thread(_retrieve_and_rerank, state["query"], state["plan"])
    return {"docs": docs}


def n_check_retrieval(state: GraphState) -> dict:
    """Reduce: dedupe across every sub-query's results (join point)."""
    seen = set()
    deduped: List[Document] = []
    for doc in state.get("docs", []):
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            deduped.append(doc)

    modes = sorted({d.metadata.get("retrieval_mode", "hybrid") for d in deduped})
    return {"final_docs": deduped, "retrieval_mode": "+".join(modes) if modes else "none"}


def route_after_check(state: GraphState) -> str:
    return "generate" if state.get("final_docs") else "no_results"


def n_no_results(state: GraphState) -> dict:
    return {"rejected": True, "message": "No relevant documents found for this query."}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def n_generate(state: GraphState) -> dict:
    docs = state["final_docs"]
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    working_query = state.get("working_query", state["query"])
    sub_queries = state.get("sub_queries", [])
    model = state["model_choice"]

    if state["decision"].action == RouterAction.DECOMPOSE and len(sub_queries) > 1:
        sub_list = "\n".join(f"  {i}. {sq}" for i, sq in enumerate(sub_queries, 1))
        prompt = _DECOMPOSED_PROMPT.format(
            query=working_query,
            sub_answers=f"Sub-questions considered:\n{sub_list}\n\nCombined context:\n{context}",
        )
    else:
        prompt = _RAG_PROMPT.format(context=context, query=working_query)

    with timed("llm_generation", model=model.model_name, tier=model.tier, node="generate"):
        answer = await asyncio.to_thread(get_llm(model.model_name).invoke, prompt)

    return {"answer": answer.strip() if isinstance(answer, str) else str(answer)}


# ---------------------------------------------------------------------------
# Build + compile the graph
# ---------------------------------------------------------------------------

def _build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("classify_safety", n_classify_safety)
    builder.add_node("safety_barrier", n_safety_barrier)
    builder.add_node("classify_intent", n_classify_intent)
    builder.add_node("classify_complexity", n_classify_complexity)
    builder.add_node("classify_context", n_classify_context)
    builder.add_node("classify_quality", n_classify_quality)
    builder.add_node("supervisor", n_supervisor)

    builder.add_node("reject", n_reject)
    builder.add_node("clarify", n_clarify)
    builder.add_node("direct_answer", n_direct_answer)
    builder.add_node("rewrite", n_rewrite)
    builder.add_node("decompose", n_decompose)
    builder.add_node("direct_rag_prepare", n_direct_rag_prepare)

    builder.add_node("retrieve_one", n_retrieve_one)
    builder.add_node("check_retrieval", n_check_retrieval)
    builder.add_node("no_results", n_no_results)
    builder.add_node("generate", n_generate)

    # ── Parallel classification fan-out from START ──────────────────────
    builder.add_edge(START, "classify_safety")
    builder.add_edge(START, "classify_intent")
    builder.add_edge(START, "classify_context")
    builder.add_edge("classify_intent", "classify_complexity")
    builder.add_edge("classify_context", "classify_quality")

    # ── Join: supervisor waits for all three terminal classifier nodes ──
    # (classify_safety is routed through a 1-hop barrier so it lands in
    # the same superstep as the two dependent classifier chains below.)
    builder.add_edge("classify_safety", "safety_barrier")
    builder.add_edge("safety_barrier", "supervisor")
    builder.add_edge("classify_complexity", "supervisor")
    builder.add_edge("classify_quality", "supervisor")

    # ── Conditional routing (Feature 2) ──────────────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "reject": "reject",
            "clarify": "clarify",
            "direct_answer": "direct_answer",
            "rewrite": "rewrite",
            "decompose": "decompose",
            "direct_rag_prepare": "direct_rag_prepare",
        },
    )

    builder.add_edge("reject", END)
    builder.add_edge("clarify", END)
    builder.add_edge("direct_answer", END)

    # ── Every query-transformation path fans out to retrieval the same way ─
    builder.add_conditional_edges("rewrite", fan_out_retrieval)
    builder.add_conditional_edges("decompose", fan_out_retrieval)
    builder.add_conditional_edges("direct_rag_prepare", fan_out_retrieval)

    builder.add_edge("retrieve_one", "check_retrieval")
    builder.add_conditional_edges(
        "check_retrieval", route_after_check, {"generate": "generate", "no_results": "no_results"}
    )
    builder.add_edge("no_results", END)
    builder.add_edge("generate", END)

    return builder.compile()


_COMPILED_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point — same signature as the pre-LangGraph pipeline had,
# so api/app.py needs no changes.
# ---------------------------------------------------------------------------

async def run_agentic_query(
    query: str,
    session_id: Optional[str] = None,
    top_k_override: Optional[int] = None,
    top_n_override: Optional[int] = None,
) -> AgenticResult:
    history: List[Turn] = get_history(session_id)

    initial_state: GraphState = {
        "query": query,
        "session_id": session_id,
        "history": history,
        "top_k_override": top_k_override,
        "top_n_override": top_n_override,
        "docs": [],
    }

    final_state = await _COMPILED_GRAPH.ainvoke(initial_state)

    decision = final_state.get("decision")
    model_choice = final_state.get("model_choice")

    result = AgenticResult(
        rejected=final_state.get("rejected", False),
        needs_clarification=final_state.get("needs_clarification", False),
        message=final_state.get("message"),
        answer=final_state.get("answer"),
        router_action=decision.action if decision else None,
        router_reason=decision.reason if decision else None,
        original_query=query,
        final_query=final_state.get("working_query", query),
        sub_queries=final_state.get("sub_queries", []),
        retrieval_mode=final_state.get("retrieval_mode"),
        model_used=model_choice.model_name if model_choice else None,
        model_tier=model_choice.tier if model_choice else None,
        classification=final_state.get("classification"),
        sources=final_state.get("final_docs", []),
    )

    if result.answer is not None:
        append_turn(session_id, query, result.answer)

    return result
