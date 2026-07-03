# evaluator.py
"""
RAG evaluation suite — fully local, no OpenAI key required.

Metrics

  faithfulness        Does the answer contain only claims supported by the
                      retrieved context?  (LLM-as-judge via Ollama)

  answer_relevancy    Is the answer actually on-topic for the question?
                      (cosine similarity between query and answer embeddings)

  context_precision   Of the chunks retrieved, what fraction were actually
                      useful for generating the answer?
                      (LLM-as-judge, chunk-by-chunk)

  context_recall      Did the retrieved chunks contain enough information to
                      answer the question?
                      (LLM-as-judge comparing answer to expected answer)

  latency_ms          End-to-end query time in milliseconds.

Usage


    # Run against a list of EvalSample objects
    from evaluator import Evaluator, EvalSample
    samples = [
        EvalSample(
            question="What is the capital of France?",
            expected_answer="Paris",   # optional but enables recall metric
        ),
        ...
    ]
    evaluator = Evaluator()
    report = evaluator.run(samples)
    report.print_summary()
    report.save("./eval_results")

    # OR auto-generate Q&A pairs from the knowledge base and evaluate
    report = evaluator.run_auto()
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document

from config import EVAL_AUTO_QA_COUNT, EVAL_OUTPUT_DIR, RERANK_TOP_N, RETRIEVAL_TOP_K
from core.embeddings import get_embeddings
from services.hybrid_retriever import HybridRetriever
from services.llm import get_llm
from services.reranker import rerank


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """One question (+ optional ground-truth answer) to evaluate."""
    question: str
    expected_answer: Optional[str] = None


@dataclass
class EvalResult:
    """Per-question evaluation result."""
    question: str
    answer: str
    expected_answer: Optional[str]
    faithfulness: float          # 0–1
    answer_relevancy: float      # 0–1  (cosine similarity)
    context_precision: float     # 0–1
    context_recall: float        # 0–1 (NaN if no expected_answer)
    latency_ms: float
    retrieved_chunks: int
    sources: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """
        Harmonic-mean-inspired composite score.
        context_recall is excluded when no expected answer is available.
        """
        metrics = [self.faithfulness, self.answer_relevancy, self.context_precision]
        if self.expected_answer:
            metrics.append(self.context_recall)
        return sum(metrics) / len(metrics)


@dataclass
class EvalReport:
    """Aggregated results across all samples."""
    results: List[EvalResult]

    @property
    def avg_faithfulness(self) -> float:
        return _mean([r.faithfulness for r in self.results])

    @property
    def avg_answer_relevancy(self) -> float:
        return _mean([r.answer_relevancy for r in self.results])

    @property
    def avg_context_precision(self) -> float:
        return _mean([r.context_precision for r in self.results])

    @property
    def avg_context_recall(self) -> float:
        scored = [r.context_recall for r in self.results if r.expected_answer]
        return _mean(scored) if scored else float("nan")

    @property
    def avg_latency_ms(self) -> float:
        return _mean([r.latency_ms for r in self.results])

    @property
    def avg_overall(self) -> float:
        return _mean([r.overall_score for r in self.results])

    def print_summary(self) -> None:
        """Pretty-print the evaluation summary to stdout."""
        width = 50
        print("\n" + "=" * width)
        print("  RAG EVALUATION REPORT")
        print("=" * width)
        print(f"  Samples evaluated  : {len(self.results)}")
        print(f"  Faithfulness       : {self.avg_faithfulness:.3f}")
        print(f"  Answer relevancy   : {self.avg_answer_relevancy:.3f}")
        print(f"  Context precision  : {self.avg_context_precision:.3f}")
        if not (self.avg_context_recall != self.avg_context_recall):  # NaN check
            print(f"  Context recall     : {self.avg_context_recall:.3f}")
        else:
            print(f"  Context recall     : N/A (no expected answers provided)")
        print(f"  Avg latency        : {self.avg_latency_ms:.0f} ms")
        print(f"  Overall score      : {self.avg_overall:.3f}")
        print("=" * width + "\n")

        # Per-question breakdown
        for i, r in enumerate(self.results, 1):
            print(f"  [{i}] Q: {r.question[:70]}")
            print(f"      A: {r.answer[:120]}")
            print(
                f"      faith={r.faithfulness:.2f} | rel={r.answer_relevancy:.2f} "
                f"| prec={r.context_precision:.2f} | "
                + (f"rec={r.context_recall:.2f} | " if r.expected_answer else "")
                + f"latency={r.latency_ms:.0f}ms | overall={r.overall_score:.2f}"
            )
            print()

    def save(self, output_dir: str = EVAL_OUTPUT_DIR) -> None:
        """Save full JSON results and a CSV summary to *output_dir*."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # JSON — full detail
        json_path = os.path.join(output_dir, f"eval_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        # CSV — summary row per question
        csv_path = os.path.join(output_dir, f"eval_{timestamp}.csv")
        headers = [
            "question", "faithfulness", "answer_relevancy",
            "context_precision", "context_recall", "latency_ms", "overall_score",
        ]
        with open(csv_path, "w") as f:
            f.write(",".join(headers) + "\n")
            for r in self.results:
                row = [
                    f'"{r.question}"',
                    f"{r.faithfulness:.4f}",
                    f"{r.answer_relevancy:.4f}",
                    f"{r.context_precision:.4f}",
                    f"{r.context_recall:.4f}" if r.expected_answer else "N/A",
                    f"{r.latency_ms:.1f}",
                    f"{r.overall_score:.4f}",
                ]
                f.write(",".join(row) + "\n")

        print(f"  Results saved → {json_path}")
        print(f"             → {csv_path}")


# ---------------------------------------------------------------------------
# Scoring helpers (LLM-as-judge)
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


import math


def _llm_score(prompt: str) -> float:
    """
    Ask the LLM to rate something on a 0–1 scale.
    We parse the first float found in the response.
    Falls back to 0.5 if parsing fails.
    """
    llm = get_llm()
    try:
        response = llm.invoke(prompt).strip()
        # Try to find a number between 0 and 1
        import re
        numbers = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", response)
        if numbers:
            return float(numbers[0])
        # Also try 0–10 scale and normalise
        numbers_10 = re.findall(r"\b([0-9]|10)\b", response)
        if numbers_10:
            return float(numbers_10[0]) / 10.0
    except Exception:
        pass
    return 0.5


def _score_faithfulness(question: str, answer: str, context: str) -> float:
    prompt = f"""You are an impartial evaluator. Score how faithful the answer is to the context on a scale of 0 to 1.
A score of 1.0 means every claim in the answer is supported by the context.
A score of 0.0 means the answer contains information not found in or contradicted by the context.
Respond with ONLY a single number between 0 and 1.

Context:
{context}

Question: {question}
Answer: {answer}

Faithfulness score (0-1):"""
    return _llm_score(prompt)


def _score_context_precision(question: str, answer: str, chunks: List[Document]) -> float:
    """Fraction of retrieved chunks that were actually relevant to generating the answer."""
    if not chunks:
        return 0.0
    useful = 0
    for chunk in chunks:
        prompt = f"""Was the following context chunk useful for answering the question?
Respond with ONLY 1 (useful) or 0 (not useful).

Question: {question}
Answer: {answer}
Context chunk: {chunk.page_content[:500]}

Score (0 or 1):"""
        score = _llm_score(prompt)
        if score >= 0.5:
            useful += 1
    return useful / len(chunks)


def _score_context_recall(question: str, expected_answer: str, context: str) -> float:
    prompt = f"""You are an impartial evaluator. Score how well the retrieved context supports generating the expected answer on a scale of 0 to 1.
A score of 1.0 means the context contains all the information needed.
A score of 0.0 means the context is missing key information.
Respond with ONLY a single number between 0 and 1.

Question: {question}
Expected answer: {expected_answer}
Retrieved context:
{context}

Context recall score (0-1):"""
    return _llm_score(prompt)


def _score_answer_relevancy(question: str, answer: str) -> float:
    """Cosine similarity between query embedding and answer embedding."""
    emb = get_embeddings()
    q_vec = emb.embed_query(question)
    a_vec = emb.embed_query(answer)
    return max(0.0, _cosine_similarity(q_vec, a_vec))


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs the full RAG pipeline for each sample and computes all metrics.
    """

    def __init__(
        self,
        top_k: int = RETRIEVAL_TOP_K,
        top_n: int = RERANK_TOP_N,
    ):
        self.top_k = top_k
        self.top_n = top_n

    def _run_pipeline(self, question: str):
        """Run hybrid retrieval → rerank → LLM for one question."""
        retriever = HybridRetriever.from_vectorstore()
        candidates = retriever.retrieve(question, k=self.top_k)
        top_docs = rerank(question, candidates, top_n=self.top_n)

        context = "\n\n---\n\n".join(doc.page_content for doc in top_docs)
        prompt = (
            f"Use only the context below to answer the question. "
            f"If the answer is not in the context, say \"I don't know.\"\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        answer = get_llm().invoke(prompt)
        return answer, top_docs, context

    def evaluate_single(self, sample: EvalSample) -> EvalResult:
        """Evaluate one Q&A sample and return an EvalResult."""
        print(f"  Evaluating: {sample.question[:80]}...")

        t0 = time.perf_counter()
        answer, top_docs, context = self._run_pipeline(sample.question)
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"    → answer generated ({latency_ms:.0f}ms), scoring...")

        faithfulness = _score_faithfulness(sample.question, answer, context)
        answer_relevancy = _score_answer_relevancy(sample.question, answer)
        context_precision = _score_context_precision(sample.question, answer, top_docs)
        context_recall = (
            _score_context_recall(sample.question, sample.expected_answer, context)
            if sample.expected_answer
            else float("nan")
        )

        return EvalResult(
            question=sample.question,
            answer=answer,
            expected_answer=sample.expected_answer,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            latency_ms=latency_ms,
            retrieved_chunks=len(top_docs),
            sources=list({doc.metadata.get("source", "unknown") for doc in top_docs}),
        )

    def run(self, samples: List[EvalSample], save: bool = True) -> EvalReport:
        """Evaluate all samples and return an EvalReport."""
        print(f"\nRunning RAG evaluation on {len(samples)} sample(s)...\n")
        results = [self.evaluate_single(s) for s in samples]
        report = EvalReport(results=results)
        report.print_summary()
        if save:
            report.save()
        return report

    def run_auto(
        self,
        count: int = EVAL_AUTO_QA_COUNT,
        save: bool = True,
    ) -> EvalReport:
        """
        Auto-generate `count` Q&A pairs from the knowledge base using the
        LLM, then run evaluation against them.

        This is useful when you don't have a hand-labelled test set.
        Quality of auto-generated pairs depends on the LLM and the documents.
        """
        print(f"\nAuto-generating {count} Q&A pairs from the knowledge base...\n")
        samples = _generate_qa_pairs(count=count)
        print(f"  Generated {len(samples)} pairs.\n")
        return self.run(samples, save=save)


# ---------------------------------------------------------------------------
# Auto Q&A generation
# ---------------------------------------------------------------------------

def _generate_qa_pairs(count: int = EVAL_AUTO_QA_COUNT) -> List[EvalSample]:
    """
    Sample random chunks from ChromaDB and ask the LLM to generate
    question + answer pairs from each chunk's content.
    """
    from vectorstore import get_vectorstore
    import random

    vs = get_vectorstore()
    raw = vs._collection.get(include=["documents"])
    all_texts: List[str] = raw["documents"]

    if not all_texts:
        raise ValueError("Knowledge base is empty — upload documents first.")

    # Sample without replacement (or with, if count > corpus size)
    sample_size = min(count, len(all_texts))
    sampled_texts = random.sample(all_texts, sample_size)

    llm = get_llm()
    samples: List[EvalSample] = []

    for text in sampled_texts:
        prompt = f"""Read the context below and write one factual question that can be answered from it.
Then write the correct answer.
Format your response EXACTLY as:
QUESTION: <your question>
ANSWER: <your answer>

Context:
{text[:600]}

QUESTION:"""
        try:
            response = llm.invoke(prompt)
            lines = response.strip().splitlines()
            question, answer = None, None
            for line in lines:
                if line.upper().startswith("QUESTION:"):
                    question = line.split(":", 1)[1].strip()
                elif line.upper().startswith("ANSWER:"):
                    answer = line.split(":", 1)[1].strip()
            # Handle case where LLM only returns the answer (question was in the prompt)
            if not question and lines:
                question = lines[0].strip()
            if question and answer:
                samples.append(EvalSample(question=question, expected_answer=answer))
            elif question:
                samples.append(EvalSample(question=question))
        except Exception as e:
            print(f"  Warning: Q&A generation failed for a chunk ({e})")
            continue

    return samples
