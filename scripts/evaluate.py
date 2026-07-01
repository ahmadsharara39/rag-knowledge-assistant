"""Evaluate retrieval and answer quality against a labelled QA set.

Retrieval metrics (using each QA pair's expected source document):
  - precision@k, recall@k, MRR

Answer metric:
  - citation coverage: fraction of answers that include at least one [n] marker
  - keyword faithfulness: fraction of expected keywords present in the answer

Usage:
    python -m scripts.evaluate
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import settings
from app.rag.pipeline import RAGPipeline
from scripts.ingest import load_documents

EVAL_FILE = Path("eval/qa_pairs.json")
CITATION = re.compile(r"\[\d+\]")


def _reciprocal_rank(sources: list[str], expected: str) -> float:
    for rank, src in enumerate(sources, start=1):
        if src == expected:
            return 1.0 / rank
    return 0.0


def main() -> None:
    if not EVAL_FILE.exists():
        raise SystemExit(f"Eval file not found: {EVAL_FILE}")

    qa_pairs = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    # Fresh in-memory pipeline seeded from the sample docs.
    pipeline = RAGPipeline(settings)
    docs = load_documents(Path("data/sample_docs"))
    pipeline.ingest_documents(docs)

    top_k = settings.default_top_k
    n = len(qa_pairs)
    precision_sum = recall_sum = mrr_sum = 0.0
    cited = 0
    keyword_hits = keyword_total = 0

    print(f"Evaluating {n} question(s) at top_k={top_k}\n" + "-" * 60)
    for qa in qa_pairs:
        question = qa["question"]
        expected_source = qa["expected_source"]
        expected_keywords = [k.lower() for k in qa.get("expected_keywords", [])]

        hits = pipeline.retrieve(question, top_k)
        sources = [h.chunk.source for h in hits]
        relevant = sum(1 for s in sources if s == expected_source)

        precision = relevant / len(sources) if sources else 0.0
        recall = 1.0 if expected_source in sources else 0.0
        rr = _reciprocal_rank(sources, expected_source)
        precision_sum += precision
        recall_sum += recall
        mrr_sum += rr

        answer, citations = pipeline.answer(question, top_k)
        if CITATION.search(answer) or citations:
            cited += 1
        for kw in expected_keywords:
            keyword_total += 1
            if kw in answer.lower():
                keyword_hits += 1

        status = "OK " if recall else "MISS"
        print(f"[{status}] p@{top_k}={precision:.2f} rr={rr:.2f} | {question}")

    print("-" * 60)
    print("Retrieval:")
    print(f"  precision@{top_k}: {precision_sum / n:.3f}")
    print(f"  recall@{top_k}:    {recall_sum / n:.3f}")
    print(f"  MRR:             {mrr_sum / n:.3f}")
    print("Answers:")
    print(f"  citation coverage:    {cited / n:.3f}")
    if keyword_total:
        print(f"  keyword faithfulness: {keyword_hits / keyword_total:.3f}")
    print(f"Providers: {settings.active_providers()}")


if __name__ == "__main__":
    main()
