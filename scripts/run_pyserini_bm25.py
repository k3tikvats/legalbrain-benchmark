"""BM25 sanity-check: rank_bm25.BM25Okapi vs the custom BM25 implementation.

Java / Pyserini is unavailable on this machine; we use rank_bm25.BM25Okapi as a
standard reference implementation (same BM25 Okapi formula, k1=1.5, b=0.75)
to confirm the custom retriever in legal_benchmark/retrieval.py produces
consistent results.

Usage:
    cd <repo_root>
    python scripts/run_pyserini_bm25.py

Outputs a JSON row to stdout and appends a note to
paper_artifacts/bm25_sanity_check.json.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EVAL_FILE = REPO / "paper_artifacts" / "heldout_benchmark" / "eval.jsonl"
CORPUS_FILE = REPO / "paper_artifacts" / "heldout_benchmark" / "corpus.jsonl"
OUT_FILE = REPO / "paper_artifacts" / "bm25_sanity_check.json"

K1 = 1.5
B = 0.75
TOP_K = 10


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", str(text).lower())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(hits_per_query: list[list[int]], gold_ids: list[int], top_k: int) -> dict:
    r1 = r5 = r10 = mrr = 0.0
    n = len(gold_ids)
    for gold, hits in zip(gold_ids, hits_per_query):
        if gold in hits[:1]:
            r1 += 1
        if gold in hits[:5]:
            r5 += 1
        if gold in hits[:10]:
            r10 += 1
        if gold in hits:
            rank = hits.index(gold) + 1
            mrr += 1.0 / rank
    return {
        "R@1": round(r1 / n, 4),
        "R@5": round(r5 / n, 4),
        "R@10": round(r10 / n, 4),
        "MRR": round(mrr / n, 4),
        "n_queries": n,
    }


def run_rank_bm25(eval_rows: list[dict], corpus_rows: list[dict]) -> dict:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 not installed; pip install rank_bm25", file=sys.stderr)
        sys.exit(1)

    # Build corpus: corpus rows + gold contexts inserted
    corpus_ctx_to_idx: dict[str, int] = {}
    all_contexts: list[str] = []
    for row in corpus_rows:
        ctx = row.get("context") or row.get("text") or ""
        if ctx not in corpus_ctx_to_idx:
            corpus_ctx_to_idx[ctx] = len(all_contexts)
            all_contexts.append(ctx)

    # Insert gold contexts (mirroring the custom BM25 evaluation)
    gold_ids: list[int] = []
    for row in eval_rows:
        ctx = row.get("context") or ""
        if ctx not in corpus_ctx_to_idx:
            corpus_ctx_to_idx[ctx] = len(all_contexts)
            all_contexts.append(ctx)
        gold_ids.append(corpus_ctx_to_idx[ctx])

    print(f"rank_bm25 corpus size: {len(all_contexts)}", flush=True)
    t0 = time.time()
    tokenized = [simple_tokenize(c) for c in all_contexts]
    bm25 = BM25Okapi(tokenized, k1=K1, b=B)
    print(f"  Index built in {time.time()-t0:.1f}s", flush=True)

    hits_per_query: list[list[int]] = []
    for i, row in enumerate(eval_rows):
        q_tokens = simple_tokenize(row.get("question") or "")
        scores = bm25.get_scores(q_tokens)
        top_idxs = np.argsort(-scores)[:TOP_K].tolist()
        hits_per_query.append(top_idxs)
        if (i + 1) % 100 == 0:
            print(f"  Queried {i+1}/{len(eval_rows)}", flush=True)

    return compute_metrics(hits_per_query, gold_ids, TOP_K)


def main() -> None:
    print("Loading data...", flush=True)
    eval_rows = load_jsonl(EVAL_FILE)
    corpus_rows = load_jsonl(CORPUS_FILE)
    print(f"  {len(eval_rows)} eval rows, {len(corpus_rows)} corpus rows", flush=True)

    print("\nRunning rank_bm25.BM25Okapi (k1=1.5, b=0.75)...", flush=True)
    metrics = run_rank_bm25(eval_rows, corpus_rows)
    print(f"\nResults: {metrics}", flush=True)

    # Compare to known custom BM25 results
    custom_bm25 = {"R@1": 0.706, "R@5": 0.826, "R@10": 0.870, "MRR": 0.761}
    print("\nComparison (custom vs rank_bm25):")
    for k in ["R@1", "R@5", "R@10", "MRR"]:
        diff = metrics[k] - custom_bm25[k]
        print(f"  {k}: custom={custom_bm25[k]}, rank_bm25={metrics[k]}, diff={diff:+.4f}")

    result = {
        "description": "rank_bm25.BM25Okapi sanity check vs custom BM25 in legal_benchmark/retrieval.py",
        "note": "Java/Pyserini unavailable; rank_bm25.BM25Okapi used as standard reference",
        "k1": K1, "b": B,
        "rank_bm25_metrics": metrics,
        "custom_bm25_metrics": custom_bm25,
        "diffs": {k: round(metrics[k] - custom_bm25[k], 4) for k in ["R@1", "R@5", "R@10", "MRR"]},
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote sanity check results to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
