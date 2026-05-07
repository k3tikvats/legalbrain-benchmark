"""
contamination_check.py
======================
Lightweight within-dataset contamination screen for the LegalBrain benchmark.

Checks whether any evaluation question (from eval.jsonl) has high n-gram
overlap with questions in a background corpus sample (corpus.jsonl or a
separate background JSONL).  This tests for *within-dataset* contamination
(e.g. if eval rows were accidentally drawn from the same question pool as the
corpus).  It does NOT check for contamination against external LLM training
data (which would require access to that data).

Usage
-----
    python scripts/contamination_check.py \
        --eval paper_artifacts/heldout_benchmark/eval.jsonl \
        --corpus paper_artifacts/heldout_benchmark/corpus.jsonl \
        --ngram 8 \
        --threshold 0.15 \
        --sample 5000

Output: JSON report to stdout and optionally to --output path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Within-dataset contamination check.")
    parser.add_argument("--eval", default="paper_artifacts/heldout_benchmark/eval.jsonl")
    parser.add_argument("--corpus", default="paper_artifacts/heldout_benchmark/corpus.jsonl")
    parser.add_argument("--ngram", type=int, default=8, help="n-gram size for overlap")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Jaccard threshold above which a pair is flagged")
    parser.add_argument("--sample", type=int, default=5000,
                        help="Max corpus rows to load (for speed)")
    parser.add_argument("--output", default=None, help="Write JSON report to this path")
    args = parser.parse_args()

    eval_path = Path(args.eval)
    corpus_path = Path(args.corpus)

    if not eval_path.exists():
        print(f"ERROR: eval file not found: {eval_path}", file=sys.stderr)
        sys.exit(1)
    if not corpus_path.exists():
        print(f"ERROR: corpus file not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    eval_rows = load_jsonl(eval_path)
    corpus_rows = load_jsonl(corpus_path, limit=args.sample)

    print(f"Loaded {len(eval_rows)} eval rows, {len(corpus_rows)} corpus rows.")
    print(f"Running {args.ngram}-gram Jaccard contamination check (threshold={args.threshold})...")

    # Pre-compute corpus question n-gram sets
    corpus_ngrams = [
        ngrams(normalize(r.get("question", "")), args.ngram) for r in corpus_rows
    ]

    flagged = []
    max_overlaps = []

    for eval_row in eval_rows:
        q_tokens = normalize(eval_row.get("question", ""))
        q_ng = ngrams(q_tokens, args.ngram)
        if not q_ng:
            max_overlaps.append(0.0)
            continue

        best_j = 0.0
        best_idx = -1
        for idx, c_ng in enumerate(corpus_ngrams):
            j = jaccard(q_ng, c_ng)
            if j > best_j:
                best_j = j
                best_idx = idx

        max_overlaps.append(best_j)
        if best_j >= args.threshold:
            flagged.append({
                "eval_row_id": eval_row.get("row_id"),
                "eval_question": eval_row.get("question"),
                "best_corpus_question": corpus_rows[best_idx].get("question"),
                "jaccard": round(best_j, 4),
            })

    n_flagged = len(flagged)
    mean_overlap = sum(max_overlaps) / max(len(max_overlaps), 1)
    pct_above = 100 * n_flagged / max(len(eval_rows), 1)

    report = {
        "ngram": args.ngram,
        "threshold": args.threshold,
        "eval_rows": len(eval_rows),
        "corpus_rows_sampled": len(corpus_rows),
        "flagged_pairs": n_flagged,
        "flagged_pct": round(pct_above, 2),
        "mean_max_jaccard": round(mean_overlap, 4),
        "flagged_examples": flagged[:20],  # show first 20
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport written to {out_path}")

    if n_flagged == 0:
        print(f"\nNo contamination detected above threshold={args.threshold} "
              f"(mean max Jaccard={mean_overlap:.4f}).")
    else:
        print(f"\nWARNING: {n_flagged} eval questions ({pct_above:.1f}%) "
              f"have Jaccard >= {args.threshold} with a corpus question. "
              f"Review flagged_examples above.")


if __name__ == "__main__":
    main()
