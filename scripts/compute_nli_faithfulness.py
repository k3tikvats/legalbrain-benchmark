"""Compute NLI faithfulness scores per row for existing predictions.jsonl files.

Reads (context, generated_answer) pairs and produces per-row NLI entailment
probability via cross-encoder/nli-deberta-v3-small. Also computes correlation
between NLI faithfulness and the lexical grounding score already stored in
predictions.

Usage:
    python scripts/compute_nli_faithfulness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from legal_benchmark.metrics import batch_nli_faithfulness

EVAL_FILE = REPO_ROOT / "paper_artifacts" / "heldout_benchmark" / "eval.jsonl"


def load_eval_contexts() -> dict[int, str]:
    """Map row_id -> gold context for the held-out eval split."""
    out = {}
    with EVAL_FILE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["row_id"]] = row["context"]
    return out


def load_predictions(run_dir: Path) -> list[dict]:
    pred_file = run_dir / "predictions.jsonl"
    with pred_file.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def report(run_name: str, predictions: list[dict], gold_contexts: dict[int, str]) -> dict:
    """Run NLI on each prediction (using gold context as premise, generated answer as hypothesis)."""
    answers = [p.get("generated_answer", "") or "" for p in predictions]
    # Use gold context if available, otherwise the extractive_answer field as a fallback context
    contexts = []
    for p in predictions:
        ctx = gold_contexts.get(p["row_id"])
        if not ctx:
            ctx = p.get("extractive_answer", "") or ""
        contexts.append(ctx)
    print(f"  Computing NLI on {len(answers)} answers (this may take a few minutes on CPU)...")
    nli_scores = batch_nli_faithfulness(answers, contexts)
    nli_arr = np.array(nli_scores)
    grounding_arr = np.array([p.get("metrics", {}).get("context_support", 0.0) for p in predictions])
    f1_arr = np.array([p.get("metrics", {}).get("token_f1", 0.0) for p in predictions])

    # Pearson and Spearman correlations
    from scipy.stats import pearsonr, spearmanr
    pearson_g = float(pearsonr(nli_arr, grounding_arr)[0])
    spearman_g = float(spearmanr(nli_arr, grounding_arr).correlation)
    pearson_f1 = float(pearsonr(nli_arr, f1_arr)[0])
    spearman_f1 = float(spearmanr(nli_arr, f1_arr).correlation)

    return {
        "run": run_name,
        "n": len(answers),
        "nli_mean": float(nli_arr.mean()),
        "nli_std": float(nli_arr.std()),
        "grounding_mean": float(grounding_arr.mean()),
        "pearson_nli_grounding": round(pearson_g, 4),
        "spearman_nli_grounding": round(spearman_g, 4),
        "pearson_nli_token_f1": round(pearson_f1, 4),
        "spearman_nli_token_f1": round(spearman_f1, 4),
    }


def main():
    runs = [
        "heldout_extended_flan_t5_base_500_20k",  # FLAN-T5 with gold context (oracle)
        "heldout_extended_bm25_flan_500_20k",     # FLAN-T5 with retrieved context
        "heldout_tinyllama_gold_500_20k",
        "heldout_tinyllama_bm25_500_20k",
        "heldout_groq_gold_500_20k",
        "heldout_groq_bm25_500_20k",
    ]
    gold_contexts = load_eval_contexts()
    print(f"Loaded {len(gold_contexts)} gold contexts.")
    results = []
    for run in runs:
        run_dir = REPO_ROOT / "benchmark_outputs" / run
        if not (run_dir / "predictions.jsonl").exists():
            print(f"SKIP {run} (no predictions)")
            continue
        print(f"\n=== {run} ===")
        preds = load_predictions(run_dir)
        r = report(run, preds, gold_contexts)
        results.append(r)
        print(f"  NLI mean: {r['nli_mean']:.4f} | Grounding mean: {r['grounding_mean']:.4f}")
        print(f"  Pearson(NLI, Grounding): {r['pearson_nli_grounding']:+.4f}  "
              f"Spearman: {r['spearman_nli_grounding']:+.4f}")
        print(f"  Pearson(NLI, TokenF1):   {r['pearson_nli_token_f1']:+.4f}  "
              f"Spearman: {r['spearman_nli_token_f1']:+.4f}")

    out_file = REPO_ROOT / "paper_artifacts" / "nli_faithfulness.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
