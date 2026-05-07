"""Save bootstrap CIs to JSON for paper consumption.

Per-row predictions schemas in this repo are not uniform:

  * Most retrievers store `metrics.retrieval_rank` 1-indexed (1=best, 0=miss).
    The `retrieval_metrics()` function from `bootstrap_ci` assumes this.
  * `run_e5_retrieval.py` stores `retrieval_rank` at the top level, 0-indexed
    (0=best, -1=miss) and does NOT populate a `metrics` block.
  * `run_crossencoder_reranker.py` stores both `retrieval_rank` (0-indexed,
    None=miss) at the top level AND `metrics.retrieval_rank` (0-indexed,
    -1=miss).

Before computing CIs, this module rewrites E5/CrossEncoder prediction rows
in-memory into the canonical 1-indexed `metrics.retrieval_rank` format so
the CI numbers match the run-level summary.json (and Table 1 in the paper).


Per-row predictions live in benchmark_outputs/<run>/predictions.jsonl. For each
of the runs reported in the paper:

  * Retrieval CIs (R@1/R@5/R@10/MRR) are computed for every run that has a
    `metrics.retrieval_rank` field with at least one non-zero value.
  * Generation CIs (EM/F1/ROUGE-L/Grounding) are computed ONLY for runs whose
    predictions carry a non-trivial `metrics.token_f1` distribution. This
    avoids the historical bug where retrieval-only runs (TF-IDF, BM25, MiniLM,
    BGE, E5, InLegalBERT, Hybrid, CrossEncoder) emitted bogus identical
    "F1=0.2429, Grounding=1.0±0" generation entries.

  * E5-base-v2 and BM25+CrossEncoder are included; the legacy artifact
    omitted both.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from bootstrap_ci import bootstrap_ci, retrieval_metrics, generation_metrics, load_predictions

OUT_FILE = REPO_ROOT / "paper_artifacts" / "bootstrap_cis.json"


def metric_dict(values: np.ndarray) -> dict:
    mean, lo, hi = bootstrap_ci(values)
    return {
        "mean": round(mean, 4),
        "lo_95": round(lo, 4),
        "hi_95": round(hi, 4),
        "half_95": round((hi - lo) / 2, 4),
    }


def has_real_generation(preds: list[dict]) -> bool:
    """True iff predictions look like they came from a generation run.

    A run is "real generation" only if at least one row has a non-empty
    `generated_answer`. Retrieval-only runs in this codebase still populate a
    gold-context extractive baseline (`extractive_answer`) and the same
    `metrics.token_f1` numbers regardless of which retriever was used; those
    must NOT be reported as that retriever's generation CIs.
    """
    if not preds:
        return False
    sample_metrics = preds[0].get("metrics") or {}
    if "token_f1" not in sample_metrics:
        return False
    return any(
        bool(str(p.get("generated_answer", "") or "").strip()) for p in preds
    )


def has_real_retrieval(preds: list[dict]) -> bool:
    if not preds:
        return False
    sample_metrics = preds[0].get("metrics") or {}
    return "retrieval_rank" in sample_metrics or "retrieval_rank" in preds[0]


def normalize_retrieval_predictions(run_name: str, preds: list[dict]) -> list[dict]:
    """Rewrite legacy E5/CrossEncoder predictions to the canonical
    1-indexed `metrics.retrieval_rank` schema used elsewhere.
    """
    if run_name not in {
        "heldout_e5_base_500_20k",
        "heldout_crossencoder_bm25_500_20k",
    }:
        return preds
    fixed = []
    for p in preds:
        legacy_rank = p.get("retrieval_rank")
        if legacy_rank is None and (p.get("metrics") or {}).get("retrieval_rank") is not None:
            legacy_rank = p["metrics"]["retrieval_rank"]
        # legacy convention: 0 = best, -1 (or None) = miss; convert to 1-indexed.
        if legacy_rank is None or legacy_rank < 0:
            canonical = 0  # miss
        else:
            canonical = int(legacy_rank) + 1
        new = dict(p)
        new["metrics"] = {**(p.get("metrics") or {}), "retrieval_rank": canonical}
        fixed.append(new)
    return fixed


def main():
    runs_of_interest = [
        # Retrieval-only baselines.
        "heldout_tfidf_500_20k",
        "heldout_bm25_500_20k",
        "heldout_dense_minilm_500_20k",
        "heldout_bge_base_500_20k",
        "heldout_e5_base_500_20k",
        "heldout_inlegalbert_500_20k",
        "heldout_hybrid_bm25_bge_500_20k",
        "heldout_crossencoder_bm25_500_20k",
        # End-to-end retrieval + generation runs.
        "heldout_extended_flan_t5_base_500_20k",
        "heldout_extended_bm25_flan_500_20k",
        "heldout_tinyllama_gold_500_20k",
        "heldout_tinyllama_bm25_500_20k",
        "heldout_groq_gold_500_20k",
        "heldout_groq_bm25_500_20k",
    ]
    out: dict = {"n_resamples": 2000, "alpha": 0.05, "seed": 2026, "runs": {}}
    for run_name in runs_of_interest:
        run_dir = REPO_ROOT / "benchmark_outputs" / run_name
        preds = load_predictions(run_dir)
        if not preds:
            print(f"[skip] {run_name}: no predictions")
            continue
        preds = normalize_retrieval_predictions(run_name, preds)
        run_out: dict = {"n": len(preds)}
        if has_real_retrieval(preds):
            run_out["retrieval"] = {
                k: metric_dict(v) for k, v in retrieval_metrics(preds).items()
            }
        if has_real_generation(preds):
            run_out["generation"] = {
                k: metric_dict(v) for k, v in generation_metrics(preds).items()
            }
        out["runs"][run_name] = run_out
        sections = [k for k in run_out if k != "n"]
        print(f"[ok ] {run_name}: n={len(preds)} sections={sections}")

    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
