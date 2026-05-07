"""Bootstrap 95% confidence intervals for benchmark metrics.

Reads per-row predictions.jsonl files from benchmark_outputs/, resamples
with replacement 2000 times, and reports mean and percentile-bootstrap
95% CI for each metric.

Usage:
    python scripts/bootstrap_ci.py
    python scripts/bootstrap_ci.py --runs heldout_bm25_500_20k heldout_extended_flan_t5_base_500_20k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS = REPO_ROOT / "benchmark_outputs"


def bootstrap_ci(values: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05,
                 seed: int = 2026) -> tuple[float, float, float]:
    """Return (mean, lo_95, hi_95) using percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(values.mean())
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = values[idx].mean(axis=1)
    lo = float(np.quantile(resampled, alpha / 2))
    hi = float(np.quantile(resampled, 1 - alpha / 2))
    return (mean, lo, hi)


def retrieval_metrics(predictions: list[dict]) -> dict[str, np.ndarray]:
    """Compute per-row R@1, R@5, R@10, RR from retrieval_rank field.

    retrieval_rank is the (1-indexed) rank of the gold context, or 0 if not in top-k.
    """
    ranks = np.array([p.get("metrics", {}).get("retrieval_rank", 0) for p in predictions])
    # Some runs may store rank under "retrieval_rank" at top level (older format)
    if (ranks == 0).all():
        ranks = np.array([p.get("retrieval_rank", 0) for p in predictions])
    r_at_1 = (ranks == 1).astype(float)
    r_at_5 = ((ranks >= 1) & (ranks <= 5)).astype(float)
    r_at_10 = ((ranks >= 1) & (ranks <= 10)).astype(float)
    rr = np.where(ranks >= 1, 1.0 / np.maximum(ranks, 1), 0.0)
    return {"R@1": r_at_1, "R@5": r_at_5, "R@10": r_at_10, "MRR": rr}


def generation_metrics(predictions: list[dict]) -> dict[str, np.ndarray]:
    """Per-row EM, Token F1, ROUGE-L, grounding."""
    em = np.array([p.get("metrics", {}).get("exact_match", 0.0) for p in predictions])
    f1 = np.array([p.get("metrics", {}).get("token_f1", 0.0) for p in predictions])
    rouge = np.array([p.get("metrics", {}).get("rouge_l", 0.0) for p in predictions])
    ground = np.array([p.get("metrics", {}).get("context_support", 0.0) for p in predictions])
    return {"EM": em, "F1": f1, "ROUGE-L": rouge, "Grounding": ground}


def load_predictions(run_dir: Path) -> list[dict]:
    pred_file = run_dir / "predictions.jsonl"
    if not pred_file.exists():
        return []
    with pred_file.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_ci(mean: float, lo: float, hi: float) -> str:
    """Format as 'M [lo, hi]' compact for tables, M.MMM ±half (95%)."""
    half = (hi - lo) / 2
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}] ±{half:.3f}"


def report_run(run_dir: Path) -> None:
    predictions = load_predictions(run_dir)
    if not predictions:
        print(f"  (no predictions found in {run_dir.name})")
        return
    n = len(predictions)
    print(f"\n=== {run_dir.name}  (n={n}) ===")

    # Determine which metrics make sense for this run
    sample = predictions[0]
    has_retrieval = "retrieval_rank" in sample.get("metrics", {}) or "retrieval_rank" in sample
    has_generation = "token_f1" in sample.get("metrics", {})

    if has_retrieval:
        print("  Retrieval:")
        for name, vals in retrieval_metrics(predictions).items():
            mean, lo, hi = bootstrap_ci(vals)
            print(f"    {name:<8s}: {format_ci(mean, lo, hi)}")

    if has_generation:
        print("  Generation:")
        for name, vals in generation_metrics(predictions).items():
            mean, lo, hi = bootstrap_ci(vals)
            print(f"    {name:<10s}: {format_ci(mean, lo, hi)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default=str(DEFAULT_OUTPUTS),
                        help="Directory containing benchmark run subdirectories.")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Specific run subdirectories to process (default: all).")
    parser.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"ERROR: {outputs_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.runs:
        run_dirs = [outputs_dir / r for r in args.runs]
    else:
        run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()])

    print(f"Bootstrap 95% CIs (n_resamples={args.n_resamples}, seed=2026)")
    for run_dir in run_dirs:
        report_run(run_dir)


if __name__ == "__main__":
    main()
