"""Post-processing script: compute InLegalBERT Semantic Answer Similarity
for any existing benchmark_outputs/ directory without re-running inference.

Usage:
    python scripts/compute_inlegal_sim.py \
        --output-dirs benchmark_outputs/heldout_* \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_benchmark.metrics import batch_inlegal_sim


def process_dir(output_dir: Path, device: str) -> None:
    pred_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    out_path = output_dir / "inlegal_sim.json"

    if not pred_path.exists():
        print(f"  SKIP {output_dir.name} — no predictions.jsonl")
        return
    if out_path.exists():
        print(f"  SKIP {output_dir.name} — inlegal_sim.json already exists")
        return

    print(f"  Processing {output_dir.name} ...", flush=True)
    records = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    preds = [r.get("generated_answer") or r.get("extractive_answer", "") for r in records]
    refs  = [r.get("reference", "") for r in records]

    scores = batch_inlegal_sim(preds, refs, device=device)
    result = {
        "mean_inlegal_sim": round(sum(scores) / max(len(scores), 1), 4),
        "model": "law-ai/InLegalBERT",
        "n": len(scores),
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"    -> mean_inlegal_sim = {result['mean_inlegal_sim']} (n={result['n']})")

    # Patch into summary.json if it exists
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["inlegal_sim"] = result
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    -> patched summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill InLegalBERT SAS scores for existing benchmark outputs.")
    parser.add_argument("--output-dirs", nargs="+", required=True, help="One or more benchmark output directories (globs ok via shell expansion)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    import torch
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Using device: {device}")

    for d in args.output_dirs:
        process_dir(Path(d), device)

    print("\nDone.")


if __name__ == "__main__":
    main()
