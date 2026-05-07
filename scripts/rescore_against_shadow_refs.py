"""Re-score the six generation runs' predictions against the Gemma2 shadow
references instead of the original Llama-3.1-8B-Instruct-generated references.

Reads:
    paper_artifacts/shadow_references_gemma2_500.jsonl
    benchmark_outputs/heldout_extended_flan_t5_base_500_20k/predictions.jsonl
    benchmark_outputs/heldout_extended_bm25_flan_500_20k/predictions.jsonl
    benchmark_outputs/heldout_tinyllama_gold_500_20k/predictions.jsonl
    benchmark_outputs/heldout_tinyllama_bm25_500_20k/predictions.jsonl
    benchmark_outputs/heldout_groq_gold_500_20k/predictions.jsonl
    benchmark_outputs/heldout_groq_bm25_500_20k/predictions.jsonl

Writes:
    paper_artifacts/shadow_rescore_summary.json

Reports per-run mean Token F1 / ROUGE-L against original vs shadow references,
and the model-ordering check.
"""
from __future__ import annotations

import json
from pathlib import Path

from legal_benchmark.metrics import token_f1, rouge_l

REPO = Path(__file__).resolve().parent.parent
SHADOW = REPO / "paper_artifacts" / "shadow_references_gemma2_500.jsonl"
OUT = REPO / "paper_artifacts" / "shadow_rescore_summary.json"

RUNS = [
    "heldout_extended_flan_t5_base_500_20k",
    "heldout_extended_bm25_flan_500_20k",
    "heldout_tinyllama_gold_500_20k",
    "heldout_tinyllama_bm25_500_20k",
    "heldout_groq_gold_500_20k",
    "heldout_groq_bm25_500_20k",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    if not SHADOW.exists():
        raise SystemExit(f"Shadow refs not yet produced: {SHADOW}")
    shadow = {int(r["row_id"]): r for r in load_jsonl(SHADOW)}
    print(f"Loaded {len(shadow)} shadow references")

    summary: dict = {
        "n_shadow": len(shadow),
        "shadow_model": next(iter(shadow.values())).get("model"),
        "runs": {},
    }
    for run_name in RUNS:
        pred_path = REPO / "benchmark_outputs" / run_name / "predictions.jsonl"
        if not pred_path.exists():
            print(f"  [skip] {run_name}: no predictions")
            continue
        preds = load_jsonl(pred_path)
        original_f1 = []
        original_rouge = []
        shadow_f1 = []
        shadow_rouge = []
        n_matched = 0
        for p in preds:
            rid = int(p.get("row_id", -1))
            answer = (p.get("generated_answer") or p.get("extractive_answer") or "").strip()
            if not answer:
                continue
            ref_orig = (p.get("reference") or "").strip()
            original_f1.append(token_f1(answer, ref_orig))
            original_rouge.append(rouge_l(answer, ref_orig))
            sh = shadow.get(rid)
            if sh is None:
                continue
            ref_shadow = (sh.get("shadow_response") or "").strip()
            if not ref_shadow:
                continue
            shadow_f1.append(token_f1(answer, ref_shadow))
            shadow_rouge.append(rouge_l(answer, ref_shadow))
            n_matched += 1

        def avg(xs: list[float]) -> float:
            return round(sum(xs) / max(len(xs), 1), 4)

        summary["runs"][run_name] = {
            "n_predictions": len(preds),
            "n_matched_shadow": n_matched,
            "f1_original_ref": avg(original_f1),
            "f1_shadow_ref": avg(shadow_f1),
            "f1_delta_shadow_minus_original": round(avg(shadow_f1) - avg(original_f1), 4),
            "rouge_l_original_ref": avg(original_rouge),
            "rouge_l_shadow_ref": avg(shadow_rouge),
        }
        print(f"  [ok] {run_name}: F1_orig={avg(original_f1)} F1_shadow={avg(shadow_f1)} (n={n_matched})")

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}")
    print()
    print("Model ordering check (Token F1, sorted descending):")
    ranked_orig = sorted(summary["runs"].items(), key=lambda kv: -kv[1]["f1_original_ref"])
    ranked_shadow = sorted(summary["runs"].items(), key=lambda kv: -kv[1]["f1_shadow_ref"])
    print("  Original references:")
    for name, m in ranked_orig:
        print(f"    {m['f1_original_ref']:.4f}  {name}")
    print("  Shadow references (Gemma2-9b-it):")
    for name, m in ranked_shadow:
        print(f"    {m['f1_shadow_ref']:.4f}  {name}")


if __name__ == "__main__":
    main()
