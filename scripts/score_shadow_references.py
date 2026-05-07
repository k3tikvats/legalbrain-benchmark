"""Score generation predictions against Qwen3-32b shadow references.

Computes Token F1 and ROUGE-L for each of the six generation runs using
the Qwen3-32b shadow references instead of the original Llama-generated
references, to test robustness against the reference-generator confound.

Usage:
    python scripts/score_shadow_references.py

Writes results to paper_artifacts/shadow_scores_qwen3.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHADOW_FILE = REPO / "paper_artifacts" / "shadow_references_qwen3_500.jsonl"
EVAL_FILE = REPO / "paper_artifacts" / "heldout_benchmark" / "eval.jsonl"
OUT_FILE = REPO / "paper_artifacts" / "shadow_scores_qwen3.json"

# Map run directory name → human label and original Token F1
RUN_INFO: dict[str, dict] = {
    "heldout_extended_flan_t5_base_500_20k": {
        "label": "FLAN-T5-base (BM25 retrieved)",
        "orig_f1": 0.163,
        "orig_rl": 0.146,
    },
    "heldout_gold_bm25_flan_t5_base_500_20k": {
        "label": "FLAN-T5-base (gold context)",
        "orig_f1": 0.192,
        "orig_rl": 0.171,
    },
    "heldout_tinyllama_bm25_500_20k": {
        "label": "TinyLlama-1.1B (BM25 retrieved)",
        "orig_f1": 0.472,
        "orig_rl": 0.406,
    },
    "heldout_tinyllama_gold_500_20k": {
        "label": "TinyLlama-1.1B (gold context)",
        "orig_f1": 0.498,
        "orig_rl": 0.430,
    },
    "heldout_groq_bm25_500_20k": {
        "label": "Llama-3.1-8B / Groq (BM25 retrieved)",
        "orig_f1": 0.474,
        "orig_rl": 0.397,
    },
    "heldout_groq_gold_500_20k": {
        "label": "Llama-3.1-8B / Groq (gold context)",
        "orig_f1": 0.534,
        "orig_rl": 0.451,
    },
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    import re, string
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def token_f1(pred: str, ref: str) -> float:
    from collections import Counter
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    return 2 * common / (len(p) + len(r))


def rouge_l(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    m, n = len(p), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if p[i-1] == r[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def score_run(run_dir: Path, shadow_by_row_id: dict[int, str]) -> dict:
    pred_file = run_dir / "predictions.jsonl"
    if not pred_file.exists():
        return {}
    preds = load_jsonl(pred_file)
    f1s, rls = [], []
    matched = 0
    for row in preds:
        rid = int(row.get("row_id", -1))
        shadow = shadow_by_row_id.get(rid)
        if shadow is None:
            continue
        pred = row.get("generated_answer") or row.get("predicted_answer") or row.get("prediction") or ""
        f1s.append(token_f1(pred, shadow))
        rls.append(rouge_l(pred, shadow))
        matched += 1
    if not f1s:
        return {}
    return {
        "shadow_token_f1": round(sum(f1s) / len(f1s), 4),
        "shadow_rouge_l": round(sum(rls) / len(rls), 4),
        "n_matched": matched,
    }


def main() -> None:
    if not SHADOW_FILE.exists() or SHADOW_FILE.stat().st_size == 0:
        print(f"Shadow file not found or empty: {SHADOW_FILE}")
        return

    shadow_rows = load_jsonl(SHADOW_FILE)
    shadow_by_row_id = {int(r["row_id"]): r["shadow_response"] for r in shadow_rows}
    print(f"Loaded {len(shadow_by_row_id)} shadow references (Qwen3-32b)")

    outputs_dir = REPO / "benchmark_outputs"
    results: dict[str, dict] = {}

    for run_name, info in RUN_INFO.items():
        run_dir = outputs_dir / run_name
        if not run_dir.exists():
            print(f"  SKIP {run_name} (dir not found)")
            continue
        scores = score_run(run_dir, shadow_by_row_id)
        if not scores:
            print(f"  SKIP {run_name} (no predictions)")
            continue
        delta_f1 = round(scores["shadow_token_f1"] - info["orig_f1"], 4)
        delta_rl = round(scores["shadow_rouge_l"] - info["orig_rl"], 4)
        results[run_name] = {
            "label": info["label"],
            "orig_token_f1": info["orig_f1"],
            "orig_rouge_l": info["orig_rl"],
            **scores,
            "delta_f1": delta_f1,
            "delta_rouge_l": delta_rl,
        }
        print(
            f"  {info['label']}: orig_F1={info['orig_f1']}, shadow_F1={scores['shadow_token_f1']} "
            f"(Δ={delta_f1:+.4f}), n_matched={scores['n_matched']}"
        )

    if not results:
        print("No results computed.")
        return

    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}")

    # Print ranking preservation check
    print("\n--- Ranking preservation (Token F1 on shadow refs) ---")
    ranked = sorted(results.items(), key=lambda x: x[1]["shadow_token_f1"], reverse=True)
    for name, r in ranked:
        print(f"  {r['shadow_token_f1']:.4f}  {r['label']}")


if __name__ == "__main__":
    main()
