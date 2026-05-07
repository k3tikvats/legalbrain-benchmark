"""Recompute the domain-partitioned Table 3 numbers using the tightened
Constitutional regex (and any other regex changes in legal_benchmark/tasks.py).

Reads the held-out eval rows + the saved predictions for the three generation
runs reported in the paper, applies `classify_legal_domain` to each context,
and writes the per-domain n / R@1 / MRR / Token-F1 to
`paper_artifacts/domain_partition_recomputed.json`.

Also recomputes Cohen's kappa against the existing 80-row LLM-judge sample at
`paper_artifacts/domain_validation_llm_judge.json`. The judge labels are kept
fixed (no Groq calls); only the regex labels are recomputed under the new
patterns. The 80 sampled row_ids are looked up first in eval.jsonl then
corpus.jsonl (since the original sample was stratified across dataset rows).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from legal_benchmark.tasks import classify_legal_domain

REPO = Path(__file__).resolve().parent.parent
HELDOUT = REPO / "paper_artifacts" / "heldout_benchmark"
OUTPUTS = REPO / "benchmark_outputs"
ARTIFACTS = REPO / "paper_artifacts"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_row_id(rows: list[dict]) -> dict:
    idx = {}
    for r in rows:
        rid = r.get("row_id")
        if rid is not None:
            idx[int(rid)] = r
    return idx


def build_context_lookup() -> dict:
    eval_rows = load_jsonl(HELDOUT / "eval.jsonl")
    corpus_rows = load_jsonl(HELDOUT / "corpus.jsonl")
    lookup = index_by_row_id(eval_rows)
    for rid, row in index_by_row_id(corpus_rows).items():
        lookup.setdefault(rid, row)
    return lookup, eval_rows


def predictions_to_metric_map(run_dir: Path) -> dict[int, dict]:
    """row_id -> {token_f1, retrieval_rank}."""
    out = {}
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        return out
    for p in load_jsonl(pred_path):
        rid = p.get("row_id")
        if rid is None:
            continue
        m = p.get("metrics", {}) or {}
        out[int(rid)] = {
            "token_f1": float(m.get("token_f1", 0.0)),
            "retrieval_rank": int(m.get("retrieval_rank", 0) or 0),
        }
    return out


def domain_table(eval_rows: list[dict], runs: dict[str, dict[int, dict]]) -> dict:
    by_domain: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "r1": [],
        "rr": [],
        "f1_per_run": defaultdict(list),
    })
    for row in eval_rows:
        rid = int(row["row_id"])
        domain = classify_legal_domain(row.get("context", ""))
        bucket = by_domain[domain]
        bucket["n"] += 1
        # Take retrieval rank from the BM25-FLAN run (canonical "BM25 retrieval" run).
        rank_run = runs.get("heldout_extended_bm25_flan_500_20k") or {}
        rank = rank_run.get(rid, {}).get("retrieval_rank", 0)
        bucket["r1"].append(1.0 if rank == 1 else 0.0)
        bucket["rr"].append(0.0 if rank == 0 else 1.0 / rank)
        for run_name, m in runs.items():
            tf = m.get(rid, {}).get("token_f1", 0.0)
            bucket["f1_per_run"][run_name].append(tf)
    out = {}
    for domain, b in sorted(by_domain.items()):
        n = b["n"]
        r1 = sum(b["r1"]) / n if n else 0.0
        mrr = sum(b["rr"]) / n if n else 0.0
        f1 = {k: round(sum(v) / max(len(v), 1), 4) for k, v in b["f1_per_run"].items()}
        out[domain] = {"n": n, "R@1": round(r1, 4), "MRR": round(mrr, 4), "F1": f1}
    # Macro avg over all listed domains (including Other) for transparency.
    if out:
        macro_r1 = sum(d["R@1"] for d in out.values()) / len(out)
        macro_mrr = sum(d["MRR"] for d in out.values()) / len(out)
        macro_f1 = {}
        for run_name in next(iter(out.values()))["F1"].keys():
            macro_f1[run_name] = round(
                sum(d["F1"][run_name] for d in out.values()) / len(out), 4
            )
        out["macro_avg"] = {
            "n": sum(d["n"] for d in out.values()),
            "R@1": round(macro_r1, 4),
            "MRR": round(macro_mrr, 4),
            "F1": macro_f1,
        }
    return out


def cohen_kappa(rater_a: list[str], rater_b: list[str]) -> tuple[float, float]:
    """Return (kappa, raw_agreement)."""
    assert len(rater_a) == len(rater_b)
    n = len(rater_a)
    if n == 0:
        return 0.0, 0.0
    labels = sorted(set(rater_a) | set(rater_b))
    agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    po = agree / n
    pe = 0.0
    for lab in labels:
        pe += (rater_a.count(lab) / n) * (rater_b.count(lab) / n)
    if pe >= 1.0:
        return 1.0, po
    kappa = (po - pe) / (1.0 - pe)
    return kappa, po


def recompute_kappa(judge_artifact: Path, ctx_lookup: dict) -> dict:
    data = json.loads(judge_artifact.read_text(encoding="utf-8"))
    pairs = data.get("per_pair", [])
    judge_labels = []
    new_regex_labels = []
    skipped = []
    for p in pairs:
        rid = int(p["row_id"])
        row = ctx_lookup.get(rid)
        if row is None:
            skipped.append(rid)
            continue
        new_regex_labels.append(classify_legal_domain(row.get("context", "")))
        judge_labels.append(p["llm_label"])
    kappa, raw = cohen_kappa(new_regex_labels, judge_labels)
    return {
        "n": len(judge_labels),
        "n_skipped_missing_row": len(skipped),
        "missing_row_ids": skipped[:10],
        "raw_agreement_pct": round(100 * raw, 2),
        "cohen_kappa": round(kappa, 4),
        "regex_labels": new_regex_labels,
        "judge_labels": judge_labels,
    }


def main() -> None:
    ctx_lookup, eval_rows = build_context_lookup()

    runs = {
        "heldout_extended_flan_t5_base_500_20k": predictions_to_metric_map(
            OUTPUTS / "heldout_extended_flan_t5_base_500_20k"
        ),
        "heldout_tinyllama_bm25_500_20k": predictions_to_metric_map(
            OUTPUTS / "heldout_tinyllama_bm25_500_20k"
        ),
        "heldout_groq_bm25_500_20k": predictions_to_metric_map(
            OUTPUTS / "heldout_groq_bm25_500_20k"
        ),
        # Used for retrieval-rank assignment in domain_table().
        "heldout_extended_bm25_flan_500_20k": predictions_to_metric_map(
            OUTPUTS / "heldout_extended_bm25_flan_500_20k"
        ),
    }

    table = domain_table(eval_rows, runs)
    judge = recompute_kappa(
        ARTIFACTS / "domain_validation_llm_judge.json", ctx_lookup
    )

    out = {"domain_table": table, "kappa_against_judge_80": judge}
    target = ARTIFACTS / "domain_partition_recomputed.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {target}")


if __name__ == "__main__":
    main()
