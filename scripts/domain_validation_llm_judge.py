"""LLM-as-judge validation of the regex-based domain classifier.

For a stratified sample of 70 rows (10 per regex-assigned domain), we ask
Llama-3.1-8B (Groq, free tier) to independently classify each context-question
pair into one of the 7 + Other domains using the same taxonomy. We then compute
Cohen's kappa and a confusion matrix between the regex labels and the LLM
labels.

This is an inter-rater-style validation (not ground truth), but it provides a
secondary, independently-derived signal of domain-classifier consistency.

Usage:
    GROQ_API_KEY=... python scripts/domain_validation_llm_judge.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from legal_benchmark.tasks import classify_legal_domain as classify_domain_text  # regex classifier

def classify_domain(context: str, question: str) -> str:
    return classify_domain_text(f"{context}\n\n{question}")

DOMAINS = ["Criminal", "Civil", "Constitutional", "Contract", "Family", "Property", "Service", "Other"]

JUDGE_PROMPT = """You are an Indian-law expert classifying a legal text into ONE of these eight domains:

- Criminal: IPC offences, criminal procedure, FIR, cognizable matters, sentencing, bail, arrest.
- Civil: civil suits, contracts in dispute, civil procedure (CPC), damages, suits for declaration.
- Constitutional: Articles of the Constitution, fundamental rights, writs, judicial review.
- Contract: contractual obligations, breach, consideration, formation, agency, sale of goods.
- Family: marriage, divorce, maintenance, adoption, child custody, succession, inheritance.
- Property: title, possession, immovable property, transfer, easements, mortgage, tenancy.
- Service: government employment, service rules, tribunals, transfer/promotion disputes, retirement.
- Other: anything else (tax, labour, company, IP, environmental, IT/cyber, etc.).

Pick the SINGLE most appropriate domain. Reply with EXACTLY one word from the list above. No punctuation, no explanation.

Context (truncated):
{context}

Question:
{question}

Domain:"""


def stratified_sample(eval_rows: list[dict], n_per_domain: int = 10) -> list[tuple[dict, str]]:
    """Sample rows stratified by regex-assigned domain."""
    by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for row in eval_rows:
        regex_label = classify_domain(row["context"], row["question"])
        if regex_label not in by_domain:
            regex_label = "Other"
        by_domain[regex_label].append((row, regex_label))

    import random
    rng = random.Random(2026)
    sample: list[tuple[dict, str]] = []
    for d in DOMAINS:
        rows = by_domain[d]
        rng.shuffle(rows)
        sample.extend([(r, lab) for r, lab in rows[:n_per_domain]])
    return sample


def judge_label(client, context: str, question: str) -> str:
    prompt = JUDGE_PROMPT.format(context=context[:1800], question=question[:300])
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8,
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip().split()[0].rstrip(".,;:")
    # Snap to nearest valid domain
    for d in DOMAINS:
        if text.lower().startswith(d.lower()):
            return d
    return "Other"


def cohen_kappa(labels1: list[str], labels2: list[str], categories: list[str]) -> float:
    """Cohen's kappa between two labelings."""
    n = len(labels1)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    confusion = [[0] * k for _ in range(k)]
    for a, b in zip(labels1, labels2):
        confusion[cat_to_idx[a]][cat_to_idx[b]] += 1
    p_o = sum(confusion[i][i] for i in range(k)) / n
    rows = [sum(confusion[i]) / n for i in range(k)]
    cols = [sum(confusion[i][j] for i in range(k)) / n for j in range(k)]
    p_e = sum(rows[i] * cols[i] for i in range(k))
    if p_e >= 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def main():
    if "GROQ_API_KEY" not in os.environ:
        sys.exit("Set GROQ_API_KEY env var")
    from groq import Groq
    client = Groq()

    eval_file = REPO_ROOT / "paper_artifacts" / "heldout_benchmark" / "eval.jsonl"
    rows = [json.loads(line) for line in eval_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(rows)} eval rows")

    sample = stratified_sample(rows, n_per_domain=10)
    print(f"Sampled {len(sample)} rows (10 per regex-assigned domain)")

    regex_labels = []
    judge_labels = []
    by_pair = []
    for i, (row, regex_label) in enumerate(sample):
        try:
            llm_label = judge_label(client, row["context"], row["question"])
        except Exception as e:
            print(f"  row {i}: judge error: {e}")
            llm_label = "Other"
        regex_labels.append(regex_label)
        judge_labels.append(llm_label)
        by_pair.append({
            "row_id": row["row_id"],
            "regex_label": regex_label,
            "llm_label": llm_label,
            "agree": regex_label == llm_label,
        })
        print(f"  [{i+1}/{len(sample)}] regex={regex_label}, llm={llm_label}, "
              f"{'agree' if regex_label == llm_label else 'DISAGREE'}")

    kappa = cohen_kappa(regex_labels, judge_labels, DOMAINS)
    agree = sum(1 for a, b in zip(regex_labels, judge_labels) if a == b)
    pct_agree = 100 * agree / len(sample)

    print("\n=== Domain Classifier Validation ===")
    print(f"  n = {len(sample)} (stratified, 10 per regex-assigned domain)")
    print(f"  Raw agreement: {agree}/{len(sample)} = {pct_agree:.1f}%")
    print(f"  Cohen's kappa (regex vs Llama-3.1-8B judge): {kappa:.3f}")

    out_file = REPO_ROOT / "paper_artifacts" / "domain_validation_llm_judge.json"
    out_file.write_text(json.dumps({
        "n": len(sample),
        "stratification": "10 per regex-assigned domain across 7+Other domains",
        "judge_model": "llama-3.1-8b-instant via Groq",
        "regex_labels": regex_labels,
        "judge_labels": judge_labels,
        "raw_agreement": agree,
        "raw_agreement_pct": round(pct_agree, 2),
        "cohen_kappa": round(kappa, 4),
        "per_pair": by_pair,
    }, indent=2))
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
