"""Empirical PII audit over the heldout split (500 eval + 20k corpus).

Counts incidence of PII patterns across all rows: phone numbers, email addresses,
postal addresses, FIR numbers, neutral citations, named-entity counts, and
common Indian legal sensitive identifiers (Aadhaar-like, PAN-like).

Usage:
    python scripts/pii_audit.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = REPO_ROOT / "paper_artifacts" / "heldout_benchmark" / "eval.jsonl"
CORPUS_FILE = REPO_ROOT / "paper_artifacts" / "heldout_benchmark" / "corpus.jsonl"


PII_PATTERNS = {
    "indian_mobile": re.compile(r"\b[6-9]\d{9}\b"),
    "phone_dashed": re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),
    "phone_paren": re.compile(r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}"),
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "aadhaar_12digit": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan_card": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "fir_number": re.compile(r"\bFIR\s+(?:No\.?|Number)?\s*:?\s*\d+", re.IGNORECASE),
    "case_number_neutral": re.compile(r"\b\d{4}\s+SCC\s+\d+\b"),
    "case_number_air": re.compile(r"\bAIR\s+\d{4}\s+SC\s+\d+\b"),
    "address_pin": re.compile(r"\b\d{6}\b"),
    "url": re.compile(r"https?://\S+"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


SENSITIVE_TERMS = [
    "POCSO", "rape", "sexual assault", "minor child", "victim's name",
    "prosecutrix", "molestation", "abducted",
]


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def audit_text(text: str) -> dict[str, int]:
    counts = {}
    for name, pat in PII_PATTERNS.items():
        counts[name] = len(pat.findall(text))
    counts["sensitive_terms"] = sum(text.lower().count(term.lower()) for term in SENSITIVE_TERMS)
    return counts


def audit_split(rows: list[dict], name: str) -> dict:
    rows_with_any_pii = 0
    total_counts = Counter()
    per_field_rows_with_pii = {"context": 0, "question": 0, "response": 0}

    for row in rows:
        any_field_has_pii = False
        for field in ["context", "question", "response"]:
            text = row.get(field, "") or ""
            counts = audit_text(text)
            for k, v in counts.items():
                total_counts[k] += v
            if sum(counts.values()) > 0:
                per_field_rows_with_pii[field] += 1
                any_field_has_pii = True
        if any_field_has_pii:
            rows_with_any_pii += 1

    return {
        "split_name": name,
        "n_rows": len(rows),
        "rows_with_any_pii_pattern": rows_with_any_pii,
        "rows_with_any_pii_pct": round(100 * rows_with_any_pii / max(len(rows), 1), 2),
        "per_field_rows_with_pii": per_field_rows_with_pii,
        "total_pattern_matches": dict(total_counts),
    }


def main():
    out = {"description": "Empirical PII audit on heldout split"}
    if EVAL_FILE.exists():
        eval_rows = load_rows(EVAL_FILE)
        out["eval_500"] = audit_split(eval_rows, "eval_500")
    if CORPUS_FILE.exists():
        corpus_rows = load_rows(CORPUS_FILE)
        out["corpus_20k"] = audit_split(corpus_rows, "corpus_20k")

    out_file = REPO_ROOT / "paper_artifacts" / "pii_audit.json"
    out_file.write_text(json.dumps(out, indent=2))

    # Console-friendly summary
    for split_name in ["eval_500", "corpus_20k"]:
        if split_name not in out:
            continue
        s = out[split_name]
        print(f"\n=== {split_name} (n={s['n_rows']}) ===")
        print(f"  Rows with any PII pattern: {s['rows_with_any_pii_pattern']} / {s['n_rows']} "
              f"({s['rows_with_any_pii_pct']}%)")
        print("  Per-field row counts (rows containing >=1 PII pattern in field):")
        for field, count in s["per_field_rows_with_pii"].items():
            print(f"    {field:<10s}: {count}")
        print("  Total pattern matches across all rows:")
        for pat, count in sorted(s["total_pattern_matches"].items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"    {pat:<24s}: {count}")
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
