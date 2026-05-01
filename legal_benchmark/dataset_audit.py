from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from legal_benchmark.data import load_rows
from legal_benchmark.metrics import token_list
from legal_benchmark.split_builder import context_fingerprint


INDIC_RANGES = {
    "devanagari": ("\u0900", "\u097f"),
    "bengali": ("\u0980", "\u09ff"),
    "odia": ("\u0b00", "\u0b7f"),
    "tamil": ("\u0b80", "\u0bff"),
    "telugu": ("\u0c00", "\u0c7f"),
    "kannada": ("\u0c80", "\u0cff"),
}


def char_script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char in text:
        if "a" <= char.lower() <= "z":
            counts["latin"] += 1
            continue
        for script, (start, end) in INDIC_RANGES.items():
            if start <= char <= end:
                counts[script] += 1
                break
    return counts


def dominant_script(text: str) -> str:
    counts = char_script_counts(text)
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(values[low])
    return values[low] * (high - index) + values[high] * (index - low)


def length_summary(values: list[int]) -> dict:
    return {
        "mean": sum(values) / max(len(values), 1),
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0,
    }


def legal_signal(row: dict) -> bool:
    text = f"{row['context']} {row['question']} {row['response']}".lower()
    patterns = [
        r"\bsection\b",
        r"\barticle\b",
        r"\bact\b",
        r"\bcourt\b",
        r"\bpetition\b",
        r"\brespondent\b",
        r"\bappellant\b",
        r"\bjudgment\b",
        r"\bconstitution\b",
        r"\bwrit\b",
        r"\bbench\b",
        r"\bplaintiff\b",
        r"\bdefendant\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset sample statistics.")
    parser.add_argument("--dataset", default="Prarabdha/indian-legal-supervised-fine-tuning-data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="paper_artifacts/dataset_audit")
    args = parser.parse_args()

    rows = load_rows(args.dataset, args.split, args.sample_size, args.seed, streaming=True)

    context_lens = [len(token_list(row["context"])) for row in rows]
    question_lens = [len(token_list(row["question"])) for row in rows]
    response_lens = [len(token_list(row["response"])) for row in rows]
    scripts = Counter(dominant_script(row["context"]) for row in rows)
    legal_like = sum(legal_signal(row) for row in rows)

    fingerprints = [context_fingerprint(row["context"]) for row in rows]
    fingerprint_counts = Counter(fingerprints)
    duplicate_rows = sum(count - 1 for count in fingerprint_counts.values() if count > 1)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "sample_size": len(rows),
        "context_tokens": length_summary(context_lens),
        "question_tokens": length_summary(question_lens),
        "response_tokens": length_summary(response_lens),
        "dominant_script_counts": dict(scripts),
        "legal_signal_rows": legal_like,
        "legal_signal_rate": legal_like / max(len(rows), 1),
        "duplicate_context_fingerprint_rows": duplicate_rows,
        "duplicate_context_fingerprint_rate": duplicate_rows / max(len(rows), 1),
        "fingerprint_method": "blake2b over normalized first 180 context tokens",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Dataset Audit Summary",
        "",
        f"Sample size: {summary['sample_size']}",
        "",
        "| Field | Mean | P05 | P50 | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field, key in [
        ("Context tokens", "context_tokens"),
        ("Question tokens", "question_tokens"),
        ("Response tokens", "response_tokens"),
    ]:
        stats = summary[key]
        lines.append(
            f"| {field} | {stats['mean']:.1f} | {stats['p05']:.0f} | {stats['p50']:.0f} | {stats['p95']:.0f} | {stats['max']} |"
        )
    lines.extend(
        [
            "",
            "## Dominant Script Counts",
            "",
            "| Script | Count |",
            "|---|---:|",
        ]
    )
    for script, count in scripts.most_common():
        lines.append(f"| {script} | {count} |")
    lines.extend(
        [
            "",
            f"Legal-signal rate: {summary['legal_signal_rate']:.3f}",
            f"Duplicate context fingerprint rate: {summary['duplicate_context_fingerprint_rate']:.3f}",
        ]
    )
    (output_dir / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
