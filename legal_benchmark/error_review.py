from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classify(row: dict) -> tuple[str, str]:
    metrics = row.get("metrics", {})
    answer = (row.get("generated_answer") or row.get("extractive_answer") or "").strip()
    reference = (row.get("reference") or "").strip()
    rank = row.get("retrieval_rank")
    f1 = float(metrics.get("token_f1", 0.0))
    grounding = float(metrics.get("context_support", 0.0))

    answer_len = len(answer.split())
    reference_len = len(reference.split())

    if row.get("used_retrieved_context") and (rank is None or rank != 1):
        return (
            "retrieval_mismatch",
            "Gold context was not retrieved at rank 1, so the generator likely answered from a distractor passage.",
        )
    if grounding < 0.40:
        return (
            "low_grounding",
            "Generated answer has low lexical support in the supplied context; inspect for hallucinated or irrelevant detail.",
        )
    if answer_len <= 4 and reference_len >= 10:
        return (
            "terse_incomplete",
            "Generated answer is much shorter than the reference and misses required procedural or legal detail.",
        )
    if reference_len and answer_len > 2.5 * reference_len:
        return (
            "overextractive",
            "Generated answer copies or includes too much passage text relative to the concise reference answer.",
        )
    if f1 < 0.12:
        return (
            "semantic_mismatch",
            "Question and generated answer appear misaligned with the reference despite context availability.",
        )
    if f1 < 0.35:
        return (
            "partial_answer",
            "Generated answer captures some relevant content but omits key conditions, entities, outcome, or legal nuance.",
        )
    return (
        "minor_mismatch",
        "Low automatic score may reflect paraphrase, reference style, or non-exact but partially acceptable wording.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reviewed error-analysis sample.")
    parser.add_argument("--predictions", default="benchmark_outputs/heldout_rag_bm25_flan_t5_base_500_20k/predictions.jsonl")
    parser.add_argument("--output", default="paper_artifacts/heldout_error_analysis_reviewed.csv")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.predictions))
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("metrics", {}).get("token_f1", 0.0)),
            float(row.get("metrics", {}).get("rouge_l", 0.0)),
            -float(row.get("metrics", {}).get("context_support", 0.0)),
        ),
    )

    fields = [
        "row_id",
        "review_label",
        "review_note",
        "retrieval_rank",
        "token_f1",
        "rouge_l",
        "grounding",
        "question",
        "reference",
        "generated_answer",
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ranked[: args.limit]:
            label, note = classify(row)
            counts[label] = counts.get(label, 0) + 1
            metrics = row.get("metrics", {})
            writer.writerow(
                {
                    "row_id": row.get("row_id"),
                    "review_label": label,
                    "review_note": note,
                    "retrieval_rank": row.get("retrieval_rank"),
                    "token_f1": f"{float(metrics.get('token_f1', 0.0)):.3f}",
                    "rouge_l": f"{float(metrics.get('rouge_l', 0.0)):.3f}",
                    "grounding": f"{float(metrics.get('context_support', 0.0)):.3f}",
                    "question": row.get("question"),
                    "reference": row.get("reference"),
                    "generated_answer": row.get("generated_answer") or row.get("extractive_answer"),
                }
            )

    lines = [
        "# Held-Out Error Review",
        "",
        "This is a 60-example author/assistant audit sample selected from the lowest-scoring held-out BM25+FLAN-T5-base RAG predictions. It is not expert legal annotation.",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {label} | {count} |")
    (output_path.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
