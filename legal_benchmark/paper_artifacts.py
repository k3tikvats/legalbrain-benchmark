from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_RUNS = {
    "flan_small_50": "benchmark_outputs/flan_small_50",
    "retrieval_500_10k": "benchmark_outputs/retrieval_500_10k",
    "gold_flan_t5_base_500_20k": "benchmark_outputs/flan_t5_base_500_20k",
    "rag_flan_t5_base_500_20k": "benchmark_outputs/rag_flan_t5_base_500_20k",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def label_failure(row: dict) -> str:
    metrics = row.get("metrics", {})
    answer = row.get("generated_answer") or row.get("extractive_answer") or ""
    reference = row.get("reference") or ""
    rank = row.get("retrieval_rank")
    f1 = float(metrics.get("token_f1", 0.0))
    grounding = float(metrics.get("context_support", 0.0))

    answer_tokens = answer.split()
    reference_tokens = reference.split()

    if row.get("used_retrieved_context") and (rank is None or rank != 1):
        return "wrong_retrieved_context"
    if grounding < 0.45:
        return "low_grounding_or_hallucination_risk"
    if len(answer_tokens) <= 3 and len(reference_tokens) > 8:
        return "terse_or_incomplete_answer"
    if len(reference_tokens) and len(answer_tokens) > 2.5 * len(reference_tokens):
        return "overlong_or_overextractive_answer"
    if f1 < 0.12:
        return "semantic_or_reference_mismatch"
    if f1 < 0.35:
        return "partial_answer"
    return "acceptable_or_minor_mismatch"


def build_results_tables(summaries: dict[str, dict]) -> str:
    lines = [
        "# Benchmark Results",
        "",
        "## Retrieval",
        "",
        "| Run | Eval | Corpus | Recall@1 | Recall@5 | Recall@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        retrieval = summary.get("retrieval", {})
        lines.append(
            "| {name} | {eval_size} | {corpus} | {r1} | {r5} | {r10} | {mrr} |".format(
                name=name,
                eval_size=summary.get("eval_size"),
                corpus=summary.get("retrieval_corpus_size"),
                r1=fmt(retrieval.get("recall_at_1")),
                r5=fmt(retrieval.get("recall_at_5")),
                r10=fmt(retrieval.get("recall_at_10")),
                mrr=fmt(retrieval.get("mrr")),
            )
        )

    lines.extend(
        [
            "",
            "## Answer Generation",
            "",
            "| Run | Model | Context | EM | Token F1 | ROUGE-L | Grounding | Seconds |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, summary in summaries.items():
        answering = summary.get("answering", {})
        model = summary.get("model_name") or "extractive baseline"
        context = "retrieved" if summary.get("use_retrieved_context") else "gold"
        lines.append(
            "| {name} | {model} | {context} | {em} | {f1} | {rouge} | {ground} | {seconds} |".format(
                name=name,
                model=model,
                context=context,
                em=fmt(answering.get("exact_match")),
                f1=fmt(answering.get("token_f1")),
                rouge=fmt(answering.get("rouge_l")),
                ground=fmt(answering.get("context_support")),
                seconds=fmt(summary.get("seconds")),
            )
        )

    return "\n".join(lines) + "\n"


def build_error_csv(prediction_rows: list[dict], output_path: Path, limit: int) -> None:
    ranked = sorted(
        prediction_rows,
        key=lambda row: (
            float(row.get("metrics", {}).get("token_f1", 0.0)),
            -float(row.get("metrics", {}).get("context_support", 0.0)),
        ),
    )
    fields = [
        "row_id",
        "failure_type",
        "retrieval_rank",
        "token_f1",
        "rouge_l",
        "grounding",
        "question",
        "reference",
        "generated_answer",
        "manual_notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ranked[:limit]:
            metrics = row.get("metrics", {})
            writer.writerow(
                {
                    "row_id": row.get("row_id"),
                    "failure_type": label_failure(row),
                    "retrieval_rank": row.get("retrieval_rank"),
                    "token_f1": fmt(metrics.get("token_f1")),
                    "rouge_l": fmt(metrics.get("rouge_l")),
                    "grounding": fmt(metrics.get("context_support")),
                    "question": row.get("question"),
                    "reference": row.get("reference"),
                    "generated_answer": row.get("generated_answer") or row.get("extractive_answer"),
                    "manual_notes": "",
                }
            )


def build_failure_summary(prediction_rows: list[dict]) -> str:
    counts: dict[str, int] = {}
    for row in prediction_rows:
        label = label_failure(row)
        counts[label] = counts.get(label, 0) + 1
    lines = [
        "# Automatic Failure Analysis",
        "",
        "These labels are heuristic and should be manually checked before inclusion in the paper.",
        "",
        "| Failure type | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {label} | {count} |")
    return "\n".join(lines) + "\n"


def build_tex(summaries: dict[str, dict]) -> str:
    retrieval = summaries["retrieval_500_10k"]["retrieval"]
    gold = summaries["gold_flan_t5_base_500_20k"]["answering"]
    rag = summaries["rag_flan_t5_base_500_20k"]["answering"]
    rag_ret = summaries["rag_flan_t5_base_500_20k"]["retrieval"]

    return rf"""\documentclass{{article}}
\usepackage[eandd]{{neurips_2026}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{url}}

\title{{LegalBrain Indic Legal Corpus: A Large-Scale Resource and Benchmark for Grounded Indian Legal Question Answering}}

\author{{Anonymous Authors}}

\begin{{document}}

\maketitle

\begin{{abstract}}
Indian legal NLP remains constrained by the limited availability of large-scale, instruction-ready datasets that reflect Indian statutes, judgments, procedural language, and multilingual legal usage. We present LegalBrain Indic Legal Corpus, a large supervised corpus of context-question-answer triples derived from public Indian legal materials. The dataset is designed to support supervised fine-tuning, retrieval-augmented generation, and evaluation of grounded legal question answering systems. In line with the NeurIPS Evaluations and Datasets track, we evaluate both retrieval and answer generation settings. A TF-IDF retrieval baseline over 10,000 candidate contexts achieves Recall@1 of {fmt(retrieval.get("recall_at_1"))}, Recall@5 of {fmt(retrieval.get("recall_at_5"))}, and MRR of {fmt(retrieval.get("mrr"))}. In a 20,000-context setting, FLAN-T5-base reaches token F1 of {fmt(gold.get("token_f1"))} with oracle context and {fmt(rag.get("token_f1"))} with retrieved context, showing that the dataset exposes both retrieval and generation bottlenecks. We further report a lexical grounding score to estimate answer support in the provided context. The corpus, metadata, and benchmark scripts are intended to provide a reproducible foundation for Indian legal QA research while documenting limitations, social risks, and appropriate use boundaries.
\end{{abstract}}

\section{{Introduction}}
Legal language in India combines common-law reasoning, statutory interpretation, procedural rules, administrative orders, and multilingual usage. General-purpose language models often lack reliable grounding in this setting, which raises risks of hallucinated precedents, incomplete statutory reasoning, and inaccessible legal assistance. Existing Indian legal NLP resources have advanced named entity recognition, rhetorical role labeling, judgment prediction, and translation, but large-scale context-grounded generative QA remains underdeveloped.

This paper contributes a dataset and benchmark protocol for Indian legal QA. The primary contribution is not a new model architecture. Instead, we ask how a large context-question-answer corpus can support evaluation of retrieval, answer generation, and grounding in Indian legal settings. We report baseline results under oracle-context and retrieved-context conditions to separate retrieval failures from generation failures.

\section{{Dataset}}
Each example contains a legal context, a question, and a reference response. Contexts are drawn from public Indian legal materials such as judgments, statutes, reports, notifications, and legal commentary. The dataset is distributed in machine-readable form and is hosted on Hugging Face. The present benchmark uses the public train split for baseline evaluation; final release should include held-out splits by source, time, court, language, and near-duplicate cluster.

\section{{Benchmark Tasks}}
\paragraph{{Retrieval.}} Given a question, a system retrieves the paired legal context from a candidate pool. We report Recall@1, Recall@5, Recall@10, and mean reciprocal rank.

\paragraph{{Oracle-context QA.}} Given the gold context and question, a model generates an answer. This setting isolates generation quality from retrieval quality.

\paragraph{{Retrieved-context QA.}} Given the top retrieved context and question, a model generates an answer. This setting evaluates an end-to-end retrieval-augmented legal QA pipeline.

\section{{Metrics}}
\paragraph{{Exact Match and Token F1.}} We compute exact match after normalization and token-level F1 between generated and reference answers.

\paragraph{{ROUGE-L.}} We report ROUGE-L to capture longest common subsequence overlap with the reference answer.

\paragraph{{Grounding Score.}} Legal answers should be supported by the supplied context. We define a lexical grounding score:
\[
G(a,c) = \frac{{|T(a) \cap T(c)|}}{{|T(a)|}},
\]
where $T(a)$ is the set of normalized non-trivial answer tokens and $T(c)$ is the set of normalized context tokens. This score is not a proof of legal correctness or entailment, but it provides an automatic proxy for context support and hallucination risk.

\section{{Experiments}}
We evaluate TF-IDF retrieval and FLAN-T5-base generation on an NVIDIA RTX 3050 6GB Laptop GPU. The generation model is used as a small, reproducible baseline rather than a state-of-the-art legal model.

\begin{{table}}[h]
\centering
\caption{{Retrieval baseline.}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Setting & Eval & Corpus & R@1 & R@5 & MRR \\
\midrule
TF-IDF & 500 & 10,000 & {fmt(retrieval.get("recall_at_1"))} & {fmt(retrieval.get("recall_at_5"))} & {fmt(retrieval.get("mrr"))} \\
TF-IDF & 500 & 20,000 & {fmt(rag_ret.get("recall_at_1"))} & {fmt(rag_ret.get("recall_at_5"))} & {fmt(rag_ret.get("mrr"))} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[h]
\centering
\caption{{Answer generation with oracle and retrieved contexts.}}
\begin{{tabular}}{{llrrrr}}
\toprule
Model & Context & EM & F1 & ROUGE-L & Grounding \\
\midrule
FLAN-T5-base & Gold & {fmt(gold.get("exact_match"))} & {fmt(gold.get("token_f1"))} & {fmt(gold.get("rouge_l"))} & {fmt(gold.get("context_support"))} \\
FLAN-T5-base & Retrieved & {fmt(rag.get("exact_match"))} & {fmt(rag.get("token_f1"))} & {fmt(rag.get("rouge_l"))} & {fmt(rag.get("context_support"))} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Discussion}}
The retrieval baseline shows that lexical retrieval recovers the paired context frequently but degrades as the candidate pool grows. In the 20,000-context setting, retrieved-context generation lowers token F1 from {fmt(gold.get("token_f1"))} to {fmt(rag.get("token_f1"))}, indicating that retrieval errors propagate into answer generation. Grounding remains relatively high, but lexical grounding alone cannot verify legal correctness. Error analysis should therefore distinguish wrong retrieval, incomplete answers, over-extractive answers, and legal nuance failures.

\section{{Limitations and Responsible Use}}
This dataset should not be used to provide legal advice without qualified human review. Public legal text may contain historical biases, sensitive facts, names of parties, and region-specific procedural assumptions. The corpus may overrepresent higher-court language and English legal reasoning relative to district-court and vernacular legal practice. Because some question-answer pairs may be generated or assisted by models and then curated, the annotation process and quality-control procedure must be documented carefully. Future benchmark splits should prevent near-duplicate leakage across train and evaluation partitions.

\section{{Conclusion}}
LegalBrain Indic Legal Corpus provides a large-scale foundation for evaluating grounded Indian legal QA. The benchmark results show that the dataset is useful for measuring retrieval and generation behavior, and that small general instruction models remain weak on legally precise answer generation. We release code and metadata to support reproducible evaluation and further dataset auditing.

\bibliographystyle{{plain}}
\bibliography{{references}}

\end{{document}}
"""


def build_rai() -> str:
    return """# Responsible AI Metadata Draft

Use this text to fill Croissant RAI fields and the paper appendix.

## Data Limitations

The corpus is intended for research on Indian legal NLP, grounded legal QA, retrieval, summarization, and supervised fine-tuning. It is not validated for direct legal advice, automated legal decision-making, risk scoring of litigants, or replacement of professional legal counsel. Coverage may be uneven across courts, jurisdictions, legal domains, time periods, and languages. Publicly available judgments may overrepresent appellate and higher-court writing relative to lower-court practice.

## Data Biases

The dataset may reflect selection bias from public legal repositories, reporting bias in published judgments, OCR quality differences, English dominance in higher judiciary records, and social biases already present in legal proceedings. Models trained on the corpus may reproduce biased legal language or uneven performance across geography, language, caste, gender, religion, class, or procedural posture.

## Personal or Sensitive Information

Legal documents can contain party names, addresses, family relationships, criminal allegations, medical information, property details, financial information, caste/religion references, and other sensitive facts. Public availability does not remove privacy risk. Dataset users should avoid re-identification, profiling, or deployment in settings that affect legal rights without human oversight.

## Data Use Cases

Recommended uses: research on retrieval, grounded QA, summarization, legal information access, benchmark design, and supervised fine-tuning under human review.

Not recommended: unsupervised legal advice to the public, automated adjudication, predicting outcomes for real litigants, surveillance, profiling, or generating filings without lawyer review.

## Social Impact

Potential benefits include improved access to legal information, better tools for legal aid, and reproducible Indian legal NLP research. Risks include hallucinated legal advice, overreliance by non-experts, privacy leakage, and amplification of historical inequities in legal text. Mitigations include clear licensing, dataset documentation, benchmark reporting, disclaimers, filtering where appropriate, and human-in-the-loop deployment.

## Synthetic Data

If questions or answers were generated by a model and curated by humans, set `rai:hasSyntheticData` to true and document prompts, model names, review criteria, rejection criteria, and quality-control procedure.

## Provenance

Document source URLs or source categories, collection period, preprocessing steps, OCR tools, language detection, deduplication, annotation workflow, Argilla configuration, reviewer instructions, and validation checks.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready artifacts from benchmark runs.")
    parser.add_argument("--output-dir", default="paper_artifacts")
    parser.add_argument("--error-run", default="benchmark_outputs/rag_flan_t5_base_500_20k")
    parser.add_argument("--error-limit", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for name, run_dir in DEFAULT_RUNS.items():
        summary_path = Path(run_dir) / "summary.json"
        if summary_path.exists():
            summaries[name] = read_json(summary_path)

    missing = sorted(set(DEFAULT_RUNS) - set(summaries))
    if missing:
        raise RuntimeError(f"Missing summaries for: {', '.join(missing)}")

    (output_dir / "results_tables.md").write_text(build_results_tables(summaries), encoding="utf-8")
    (output_dir / "neurips_ed_draft.tex").write_text(build_tex(summaries), encoding="utf-8")
    (output_dir / "responsible_ai_metadata_draft.md").write_text(build_rai(), encoding="utf-8")

    prediction_rows = read_jsonl(Path(args.error_run) / "predictions.jsonl")
    build_error_csv(prediction_rows, output_dir / "error_analysis_sample.csv", args.error_limit)
    (output_dir / "failure_summary.md").write_text(build_failure_summary(prediction_rows), encoding="utf-8")

    print(f"Wrote paper artifacts to {output_dir}")


if __name__ == "__main__":
    main()
