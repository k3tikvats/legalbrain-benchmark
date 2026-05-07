"""Extended benchmark tasks for LegalBrain Indian Legal Corpus.

This module implements three novel evaluation protocols that go beyond
standard retrieval + extractive QA:

1. **UnanswerableQATask** — SQuAD-2.0-style unanswerable detection benchmark,
   automatically constructed by swapping question-context pairs.  Models must
   distinguish answerable from unanswerable queries.

2. **LegalDomainTask** — Legal-domain routing / partitioned-retrieval benchmark.
   Each QA pair is auto-labelled into one of seven Indian law domains (Criminal,
   Civil, Constitutional, Contract, Family, Property, Service) via keyword
   patterns extracted from the context.  Retrieval and QA metrics are reported
   per-domain and as macro-averages, revealing domain-specific difficulty.

3. **StatuteCitationTask** — Statute/section span-extraction benchmark.
   Regex patterns extract gold citation spans ("Section 302 of the Indian Penal
   Code", "Article 21 of the Constitution", etc.) from gold contexts.  Models
   are scored on exact citation match and partial citation token-F1.

All tasks operate on the *already-built* eval/corpus lists that are produced
by ``split_builder.py`` and ``data.py``, requiring no new human annotation.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from legal_benchmark.metrics import (
    aggregate_metric_dict,
    exact_match,
    normalize_text,
    token_f1,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Task 1: Unanswerable QA Detection
# ---------------------------------------------------------------------------

_UNANSWERABLE_SIGNAL = "The context does not provide enough information."


def build_unanswerable_pairs(
    eval_rows: list[dict],
    neg_ratio: float = 1.0,
    seed: int = 2026,
) -> list[dict]:
    """Create unanswerable QA pairs by mismatching questions with wrong contexts.

    For each eval row we sample a context from a *different* row.  The result
    is a mixed dataset of ``answerable`` and ``unanswerable`` items at the
    requested ratio.

    Args:
        eval_rows: List of dicts with keys ``row_id``, ``question``,
            ``context``, ``response``.
        neg_ratio: Ratio of negative (unanswerable) to positive (answerable)
            samples.  Default 1.0 → balanced 50/50 dataset.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with keys: ``row_id``, ``question``, ``context``,
        ``reference_answer``, ``is_answerable`` (bool).
    """
    rng = random.Random(seed)
    n = len(eval_rows)
    n_neg = round(n * neg_ratio)

    positives = [
        {
            "row_id": row["row_id"],
            "question": row["question"],
            "context": row["context"],
            "reference_answer": row["response"],
            "is_answerable": True,
        }
        for row in eval_rows
    ]

    # Build hard negatives: same question, context from a different row
    # ensuring context does not belong to the same question
    indices = list(range(n))
    negatives = []
    for i, row in enumerate(eval_rows[:n_neg]):
        candidates = [j for j in indices if j != i]
        j = rng.choice(candidates)
        negatives.append(
            {
                "row_id": f"{row['row_id']}_neg",
                "question": row["question"],
                "context": eval_rows[j]["context"],
                "reference_answer": _UNANSWERABLE_SIGNAL,
                "is_answerable": False,
            }
        )

    return positives + negatives


def score_unanswerable_predictions(
    predictions: list[dict],
) -> dict:
    """Compute F1 for has-answer and no-answer classes.

    Each prediction dict must have:
    - ``is_answerable`` (bool): gold label
    - ``predicted_answer`` (str): model output string

    A prediction is deemed "unanswerable" if the generated text contains
    canonical refuse phrases (case-insensitive substring match).

    Confusion-matrix definitions (treating "answerable" as the positive class
    for ``has_answer_f1`` and "unanswerable" as the positive class for
    ``no_answer_f1``):

        gold=T, pred=T  -> tp_ans, tn_una
        gold=T, pred=F  -> fn_ans, fp_una   (model wrongly abstained)
        gold=F, pred=T  -> fp_ans, fn_una   (model missed an abstention opportunity)
        gold=F, pred=F  -> tn_ans, tp_una
    """
    _REFUSE_PATTERNS = [
        "does not provide",
        "not in the context",
        "cannot be determined",
        "unanswerable",
        "no information",
        "insufficient",
    ]

    def _is_refused(text: str) -> bool:
        text_lower = text.lower()
        return any(p in text_lower for p in _REFUSE_PATTERNS)

    tp_ans = fp_ans = fn_ans = 0
    tp_una = fp_una = fn_una = 0

    for pred in predictions:
        gold_ans = bool(pred["is_answerable"])
        pred_ans = not _is_refused(pred.get("predicted_answer", ""))

        if gold_ans and pred_ans:
            tp_ans += 1
        elif gold_ans and not pred_ans:
            fn_ans += 1
            fp_una += 1
        elif not gold_ans and pred_ans:
            fp_ans += 1
            fn_una += 1
        else:  # gold=F, pred=F
            tp_una += 1

    def _f1(tp: int, fp: int, fn: int) -> float:
        if tp == 0:
            return 0.0
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    has_ans_f1 = _f1(tp_ans, fp_ans, fn_ans)
    no_ans_f1 = _f1(tp_una, fp_una, fn_una)
    return {
        "has_answer_f1": round(has_ans_f1, 4),
        "no_answer_f1": round(no_ans_f1, 4),
        "overall_f1": round((has_ans_f1 + no_ans_f1) / 2, 4),
        "n_total": len(predictions),
        "n_answerable": sum(bool(p["is_answerable"]) for p in predictions),
        "n_unanswerable": sum(not bool(p["is_answerable"]) for p in predictions),
        "tp_ans": tp_ans, "fp_ans": fp_ans, "fn_ans": fn_ans,
        "tp_una": tp_una, "fp_una": fp_una, "fn_una": fn_una,
    }


# ---------------------------------------------------------------------------
# Task 2: Legal Domain Classification + Domain-Partitioned Retrieval
# ---------------------------------------------------------------------------

# Keyword patterns derived from standard Indian legal taxonomy
_DOMAIN_PATTERNS: dict[str, list[str]] = {
    "Criminal": [
        r"\b(ipc|crpc|indian penal code|criminal procedure|section 3\d\d|murder|theft|rape|assault|"
        r"bail|cognizable|fir|accused|prosecution|conviction|acquittal|punishment|sentence)\b",
    ],
    "Constitutional": [
        # Tightened: drop generic court/government terms (supreme court, high court,
        # judicial review, parliament, legislature, governor, president) that appear
        # in nearly every Indian judgment regardless of substantive area, and require
        # constitution-specific anchors. "amendment" qualified to constitutional amendment.
        r"\b(constitution of india|article \d+|fundamental right|directive principle|"
        r"writ petition|habeas corpus|mandamus|certiorari|quo warranto|"
        r"basic structure|constitutional bench|constitutional amendment|"
        r"constitutionally|preamble|seventh schedule)\b",
    ],
    "Civil": [
        r"\b(cpc|civil procedure|suit|plaintiff|defendant|decree|injunction|specific performance|"
        r"tort|damages|negligence|trespass|nuisance|limitation act)\b",
    ],
    "Contract": [
        r"\b(contract|agreement|breach|consideration|offer|acceptance|specific relief|"
        r"arbitration|award|indemnity|guarantee|sale of goods|transfer of property)\b",
    ],
    "Family": [
        r"\b(marriage|divorce|custody|maintenance|adoption|succession|inheritance|"
        r"hindu marriage|muslim personal law|christian marriage|dowry|domestic violence)\b",
    ],
    "Property": [
        r"\b(land|property|immovable|deed|mortgage|easement|registration act|"
        r"stamp duty|tenancy|landlord|tenant|revenue|mutation|title)\b",
    ],
    "Service": [
        r"\b(service|employee|employer|termination|dismissal|departmental inquiry|"
        r"government servant|pensioner|seniority|promotion|central administrative tribunal|"
        r"labour|industrial dispute|workman)\b",
    ],
}

_COMPILED_PATTERNS: dict[str, re.Pattern] = {
    domain: re.compile("|".join(patterns), re.IGNORECASE)
    for domain, patterns in _DOMAIN_PATTERNS.items()
}


def classify_legal_domain(text: str) -> str:
    """Return the most-likely legal domain for a context string.

    Counts keyword hits per domain (case-insensitive) and returns the domain
    with the most hits.  Returns ``"Other"`` if no keywords match.
    """
    scores: dict[str, int] = {}
    for domain, pattern in _COMPILED_PATTERNS.items():
        scores[domain] = len(pattern.findall(text))
    best_domain = max(scores, key=lambda d: scores[d])
    return best_domain if scores[best_domain] > 0 else "Other"


def domain_partition_metrics(
    eval_rows: list[dict],
    ranks: list[int | None],
    metric_rows: list[dict],
) -> dict:
    """Compute retrieval and QA metrics partitioned by legal domain.

    Args:
        eval_rows: Eval rows (same order as ``ranks`` and ``metric_rows``).
        ranks: Gold context ranks from retrieval (None = not found in top-10).
        metric_rows: Per-row metric dicts (exact_match, token_f1, rouge_l, …).

    Returns:
        Dict mapping domain → {recall_at_1, mrr, token_f1, rouge_l, n}.
        Also includes a ``macro_avg`` key.
    """
    domain_data: dict[str, dict[str, list]] = defaultdict(
        lambda: {"ranks": [], "metrics": []}
    )

    for row, rank, mrow in zip(eval_rows, ranks, metric_rows):
        domain = classify_legal_domain(row.get("context", ""))
        domain_data[domain]["ranks"].append(rank)
        domain_data[domain]["metrics"].append(mrow)

    results: dict[str, dict] = {}
    for domain, data in sorted(domain_data.items()):
        n = len(data["ranks"])
        r_at_1 = sum(r is not None and r <= 1 for r in data["ranks"]) / n
        mrr_val = sum(0.0 if r is None else 1.0 / r for r in data["ranks"]) / n
        agg = aggregate_metric_dict(data["metrics"])
        results[domain] = {
            "n": n,
            "recall_at_1": round(r_at_1, 4),
            "mrr": round(mrr_val, 4),
            "token_f1": round(agg.get("token_f1", 0.0), 4),
            "rouge_l": round(agg.get("rouge_l", 0.0), 4),
        }

    # Macro average over domains (excluding "Other")
    core_domains = [d for d in results if d != "Other"]
    if core_domains:
        macro: dict[str, float] = {}
        for key in ("recall_at_1", "mrr", "token_f1", "rouge_l"):
            macro[key] = round(
                sum(results[d][key] for d in core_domains) / len(core_domains), 4
            )
        results["macro_avg"] = macro

    return results


# ---------------------------------------------------------------------------
# Task 3: Statute Citation Extraction
# ---------------------------------------------------------------------------

_CITATION_PATTERN = re.compile(
    r"""
    (?:
        # "Section/Article/Order/Rule N[A-Z]? of the XYZ Act/Code/Rules YYYY"
        (?:section|article|order|rule|schedule|clause)\s+
        \d+[a-zA-Z]?(?:\s*\(\d+\))?
        (?:\s+(?:of\s+(?:the\s+)?)?
            (?:[A-Z][a-z]+\s+){1,8}
            (?:Act|Code|Rules|Regulations|Constitution|Ordinance)
            (?:\s*,?\s*\d{4})?
        )?
    |
        # Bare "IPC", "CrPC", "CPC" references
        \b(?:IPC|CrPC|CPC|IEA|CRPC)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_citations(text: str) -> list[str]:
    """Return a de-duplicated list of statute citation strings from ``text``."""
    matches = _CITATION_PATTERN.findall(text)
    seen: set[str] = set()
    unique: list[str] = []
    for match in matches:
        normalised = re.sub(r"\s+", " ", match).strip().lower()
        if normalised and normalised not in seen:
            seen.add(normalised)
            unique.append(match.strip())
    return unique


def build_citation_pairs(eval_rows: list[dict]) -> list[dict]:
    """Build citation extraction pairs from eval rows.

    For each row that has at least one extractable citation in the context,
    creates a pair: (question="What statute or section applies?",
    context=row["context"], gold_citations=[...]).

    Rows with no citations are excluded (no signal to evaluate against).
    """
    pairs = []
    for row in eval_rows:
        citations = extract_citations(row.get("context", ""))
        if citations:
            pairs.append(
                {
                    "row_id": row["row_id"],
                    "question": "Which statute or legal provision is cited in the following legal context?",
                    "context": row["context"],
                    "gold_citations": citations,
                    "gold_citation_text": " | ".join(citations),
                }
            )
    return pairs


def score_citation_predictions(predictions: list[dict]) -> dict:
    """Score predicted citation strings.

    Each prediction dict should have:
    - ``gold_citations``: list of gold citation strings
    - ``predicted_answer``: model output string

    Metrics:
    - ``exact_citation_match``: 1 if any gold citation appears verbatim in
      the predicted text (case-insensitive).
    - ``citation_token_f1``: token-level F1 between predicted text and the
      concatenated gold citations.
    """
    em_scores = []
    f1_scores = []

    for pred in predictions:
        gold_text = " ".join(pred.get("gold_citations", []))
        predicted = pred.get("predicted_answer", "")
        predicted_lower = predicted.lower()

        em = any(
            cit.lower() in predicted_lower
            for cit in pred.get("gold_citations", [])
        )
        em_scores.append(float(em))
        f1_scores.append(token_f1(predicted, gold_text))

    n = len(predictions)
    return {
        "n_pairs": n,
        "exact_citation_match": round(sum(em_scores) / max(n, 1), 4),
        "citation_token_f1": round(sum(f1_scores) / max(n, 1), 4),
    }


# ---------------------------------------------------------------------------
# Convenience runner: run all three tasks and write results
# ---------------------------------------------------------------------------

def run_all_tasks(
    eval_rows: list[dict],
    ranks: list[int | None],
    metric_rows: list[dict],
    answerer=None,
    output_dir: Optional[Path] = None,
    seed: int = 2026,
) -> dict:
    """Run all three extended tasks and return a combined results dict.

    Args:
        eval_rows: Eval split rows.
        ranks: Retrieval ranks corresponding to eval_rows.
        metric_rows: Per-row QA metrics from the main pipeline.
        answerer: Optional answerer object with ``.answer(question, context)``
            method.  If None, only auto-scored tasks (domain, citations without
            generation) are run.
        output_dir: If provided, write per-task JSON files here.
        seed: RNG seed for unanswerable pair construction.
    """
    results: dict = {}

    # Task 2 always runs (no generation needed)
    domain_results = domain_partition_metrics(eval_rows, ranks, metric_rows)
    results["domain_partitioned"] = domain_results
    if output_dir:
        _write_json(Path(output_dir) / "task_domain.json", domain_results)

    # Task 3: citation extraction coverage (no generation needed for coverage)
    citation_pairs = build_citation_pairs(eval_rows)
    citation_coverage = {
        "n_rows_with_citations": len(citation_pairs),
        "n_total_rows": len(eval_rows),
        "citation_coverage_pct": round(
            100 * len(citation_pairs) / max(len(eval_rows), 1), 1
        ),
    }
    if answerer is not None:
        cite_ckpt_path = Path(output_dir) / "task_citations_predictions.jsonl" if output_dir else None
        cite_done: dict = {}
        if cite_ckpt_path and cite_ckpt_path.exists():
            for line in cite_ckpt_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    cite_done[str(rec["row_id"])] = rec
            print(f"  Citation checkpoint: {len(cite_done)} already done, resuming.")
        cite_ckpt_handle = cite_ckpt_path.open("a", encoding="utf-8") if cite_ckpt_path else None
        citation_preds = list(cite_done.values())
        try:
            for pair in tqdm(citation_pairs, desc="Citation extraction"):
                if str(pair["row_id"]) in cite_done:
                    continue
                predicted = answerer.answer(pair["question"], pair["context"])
                rec = {**pair, "predicted_answer": predicted}
                citation_preds.append(rec)
                if cite_ckpt_handle:
                    cite_ckpt_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cite_ckpt_handle.flush()
        finally:
            if cite_ckpt_handle:
                cite_ckpt_handle.close()
        citation_coverage.update(score_citation_predictions(citation_preds))

    results["statute_citation"] = citation_coverage
    if output_dir:
        _write_json(Path(output_dir) / "task_citations.json", citation_coverage)

    # Task 1: unanswerable detection requires generation
    if answerer is not None:
        unanswerable_pairs = build_unanswerable_pairs(eval_rows, seed=seed)
        una_ckpt_path = Path(output_dir) / "task_unanswerable_predictions.jsonl" if output_dir else None
        una_done: dict = {}
        if una_ckpt_path and una_ckpt_path.exists():
            for line in una_ckpt_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    una_done[str(rec["row_id"])] = rec
            print(f"  Unanswerable checkpoint: {len(una_done)} already done, resuming.")
        una_ckpt_handle = una_ckpt_path.open("a", encoding="utf-8") if una_ckpt_path else None
        una_preds = list(una_done.values())
        try:
            for pair in tqdm(unanswerable_pairs, desc="Unanswerable QA"):
                if str(pair["row_id"]) in una_done:
                    continue
                predicted = answerer.answer(pair["question"], pair["context"])
                rec = {**pair, "predicted_answer": predicted}
                una_preds.append(rec)
                if una_ckpt_handle:
                    una_ckpt_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    una_ckpt_handle.flush()
        finally:
            if una_ckpt_handle:
                una_ckpt_handle.close()
        una_results = score_unanswerable_predictions(una_preds)
        results["unanswerable_detection"] = una_results
        if output_dir:
            _write_json(Path(output_dir) / "task_unanswerable.json", una_results)

    return results
