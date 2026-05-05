from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from legal_benchmark.answering import (
    CausalLMAnswerer,
    ExtractiveSentenceAnswerer,
    GroqAnswerer,
    Seq2SeqAnswerer,
    truncate_text,
)
from legal_benchmark.data import load_rows, split_eval_and_corpus
from legal_benchmark.metrics import (
    aggregate_metric_dict,
    batch_bertscore,
    batch_inlegal_sim,
    batch_nli_faithfulness,
    context_support_ratio,
    exact_match,
    rouge_l,
    token_f1,
)
from legal_benchmark.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    InLegalBERTRetriever,
    TfidfRetriever,
    retrieval_metrics,
)
from legal_benchmark.tasks import run_all_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Indian legal QA datasets.")
    parser.add_argument("--dataset", default="CyCrawwler/LegalBrain-Indic-Legal-Corpus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--retrieval-corpus-size", type=int, default=1000)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--corpus-file", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="benchmark_outputs/run")
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument(
        "--retriever",
        default="tfidf",
        choices=["tfidf", "bm25", "dense", "inlegal", "hybrid"],
    )
    parser.add_argument("--dense-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument(
        "--answerer",
        default="seq2seq",
        choices=["seq2seq", "causal", "groq"],
        help=(
            "Generation backend. 'seq2seq' → FLAN-T5 family; "
            "'causal' → decoder-only LMs (TinyLlama, Phi-3, LLaMA-3); "
            "'groq' → Groq free API (requires GROQ_API_KEY env var)."
        ),
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit NF4 quant for causal models (bitsandbytes)")
    parser.add_argument("--run-extended-tasks", action="store_true", help="Run domain/unanswerable/citation tasks")
    parser.add_argument("--nli-faithfulness", action="store_true", help="Compute NLI faithfulness score (requires sentence-transformers)")
    parser.add_argument("--use-retrieved-context", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=3500)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing predictions.jsonl, skipping already-completed rows")
    parser.add_argument("--inlegal-sim", action="store_true", help="Compute domain-adapted Semantic Answer Similarity via InLegalBERT embeddings")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch cannot see a CUDA device.")

    actual_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if actual_device == "auto":
        actual_device = "cpu"

    if args.eval_file and args.corpus_file:
        print(f"Loading held-out eval rows from {args.eval_file} and corpus rows from {args.corpus_file}...")
        eval_rows = load_rows(args.eval_file, args.split, args.eval_size, args.seed, streaming=False)
        corpus_rows = load_rows(args.corpus_file, args.split, args.retrieval_corpus_size, args.seed, streaming=False)
    else:
        needed_rows = args.eval_size + args.retrieval_corpus_size
        print(f"Loading {needed_rows} rows from {args.dataset}...")
        rows = load_rows(
            dataset=args.dataset,
            split=args.split,
            needed_rows=needed_rows,
            seed=args.seed,
            streaming=not args.no_streaming,
        )
        eval_rows, corpus_rows = split_eval_and_corpus(rows, args.eval_size)

    # Add the paired gold contexts to the retrieval corpus so each query has a known target.
    retrieval_rows = [dict(row, corpus_id=i) for i, row in enumerate(eval_rows + corpus_rows)]
    gold_corpus_ids = {row["row_id"]: i for i, row in enumerate(eval_rows)}

    if args.retriever == "tfidf":
        retriever = TfidfRetriever()
    elif args.retriever == "bm25":
        retriever = BM25Retriever()
    elif args.retriever == "inlegal":
        retriever = InLegalBERTRetriever(device=actual_device)
    elif args.retriever == "hybrid":
        retriever = HybridRetriever(
            dense_model_name=args.dense_model_name,
            device=actual_device,
            batch_size=args.dense_batch_size,
        )
    else:
        retriever = DenseRetriever(
            model_name=args.dense_model_name,
            device=actual_device,
            batch_size=args.dense_batch_size,
        )
    retriever.fit([row["context"] for row in retrieval_rows])

    ranks: list[int | None] = []
    retrieved_context_by_row_id: dict[int, str] = {}
    for row in tqdm(eval_rows, desc="Retrieval"):
        hits = retriever.search(row["question"], top_k=10)
        gold_id = gold_corpus_ids[row["row_id"]]
        rank = None
        for position, (hit_id, _score) in enumerate(hits, start=1):
            if hit_id == gold_id:
                rank = position
                break
        ranks.append(rank)
        retrieved_context_by_row_id[row["row_id"]] = retrieval_rows[hits[0][0]]["context"]

    retrieval_summary = retrieval_metrics(ranks)

    prediction_path = output_dir / "predictions.jsonl"
    metric_rows = []

    # Resume: load already-completed predictions so we can skip them
    already_done: dict[int, dict] = {}
    if args.resume and prediction_path.exists():
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                already_done[rec["row_id"]] = rec
        print(f"Resuming: {len(already_done)} predictions already completed, skipping those rows.")

    extractive = ExtractiveSentenceAnswerer()
    generator = None
    if not args.skip_generation:
        if args.answerer == "causal":
            generator = CausalLMAnswerer(
                model_name=args.model_name,
                device=actual_device,
                max_context_chars=args.max_context_chars,
                max_new_tokens=args.max_new_tokens,
                load_in_4bit=args.load_in_4bit,
            )
        elif args.answerer == "groq":
            generator = GroqAnswerer(
                model_name=args.model_name,
                max_context_chars=args.max_context_chars,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            generator = Seq2SeqAnswerer(
                model_name=args.model_name,
                device=actual_device,
                max_context_chars=args.max_context_chars,
                max_new_tokens=args.max_new_tokens,
            )

    # Pre-populate metric_rows with already-done predictions (in eval order)
    for row in eval_rows:
        if row["row_id"] in already_done:
            metric_rows.append(already_done[row["row_id"]]["metrics"])

    open_mode = "a" if already_done else "w"
    with prediction_path.open(open_mode, encoding="utf-8") as handle:
        for row, rank in tqdm(list(zip(eval_rows, ranks)), desc="Answering"):
            if row["row_id"] in already_done:
                continue  # already written on a previous run
            context = (
                retrieved_context_by_row_id[row["row_id"]]
                if args.use_retrieved_context
                else row["context"]
            )
            extractive_answer = extractive.answer(row["question"], truncate_text(context, args.max_context_chars))
            generated_answer = ""
            if generator is not None:
                generated_answer = generator.answer(row["question"], context)

            answer_for_primary_metrics = generated_answer or extractive_answer
            metrics = {
                "exact_match": exact_match(answer_for_primary_metrics, row["response"]),
                "token_f1": token_f1(answer_for_primary_metrics, row["response"]),
                "rouge_l": rouge_l(answer_for_primary_metrics, row["response"]),
                "context_support": context_support_ratio(answer_for_primary_metrics, context),
                "retrieval_rank": 0 if rank is None else rank,
            }
            metric_rows.append(metrics)

            record = {
                "row_id": row["row_id"],
                "question": row["question"],
                "reference": row["response"],
                "retrieval_rank": rank,
                "used_retrieved_context": args.use_retrieved_context,
                "extractive_answer": extractive_answer,
                "generated_answer": generated_answer,
                "metrics": metrics,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    answer_summary = aggregate_metric_dict(metric_rows)

    # BERTScore pass (roberta-large, semantic similarity to reference)
    bertscore_summary: dict = {}
    if not args.skip_generation:
        print("Computing BERTScore (roberta-large)...")
        preds = []
        refs = []
        for row, mr in zip(eval_rows, metric_rows):
            # re-read generated answers from predictions file
            preds.append("")
            refs.append(row.get("response", ""))
        # Re-collect from metric_rows (predictions already written to file)
        # Read them back from the written predictions file
        pred_lines = [json.loads(l) for l in prediction_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        preds = [r.get("generated_answer") or r.get("extractive_answer", "") for r in pred_lines]
        refs = [r.get("reference", "") for r in pred_lines]
        bs_scores = batch_bertscore(preds, refs, device=actual_device)
        bertscore_summary = {
            "mean_bertscore_f1": round(sum(bs_scores) / max(len(bs_scores), 1), 4),
            "model_type": "roberta-large",
            "n": len(bs_scores),
        }
        write_json(output_dir / "bertscore.json", bertscore_summary)
        print(f"BERTScore F1 (mean): {bertscore_summary['mean_bertscore_f1']}")

    # Optional InLegalBERT Semantic Answer Similarity pass
    inlegal_sim_summary: dict = {}
    if getattr(args, 'inlegal_sim', False) and not args.skip_generation:
        print("Computing InLegalBERT Semantic Answer Similarity (law-ai/InLegalBERT)...")
        pred_lines2 = [json.loads(l) for l in prediction_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        preds2 = [r.get("generated_answer") or r.get("extractive_answer", "") for r in pred_lines2]
        refs2 = [r.get("reference", "") for r in pred_lines2]
        sim_scores = batch_inlegal_sim(preds2, refs2, device=actual_device)
        inlegal_sim_summary = {
            "mean_inlegal_sim": round(sum(sim_scores) / max(len(sim_scores), 1), 4),
            "model": "law-ai/InLegalBERT",
            "n": len(sim_scores),
        }
        write_json(output_dir / "inlegal_sim.json", inlegal_sim_summary)
        print(f"InLegalBERT SAS (mean cosine): {inlegal_sim_summary['mean_inlegal_sim']}")

    # Optional NLI faithfulness pass (cross-encoder/nli-deberta-v3-small)
    nli_summary: dict = {}
    if args.nli_faithfulness and not args.skip_generation:
        print("Computing NLI faithfulness scores...")
        answered = [mr for mr in metric_rows]
        contexts_for_nli: list[str] = []
        answers_for_nli: list[str] = []
        for row, mr in zip(eval_rows, metric_rows):
            ctx = (
                retrieved_context_by_row_id[row["row_id"]]
                if args.use_retrieved_context
                else row["context"]
            )
            contexts_for_nli.append(ctx)
            answers_for_nli.append(row.get("response", ""))
        nli_scores = batch_nli_faithfulness(answers_for_nli, contexts_for_nli)
        nli_summary = {
            "mean_nli_faithfulness": round(sum(nli_scores) / max(len(nli_scores), 1), 4),
            "n": len(nli_scores),
        }
        write_json(output_dir / "nli_faithfulness.json", nli_summary)

    # Optional extended tasks: domain partitioning, citations, unanswerable
    extended_summary: dict = {}
    if args.run_extended_tasks:
        print("Running extended benchmark tasks...")
        extended_summary = run_all_tasks(
            eval_rows=eval_rows,
            ranks=ranks,
            metric_rows=metric_rows,
            answerer=generator,
            output_dir=output_dir,
        )

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "eval_size": args.eval_size,
        "retrieval_corpus_size": args.retrieval_corpus_size,
        "model_name": None if args.skip_generation else args.model_name,
        "answerer": args.answerer,
        "retriever": args.retriever,
        "dense_model_name": args.dense_model_name if args.retriever in ("dense", "hybrid") else None,
        "eval_file": args.eval_file,
        "corpus_file": args.corpus_file,
        "device": actual_device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "use_retrieved_context": args.use_retrieved_context,
        "retrieval": retrieval_summary,
        "answering": answer_summary,
        "bertscore": bertscore_summary,
        "inlegal_sim": inlegal_sim_summary,
        "nli_faithfulness": nli_summary,
        "extended_tasks": extended_summary,
        "seconds": round(time.time() - started, 2),
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "config.json", vars(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
