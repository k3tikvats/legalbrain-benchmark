from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from legal_benchmark.answering import ExtractiveSentenceAnswerer, Seq2SeqAnswerer, truncate_text
from legal_benchmark.data import load_rows, split_eval_and_corpus
from legal_benchmark.metrics import (
    aggregate_metric_dict,
    context_support_ratio,
    exact_match,
    rouge_l,
    token_f1,
)
from legal_benchmark.retrieval import BM25Retriever, DenseRetriever, TfidfRetriever, retrieval_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Indian legal QA datasets.")
    parser.add_argument("--dataset", default="Prarabdha/indian-legal-supervised-fine-tuning-data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--retrieval-corpus-size", type=int, default=1000)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--corpus-file", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="benchmark_outputs/run")
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--retriever", default="tfidf", choices=["tfidf", "bm25", "dense"])
    parser.add_argument("--dense-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--use-retrieved-context", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=3500)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-streaming", action="store_true")
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

    extractive = ExtractiveSentenceAnswerer()
    generator = None
    if not args.skip_generation:
        generator = Seq2SeqAnswerer(
            model_name=args.model_name,
            device=actual_device,
            max_context_chars=args.max_context_chars,
            max_new_tokens=args.max_new_tokens,
        )

    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, rank in tqdm(list(zip(eval_rows, ranks)), desc="Answering"):
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
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "eval_size": args.eval_size,
        "retrieval_corpus_size": args.retrieval_corpus_size,
        "model_name": None if args.skip_generation else args.model_name,
        "retriever": args.retriever,
        "dense_model_name": args.dense_model_name if args.retriever == "dense" else None,
        "eval_file": args.eval_file,
        "corpus_file": args.corpus_file,
        "device": actual_device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "use_retrieved_context": args.use_retrieved_context,
        "retrieval": retrieval_summary,
        "answering": answer_summary,
        "seconds": round(time.time() - started, 2),
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "config.json", vars(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
