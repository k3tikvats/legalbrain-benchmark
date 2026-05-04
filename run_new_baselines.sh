#!/usr/bin/env bash
# Run all new baselines for the NeurIPS 2026 paper.
# Execute from the LEGAL-PAPER root: bash run_new_baselines.sh
# Requires: .venv with torch+cu121, sentence-transformers, groq, bitsandbytes
# Set GROQ_API_KEY before running if you want the Groq Llama-3.1-8B run.

set -e
PYTHON=".venv/Scripts/python.exe"
EVAL="paper_artifacts/heldout_benchmark/eval.jsonl"
CORPUS="paper_artifacts/heldout_benchmark/corpus.jsonl"
SIZE="--eval-size 500 --retrieval-corpus-size 20000"

echo "=== InLegalBERT retrieval ==="
$PYTHON -m legal_benchmark.run_benchmark \
  --eval-file $EVAL --corpus-file $CORPUS $SIZE \
  --retriever inlegal --skip-generation \
  --output-dir benchmark_outputs/heldout_inlegalbert_500_20k

echo "=== Hybrid BM25+BGE retrieval ==="
$PYTHON -m legal_benchmark.run_benchmark \
  --eval-file $EVAL --corpus-file $CORPUS $SIZE \
  --retriever hybrid --dense-model-name BAAI/bge-base-en-v1.5 --skip-generation \
  --output-dir benchmark_outputs/heldout_hybrid_bm25_bge_500_20k

echo "=== Extended tasks (domain + citation) with BM25 ==="
$PYTHON -m legal_benchmark.run_benchmark \
  --eval-file $EVAL --corpus-file $CORPUS $SIZE \
  --retriever bm25 --skip-generation --run-extended-tasks \
  --output-dir benchmark_outputs/heldout_extended_tasks_500_20k

echo "=== TinyLlama-1.1B-Chat 4-bit NF4 generation ==="
$PYTHON -m legal_benchmark.run_benchmark \
  --eval-file $EVAL --corpus-file $CORPUS $SIZE \
  --retriever bm25 --answerer causal \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --load-in-4bit \
  --output-dir benchmark_outputs/heldout_tinyllama_bm25_500_20k

if [ -n "$GROQ_API_KEY" ]; then
  echo "=== Groq Llama-3.1-8B generation ==="
  $PYTHON -m legal_benchmark.run_benchmark \
    --eval-file $EVAL --corpus-file $CORPUS $SIZE \
    --retriever bm25 --answerer groq \
    --model-name llama-3.1-8b-instant \
    --output-dir benchmark_outputs/heldout_groq_llama_bm25_500_20k
else
  echo "GROQ_API_KEY not set — skipping Groq run. Get key at console.groq.com."
fi

echo "=== All runs complete ==="
