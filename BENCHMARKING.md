# LegalBrain Benchmark Harness

This benchmark is designed for the NeurIPS 2026 Evaluations & Datasets track framing: it measures how useful the corpus is for Indian legal QA, retrieval, and grounded answer generation.

## What It Evaluates

- Retrieval: can a question retrieve its paired legal context from a corpus?
- Answer generation: can a small model answer from the provided context?
- Faithfulness proxy: how much of the generated answer is supported by the context?
- Reference similarity: exact match, token F1, ROUGE-L, and context-support ratio.

## Local RTX 3050 6GB Recommendation

Start small:

```powershell
.\run_local_benchmark.ps1 -EvalSize 50 -RetrievalCorpusSize 1000
```

Use `-OutputDir benchmark_outputs/my_run_name` to keep multiple runs.

Retrieval-only is much cheaper:

```powershell
.\run_local_benchmark.ps1 -EvalSize 500 -RetrievalCorpusSize 10000 -SkipGeneration
```

The local runner installs a CUDA 12.1 PyTorch wheel by default, which works with newer NVIDIA drivers. Your GPU should handle `google/flan-t5-small`. If CUDA memory fails, rerun with smaller `EvalSize`, or use Kaggle for `flan-t5-base` or larger.

For CPU-only setup:

```powershell
.\run_local_benchmark.ps1 -EvalSize 50 -RetrievalCorpusSize 1000 -CpuOnly
```

## Outputs

Each run writes:

- `summary.json`: aggregate metrics
- `predictions.jsonl`: per-example predictions
- `config.json`: exact run settings

These are paper-ready artifacts for tables and error analysis.
