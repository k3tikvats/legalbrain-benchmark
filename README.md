# LegalBrain Indic Legal Corpus — NeurIPS 2026 E&D Benchmark

**Submission**: NeurIPS 2026 Evaluations & Datasets Track (anonymous + single-blind)
**Dataset**: [CyCrawwler/LegalBrain-Indic-Legal-Corpus](https://huggingface.co/datasets/CyCrawwler/LegalBrain-Indic-Legal-Corpus) 
**Code**: [github.com/k3tikvats/legalbrain-benchmark](https://github.com/k3tikvats/legalbrain-benchmark)
**Licenses**: dataset CC BY 4.0 (derivative annotation layer); benchmark code Apache-2.0

---

## What This Repo Contains

```
legal_benchmark/        # Benchmark harness (retrieval + generation + metrics)
paper_artifacts/
  NeurIPS_2026/         # LaTeX paper source (.tex, .sty, .bib, checklist)
  dataset_audit/        # 20k-row audit summary (stats, language distribution)
  legalbrain_croissant_rai.json   # Croissant metadata with RAI fields
  responsible_ai_metadata_draft.md
  results_tables.md     # All benchmark results in one place
  failure_summary.md    # Error analysis (60-sample heuristic audit)
  heldout_benchmark/    # manifest.json only; eval/corpus jsonl not in repo
benchmark_outputs/      # summary.json per run (predictions.jsonl excluded)
requirements.txt        # CPU install
requirements-cuda.txt   # CUDA 12.1 install (RTX 30xx series)
run_local_benchmark.ps1 # Windows PowerShell runner (auto-installs via uv)
BENCHMARKING.md         # Full benchmark documentation
KAGGLE_RUN.md           # Kaggle notebook instructions
```

---

## Quick Start (Windows, RTX 3050 or similar)

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11.

```powershell
# Small smoke test (50 examples, 1k corpus, ~2 min)
.\run_local_benchmark.ps1 -EvalSize 50 -RetrievalCorpusSize 1000

# Retrieval-only at paper scale (500 examples, 20k corpus, ~2 min)
.\run_local_benchmark.ps1 -EvalSize 500 -RetrievalCorpusSize 20000 -SkipGeneration

# Full oracle-context QA at paper scale (~8 min on RTX 3050 6GB)
.\run_local_benchmark.ps1 -EvalSize 500 -RetrievalCorpusSize 20000 -Model google/flan-t5-base -OutputDir benchmark_outputs/my_run
```

For CPU-only:

```powershell
.\run_local_benchmark.ps1 -EvalSize 50 -RetrievalCorpusSize 500 -CpuOnly
```

---

## Reproducing Paper Results (Heldout Split)

The held-out benchmark split (500 eval + 20,000 corpus) is built from the dataset using a fixed seed and context-fingerprint deduplication. Rebuild it with:

```powershell
uv run --python 3.11 --with-requirements requirements-cuda.txt `
  --extra-index-url https://download.pytorch.org/whl/cu121 `
  --index-strategy unsafe-best-match `
  python -m legal_benchmark.split_builder `
  --output-dir paper_artifacts/heldout_benchmark
```

Then run retrieval baselines:

```powershell
# BM25 retrieval (paper Table 1 — R@1=0.706, MRR=0.761)
uv run --python 3.11 --with-requirements requirements-cuda.txt `
  --extra-index-url https://download.pytorch.org/whl/cu121 `
  --index-strategy unsafe-best-match `
  python -m legal_benchmark.run_benchmark `
  --eval-file paper_artifacts/heldout_benchmark/eval.jsonl `
  --corpus-file paper_artifacts/heldout_benchmark/corpus.jsonl `
  --retriever bm25 --skip-generation `
  --output-dir benchmark_outputs/heldout_bm25

# FLAN-T5-base oracle-context QA (paper Table 2 — F1=0.192)
uv run --python 3.11 --with-requirements requirements-cuda.txt `
  --extra-index-url https://download.pytorch.org/whl/cu121 `
  --index-strategy unsafe-best-match `
  python -m legal_benchmark.run_benchmark `
  --eval-file paper_artifacts/heldout_benchmark/eval.jsonl `
  --corpus-file paper_artifacts/heldout_benchmark/corpus.jsonl `
  --retriever bm25 --model-name google/flan-t5-base `
  --output-dir benchmark_outputs/heldout_gold_bm25_flan
```

---

## Paper Results Summary

### Table 1 — Retrieval (500 eval, 20k corpus, heldout split)

| Retriever    | R@1       | R@5       | R@10      | MRR       |
| ------------ | --------- | --------- | --------- | --------- |
| TF-IDF       | 0.492     | 0.694     | 0.768     | 0.579     |
| **BM25**     | **0.706** | **0.826** | **0.870** | **0.761** |
| MiniLM Dense | 0.394     | 0.584     | 0.626     | 0.473     |

### Table 2 — Generation (FLAN-T5-base, 500 eval, heldout split)

| Context        | EM    | Token F1 | ROUGE-L | Grounding |
| -------------- | ----- | -------- | ------- | --------- |
| Gold           | 0.010 | 0.192    | 0.171   | 0.594     |
| BM25 Retrieved | 0.010 | 0.163    | 0.146   | 0.574     |

All runs on NVIDIA GeForce RTX 3050 6GB Laptop GPU.

---

## Dataset

~6.06M context-question-answer triples (~9.1 GB Parquet) from publicly accessible Indian legal sources (Supreme Court, High Courts, Law Commission, Government acts). Released under **CC BY 4.0** at the derivative-dataset level on Hugging Face; the benchmark **code** in this repository is released separately under **Apache-2.0**. A 100-row JSONL sample for reviewer inspection lives at `paper_artifacts/NeurIPS_2026/legalbrain_sample_100.jsonl`.

Croissant metadata with Responsible AI fields: `paper_artifacts/legalbrain_croissant_rai.json`

---

## For Reviewers

- Paper PDF: see `paper_artifacts/NeurIPS_2026/` (compile `legalbrain_neurips_2026.tex`)
- All benchmark configs: `benchmark_outputs/*/summary.json`
- Error analysis: `paper_artifacts/failure_summary.md`
- Dataset statistics: `paper_artifacts/dataset_audit/audit_summary.md`
