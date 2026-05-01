# Kaggle Benchmark Run

Use Kaggle if local setup cannot install Python 3.11/CUDA PyTorch, or if you want larger runs than an RTX 3050 6GB can comfortably handle.

1. Create a Kaggle notebook with GPU enabled.
2. Upload this folder as a dataset or copy the files into `/kaggle/working/legal-benchmark`.
3. Run:

```bash
cd /kaggle/working/legal-benchmark
pip install -r requirements.txt
python -m legal_benchmark.run_benchmark \
  --dataset Prarabdha/indian-legal-supervised-fine-tuning-data \
  --eval-size 500 \
  --retrieval-corpus-size 20000 \
  --model-name google/flan-t5-base \
  --output-dir benchmark_outputs/kaggle_flan_t5_base \
  --device auto
```

For a fast smoke test:

```bash
python -m legal_benchmark.run_benchmark \
  --eval-size 25 \
  --retrieval-corpus-size 500 \
  --model-name google/flan-t5-small \
  --output-dir benchmark_outputs/smoke
```

Recommended NeurIPS-scale runs:

- Retrieval: `--eval-size 1000 --retrieval-corpus-size 50000 --skip-generation`
- Generation on 6GB GPU: `google/flan-t5-small`, batch size 1-4
- Generation on Kaggle T4/P100: `google/flan-t5-base`, batch size 2-8

Do not use the training split as both train and test in the paper without careful leakage controls. For the final paper, create held-out splits by source court, time period, language, and near-duplicate clusters.
