# Benchmark Results

## Retrieval

| Run | Eval | Corpus | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| flan_small_50 | 50 | 1000 | 0.880 | 0.940 | 0.960 | 0.908 |
| retrieval_500_10k | 500 | 10000 | 0.644 | 0.830 | 0.874 | 0.722 |
| gold_flan_t5_base_500_20k | 500 | 20000 | 0.576 | 0.770 | 0.844 | 0.660 |
| rag_flan_t5_base_500_20k | 500 | 20000 | 0.576 | 0.770 | 0.844 | 0.660 |

## Answer Generation

| Run | Model | Context | EM | Token F1 | ROUGE-L | Grounding | Seconds |
|---|---|---|---:|---:|---:|---:|---:|
| flan_small_50 | google/flan-t5-small | gold | 0.000 | 0.165 | 0.146 | 0.823 | 44.410 |
| retrieval_500_10k | extractive baseline | gold | 0.000 | 0.267 | 0.213 | 1.000 | 36.130 |
| gold_flan_t5_base_500_20k | google/flan-t5-base | gold | 0.008 | 0.213 | 0.188 | 0.720 | 304.750 |
| rag_flan_t5_base_500_20k | google/flan-t5-base | retrieved | 0.006 | 0.152 | 0.131 | 0.683 | 251.470 |
