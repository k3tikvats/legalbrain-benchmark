# Held-Out Benchmark Results

## Derived Held-Out Split

The upstream Hugging Face release exposes a public `train` split. For benchmark evaluation, we derived a local held-out split:

- streamed 30,000 valid rows
- removed 33 duplicate context fingerprints
- reserved 500 evaluation rows
- reserved 20,000 unique distractor contexts
- fingerprint: BLAKE2b over normalized first 180 context tokens

## Dataset Audit

Sample size: 20,000 rows.

| Field | Mean | P05 | P50 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Context tokens | 374.3 | 112 | 416 | 455 | 1250 |
| Question tokens | 20.9 | 13 | 20 | 32 | 66 |
| Response tokens | 47.1 | 12 | 42 | 101 | 361 |

| Audit item | Value |
|---|---:|
| Legal-signal rate | 0.937 |
| Duplicate context fingerprint rate | 0.001 |
| Latin-script dominant rows | 19,908 |
| Devanagari dominant rows | 89 |
| Unknown-script rows | 3 |

## Retrieval

| Retriever | Eval | Corpus | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 500 | 20,000 | 0.492 | 0.694 | 0.768 | 0.579 |
| BM25 | 500 | 20,000 | 0.706 | 0.826 | 0.870 | 0.761 |
| MiniLM dense | 500 | 20,000 | 0.394 | 0.584 | 0.626 | 0.473 |

## Generation

| Model | Context | Retriever | EM | Token F1 | ROUGE-L | Grounding |
|---|---|---|---:|---:|---:|---:|
| FLAN-T5-base | Gold | BM25 used only for retrieval reporting | 0.010 | 0.192 | 0.171 | 0.594 |
| FLAN-T5-base | Retrieved | BM25 | 0.010 | 0.163 | 0.146 | 0.574 |

## Error Review

60 lowest-scoring held-out BM25+FLAN-T5-base RAG predictions:

| Label | Count |
|---|---:|
| retrieval_mismatch | 25 |
| low_grounding | 21 |
| terse_incomplete | 13 |
| semantic_mismatch | 1 |
