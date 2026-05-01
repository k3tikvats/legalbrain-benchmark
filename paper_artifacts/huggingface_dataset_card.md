---
language:
  - en
  - hi
  - ta
  - te
  - mr
  - bn
  - kn
  - or
license: apache-2.0
task_categories:
  - question-answering
  - text-generation
  - summarization
  - sentence-similarity
pretty_name: LegalBrain Indic Legal Corpus
size_categories:
  - 1M<n<10M
---

# LegalBrain Indic Legal Corpus

LegalBrain Indic Legal Corpus is a large-scale Indian legal dataset for research on grounded legal question answering, supervised fine-tuning, retrieval-augmented generation, legal summarization, and legal information access.

The dataset contains context-question-response triples derived from public Indian legal materials. Each example is designed so that the response should be answerable from the provided context.

## Dataset Structure

Each row contains:

| Field | Description |
|---|---|
| `context` | Legal passage, judgment excerpt, statute excerpt, report text, or commentary passage |
| `question` | Legal or interpretive query grounded in the context |
| `response` | Reference answer supported by the context |

## Intended Uses

Recommended research uses:

- grounded Indian legal QA
- retrieval-augmented generation
- legal passage retrieval
- legal summarization
- supervised fine-tuning of legal assistants
- hallucination and grounding evaluation

Not recommended:

- direct legal advice without qualified human review
- automated adjudication
- legal risk scoring of people or organizations
- surveillance or profiling
- production systems that affect legal rights without expert oversight

## Baseline Benchmark Results

Derived held-out retrieval over 500 evaluation examples and 20,000 distractor contexts:

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF | 0.492 | 0.694 | 0.768 | 0.579 |
| BM25 | 0.706 | 0.826 | 0.870 | 0.761 |
| MiniLM dense | 0.394 | 0.584 | 0.626 | 0.473 |

FLAN-T5-base QA over 500 examples:

| Context setting | EM | Token F1 | ROUGE-L | Grounding |
|---|---:|---:|---:|---:|
| Gold context | 0.010 | 0.192 | 0.171 | 0.594 |
| BM25 retrieved context | 0.010 | 0.163 | 0.146 | 0.574 |

Grounding is computed as the fraction of generated answer tokens appearing in the supplied context. It is a lexical proxy for context support, not a guarantee of legal correctness.

## Limitations

The dataset may have uneven coverage across courts, languages, regions, legal domains, and time periods. It may reflect biases present in public legal records and in the selection of accessible sources. Legal documents may contain sensitive facts about real people. Public availability does not eliminate privacy or misuse risk.

Users should document downstream use, evaluate for bias and hallucination, and keep qualified humans in the loop for legal applications.

## Citation

TODO: Add citation after paper submission.
