# NeurIPS E&D Submission Checklist

## Must Finish Before Submission

- Replace `Anonymous Authors` in `neurips_ed_draft.tex` only if using single-blind review.
- Decide review mode:
  - Double-blind if dataset/code can be anonymized.
  - Single-blind if the public dataset identity cannot be reasonably anonymized.
- Use the official NeurIPS 2026 LaTeX files:
  - `neurips_2026.sty`
  - `neurips_2026.tex` checklist section
- Add the mandatory NeurIPS checklist to the final PDF.
- Add exact source provenance:
  - source websites or repositories
  - collection dates
  - license / terms-of-use reasoning
  - OCR and preprocessing details
  - annotation instructions
  - Argilla review workflow
- Add dataset statistics:
  - total rows
  - language distribution
  - context/question/response length distribution
  - source/domain distribution
  - duplicate removal statistics
- Create held-out splits for the final benchmark if time permits:
  - by court/source
  - by date
  - by language
  - by near-duplicate cluster
- Download Hugging Face Croissant metadata.
- Add minimal Responsible AI fields.
- Validate Croissant JSON before OpenReview upload.
- Upload dataset URL and validated Croissant file to OpenReview.

## Current Results To Include

Retrieval, 500 examples, 10k candidates:

- Recall@1: 0.644
- Recall@5: 0.830
- Recall@10: 0.874
- MRR: 0.722

Retrieval, 500 examples, 20k candidates:

- Recall@1: 0.576
- Recall@5: 0.770
- Recall@10: 0.844
- MRR: 0.660

FLAN-T5-base oracle-context QA:

- EM: 0.008
- Token F1: 0.213
- ROUGE-L: 0.188
- Grounding: 0.720

FLAN-T5-base retrieved-context QA:

- EM: 0.006
- Token F1: 0.152
- ROUGE-L: 0.131
- Grounding: 0.683

## Claims That Are Safe

- The corpus supports reproducible evaluation of Indian legal retrieval and grounded QA.
- Lexical retrieval is useful but degrades as candidate pool size increases.
- Oracle-context and retrieved-context results expose separate generation and retrieval bottlenecks.
- Small general instruction models remain weak on precise Indian legal QA.
- Grounding score is useful as an automatic proxy, but not a substitute for legal correctness or semantic entailment.

## Claims To Avoid

- Do not claim the dataset solves legal AI.
- Do not claim models trained on this corpus can provide legal advice.
- Do not claim grounding score proves correctness.
- Do not claim full representativeness of Indian law unless you have source and language distribution evidence.
- Do not claim all data is risk-free because it is public.
