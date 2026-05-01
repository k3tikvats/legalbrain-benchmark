# Provenance And Licensing Due Diligence

This is the material to convert into the paper appendix and dataset card.

## Current Public Dataset Claim

The Hugging Face dataset card states that the corpus is built from publicly and legally accessible Indian legal sources, including:

- Supreme Court judgments
- High Court decisions
- Law Commission reports
- public legal textbooks and commentaries
- open legal news archives
- public-domain legal Q&A portals
- government acts, rules, and notifications

## Submission Risk

The dataset is released under Apache-2.0, but a permissive dataset-level license is only defensible if upstream sources permit redistribution and derivative dataset release. Public accessibility is not the same as permission to redistribute. This is a likely reviewer/legal concern.

## Before Submission, Fill This Table

| Source category | Example URLs/repositories | Collection dates | Upstream license/terms | Redistribution allowed? | Notes |
|---|---|---|---|---|---|
| Supreme Court judgments | TODO | TODO | TODO | TODO | TODO |
| High Court judgments | TODO | TODO | TODO | TODO | TODO |
| Law Commission reports | TODO | TODO | TODO | TODO | TODO |
| Government acts/rules/notifications | TODO | TODO | TODO | TODO | TODO |
| Public legal Q&A | TODO | TODO | TODO | TODO | TODO |
| Legal textbooks/commentaries | TODO | TODO | TODO | TODO | Highest-risk category; verify public-domain status or remove. |
| Legal news archives | TODO | TODO | TODO | TODO | Highest-risk category; verify license or remove. |

## Recommended Paper Language

Use cautious wording:

> The dataset is distributed under Apache-2.0 at the derived dataset level. Because upstream legal sources may differ in copyright and terms of use, we document source categories, collection dates, and redistribution assumptions, and recommend that downstream users verify source-specific legal constraints before commercial deployment.

Avoid:

> All public legal data is free to redistribute.

## Privacy/PII Risk

Indian judgments and public legal documents may include names, addresses, family relationships, criminal allegations, medical facts, property details, caste/religion references, and other sensitive information. Public access does not eliminate downstream privacy risk. The release should include explicit warnings against re-identification, profiling, automated adjudication, or unsupervised legal advice.
