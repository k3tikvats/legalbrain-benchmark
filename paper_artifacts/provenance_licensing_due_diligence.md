# Provenance And Licensing Due Diligence

Supplementary documentation for the LegalBrain Indic Legal Corpus submission to NeurIPS 2026 Evaluations & Datasets Track (Submission #4193). This document accompanies the paper §3 (Provenance) and §Limitations (Licensing of upstream sources).

## Dataset Release License

The derivative LegalBrain Indic Legal Corpus is released under **Creative Commons Attribution 4.0 International (CC BY 4.0)** at the derived-dataset level. The CC BY 4.0 release covers the curated context-question-answer triples, annotations, splits, and benchmark scaffolding produced by the authors. Upstream legal source materials retain their original terms; users redistributing or commercialising any portion of the corpus must independently verify upstream source terms and attribute accordingly.

## Per-Source Provenance and Licensing Matrix

Collection period: 2024–2025. Sources accessed exclusively through public-facing portals without authentication or paid subscription. Cases requiring mandatory victim-identity protection (POCSO Act, sexual-offence proceedings, juvenile cases) were excluded at collection time per Indian Supreme Court guidance (cf. *Nipun Saxena v. Union of India*, 2018).

| Source category | Example portals / repositories | Collection period | Upstream license / terms | Redistribution position | Notes |
|---|---|---|---|---|---|
| Supreme Court of India judgments | `main.sci.gov.in`, eCourts SC India | 2024-2025 | SCI website terms: public records, free for personal/research/non-commercial reuse; commercial reuse not addressed | Permitted for non-commercial research; commercial users should consult SCI directly | Public records; no commercial reuse claim made |
| High Court judgments | High Court eCourts portals (Delhi, Bombay, Madras, etc.) | 2024–2025 | Varies by court; most HCs publish judgments as public records under HC-specific reuse policies | Court-specific verification required for redistribution | Different HCs apply different terms; redistributors should attribute the originating court |
| Law Commission of India reports | `lawcommissionofindia.nic.in` | 2024–2025 | Government Open Data License – India (GODL) | Permitted with attribution under GODL | Government publication; non-modification clause does not apply to derivative annotations |
| Government acts, rules, notifications | `indiacode.nic.in`, `egazette.nic.in` | 2024–2025 | GODL – India / IndiaCode terms | Permitted with attribution | Statutory text is generally non-copyrightable as government material |
| Public-domain legal Q&A excerpts | Selected open legal information portals | 2024–2025 | Source-specific; included only where portal terms allowed reproduction | Limited inclusion; per-row source attribution where available | Subset deliberately small; flagged-risk sources excluded |
| Legal textbooks / commentaries | Public-domain or open-license commentary sources only | 2024–2025 | Public-domain or explicit open-license only | Included only where public-domain status was verifiable | Flagged-risk copyrighted textbooks were excluded |
| Legal news archives | **Excluded** | — | — | — | Flagged-risk category; removed at collection time to avoid ambiguous terms-of-use |

## PII and DPDP Act 2023

Indian court judgments contain personal data of parties, witnesses, advocates, and other identified persons. The corpus relies on the publicly-available-information lawful basis under the Digital Personal Data Protection Act, 2023, for collection and release of derivative annotations. Categories of sensitive content the user community should be aware of:

- Names, addresses, family relationships of parties and witnesses
- Criminal allegations and case-specific facts
- Medical, financial, and property information referenced in judgments
- Caste / religion / community references where relevant to the legal record
- Procedural identifiers (FIR numbers, case numbers, neutral citations)

POCSO Act matters, sexual-offence proceedings, and juvenile cases were excluded at collection time. Downstream users should still apply additional redaction or filtering before deployment, since automated upstream classification is imperfect, and should perform DPDP Act applicability assessment for their specific use case.

## Compliance Recommendations for Downstream Users

- Cite the dataset and acknowledge upstream source attribution obligations (SCI policy, HC-specific terms, GODL).
- Do not use the corpus for re-identification, profiling, surveillance, or automated adjudication of real persons.
- For commercial redistribution, perform per-source license verification; the CC BY 4.0 release does not override upstream restrictions.
- Apply additional PII redaction and content filtering appropriate to the use case under DPDP Act 2023.
- Treat reference answers as supervised-learning targets, not as authoritative legal interpretations; the human review layer was conducted by the authors and contributors without formal credentialed legal annotation, and inter-rater reliability was not measured.

## Recommended Paper Language

> The dataset is released under CC BY 4.0 at the derived-dataset level. Upstream Indian legal materials retain their original terms (SCI website policy, HC-specific reuse rules, GODL-India for statutory text and Law Commission reports); the CC BY 4.0 release covers the derivative annotation layer. Downstream redistributors are responsible for verifying source-specific legal constraints before commercial deployment. POCSO and other mandatory-anonymisation case categories were excluded at collection time. The dataset must not be used for re-identification, profiling, automated adjudication, or unsupervised legal advice.

## Removed at Collection Time (high-risk categories)

- Legal news archives (terms-of-use ambiguity)
- Copyrighted legal textbooks and commentaries
- POCSO / sexual-offence / juvenile-anonymisation cases (Indian SC mandate)
- Materials requiring authentication or paid subscription
