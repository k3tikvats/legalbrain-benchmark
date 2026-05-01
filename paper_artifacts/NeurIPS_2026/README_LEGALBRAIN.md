# LegalBrain NeurIPS 2026 E&D Draft

Main paper file:

```text
legalbrain_neurips_2026.tex
```

Supporting files:

```text
neurips_2026.sty
references.bib
checklist_completed.tex
```

Compile with:

```bash
pdflatex legalbrain_neurips_2026
bibtex legalbrain_neurips_2026
pdflatex legalbrain_neurips_2026
pdflatex legalbrain_neurips_2026
```

or:

```bash
latexmk -pdf legalbrain_neurips_2026.tex
```

Before submission:

- Replace `add-email-before-submission@example.com`.
- Add any coauthors and affiliations.
- Decide whether to keep `\usepackage[eandd,nonanonymous]{neurips_2026}`.
  - Keep `nonanonymous` if submitting single-blind because the public dataset identity cannot be anonymized.
  - Remove `nonanonymous` if you prepare anonymized dataset/code links.
- Complete exact source provenance and annotation workflow details.
- Add manually verified error-analysis counts after filling `../error_analysis_sample.csv`.
- Validate `../legalbrain_croissant_rai.json` with the Croissant validator.
