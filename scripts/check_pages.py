"""Estimate body / appendix length to verify the 9-page NeurIPS limit."""
import re
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_artifacts" / "NeurIPS_2026" / "legalbrain_neurips_2026.tex"
text = TEX.read_text(encoding="utf-8")

m1 = text.find(r"\maketitle")
m_bib = text.find(r"\bibliographystyle")
m_app = text.find(r"\appendix")
body_end = min(x for x in [m_bib, m_app] if x > 0)
body = text[m1:body_end]
appendix = text[m_app:m_bib if m_bib > m_app else len(text)]


def words_of(s: str) -> int:
    s = re.sub(r"%[^\n]*", "", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?(\{[^}]*\})*", " ", s)
    s = re.sub(r"[\\{}]", " ", s)
    return len(s.split())


body_w = words_of(body)
ap_w = words_of(appendix)
n_tables = len(re.findall(r"\\begin\{table\}", body))
n_figures = len(re.findall(r"\\begin\{figure\}", body))
n_resize = len(re.findall(r"\\resizebox", body))

print(f"Body (excl. bib + appendix) words: {body_w}")
print(f"  rough page estimate at 600 wpm: {body_w/600:.1f}")
print(f"  rough page estimate at 550 wpm: {body_w/550:.1f}")
print(f"  tables in body: {n_tables}, figures in body: {n_figures}, resize'd tables: {n_resize}")
print(f"  (each table/figure typically eats 0.3-0.6 page beyond pure-text estimate)")
print()
print(f"Appendix words: {ap_w}  (~{ap_w/600:.1f} pages, NeurIPS allows unlimited appendix)")
