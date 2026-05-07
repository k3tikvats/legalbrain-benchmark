"""Quick static check on the .tex: braces, begin/end pairs, refs, citations."""
import re
from collections import Counter
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_artifacts" / "NeurIPS_2026" / "legalbrain_neurips_2026.tex"
BIB = TEX.parent / "references.bib"

text = TEX.read_text(encoding="utf-8")

opens = text.count("{")
closes = text.count("}")
print(f"braces: open={opens} close={closes} diff={opens - closes}")

begins = re.findall(r"\\begin\{(\w+)\}", text)
ends = re.findall(r"\\end\{(\w+)\}", text)
b = Counter(begins)
e = Counter(ends)
mm = [(env, b[env], e[env]) for env in set(b) | set(e) if b[env] != e[env]]
print(f"begin/end mismatches: {mm}")

refs = set(re.findall(r"\\ref\{([^}]+)\}", text))
labels = set(re.findall(r"\\label\{([^}]+)\}", text))
print(f"missing labels (referenced but not defined): {sorted(refs - labels)}")

bib = BIB.read_text(encoding="utf-8")
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cite_keys: set[str] = set()
for m in re.finditer(r"\\cite[pt]?\{([^}]+)\}", text):
    for k in m.group(1).split(","):
        cite_keys.add(k.strip())
print(f"missing bib entries (cited but not in references.bib): {sorted(cite_keys - bib_keys)}")
print(f"unused bib entries: {sorted(bib_keys - cite_keys)}")
