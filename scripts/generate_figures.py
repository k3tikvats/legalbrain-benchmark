"""
Generate publication-quality figures for the LegalBrain NeurIPS paper.
Outputs:
  paper_artifacts/NeurIPS_2026/fig_retrieval_recall.pdf
  paper_artifacts/NeurIPS_2026/fig_domain_heatmap.pdf
"""
import json, pathlib, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = pathlib.Path("paper_artifacts/NeurIPS_2026")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette (NeurIPS-friendly) ─────────────────────────────────────────
COLORS = {
    "TF-IDF":              "#aec6cf",  # light blue
    "BM25":                "#2166ac",  # strong blue
    "MiniLM dense":        "#f4a582",  # salmon
    "BGE-base-en":         "#d6604d",  # red-orange
    "InLegalBERT dense":   "#999999",  # grey
    "Hybrid BM25+BGE":     "#74c476",  # green
    "BM25+CrossEncoder":   "#238b45",  # dark green
    "E5-base-v2":          "#9970ab",  # purple
}

# ── Figure 1: Recall@K grouped bar chart ──────────────────────────────────────
# Data: [R@1, R@5, R@10] — E5 will be filled in after the experiment;
#   placeholder values here so the script can run before E5 completes.
#   The paper-update step will regenerate with real E5 numbers.
import sys, os
# Try to load E5 result if available
e5_summary = pathlib.Path("benchmark_outputs/heldout_e5_base_500_20k/summary.json")
if e5_summary.exists():
    e5 = json.loads(e5_summary.read_text())
    e5_r1, e5_r5, e5_r10, e5_mrr = e5["r1"], e5["r5"], e5["r10"], e5["mrr"]
    print(f"Loaded E5 results: R@1={e5_r1:.3f}, R@5={e5_r5:.3f}, R@10={e5_r10:.3f}, MRR={e5_mrr:.3f}")
    include_e5 = True
else:
    e5_r1, e5_r5, e5_r10, e5_mrr = None, None, None, None
    include_e5 = False
    print("E5 results not yet available — generating figure without E5")

RETRIEVERS_BASE = [
    ("TF-IDF",             0.492, 0.694, 0.768, 0.579),
    ("BM25",               0.706, 0.826, 0.870, 0.761),
    ("MiniLM dense",       0.394, 0.584, 0.626, 0.473),
    ("BGE-base-en",        0.460, 0.612, 0.676, 0.527),
    ("InLegalBERT dense",  0.160, 0.252, 0.318, 0.200),
    ("Hybrid BM25+BGE",    0.612, 0.764, 0.814, 0.677),
    ("BM25+CrossEncoder",  0.714, 0.818, 0.846, 0.760),
]

if include_e5:
    RETRIEVERS_BASE.insert(5, ("E5-base-v2", e5_r1, e5_r5, e5_r10, e5_mrr))

RETRIEVERS = [(name, r1, r5, r10, mrr) for name, r1, r5, r10, mrr in RETRIEVERS_BASE]

names  = [r[0] for r in RETRIEVERS]
r1_vals  = [r[1] for r in RETRIEVERS]
r5_vals  = [r[2] for r in RETRIEVERS]
r10_vals = [r[3] for r in RETRIEVERS]

x = np.arange(len(names))
width = 0.26

fig, ax = plt.subplots(figsize=(8.5, 3.5))
bars1 = ax.bar(x - width, r1_vals,  width, label="R@1",  color=[COLORS.get(n, "#888") for n in names],     alpha=1.0,  edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x,          r5_vals,  width, label="R@5",  color=[COLORS.get(n, "#888") for n in names],     alpha=0.7,  edgecolor="white", linewidth=0.5)
bars3 = ax.bar(x + width,  r10_vals, width, label="R@10", color=[COLORS.get(n, "#888") for n in names],     alpha=0.45, edgecolor="white", linewidth=0.5)

# Add value labels on top of R@1 bars only (to avoid clutter)
for bar, val in zip(bars1, r1_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
            f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

ax.set_ylabel("Recall", fontsize=10)
ax.set_title("Retrieval Recall@K — 500-example held-out split (corpus: 20k)", fontsize=10, pad=6)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Opacity legend
legend_patches = [
    mpatches.Patch(facecolor="#555", alpha=1.0,  label="Recall@1"),
    mpatches.Patch(facecolor="#555", alpha=0.7,  label="Recall@5"),
    mpatches.Patch(facecolor="#555", alpha=0.45, label="Recall@10"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=8, framealpha=0.8)

plt.tight_layout()
out1 = OUT_DIR / "fig_retrieval_recall.pdf"
fig.savefig(out1, bbox_inches="tight", dpi=200)
plt.close()
print(f"Saved: {out1}")

# ── Figure 2: Domain heatmap ─────────────────────────────────────────────────
# Rows = domains, Cols = (R@1, MRR, F1-FLAN, F1-Tiny, F1-Groq)
DOMAINS = ["Criminal", "Civil", "Constitutional", "Contract", "Family", "Property", "Service", "Other", "Macro avg"]
# Updated under tightened Constitutional regex (May 2026 revision); see
# paper_artifacts/domain_partition_recomputed.json. Constitutional dropped
# from n=171 to n=39 once generic court vocabulary was removed; Other grew
# from 38 to 97; Criminal/Civil/Contract/Property/Service expanded modestly.
N       = [97, 80, 39, 69, 13, 46, 59, 97, 500]
R1      = [0.536, 0.613, 0.795, 0.725, 0.846, 0.804, 0.864, 0.742, 0.741]
MRR_D   = [0.609, 0.669, 0.841, 0.792, 0.904, 0.846, 0.896, 0.795, 0.794]
F1_FLAN = [0.208, 0.156, 0.203, 0.191, 0.150, 0.186, 0.214, 0.198, 0.188]
F1_TINY = [0.432, 0.477, 0.446, 0.474, 0.523, 0.504, 0.466, 0.498, 0.477]
F1_GROQ = [0.420, 0.446, 0.476, 0.473, 0.541, 0.503, 0.519, 0.504, 0.485]

data = np.array([R1, MRR_D, F1_FLAN, F1_TINY, F1_GROQ]).T  # shape (9,5)

COL_LABELS = ["R@1\n(BM25)", "MRR\n(BM25)", "F1\n(FLAN-T5)", "F1\n(TinyLlama)", "F1\n(Groq-8B)"]
ROW_LABELS = [f"{d}\n(n={n})" for d, n in zip(DOMAINS[:-1], N[:-1])] + [f"Macro avg\n(n={N[-1]})"]

# Custom colormap: white → blue
cmap = LinearSegmentedColormap.from_list("legal", ["#f7fbff", "#2166ac"])

fig, ax = plt.subplots(figsize=(7, 4.2))
im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.10, vmax=0.90)

# Annotate cells
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        text_col = "white" if val > 0.62 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7.5,
                color=text_col, fontweight="bold" if i == 8 else "normal")

# Draw separator line before Macro avg row
ax.axhline(7.5, color="black", linewidth=1.5)
ax.axvline(1.5, color="#aaa", linewidth=0.8, linestyle="--")  # separate retrieval / generation

ax.set_xticks(range(len(COL_LABELS)))
ax.set_xticklabels(COL_LABELS, fontsize=8)
ax.set_yticks(range(len(ROW_LABELS)))
ax.set_yticklabels(ROW_LABELS, fontsize=8)
ax.set_title("Domain-partitioned performance across seven Indian legal areas", fontsize=9, pad=6)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Score", fontsize=8)

# Column group labels
ax.text(0.5, -0.16, "← Retrieval →", transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=7.5, color="#444", style="italic")
ax.text(3.0, -0.16, "← Generation (Token F1) →", transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=7.5, color="#444", style="italic")

plt.tight_layout()
out2 = OUT_DIR / "fig_domain_heatmap.pdf"
fig.savefig(out2, bbox_inches="tight", dpi=200)
plt.close()
print(f"Saved: {out2}")

print("\nAll figures generated successfully.")
