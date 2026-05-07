"""
Cross-encoder reranker baseline: BM25 top-20 → cross-encoder rerank → report R@1/R@5/R@10/MRR.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.
"""
import json, os, sys, time, statistics, pathlib

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_benchmark.retrieval import BM25Retriever
from sentence_transformers.cross_encoder import CrossEncoder

EVAL_FILE = "paper_artifacts/heldout_benchmark/eval.jsonl"
CORPUS_FILE = "paper_artifacts/heldout_benchmark/corpus.jsonl"
OUT_DIR = "benchmark_outputs/heldout_crossencoder_bm25_500_20k"
RERANK_TOP_K = 20  # BM25 retrieves top-20, reranker re-orders them
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading eval and corpus...")
eval_rows = [json.loads(l) for l in open(EVAL_FILE, encoding="utf-8") if l.strip()]
corpus_rows = [json.loads(l) for l in open(CORPUS_FILE, encoding="utf-8") if l.strip()]

# Build corpus id → context map
corpus_by_id = {r["row_id"]: r["context"] for r in corpus_rows}
# Also include eval contexts (gold context must be in the pool)
eval_by_id = {r["row_id"]: r["context"] for r in eval_rows}

print(f"Eval: {len(eval_rows)}, Corpus: {len(corpus_rows)}")

# Build BM25 index over corpus + eval contexts
print("Building BM25 index...")
all_ctx_ids = list(corpus_by_id.keys()) + [r for r in eval_by_id if r not in corpus_by_id]
all_ctxs = [corpus_by_id.get(rid, eval_by_id.get(rid, "")) for rid in all_ctx_ids]

from rank_bm25 import BM25Okapi
import re

def tokenize(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()

tokenized_corpus = [tokenize(c) for c in all_ctxs]
bm25_index = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
ce = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)

print("Running BM25 → CrossEncoder reranking...")
t0 = time.time()

recalls = {1: [], 5: [], 10: []}
rr_list = []
results = []

for i, row in enumerate(eval_rows):
    if i % 50 == 0:
        print(f"  [{i}/{len(eval_rows)}]", flush=True)

    qid = row["row_id"]
    question = row["question"]
    gold_ctx = row["context"]

    # BM25 top-20
    q_tokens = tokenize(question)
    scores = bm25_index.get_scores(q_tokens)
    top20_idx = sorted(range(len(scores)), key=lambda x: -scores[x])[:RERANK_TOP_K]

    # Make sure gold context is findable
    gold_idx = None
    for j, rid in enumerate(all_ctx_ids):
        if (corpus_by_id.get(rid, eval_by_id.get(rid, "")) == gold_ctx):
            gold_idx = j
            break

    # Cross-encoder rerank the top-20
    ce_pairs = [(question, all_ctxs[idx]) for idx in top20_idx]
    ce_scores = ce.predict(ce_pairs)

    # Sort by cross-encoder score
    reranked = sorted(zip(top20_idx, ce_scores), key=lambda x: -x[1])
    reranked_ctxs = [all_ctxs[idx] for idx, _ in reranked]

    # Compute retrieval rank
    retrieval_rank = None
    for rank, ctx in enumerate(reranked_ctxs):
        if ctx == gold_ctx:
            retrieval_rank = rank  # 0-indexed
            break

    for k in [1, 5, 10]:
        recalls[k].append(1 if retrieval_rank is not None and retrieval_rank < k else 0)

    rr = (1.0 / (retrieval_rank + 1)) if retrieval_rank is not None else 0.0
    rr_list.append(rr)

    results.append({
        "row_id": qid,
        "question": question,
        "retrieval_rank": retrieval_rank,
        "used_retrieved_context": None,
        "metrics": {
            "retrieval_rank": retrieval_rank if retrieval_rank is not None else -1,
        }
    })

elapsed = time.time() - t0

r1 = statistics.mean(recalls[1])
r5 = statistics.mean(recalls[5])
r10 = statistics.mean(recalls[10])
mrr = statistics.mean(rr_list)

print(f"\n=== BM25 + CrossEncoder Reranker (top-{RERANK_TOP_K} → rerank) ===")
print(f"  R@1  = {r1:.4f}")
print(f"  R@5  = {r5:.4f}")
print(f"  R@10 = {r10:.4f}")
print(f"  MRR  = {mrr:.4f}")
print(f"  Time = {elapsed:.1f}s")

# Save
summary = {
    "retriever": f"BM25_top{RERANK_TOP_K}+CrossEncoder({CROSS_ENCODER_MODEL})",
    "n_eval": len(eval_rows),
    "recall_at_1": round(r1, 4),
    "recall_at_5": round(r5, 4),
    "recall_at_10": round(r10, 4),
    "mrr": round(mrr, 4),
    "elapsed_seconds": round(elapsed, 1),
}

with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(f"{OUT_DIR}/predictions.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

config = {
    "retriever": "bm25_crossencoder_rerank",
    "cross_encoder_model": CROSS_ENCODER_MODEL,
    "rerank_top_k": RERANK_TOP_K,
    "eval_file": EVAL_FILE,
    "corpus_file": CORPUS_FILE,
}
with open(f"{OUT_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\nSaved to {OUT_DIR}/")
