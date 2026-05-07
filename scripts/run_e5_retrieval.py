"""
E5-base-v2 dense retrieval baseline.
intfloat/e5-base-v2 uses instruction prefixes:
  - queries:  "query: {text}"
  - passages: "passage: {text}"
This is the canonical E5 asymmetric retrieval setup.
"""
import json, os, sys, time, statistics, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
import torch

EVAL_FILE = "paper_artifacts/heldout_benchmark/eval.jsonl"
CORPUS_FILE = "paper_artifacts/heldout_benchmark/corpus.jsonl"
OUT_DIR = "benchmark_outputs/heldout_e5_base_500_20k"
MODEL_NAME = "intfloat/e5-base-v2"
BATCH_SIZE = 64

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading eval and corpus...")
eval_rows = [json.loads(l) for l in open(EVAL_FILE, encoding="utf-8") if l.strip()]
corpus_rows = [json.loads(l) for l in open(CORPUS_FILE, encoding="utf-8") if l.strip()]

corpus_by_id = {r["row_id"]: r["context"] for r in corpus_rows}
eval_by_id = {r["row_id"]: r["context"] for r in eval_rows}

print(f"Eval: {len(eval_rows)}, Corpus: {len(corpus_rows)}")

# Build unified pool (corpus + eval gold contexts)
all_ctx_ids = list(corpus_by_id.keys())
for rid in eval_by_id:
    if rid not in corpus_by_id:
        all_ctx_ids.append(rid)
all_ctxs = [corpus_by_id.get(rid, eval_by_id.get(rid, "")) for rid in all_ctx_ids]
ctx_to_idx = {ctx: i for i, ctx in enumerate(all_ctxs)}

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=device)

# E5 requires "passage: " prefix for corpus, "query: " prefix for queries
print(f"Encoding {len(all_ctxs)} corpus passages (batch={BATCH_SIZE})...")
t0 = time.time()
passage_texts = ["passage: " + c for c in all_ctxs]
corpus_embeddings = model.encode(
    passage_texts,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)
encode_time = time.time() - t0
print(f"Corpus encoding took {encode_time:.1f}s, shape={corpus_embeddings.shape}")

print("Running retrieval evaluation...")
t1 = time.time()
recalls = {1: [], 5: [], 10: []}
rr_list = []
results = []

for i, row in enumerate(eval_rows):
    if i % 50 == 0:
        print(f"  [{i}/{len(eval_rows)}]", flush=True)

    question = row["question"]
    gold_ctx = row["context"]

    # Encode query with "query: " prefix
    q_emb = model.encode(
        ["query: " + question],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Cosine similarity (embeddings are already normalized)
    scores = np.dot(corpus_embeddings, q_emb)

    # Find gold context index
    gold_idx = ctx_to_idx.get(gold_ctx)

    # Get top-10 ranking
    top_k = 10
    if top_k >= len(scores):
        order = np.argsort(-scores)
    else:
        order = np.argpartition(-scores, top_k)[:top_k]
        order = order[np.argsort(-scores[order])]
    top_ctxs = [all_ctxs[idx] for idx in order[:top_k]]

    retrieval_rank = None
    for rank, ctx in enumerate(top_ctxs):
        if ctx == gold_ctx:
            retrieval_rank = rank
            break

    for k in [1, 5, 10]:
        recalls[k].append(1 if retrieval_rank is not None and retrieval_rank < k else 0)

    rr = (1.0 / (retrieval_rank + 1)) if retrieval_rank is not None else 0.0
    rr_list.append(rr)

    results.append({
        "row_id": row["row_id"],
        "question": question,
        "retrieval_rank": retrieval_rank if retrieval_rank is not None else -1,
    })

elapsed = time.time() - t1

r1 = statistics.mean(recalls[1])
r5 = statistics.mean(recalls[5])
r10 = statistics.mean(recalls[10])
mrr = statistics.mean(rr_list)

print(f"\n=== E5-base-v2 Retrieval ===")
print(f"  R@1  = {r1:.4f}")
print(f"  R@5  = {r5:.4f}")
print(f"  R@10 = {r10:.4f}")
print(f"  MRR  = {mrr:.4f}")
print(f"  Query inference time = {elapsed:.1f}s")

# Save results
summary = {
    "model": MODEL_NAME,
    "eval_size": len(eval_rows),
    "corpus_size": len(corpus_rows),
    "r1": r1, "r5": r5, "r10": r10, "mrr": mrr,
    "encode_time_s": encode_time,
    "query_time_s": elapsed,
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

config = {
    "model": MODEL_NAME,
    "retriever": "e5_base_v2",
    "prefix_strategy": "query:/passage:",
    "corpus_size": len(all_ctxs),
}
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(OUT_DIR, "predictions.jsonl"), "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"\nResults saved to {OUT_DIR}/")
