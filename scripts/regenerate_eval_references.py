"""Regenerate eval-set references with a non-Llama Groq model (Qwen3-32b).

Goal: break the Llama-3.1-8B-Instruct reference-generator confound on the
500 reported eval rows. Output saved to
`paper_artifacts/shadow_references_qwen3_500.jsonl` with fields:
    row_id, question, original_response, shadow_response

Designed to checkpoint per-row so it can be resumed after rate-limit pauses.

Usage:
    $env:GROQ_API_KEY = "gsk_..."  # or export GROQ_API_KEY=... in bash
    uv run --python 3.11 --with groq python scripts/regenerate_eval_references.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_FILE = REPO / "paper_artifacts" / "heldout_benchmark" / "eval.jsonl"
OUT_FILE = REPO / "paper_artifacts" / "shadow_references_qwen3_500.jsonl"

MODEL = "qwen/qwen3-32b"
MAX_CONTEXT_CHARS = 3500
MAX_NEW_TOKENS = 256

PROMPT_SYSTEM = (
    "You are an expert Indian legal assistant. Read the legal context carefully and answer "
    "the question using only the information explicitly stated in the context. Do not infer, "
    "speculate, or use external knowledge. If the answer is not stated in the context, say "
    "'The context does not provide enough information.' Keep the answer concise (1-3 sentences) "
    "and grounded in the exact wording of the context wherever possible."
)


def main() -> None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("ERROR: GROQ_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        print("ERROR: install groq package (pip install groq)", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=key)

    eval_rows = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(eval_rows)} eval rows", flush=True)

    done_ids: set[int] = set()
    if OUT_FILE.exists():
        for line in OUT_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                done_ids.add(int(rec["row_id"]))
        print(f"Resuming: {len(done_ids)} already done, skipping", flush=True)

    open_mode = "a" if done_ids else "w"
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    backoff = 1.0
    consecutive_errors = 0
    written = 0

    with OUT_FILE.open(open_mode, encoding="utf-8") as out_handle:
        for i, row in enumerate(eval_rows):
            rid = int(row["row_id"])
            if rid in done_ids:
                continue
            context = (row.get("context") or "")[:MAX_CONTEXT_CHARS]
            question = row.get("question") or ""
            user_msg = f"Context:\n{context}\n\nQuestion: {question}"
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": PROMPT_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=MAX_NEW_TOKENS,
                    temperature=0.0,
                )
                answer = (resp.choices[0].message.content or "").strip()
                rec = {
                    "row_id": rid,
                    "question": question,
                    "original_response": row.get("response", ""),
                    "shadow_response": answer,
                    "model": MODEL,
                }
                out_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_handle.flush()
                written += 1
                consecutive_errors = 0
                backoff = 1.0
                if written % 25 == 0 or i < 5:
                    print(f"  [{i+1}/{len(eval_rows)}] row_id={rid} ok (written this run: {written})", flush=True)
            except Exception as exc:
                consecutive_errors += 1
                msg = str(exc)[:200]
                print(f"  [{i+1}/{len(eval_rows)}] row_id={rid} ERROR: {msg}", flush=True)
                if "rate" in msg.lower() or "429" in msg:
                    sleep_s = min(backoff * 2, 60)
                    print(f"    rate-limit; sleeping {sleep_s:.1f}s", flush=True)
                    time.sleep(sleep_s)
                    backoff = sleep_s
                else:
                    time.sleep(min(backoff, 10))
                if consecutive_errors >= 5:
                    print("  too many consecutive errors; aborting; rerun to resume", flush=True)
                    sys.exit(2)

    print(f"\nDone. Wrote {written} new rows to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
