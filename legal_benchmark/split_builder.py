from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from tqdm import tqdm

from legal_benchmark.data import load_rows


def normalize_for_fingerprint(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u0900-\u097f\u0980-\u09ff\u0b00-\u0b7f\u0b80-\u0bff\u0c00-\u0c7f\u0c80-\u0cff ]", "", text)
    tokens = text.split()
    return " ".join(tokens[:180])


def context_fingerprint(text: str) -> str:
    normalized = normalize_for_fingerprint(text)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=12).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a derived held-out benchmark split.")
    parser.add_argument("--dataset", default="Prarabdha/indian-legal-supervised-fine-tuning-data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--corpus-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="paper_artifacts/heldout_benchmark")
    args = parser.parse_args()

    rows = load_rows(
        dataset=args.dataset,
        split=args.split,
        needed_rows=args.sample_size,
        seed=args.seed,
        streaming=True,
    )

    seen_fingerprints: set[str] = set()
    unique_rows: list[dict] = []
    duplicate_count = 0
    for row in tqdm(rows, desc="Fingerprinting"):
        fingerprint = context_fingerprint(row["context"])
        if fingerprint in seen_fingerprints:
            duplicate_count += 1
            continue
        seen_fingerprints.add(fingerprint)
        row["context_fingerprint"] = fingerprint
        unique_rows.append(row)

    if len(unique_rows) < args.eval_size + args.corpus_size:
        raise RuntimeError(
            f"Only {len(unique_rows)} unique rows available after fingerprinting; "
            f"need {args.eval_size + args.corpus_size}."
        )

    eval_rows = unique_rows[: args.eval_size]
    corpus_rows = unique_rows[args.eval_size : args.eval_size + args.corpus_size]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    write_jsonl(output_dir / "corpus.jsonl", corpus_rows)

    manifest = {
        "dataset": args.dataset,
        "source_split": args.split,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "unique_rows": len(unique_rows),
        "duplicate_fingerprint_rows_removed": duplicate_count,
        "fingerprint_method": "blake2b over normalized first 180 context tokens",
        "eval_size": len(eval_rows),
        "corpus_size": len(corpus_rows),
        "eval_file": str(output_dir / "eval.jsonl"),
        "corpus_file": str(output_dir / "corpus.jsonl"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
