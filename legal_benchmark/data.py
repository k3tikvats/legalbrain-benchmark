from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = ("context", "question", "response")


def _clean_row(row: dict, row_id: int) -> dict | None:
    cleaned = {}
    for key in REQUIRED_COLUMNS:
        value = row.get(key)
        if value is None:
            return None
        value = str(value).strip()
        if not value or value.lower() in {"none", "nan", "null"}:
            return None
        cleaned[key] = value

    if not all(cleaned[key] for key in REQUIRED_COLUMNS):
        return None
    cleaned["row_id"] = row.get("row_id", row_id)
    return cleaned


def _load_local_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_local_json(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("data", data.get("rows", []))
    yield from data


def _load_local_csv(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def load_rows(
    dataset: str,
    split: str,
    needed_rows: int,
    seed: int,
    streaming: bool = True,
) -> list[dict]:
    path = Path(dataset)
    rows: Iterable[dict]

    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows = _load_local_jsonl(path)
        elif suffix == ".json":
            rows = _load_local_json(path)
        elif suffix == ".csv":
            rows = _load_local_csv(path)
        elif suffix == ".parquet":
            import pandas as pd

            rows = pd.read_parquet(path).to_dict("records")
        else:
            raise ValueError(f"Unsupported local dataset format: {path.suffix}")
    else:
        from datasets import load_dataset

        if streaming:
            ds = load_dataset(dataset, split=split, streaming=True)
            rows = ds.shuffle(seed=seed, buffer_size=max(needed_rows * 10, 1000))
        else:
            ds = load_dataset(dataset, split=split)
            ds = ds.shuffle(seed=seed)
            rows = ds

    selected: list[dict] = []
    for index, row in enumerate(rows):
        cleaned = _clean_row(dict(row), index)
        if cleaned is not None:
            selected.append(cleaned)
        if len(selected) >= needed_rows:
            break

    if len(selected) < needed_rows:
        raise RuntimeError(f"Only found {len(selected)} valid rows; needed {needed_rows}.")

    random.Random(seed).shuffle(selected)
    return selected


def split_eval_and_corpus(rows: list[dict], eval_size: int) -> tuple[list[dict], list[dict]]:
    eval_rows = rows[:eval_size]
    corpus_rows = rows[eval_size:]
    for idx, row in enumerate(corpus_rows):
        row["corpus_id"] = idx
    return eval_rows, corpus_rows
