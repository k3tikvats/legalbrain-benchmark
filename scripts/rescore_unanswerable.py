"""Re-score saved unanswerable-task predictions with the corrected scoring logic.

This script applies the fixed `score_unanswerable_predictions` to any
`task_unanswerable_predictions.jsonl` file under benchmark_outputs/ and writes
the corrected per-run summary back to `task_unanswerable.json` (overwriting
buggy historical numbers) and a combined report to
`paper_artifacts/unanswerable_rescored.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

from legal_benchmark.tasks import score_unanswerable_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "benchmark_outputs"
ARTIFACT = REPO_ROOT / "paper_artifacts" / "unanswerable_rescored.json"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    combined: dict[str, dict] = {}
    for run_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        pred_path = run_dir / "task_unanswerable_predictions.jsonl"
        if not pred_path.exists() or pred_path.stat().st_size == 0:
            continue
        preds = load_jsonl(pred_path)
        if not preds:
            continue
        scored = score_unanswerable_predictions(preds)
        combined[run_dir.name] = scored
        # Overwrite the per-run summary so downstream readers see correct numbers.
        (run_dir / "task_unanswerable.json").write_text(
            json.dumps(scored, indent=2), encoding="utf-8"
        )
        print(f"{run_dir.name}: {scored}")

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nWrote {ARTIFACT}")


if __name__ == "__main__":
    main()
