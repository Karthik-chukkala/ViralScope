"""Compare baseline ranking stability across chronological holdout sizes."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .train_baselines import (
    DEFAULT_DATA,
    DEFAULT_MANIFEST,
    train_baselines,
)

DEFAULT_OUTPUT = Path("artifacts/temporal-sensitivity.json")
DEFAULT_FRACTIONS = (0.15, 0.20, 0.25)


def _best_models(runs: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for metric, maximize in (("src", True), ("mae", False)):
        winners = []
        for run in runs:
            scores = {
                name: values[metric]
                for name, values in run["holdout_metrics"].items()
                if values[metric] is not None
            }
            if not scores:
                continue
            winner = (max if maximize else min)(scores, key=scores.get)
            winners.append(winner)
        best[f"{metric}_winner_counts"] = {
            model: winners.count(model)
            for model in sorted(set(winners))
        }
    return best


def run_temporal_sensitivity(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
    workers: int = 4,
    holdout_fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> dict[str, Any]:
    fractions = tuple(float(value) for value in holdout_fractions)
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("holdout fractions must be non-empty and unique")
    if any(not 0 < value < 1 for value in fractions):
        raise ValueError("each holdout fraction must be between zero and one")
    if workers < 1:
        raise ValueError("workers must be positive")

    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="viralscope-temporal-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for fraction in fractions:
            label = f"{fraction:.3f}".rstrip("0").rstrip(".").replace(".", "-")
            run_root = temporary_root / label
            report = train_baselines(
                data_path=data_path,
                manifest_path=manifest_path,
                report_path=run_root / "metrics.json",
                model_dir=run_root / "models",
                diagnostic_dir=run_root / "diagnostics",
                workers=workers,
                holdout_fraction=fraction,
            )
            runs.append(
                {
                    "holdout_fraction": fraction,
                    "cutoff_utc": report["evaluation"]["cutoff_utc"],
                    "train_rows": report["evaluation"]["train_rows"],
                    "holdout_rows": report["evaluation"]["holdout_rows"],
                    "holdout_metrics": {
                        name: values["holdout"]
                        for name, values in report["metrics"].items()
                    },
                }
            )
            print(
                f"Holdout {fraction:.0%}: {report['evaluation']['holdout_rows']} rows, "
                f"cutoff {report['evaluation']['cutoff_utc']}",
                flush=True,
            )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest,
        "protocol": (
            "chronological tail holdouts from the same sampled training shards; "
            "the holdouts are nested and are not independent test sets"
        ),
        "runs": runs,
        "best_model_winner_counts": _best_models(runs),
        "official_test_scored": False,
        "extra_models_persisted": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check model ranking stability across chronological holdout fractions."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=DEFAULT_FRACTIONS,
    )
    args = parser.parse_args(argv)
    report = run_temporal_sensitivity(
        data_path=args.data,
        manifest_path=args.manifest,
        output_path=args.output,
        workers=args.workers,
        holdout_fractions=tuple(args.fractions),
    )
    print(json.dumps(report["best_model_winner_counts"], indent=2))
    print(f"Report: {args.output}")
    print("Extra model artifacts persisted: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())