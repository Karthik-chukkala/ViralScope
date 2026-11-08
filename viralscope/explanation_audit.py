"""Check explanation repeatability and formatting invariance on held-out examples."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .demo_app import load_models
from .explanations import decompose_prediction
from .train_baselines import (
    MODEL_DISPLAY_NAMES,
    _predict_model,
    build_features,
    chronological_split,
)

DEFAULT_DATA = Path("data/train-all-safe.parquet")
DEFAULT_MANIFEST = Path("data/train-all-safe-manifest.json")
DEFAULT_REPORT = Path("artifacts/explanation-stability.json")
DEFAULT_MODEL_DIR = Path("artifacts/models/final")
DEFAULT_ROWS = 128
DEFAULT_TOLERANCE = 1e-8


def _whitespace_equivalent(features: pd.DataFrame) -> pd.DataFrame:
    variant = features.copy()
    text = str(variant.iloc[0]["text"]).strip()
    variant.iloc[0, variant.columns.get_loc("text")] = (
        re.sub(r"\s+", " \t  ", text) if text else ""
    )
    return variant


def _top_feature_map(names: np.ndarray, values: np.ndarray) -> dict[str, float]:
    indexes = np.flatnonzero(np.abs(values) > 1e-10)
    indexes = indexes[np.argsort(np.abs(values[indexes]))[-6:][::-1]]
    return {str(names[index]): float(values[index]) for index in indexes}


def _compare_top_features(
    first: dict[str, float],
    second: dict[str, float],
) -> tuple[float, float]:
    first_names = set(first)
    second_names = set(second)
    union = first_names | second_names
    jaccard = len(first_names & second_names) / len(union) if union else 1.0
    shared = first_names & second_names
    sign_agreement = (
        sum(np.sign(first[name]) == np.sign(second[name]) for name in shared) / len(shared)
        if shared
        else 1.0
    )
    return float(jaccard), float(sign_agreement)


def audit_explanation_stability(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    model_dir: Path = DEFAULT_MODEL_DIR,
    max_rows: int = DEFAULT_ROWS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sample_method") != "all_train_shards":
        raise ValueError("Explanation audit requires all pinned train shards")

    frame = pd.read_parquet(data_path)
    features, labels, timestamps = build_features(frame)
    if len(labels) != manifest.get("rows"):
        raise RuntimeError("Validated row count does not match the all-shard manifest")
    _, chronological_tail, _ = chronological_split(
        features, labels, timestamps, holdout_fraction=0.2
    )
    ordered_tail = chronological_tail[
        np.argsort(timestamps[chronological_tail], kind="stable")
    ]
    sample_indexes = ordered_tail[
        np.linspace(
            0, len(ordered_tail) - 1, min(max_rows, len(ordered_tail)), dtype=np.int64
        )
    ]
    models = load_models(model_dir)
    results: dict[str, Any] = {}

    for model_name, model in models.items():
        reconstruction_errors: list[float] = []
        prediction_deltas: list[float] = []
        top_jaccard: list[float] = []
        top_sign_agreement: list[float] = []
        deterministic_matches = 0
        for index in sample_indexes:
            original = features.iloc[[int(index)]].copy()
            variant = _whitespace_equivalent(original)
            prediction = float(_predict_model(model_name, model, original)[0])
            repeated_prediction = float(_predict_model(model_name, model, original)[0])
            variant_prediction = float(_predict_model(model_name, model, variant)[0])
            names, contributions, baseline = decompose_prediction(
                model_name, model, original
            )
            repeated_names, repeated_contributions, repeated_baseline = (
                decompose_prediction(model_name, model, original)
            )
            reconstruction_errors.append(
                abs(prediction - baseline - float(contributions.sum()))
            )
            prediction_deltas.append(abs(prediction - variant_prediction))
            original_top = _top_feature_map(names, contributions)
            variant_names, variant_contributions, _ = decompose_prediction(
                model_name, model, variant
            )
            variant_top = _top_feature_map(variant_names, variant_contributions)
            jaccard, sign_agreement = _compare_top_features(
                original_top, variant_top
            )
            top_jaccard.append(jaccard)
            top_sign_agreement.append(sign_agreement)
            same_explanation = (
                np.array_equal(names, repeated_names)
                and np.allclose(
                    contributions, repeated_contributions, rtol=0, atol=tolerance
                )
                and abs(baseline - repeated_baseline) <= tolerance
                and abs(prediction - repeated_prediction) <= tolerance
            )
            deterministic_matches += int(same_explanation)

        results[model_name] = {
            "display_name": MODEL_DISPLAY_NAMES[model_name],
            "rows_checked": len(sample_indexes),
            "deterministic_explanation_rows": deterministic_matches,
            "max_additive_reconstruction_error": max(reconstruction_errors),
            "formatting_only": {
                "max_abs_prediction_delta": max(prediction_deltas),
                "mean_top_6_feature_jaccard": float(np.mean(top_jaccard)),
                "mean_shared_feature_sign_agreement": float(
                    np.mean(top_sign_agreement)
                ),
                "minimum_top_6_feature_jaccard": min(top_jaccard),
                "minimum_shared_feature_sign_agreement": min(top_sign_agreement),
            },
            "checks_passed": (
                deterministic_matches == len(sample_indexes)
                and max(reconstruction_errors) <= tolerance
                and max(prediction_deltas) <= tolerance
                and min(top_jaccard) == 1.0
                and min(top_sign_agreement) == 1.0
            ),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repository": manifest["dataset"],
            "revision": manifest["revision"],
            "rows": len(frame),
            "sample_method": manifest["sample_method"],
        },
        "sample": {
            "rows_checked_per_model": len(sample_indexes),
            "selection": "evenly spaced rows from the chronological 20% tail",
            "raw_inputs_written": False,
        },
        "method": {
            "additivity": "base value plus all feature contributions must equal prediction",
            "repeatability": "same final model and input evaluated twice",
            "perturbation": "whitespace-only formatting change preserving text tokens",
            "top_features": 6,
            "absolute_tolerance": tolerance,
            "does_not_test": [
                "semantic paraphrase robustness",
                "causal validity of feature contributions",
                "generalization performance",
            ],
        },
        "models": results,
        "all_checks_passed": all(
            result["checks_passed"] for result in results.values()
        ),
        "official_test_scored": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit feature-contribution fidelity and repeatability."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)
    report = audit_explanation_stability(
        data_path=args.data,
        manifest_path=args.manifest,
        model_dir=args.model_dir,
        max_rows=args.rows,
        tolerance=args.tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, result in report["models"].items():
        print(
            f"{result['display_name']}: {result['rows_checked']} rows; "
            f"max reconstruction error "
            f"{result['max_additive_reconstruction_error']:.3g}; "
            f"formatting invariant {result['checks_passed']}"
        )
    print(f"All explanation checks passed: {report['all_checks_passed']}")
    print(f"Aggregate report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
