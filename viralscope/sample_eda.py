"""Generate text and temporal EDA from the existing local sample cache."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DATA = Path("data/train-sample.parquet")
DEFAULT_MANIFEST = Path("data/train-sample-manifest.json")
DEFAULT_OUTPUT = Path("artifacts/sample-eda.json")
DEFAULT_PLOT = Path("artifacts/diagnostics/sample-temporal-text.png")
EDA_COLUMNS = ("Postdate", "Category", "Title", "Alltags", "label")


def _length_summary(values: np.ndarray) -> dict[str, int | float | None]:
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.9)),
        "max": int(values.max()),
    }


def summarize_sample(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    missing_columns = set(EDA_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required sample columns: {sorted(missing_columns)}")

    postdates = pd.to_numeric(frame["Postdate"], errors="coerce")
    dates = pd.to_datetime(postdates, unit="s", utc=True, errors="coerce")
    titles = frame["Title"].fillna("").astype(str).str.strip()
    tags = frame["Alltags"].fillna("").astype(str).str.strip()
    title_present = titles.ne("")
    tags_present = tags.ne("")
    title_lengths = titles[title_present].str.len().to_numpy(dtype=np.int64)
    title_word_counts = titles[title_present].map(
        lambda value: len(re.findall(r"\S+", value))
    ).to_numpy(dtype=np.int64)
    tag_token_counts = tags[tags_present].map(
        lambda value: len(re.findall(r"\S+", value))
    ).to_numpy(dtype=np.int64)
    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    valid_labels = numeric_labels[np.isfinite(numeric_labels) & (numeric_labels >= 0)].to_numpy(
        dtype=np.float64
    )
    monthly = dates.dropna().dt.strftime("%Y-%m").value_counts().sort_index()
    category_counts = Counter(
        value for value in frame["Category"].dropna().astype(str).str.strip() if value
    )

    report = {
        "rows": int(len(frame)),
        "missing_or_blank": {
            "Postdate": int(dates.isna().sum()),
            "Category": int(frame["Category"].isna().sum()),
            "Title": int((~title_present).sum()),
            "Alltags": int((~tags_present).sum()),
            "label": int((~np.isfinite(numeric_labels)).sum()),
        },
        "invalid_timestamp_count": int(dates.isna().sum()),
        "postdate_range_utc": {
            "min": dates.min().isoformat() if dates.notna().any() else None,
            "max": dates.max().isoformat() if dates.notna().any() else None,
        },
        "monthly_posts_utc": [
            {"month": month, "count": int(count)}
            for month, count in monthly.items()
        ],
        "category_count": len(category_counts),
        "top_5_categories": [
            {"category": category, "count": int(count)}
            for category, count in category_counts.most_common(5)
        ],
        "title_character_lengths": _length_summary(title_lengths),
        "title_word_counts": _length_summary(title_word_counts),
        "tag_token_counts": _length_summary(tag_token_counts),
        "label_summary": {
            "count": int(valid_labels.size),
            "min": float(valid_labels.min()) if valid_labels.size else None,
            "median": float(np.median(valid_labels)) if valid_labels.size else None,
            "mean": float(valid_labels.mean()) if valid_labels.size else None,
            "max": float(valid_labels.max()) if valid_labels.size else None,
        },
        "privacy": {
            "raw_text_written": False,
            "user_ids_read": False,
            "post_ids_read": False,
            "image_bytes_read": False,
        },
    }
    plot_data = {
        "dates": dates,
        "title_lengths": title_lengths,
        "tag_token_counts": tag_token_counts,
    }
    return report, plot_data


def _save_plot(plot_data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    monthly = plot_data["dates"].dropna().dt.strftime("%Y-%m").value_counts().sort_index()
    axes[0].bar(monthly.index, monthly.values, color="#17685c")
    axes[0].set_title("Sample posts by month (UTC)")
    axes[0].set_ylabel("Posts")
    axes[0].tick_params(axis="x", labelrotation=55)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].hist(plot_data["title_lengths"], bins=30, color="#d7a338", edgecolor="white")
    axes[1].set_title("Title length")
    axes[1].set_xlabel("Characters")
    axes[1].set_ylabel("Posts")
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].hist(plot_data["tag_token_counts"], bins=30, color="#c34f3c", edgecolor="white")
    axes[2].set_title("Tag field length")
    axes[2].set_xlabel("Whitespace-delimited tokens")
    axes[2].set_ylabel("Posts")
    axes[2].grid(axis="y", alpha=0.2)

    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_sample_eda(
    data_path: Path = DEFAULT_DATA,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
    plot_path: Path = DEFAULT_PLOT,
) -> dict[str, Any]:
    frame = pd.read_parquet(data_path, columns=list(EDA_COLUMNS))
    summary, plot_data = summarize_sample(frame)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _save_plot(plot_data, plot_path)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repository": manifest["dataset"],
            "revision": manifest["revision"],
            "split": manifest["split"],
            "sample_method": manifest["sample_method"],
            "shards_selected": manifest["shards_selected"],
        },
        "sample_eda": summary,
        "plot": str(plot_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build aggregate text and temporal EDA from the local train sample."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args(argv)
    report = build_sample_eda(
        data_path=args.data,
        manifest_path=args.manifest,
        output_path=args.output,
        plot_path=args.plot,
    )
    summary = report["sample_eda"]
    print(f"Rows analyzed: {summary['rows']}")
    print(f"Missing/blank titles: {summary['missing_or_blank']['Title']}")
    print(f"Categories represented: {summary['category_count']}")
    print(f"Aggregate report: {args.output}")
    print(f"Plot: {args.plot}")
    print("No raw text, IDs, or images were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())