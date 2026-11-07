"""Collect dataset repository metadata without downloading dataset rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from datasets import load_dataset_builder
from huggingface_hub import HfApi, get_token

DEFAULT_REPOSITORY = "smpchallenge/SMP-Image"
DEFAULT_REVISION = "f131dba653a05c0e3c6294388f81ed3fbd9c71a0"
DEFAULT_OUTPUT = Path("artifacts/dataset-audit.json")


def _card_data_dict(card_data: Any) -> dict[str, Any]:
    if card_data is None:
        return {}
    if hasattr(card_data, "to_dict"):
        return card_data.to_dict()
    return dict(card_data)


def summarize_license(card_data: Any) -> dict[str, Any]:
    card = _card_data_dict(card_data)
    return {
        "identifier": card.get("license"),
        "name": card.get("license_name"),
        "link": card.get("license_link"),
    }


def summarize_parquet_files(files: Iterable[Any]) -> dict[str, Any]:
    parquet_files = [item for item in files if item.rfilename.endswith(".parquet")]
    bytes_by_split: dict[str, int] = defaultdict(int)
    shards_by_split: dict[str, int] = defaultdict(int)

    for item in parquet_files:
        filename = item.rfilename.rsplit("/", maxsplit=1)[-1]
        split = filename.split("-", maxsplit=1)[0]
        if split in {"train", "test"}:
            bytes_by_split[split] += item.size or 0
            shards_by_split[split] += 1

    total_bytes = sum(bytes_by_split.values())
    return {
        "shard_count": len(parquet_files),
        "shards_by_split": dict(sorted(shards_by_split.items())),
        "bytes_by_split": dict(sorted(bytes_by_split.items())),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 1024**3, 2),
    }


def summarize_viewer_splits(size_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    size = size_payload.get("size", {})
    return {
        split["split"]: {
            "rows": split.get("num_rows"),
            "parquet_bytes": split.get("num_bytes_parquet_files"),
            "memory_bytes": split.get("num_bytes_memory"),
        }
        for split in size.get("splits", [])
    }


def fetch_viewer_size(repository: str) -> dict[str, Any]:
    token = get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face authentication is required for the dataset size endpoint."
        )

    query = urlencode({"dataset": repository, "config": "default"})
    request = Request(
        f"https://datasets-server.huggingface.co/size?{query}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except HTTPError as error:
        if error.code in {401, 403}:
            raise RuntimeError(
                "Hugging Face denied access to split-size metadata; "
                "check the local login and dataset license acceptance."
            ) from None
        raise RuntimeError(
            f"Hugging Face size endpoint returned HTTP {error.code}."
        ) from None


def build_report(
    dataset_info: Any,
    builder_info: Any,
    viewer_size: dict[str, Any],
) -> dict[str, Any]:
    features = builder_info.features or {}
    size = viewer_size.get("size", {})
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repository": dataset_info.id,
            "revision": dataset_info.sha,
            "repository_file_count": len(dataset_info.siblings),
            "license": summarize_license(dataset_info.cardData),
        },
        "splits": summarize_viewer_splits(viewer_size),
        "viewer_size_metadata": {
            "partial": viewer_size.get("partial"),
            "total_rows": size.get("config", {}).get("num_rows"),
            "revision_note": (
                "The viewer endpoint does not expose a revision; the repository "
                "main SHA was checked against the requested pin immediately before query."
            ),
        },
        "features": [
            {"name": name, "type": str(feature)}
            for name, feature in features.items()
        ],
        "parquet_storage": summarize_parquet_files(dataset_info.siblings),
        "scope": {
            "dataset_rows_downloaded": False,
            "image_bytes_downloaded": False,
            "test_labels_verified": False,
        },
    }


def collect_metadata(repository: str, revision: str) -> dict[str, Any]:
    api = HfApi()
    dataset_info = api.dataset_info(
        repository,
        revision=revision,
        files_metadata=True,
    )
    main_info = api.dataset_info(repository)
    if main_info.sha != dataset_info.sha:
        raise RuntimeError(
            "The repository main revision differs from the requested pin. "
            "Choose an explicit current revision before collecting size metadata."
        )

    builder_info = load_dataset_builder(
        repository,
        revision=dataset_info.sha,
    ).info
    viewer_size = fetch_viewer_size(repository)
    report = build_report(dataset_info, builder_info, viewer_size)

    parquet_bytes = report["parquet_storage"]["bytes_by_split"]
    for split, metadata in report["splits"].items():
        if parquet_bytes.get(split) != metadata["parquet_bytes"]:
            raise RuntimeError(
                f"Parquet size mismatch for {split}; refusing to write an inconsistent report."
            )

    if sum(item["rows"] or 0 for item in report["splits"].values()) != report[
        "viewer_size_metadata"
    ]["total_rows"]:
        raise RuntimeError("Split row counts do not match the dataset total.")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a metadata-only audit report for an HF dataset."
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = collect_metadata(args.repository, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset: {report['dataset']['repository']}@{report['dataset']['revision']}")
    print(f"Splits: {report['splits']}")
    print(f"License: {report['dataset']['license']['name'] or 'not declared'}")
    print(f"Parquet size: {report['parquet_storage']['total_gib']} GiB")
    print(f"Report: {args.output}")
    print("No dataset rows or image bytes were downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())