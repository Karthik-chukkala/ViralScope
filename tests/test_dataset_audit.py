import unittest
from types import SimpleNamespace

from viralscope.dataset_audit import (
    summarize_license,
    summarize_parquet_files,
    summarize_viewer_splits,
)


class DatasetAuditTests(unittest.TestCase):
    def test_summarize_license_keeps_only_public_license_fields(self) -> None:
        card = {
            "license": "other",
            "license_name": "smp-license",
            "license_link": "LICENSE.md",
            "extra_gated_prompt": "private or lengthy agreement text",
        }

        self.assertEqual(
            summarize_license(card),
            {
                "identifier": "other",
                "name": "smp-license",
                "link": "LICENSE.md",
            },
        )

    def test_summarize_parquet_files_groups_shards_and_ignores_other_files(self) -> None:
        files = [
            SimpleNamespace(rfilename="data/train-00000.parquet", size=100),
            SimpleNamespace(rfilename="data/train-00001.parquet", size=200),
            SimpleNamespace(rfilename="data/test-00000.parquet", size=50),
            SimpleNamespace(rfilename="README.md", size=10),
        ]

        summary = summarize_parquet_files(files)

        self.assertEqual(summary["shard_count"], 3)
        self.assertEqual(summary["shards_by_split"], {"test": 1, "train": 2})
        self.assertEqual(summary["bytes_by_split"], {"test": 50, "train": 300})
        self.assertEqual(summary["total_bytes"], 350)

    def test_summarize_viewer_splits_extracts_counts_and_storage(self) -> None:
        payload = {
            "size": {
                "splits": [
                    {
                        "split": "train",
                        "num_rows": 12,
                        "num_bytes_parquet_files": 100,
                        "num_bytes_memory": 120,
                    }
                ]
            }
        }

        self.assertEqual(
            summarize_viewer_splits(payload),
            {
                "train": {
                    "rows": 12,
                    "parquet_bytes": 100,
                    "memory_bytes": 120,
                }
            },
        )


if __name__ == "__main__":
    unittest.main()