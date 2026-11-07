import unittest

from viralscope.metadata_eda import (
    MetadataAccumulator,
    select_evenly_spaced_shards,
)
from viralscope.sample_cache import CACHE_COLUMNS


class MetadataEdaTests(unittest.TestCase):
    def test_training_cache_schema_excludes_ids_and_image_fields(self) -> None:
        self.assertIn("label", CACHE_COLUMNS)
        self.assertIn("Title", CACHE_COLUMNS)
        self.assertIn("Alltags", CACHE_COLUMNS)
        self.assertNotIn("Uid", CACHE_COLUMNS)
        self.assertNotIn("Pid", CACHE_COLUMNS)
        self.assertNotIn("img_filepath", CACHE_COLUMNS)
        self.assertNotIn("Longitude", CACHE_COLUMNS)
        self.assertNotIn("user_description", CACHE_COLUMNS)

    def test_select_evenly_spaced_shards_is_deterministic_and_covers_range(self) -> None:
        shards = list(range(49))

        selected = select_evenly_spaced_shards(shards, 7)

        self.assertEqual(selected, [0, 8, 16, 24, 32, 40, 48])
        self.assertEqual(select_evenly_spaced_shards(shards, 1), [24])

    def test_merge_counts_duplicate_ids_across_shard_accumulators(self) -> None:
        first = MetadataAccumulator()
        first.add_rows([{"Pid": "shared"}, {"Pid": "first"}])
        second = MetadataAccumulator()
        second.add_rows([{"Pid": "shared"}, {"Pid": "shared"}])

        first.merge(second)

        report = first.build_report()
        self.assertEqual(report["rows_scanned"], 4)
        self.assertEqual(report["unique_post_ids"], 2)
        self.assertEqual(report["duplicate_post_id_rows"], 2)

    def test_accumulator_outputs_aggregates_without_raw_content(self) -> None:
        accumulator = MetadataAccumulator()
        accumulator.add_rows([
            {
                "Uid": "user-1",
                "Pid": "post-1",
                "Postdate": "1451606400",
                "Category": "Nature",
                "Subcategory": "Flowers",
                "Concept": "rose",
                "Title": "A red rose",
                "Alltags": "red flower garden",
            },
            {
                "Uid": "user-1",
                "Pid": "post-1",
                "Postdate": "not-a-date",
                "Category": "Nature",
                "Subcategory": "Flowers",
                "Concept": "tulip",
                "Title": "",
                "Alltags": None,
            },
        ])

        report = accumulator.build_report()

        self.assertEqual(report["rows_scanned"], 2)
        self.assertEqual(report["unique_users"], 1)
        self.assertEqual(report["duplicate_post_id_rows"], 1)
        self.assertEqual(report["invalid_timestamp_count"], 1)
        self.assertEqual(report["missing_counts"]["Title"], 1)
        self.assertEqual(report["category_frequencies"][0], {"value": "Nature", "count": 2})
        self.assertEqual(report["title_word_counts"]["mean"], 3.0)
        self.assertFalse(report["privacy"]["raw_text_written"])
        self.assertNotIn("user_ids", report)
        self.assertNotIn("A red rose", str(report))


if __name__ == "__main__":
    unittest.main()