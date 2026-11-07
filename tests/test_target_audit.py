import math
import unittest

from viralscope.target_audit import summarize_labels


class TargetAuditTests(unittest.TestCase):
    def test_summarize_labels_separates_missing_nonfinite_and_sentinel(self) -> None:
        result = summarize_labels([1.0, 2.0, 2.0, None, -1.0, math.nan, "bad", -2.0])

        self.assertEqual(result["rows_scanned"], 8)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["non_numeric_count"], 1)
        self.assertEqual(result["non_finite_count"], 1)
        self.assertEqual(result["sentinel_minus_one_count"], 1)
        self.assertEqual(result["negative_non_sentinel_count"], 1)
        self.assertEqual(result["non_sentinel_summary"]["count"], 4)
        self.assertEqual(result["non_sentinel_summary"]["min"], -2.0)
        self.assertEqual(sum(result["non_sentinel_histogram"]["counts"]), 4)

    def test_summarize_labels_handles_empty_input(self) -> None:
        result = summarize_labels([])

        self.assertEqual(result["rows_scanned"], 0)
        self.assertIsNone(result["non_sentinel_summary"]["mean"])
        self.assertEqual(result["non_sentinel_histogram"]["counts"], [])


if __name__ == "__main__":
    unittest.main()