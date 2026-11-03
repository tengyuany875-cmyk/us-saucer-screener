import unittest
from unittest.mock import patch

import pandas as pd

from market_caps import add_market_caps, build_snapshot, format_market_cap, format_market_rank


def payload(rows):
    return {"status": {"rCode": 200}, "data": {"asOf": None, "rows": rows}}


def stock(ticker, cap, name="Example Common Stock"):
    return {"symbol": ticker, "marketCap": cap, "name": name}


class MarketCapTests(unittest.TestCase):
    def snapshot(self, rows):
        with patch("market_caps.MINIMUM_REFERENCE_SIZE", 1):
            return build_snapshot(payload(rows), "2026-11-04T00:00:00+00:00")

    def test_reference_rank_is_not_candidate_rank_and_preserves_results(self):
        snapshot = self.snapshot([stock("AAA", "900"), stock("BBB", "500"), stock("CCC", "100")])
        results = pd.DataFrame({"Ticker": ["CCC", "AAA", "UNKNOWN"], "Score": [90, 70, 60], "Price": [3, 5, 9]})
        enriched = add_market_caps(results, snapshot)
        pd.testing.assert_frame_equal(enriched[results.columns], results)
        self.assertEqual(enriched["USMarketCapRank"].iloc[:2].tolist(), [3, 1])
        self.assertTrue(pd.isna(enriched["USMarketCapRank"].iloc[2]))
        self.assertNotIn("MarketCapUSD", results)

    def test_ties_share_competition_rank(self):
        entries = self.snapshot([stock("CCC", "100"), stock("BBB", "500"), stock("AAA", "500.00")])["entries"]
        self.assertEqual({ticker: row["rank"] for ticker, row in entries.items()}, {"AAA": 1, "BBB": 1, "CCC": 3})

    def test_non_ordinary_instruments_are_excluded(self):
        excluded = ["Index ETF", "Closed-End Fund", "Series A Preferred Stock", "Example Warrants", "Example Units", "Example Rights", "Senior Notes"]
        rows = [stock("KEEP", "100", "Netflix Inc. Common Stock")]
        rows += [stock(f"SKIP{i}", "1000", name) for i, name in enumerate(excluded)]
        entries = self.snapshot(rows)["entries"]
        self.assertEqual(list(entries), ["KEEP"])
        self.assertEqual(entries["KEEP"]["rank"], 1)

    def test_foreign_us_listings_and_share_classes_are_included(self):
        entries = self.snapshot([stock("ADR", "300", "Example American Depositary Shares"), stock("BRK.B", "200", "Example Class B Common Stock")])["entries"]
        self.assertEqual(entries["ADR"]["rank"], 1)
        self.assertIn("BRK-B", entries)

    def test_invalid_caps_are_missing_not_zero_rank(self):
        invalid = [None, "", "0.00", "-1", "NaN", "Infinity"]
        rows = [stock("KEEP", "100")]+[stock(f"MISS{i}", cap) for i, cap in enumerate(invalid)]
        snapshot = self.snapshot(rows)
        self.assertEqual(snapshot["missing_count"], len(invalid))
        self.assertEqual(list(snapshot["entries"]), ["KEEP"])
        for value in invalid:
            self.assertEqual(format_market_cap(value), "不明")
            self.assertEqual(format_market_rank(value), "不明")

    def test_bad_or_partial_responses_do_not_produce_ranks(self):
        with self.assertRaises(ValueError):
            build_snapshot(None, "date")
        with self.assertRaises(ValueError):
            build_snapshot({"status": {"rCode": 500}}, "date")
        with self.assertRaises(ValueError):
            build_snapshot(payload([stock("AAA", "100")]), "date")
        with self.assertRaises(ValueError):
            self.snapshot([stock("AAA", "100"), stock("AAA", "200")])

    def test_dollar_units_and_rank_format(self):
        self.assertEqual(format_market_cap("12500000000"), "125億ドル")
        self.assertEqual(format_market_cap("50000000"), "0.50億ドル")
        self.assertEqual(format_market_cap("500000"), "0.01億ドル未満")
        self.assertEqual(format_market_rank(850), "参考850位")
        self.assertEqual(format_market_rank(float("nan")), "不明")
        self.assertEqual(format_market_rank(1.5), "不明")


if __name__ == "__main__":
    unittest.main()
