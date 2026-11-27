"""Behavioral regressions for the unchanged weekly detector and renderer."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from app import render_html
from config import CONFIG
from screener import analyse_ticker, sort_results


def weekly_fixture():
    rng = np.random.default_rng(42)
    base = np.linspace(30, 70, 240)
    x = np.linspace(-1, 1, 30)
    prices = np.concatenate([base, 58 + 13 * x**2 + np.linspace(0, 4, 30)]) * (1 + rng.normal(0, 0.006, 270))
    volume = np.full(270, 1_200_000.0)
    volume[-4:] *= 1.45
    return pd.DataFrame({"Close": prices, "Volume": volume}, index=pd.date_range("2021-01-01", periods=270, freq="W-FRI"))


class WeeklyRegressionTests(unittest.TestCase):
    def test_known_weekly_pattern_and_score_are_unchanged(self):
        row = analyse_ticker("TEST", weekly_fixture())
        self.assertIsNotNone(row)
        self.assertEqual(row["Score"], 88.3)
        self.assertEqual(row["Stage"], "ブレイク接近")
        self.assertEqual(row["BottomWeeks"], 17)
        self.assertEqual(row["Price"], 75.0)
        self.assertEqual(CONFIG.interval, "1wk")
        self.assertEqual(CONFIG.history_period, "10y")

    def test_weekly_invalid_or_short_inputs_remain_rejected(self):
        self.assertIsNone(analyse_ticker("EMPTY", pd.DataFrame()))
        self.assertIsNone(analyse_ticker("SHORT", weekly_fixture().tail(20)))

    def test_weekly_sorting_is_score_then_volume(self):
        rows = [{"Ticker":"A","Score":80,"VolumeRatio4wVs12w":1.0},
                {"Ticker":"B","Score":90,"VolumeRatio4wVs12w":0.5},
                {"Ticker":"C","Score":80,"VolumeRatio4wVs12w":2.0}]
        self.assertEqual(sort_results(rows)["Ticker"].tolist(), ["B","C","A"])

    def test_weekly_html_keeps_all_candidates_and_original_links(self):
        row = analyse_ticker("TEST", weekly_fixture())
        rows = [{**row,"Ticker":f"TEST{n}","Name":"Example","Exchange":"NASDAQ"} for n in range(21)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"weekly.html"
            render_html(pd.DataFrame(rows),path,21)
            content = path.read_text(encoding="utf-8")
        self.assertEqual(content.count('<article class="candidate"'),21)
        self.assertEqual(content.count('data-bottom-weeks="17" hidden>'),1)
        self.assertIn("週足ソーサーボトム候補",content)
        self.assertIn("www.tradingview.com/chart/",content)
        self.assertIn("finance.yahoo.com/quote/",content)
        self.assertIn("40週線乖離%",content)
        self.assertNotIn('class="market-rank"',content)


if __name__ == "__main__":
    unittest.main()
