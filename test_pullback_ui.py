from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from pullback_app import assign_bucket
from pullback_data import Sessions, prepare, write_csv, write_json
from pullback_history import identify
from pullback_recent import trial
from pullback_screener import analyse_daily
from pullback_settings import config_hash, load_config
from pullback_ui import build, render_daily, render_recent, weekly_preview
from test_pullback import market_frames


class LocalUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.sessions = load_config(), Sessions()
        cls.forming, cls.confirmed, cls.spy = market_frames()

    def record(self, stage="confirmed", earnings="outside_holding"):
        day = "2026-11-23" if stage == "forming" else "2026-11-24"
        raw = self.forming if stage == "forming" else self.confirmed
        frame = prepare("TEST", raw, day, self.sessions, self.config).frame
        spy = prepare("SPY", self.spy, day, self.sessions, self.config).frame
        result = analyse_daily("TEST", frame, spy, self.config, day)
        row = assign_bucket(result["record"], {"status": earnings, "next_date": "2027-02-01"})
        row.update(Name='Example <script>alert("x")</script>', Exchange="NYSE", Selected=earnings == "outside_holding")
        return identify(row, config_hash(self.config))

    def summary(self):
        return {"as_of": "2026-11-24", "finished_at_utc": "2026-11-25T10:00:00+00:00", "market_regime": "SPYは50日線以下",
                "price_success_rate": .98, "price_success_count": 98, "target_count": 100, "analysable_count": 95,
                "elapsed_seconds": 20, "scope": "full_universe", "quality_passed": True}

    def test_states_denominators_metadata_and_safe_chart_link(self):
        formed, confirmed = self.record("forming"), self.record()
        confirmed["Ticker"] = "SECOND"
        review = self.record(earnings="unknown")
        records = [formed, confirmed, review]
        original = deepcopy(records)
        caps = {"entries": {"TEST": {"market_cap_usd": "12500000000"}}, "fetched_at_utc": "2026-11-25T11:00:00+00:00"}
        page = render_daily(self.summary(), records, self.config, "", caps)
        self.assertIn('/ 75</span>', page)
        self.assertIn('/ 100</span>', page)
        self.assertIn('id="section-main"', page)
        self.assertIn('id="section-watch"', page)
        self.assertIn('id="section-review"', page)
        self.assertIn('data-stage="confirmed" data-bucket="review"', page)
        self.assertIn('98.00%', page)
        self.assertIn('SPYは50日線以下', page)
        self.assertIn('2026-11-25 19:00 JST', page)
        self.assertIn('時価総額：125億ドル', page)
        self.assertIn('https://www.tradingview.com/chart/?symbol=NYSE%3ATEST&amp;interval=D', page)
        self.assertIn('想定リスクリワード', page)
        self.assertIn('スコア内訳', page)
        self.assertNotIn('<script>alert', page)
        self.assertEqual(records, original)

    def test_empty_main_and_review_are_honest_and_not_filled(self):
        page = render_daily(self.summary(), [self.record("forming")], self.config, "", None)
        self.assertEqual(page.count('<article class="candidate"'), 1)
        self.assertNotIn('今回は候補がありません。', page)
        self.assertNotIn('id="section-review"', page)
        self.assertIn('時価総額：不明', page)
        self.assertNotIn('recent.html', page)

    def test_compact_cards_keep_reasons_metrics_and_scores_inside_details(self):
        row = self.record()
        page = render_daily(self.summary(), [row], self.config, "", None)
        before_details = page.split('<article class="candidate"', 1)[1].split('<details class="candidate-details">', 1)[0]
        self.assertIn('時価総額：', before_details)
        self.assertIn('チャートを見る', before_details)
        self.assertNotIn('高値からの調整幅', before_details)
        self.assertNotIn('想定リスクリワード', before_details)
        self.assertNotIn('検出理由', before_details)
        details = page.split('<details class="candidate-details">', 1)[1]
        for text in ['高値からの調整幅', '想定リスクリワード', '検出理由', 'スコア内訳', 'Yahoo Finance']:
            self.assertIn(text, details)
        self.assertNotIn('<details class="candidate-details" open', page)

    def test_one_empty_state_and_unknown_earnings_stay_separate_and_collapsed(self):
        page = render_daily(self.summary(), [self.record(earnings='unknown')], self.config, "", None)
        self.assertEqual(page.count('今回は候補がありません。'), 1)
        self.assertIn('主候補 <strong>0</strong>', page)
        self.assertIn('決算不明・要確認 1件', page)
        self.assertIn('<details class="review-section">', page)
        self.assertNotIn('<details class="review-section" open', page)
        self.assertNotIn('id="section-main"', page)

    def test_historical_examples_never_present_as_current_main(self):
        data = {"limitations": "過去の決算予定は再現していません。", "dates": ["2026-11-23", "2026-11-24"]}
        page = render_recent(data, [self.record(earnings="unknown")], [], self.config, "", None)
        self.assertIn('過去日の価格条件のみ・決算未再現', page)
        self.assertIn('過去の主候補としての選出ではありません', page)
        self.assertIn('data-stage="confirmed"', page)
        self.assertNotIn('id="section-main"', page)

    def test_weekly_copy_only_adds_navigation_and_local_label(self):
        original = '<html><style>/* css */</style><main><p>weekly preserved</p></main></html>'
        rendered = weekly_preview(original)
        self.assertIn('aria-current="page">週足ソーサーボトム', rendered)
        self.assertIn('<p>weekly preserved</p></main></html>', rendered)
        self.assertEqual(original, '<html><style>/* css */</style><main><p>weekly preserved</p></main></html>')
        with self.assertRaises(ValueError):
            weekly_preview('<html>unrecognized</html>')

    def test_recent_offline_trial_links_real_detector_stages_and_preserves_live(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "output/pullback/runs/source"
            cache = root / "cache/daily"
            fingerprint = config_hash(self.config)
            write_json(run_dir / "config.json", self.config)
            write_csv(run_dir / "universe.csv", pd.DataFrame([{"Ticker": "TEST", "Name": "Example", "Exchange": "NYSE"}]))
            write_json(root / "output/pullback/latest_success.json", {"run": str(run_dir), "as_of": "2026-11-24", "config_hash": fingerprint})
            live = root / "history/pullback/signals.csv"
            live.parent.mkdir(parents=True)
            live.write_text("unchanged live history", encoding="utf-8")
            for ticker, raw in [("TEST", self.confirmed), ("SPY", self.spy)]:
                csv = cache / f"{ticker}.csv"
                write_csv(csv, raw, index=True)
                write_json(cache / f"{ticker}.json", {"schema": 1, "as_of": "2026-11-24", "history_calendar_days": self.config["data"]["history_calendar_days"],
                           "fetched_at_utc": "2026-11-25T10:00:00+00:00", "sha256": hashlib.sha256(csv.read_bytes()).hexdigest()})
            with patch("pullback_data.yf.download", side_effect=AssertionError("must not fetch")):
                result = trial(root, 2)
            self.assertEqual(result["confirmed_observations"], 1)
            self.assertEqual(result["transition_count"], 1)
            self.assertEqual(result["signals_added"], 2)
            self.assertEqual(result["repeat_added"], 0)
            self.assertTrue(result["repeat_bytes_equal"])
            history = pd.read_csv(Path(result["output_dir"]) / "history/signals.csv")
            self.assertEqual(history["SetupID"].nunique(), 1)
            self.assertEqual(history["Bucket"].unique().tolist(), ["review"])
            self.assertTrue((pd.to_datetime(history["FirstSeenAtUTC"], utc=True) > pd.Timestamp("2026-11-24", tz="UTC")).all())
            self.assertEqual(live.read_text(encoding="utf-8"), "unchanged live history")
            self.assertEqual(json.loads((root / "output/pullback/latest_success.json").read_text())["run"], str(run_dir))
        with self.assertRaises(ValueError):
            trial(Path("."), 100)

    def test_builder_rejects_bad_quality_and_refresh_failure_does_not_use_old_caps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "output/pullback/runs/source"
            summary = self.summary()
            write_json(source / "config.json", self.config)
            write_json(source / "run_status.json", {**summary, "quality_passed": False})
            write_json(root / "output/pullback/latest_success.json", {"run": str(source), "as_of": summary["as_of"], "config_hash": config_hash(self.config)})
            with self.assertRaises(ValueError):
                build(root)
            write_json(source / "run_status.json", summary)
            write_csv(source / "all_candidates.csv", pd.DataFrame([self.record()]))
            (root / "docs").mkdir()
            weekly = '<html><style>/* css */</style><main>saved weekly</main></html>'
            (root / "docs/index.html").write_text(weekly, encoding="utf-8")
            write_json(root / "output/pullback/ui_market_caps.json", {"entries": {"TEST": {"market_cap_usd": "999999999999"}}, "fetched_at_utc": "2026-11-01T00:00:00+00:00"})
            with patch("pullback_ui.fetch_snapshot", side_effect=RuntimeError("provider unavailable")):
                result = build(root, refresh_caps=True)
            self.assertIn("provider unavailable", result["market_caps_error"])
            self.assertIsNone(result["market_caps_fetched_at_utc"])
            self.assertIn('時価総額：不明', (root / "output/local-ui/daily.html").read_text(encoding="utf-8"))
            self.assertEqual((root / "docs/index.html").read_text(encoding="utf-8"), weekly)


if __name__ == "__main__":
    unittest.main()
