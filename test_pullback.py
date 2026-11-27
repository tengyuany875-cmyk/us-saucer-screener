from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pullback_app import assign_bucket, quality_passed, run, select
from pullback_data import DailyData, Quote, Sessions, classify_earnings, daily_universe, earnings, prepare, write_csv, write_json
from pullback_history import append_signals, identify, observe, read_signals, save_observations
from pullback_screener import analyse_daily, indicators, wilder
from pullback_settings import config_hash, load_config


def market_frames():
    sessions = Sessions()
    dates = sessions.between("2024-01-01", "2026-11-24")[-321:]
    close = np.linspace(50, 100, 320) + 0.55 * (-1.0) ** np.arange(320)
    close[-7:] = [107, 106, 104.5, 103, 102.5, 102, 101.2]
    frame = pd.DataFrame({"Open": close - 0.15, "High": close + 1.3, "Low": close - 1.3, "Close": close,
                          "Adj Close": close, "Volume": 3_000_000.0}, index=dates[:-1])
    final = pd.DataFrame({"Open": [101.5], "High": [102.9], "Low": [101.4], "Close": [102.7],
                          "Adj Close": [102.7], "Volume": [4_800_000.0]}, index=dates[-1:])
    full = pd.concat([frame, final])
    spy_close = np.linspace(400, 440, len(dates))
    spy = pd.DataFrame({"Open": spy_close, "High": spy_close + 1, "Low": spy_close - 1, "Close": spy_close,
                        "Adj Close": spy_close, "Volume": 10_000_000.0}, index=dates)
    return frame, full, spy


class PullbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.sessions = load_config(), Sessions()
        cls.forming, cls.confirmed, cls.spy = market_frames()

    def adjusted(self, raw, day="2026-11-24"):
        quote = prepare("TEST", raw, day, self.sessions, self.config)
        self.assertEqual(quote.status, "ok", quote.message)
        return quote.frame

    def signal(self, raw=None, day="2026-11-24", spy=None, config=None):
        raw = self.confirmed if raw is None else raw
        spy = self.spy if spy is None else spy
        return analyse_daily("TEST", self.adjusted(raw, day), self.adjusted(spy, day), config or self.config, day)

    def test_known_wilder_seed_and_rsi(self):
        values = pd.Series([1., 2., 3., 9.])
        self.assertEqual(wilder(values, 3).iloc[2], 2)
        self.assertAlmostEqual(wilder(values, 3).iloc[3], 13 / 3)
        rising = self.confirmed.copy()
        rising["Close"] = np.arange(len(rising)) + 50.0
        self.assertEqual(indicators(rising, self.config)["RSI"].iloc[-1], 100)
        rising["Close"] = 50.0
        self.assertEqual(indicators(rising, self.config)["RSI"].iloc[-1], 50)

    def test_synthetic_forming_and_confirmed_are_distinct(self):
        formed = self.signal(self.forming, "2026-11-23")
        confirmed = self.signal()
        self.assertEqual(formed["status"], "signal", formed)
        self.assertEqual(confirmed["status"], "signal", confirmed)
        self.assertEqual(formed["record"]["Stage"], "forming")
        self.assertEqual(confirmed["record"]["Stage"], "confirmed")
        self.assertEqual(formed["record"]["SetupPeakDate"], confirmed["record"]["SetupPeakDate"])
        self.assertEqual(formed["record"]["Score_rebound"], 0)
        self.assertGreater(confirmed["record"]["Score_rebound"], 0)

    def test_no_lookahead_and_no_input_mutation(self):
        original = self.confirmed.copy(deep=True)
        reference = self.signal()
        future = self.confirmed.iloc[-1:].copy()
        future.index = pd.DatetimeIndex(["2026-11-25"])
        future.loc[:, ["Open", "High", "Low", "Close", "Adj Close"]] *= 100
        extended = pd.concat([self.confirmed, future])
        self.assertEqual(self.signal(extended), reference)
        pd.testing.assert_frame_equal(self.confirmed, original)

    def test_below_spy_sma_is_information_not_exclusion(self):
        spy = self.spy.copy()
        spy.loc[spy.index[-20:], ["Open", "High", "Low", "Close", "Adj Close"]] *= 0.97
        # Keep valid OHLC after changing all price fields by the same factor.
        result = self.signal(spy=spy)
        self.assertEqual(result["status"], "signal", result)
        self.assertFalse(result["record"]["SPYAboveSMA"])

    def test_changing_config_changes_acceptance_and_fingerprint(self):
        config = deepcopy(self.config)
        config["rules"]["minimum_dollar_volume"] = 10**12
        self.assertNotEqual(config_hash(config), config_hash(self.config))
        self.assertIn("liquidity", self.signal(config=config)["reason_codes"])

    def test_score_components_total_and_reason(self):
        result = self.signal()
        self.assertEqual(result["status"], "signal", result)
        row = result["record"]
        self.assertAlmostEqual(sum(json.loads(row["ScoreBreakdown"]).values()), row["Score"])
        self.assertIn("前日高値突破", row["Reasons"])
        self.assertGreaterEqual(row["RewardRisk"], self.config["rules"]["minimum_reward_risk"])

    def test_downtrend_is_not_a_pullback(self):
        raw = self.confirmed.copy()
        raw.loc[raw.index[-1], ["Open", "High", "Low", "Close", "Adj Close"]] = [59, 61, 58, 60, 60]
        result = self.signal(raw)
        self.assertEqual(result["status"], "no_signal")
        self.assertIn("trend", result["reason_codes"])

    def test_empty_stale_short_and_missing_sessions_are_distinct(self):
        q = lambda f: prepare("TEST", f, "2026-11-24", self.sessions, self.config)
        self.assertEqual(q(pd.DataFrame()).status, "fetch_failed")
        self.assertEqual(q(self.confirmed.iloc[:-1]).status, "stale")
        short = q(self.confirmed.tail(20))
        self.assertEqual(short.status, "insufficient_history")
        self.assertTrue(short.price_success)
        self.assertEqual(q(self.confirmed.drop(self.confirmed.index[-3])).status, "missing_sessions")

    def test_corrupt_nonfinite_or_duplicate_prices_rejected(self):
        for value in [np.nan, np.inf, -1]:
            frame = self.confirmed.copy()
            frame.loc[frame.index[-1], "High"] = value
            self.assertEqual(prepare("TEST", frame, "2026-11-24", self.sessions, self.config).status, "invalid_data")
        duplicate = pd.concat([self.confirmed, self.confirmed.tail(1)])
        self.assertEqual(prepare("TEST", duplicate, "2026-11-24", self.sessions, self.config).status, "invalid_data")

    def test_consistent_ohlc_adjustment_and_nominal_adv(self):
        raw = self.confirmed.copy()
        raw["Adj Close"] *= 0.5
        frame = self.adjusted(raw)
        self.assertAlmostEqual(frame["Open"].iloc[-1], raw["Open"].iloc[-1] * 0.5)
        self.assertEqual(frame["RawClose"].iloc[-1], raw["Close"].iloc[-1])
        actual = indicators(frame, self.config)["DollarVolume"].iloc[-1]
        expected = (raw["Close"] * raw["Volume"]).tail(self.config["indicators"]["liquidity_days"]).mean()
        self.assertAlmostEqual(actual, expected)

    def test_calendar_holiday_halfday_and_dst(self):
        self.assertEqual(self.sessions.following("2026-11-25", 1)[0], pd.Timestamp("2026-11-27"))
        self.assertEqual(self.sessions.last_completed(datetime(2026,11,27,18,29,tzinfo=timezone.utc),30), "2026-11-25")
        self.assertEqual(self.sessions.last_completed(datetime(2026,11,27,18,30,tzinfo=timezone.utc),30), "2026-11-27")
        self.assertEqual(self.sessions.calendar.session_close("2026-07-01").hour, 20)
        self.assertEqual(self.sessions.calendar.session_close("2026-11-24").hour, 21)

    def test_earnings_unknown_near_and_outside(self):
        now = datetime(2026,11,25,12,tzinfo=timezone.utc)
        check = lambda dates: classify_earnings(dates, now, self.sessions, "2026-11-24", 5)
        self.assertEqual(check([])["status"], "unknown")
        self.assertEqual(check(["2026-11-23T16:00:00-05:00"])["status"], "unknown")
        self.assertEqual(check(["2026-11-25T16:30:00-05:00"])["status"], "within_holding")
        self.assertEqual(check(["2026-11-25T06:00:00-05:00","2027-02-01T16:00:00-05:00"])["status"], "within_holding")
        self.assertEqual(check(["2026-11-24T16:30:00-05:00"])["status"], "within_holding")
        self.assertEqual(check(["2026-11-24"])["status"], "within_holding")
        self.assertEqual(check(["2026-11-25"])["status"], "within_holding")
        self.assertEqual(check(["2026-12-02T16:00:00-05:00"])["status"], "within_holding")
        self.assertEqual(check(["2026-12-03T08:00:00-05:00"])["status"], "outside_holding")

    def test_unknown_earnings_never_enters_main(self):
        row = {"Stage": "confirmed"}
        self.assertEqual(assign_bucket(row, {"status":"unknown", "next_date":""})["Bucket"], "review")
        self.assertEqual(assign_bucket(row, {"status":"within_holding", "next_date":"date"})["Bucket"], "earnings_excluded")
        self.assertEqual(assign_bucket(row, {"status":"outside_holding", "next_date":"date"})["Bucket"], "main")
        self.assertEqual(assign_bucket({"Stage":"forming"}, {"status":"outside_holding", "next_date":"date"})["Bucket"], "watch")

    def test_selection_prioritises_confirmed_caps_twenty_and_does_not_fill(self):
        records = [{"SignalID":str(n), "Ticker":str(n), "Bucket":"main" if n==25 else "watch", "Score":90-n, "AverageDollarVolume":100} for n in range(26)]
        records.append({"SignalID":"review", "Ticker":"review", "Bucket":"review", "Score":100, "AverageDollarVolume":100})
        chosen = select(records,20)
        self.assertEqual(len(chosen),20)
        self.assertEqual(chosen[0]["Bucket"],"main")
        self.assertNotIn("review",{r["SignalID"] for r in chosen})
        self.assertEqual(len(select(records[:2],20)),2)
        self.assertEqual(select([],20),[])

    def test_quality_gate_empty_failure_and_successful_zero_candidates(self):
        self.assertFalse(quality_passed([],self.config,True))
        self.assertFalse(quality_passed([{"PriceSuccess":False}],self.config,True))
        self.assertFalse(quality_passed([{"PriceSuccess":True}],self.config,False))
        self.assertTrue(quality_passed([{"PriceSuccess":True}],self.config,True))

    def test_cache_hit_no_network_and_corruption_is_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            download = Mock(return_value=self.confirmed)
            config = deepcopy(self.config)
            config["data"]["pause_seconds"] = 0
            data = DailyData(Path(tmp), config, self.sessions, "2026-11-24", downloader=download)
            self.assertEqual(list(data.fetch(["TEST"]))[0].status,"ok")
            self.assertEqual(list(data.fetch(["TEST"]))[0].source,"cache")
            self.assertEqual(download.call_count,1)
            (Path(tmp)/"TEST.csv").write_text("broken",encoding="utf-8")
            data.offline=True
            self.assertEqual(list(data.fetch(["TEST"]))[0].status,"cache_miss")

    def test_partial_provider_failure_retries_only_missing_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=deepcopy(self.config)
            config["data"].update({"pause_seconds":0,"retry_seconds":0})
            batch=pd.concat({"GOOD":self.confirmed},axis=1)
            downloader=Mock(side_effect=[batch,self.confirmed])
            data=DailyData(Path(tmp),config,self.sessions,"2026-11-24",downloader=downloader)
            quotes=list(data.fetch(["GOOD","RETRY"]))
            self.assertTrue(all(q.status=="ok" for q in quotes))
            self.assertEqual(downloader.call_args_list[1].kwargs["tickers"],["RETRY"])

    def test_history_is_idempotent_and_retains_both_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first=identify({"Ticker":"TEST","SetupPeakDate":"2026-11-13","MarketDate":"2026-11-23","Stage":"forming","Bucket":"watch"},"hash")
            confirmed=identify({**first,"MarketDate":"2026-11-24","Stage":"confirmed","Bucket":"main"},"hash")
            history, added=append_signals([first,confirmed],root,self.config,"hash","2026-11-25T12:00:00+00:00")
            self.assertEqual(added,2)
            history,added=append_signals([first,confirmed],root,self.config,"hash","2026-11-25T13:00:00+00:00")
            self.assertEqual(added,0)
            self.assertEqual(len(read_signals(root)),2)
            self.assertEqual(history["SetupID"].nunique(),1)
            self.assertEqual(history["Stage"].nunique(),2)
            self.assertTrue(history["FirstSeenAtUTC"].str.contains("12:00").all())

    def test_observations_horizons_costs_and_pending_not_zero(self):
        now="2026-11-25T12:00:00+00:00"
        record={"SignalID":"s","Ticker":"TEST","Stage":"forming","Bucket":"watch","MarketDate":"2026-11-17",
                "Horizons":"[1,3,5]","CostBpsPerSide":10,"FirstSeenAtUTC":"2026-11-17T22:00:00+00:00"}
        quote=Mock(frame=self.adjusted(self.confirmed),status="ok")
        data=Mock(as_of="2026-11-24")
        data.cached.return_value=quote
        outcomes=observe(pd.DataFrame([record]),data,self.sessions,now)
        self.assertTrue(outcomes["Status"].eq("complete").all())
        row=outcomes.iloc[-1]
        expected=(row.AdjustedExitClose*.999/(row.AdjustedEntryOpen*1.001)-1)*100
        self.assertAlmostEqual(row.NetReturnPct,expected)
        record["MarketDate"]="2026-11-24"
        pending=observe(pd.DataFrame([record]),data,self.sessions,now)
        self.assertTrue(pending["Status"].eq("pending").all())
        self.assertTrue(pending["NetReturnPct"].isna().all())
        record["FirstSeenAtUTC"]="2026-11-25T16:00:00+00:00"
        self.assertTrue(observe(pd.DataFrame([record]),data,self.sessions,now)["Status"].eq("late_observation").all())

    def test_config_rejects_unknown_keys_bad_weights_and_periods(self):
        for mutate in [lambda c:c["rules"].update({"typo":1}),lambda c:c["scoring"]["weights"].update({"trend":19}),lambda c:c["indicators"].update({"rsi":0})]:
            config=deepcopy(self.config)
            mutate(config)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"config.json"
                write_json(path,config)
                with self.assertRaises(ValueError):load_config(path)

    def test_earnings_transport_failure_and_meetings_are_not_safe_dates(self):
        now=datetime(2026,11,25,12,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("pullback_data.yf.Ticker") as ticker:
                ticker.return_value.get_earnings_dates.side_effect=RuntimeError("fetch failed")
                result=earnings("TEST",Path(tmp),self.config,self.sessions,"2026-11-24",now)
                self.assertEqual(result["status"],"unknown")
                self.assertIn("fetch failed",result["error"])
            with patch("pullback_data.yf.Ticker") as ticker:
                ticker.return_value.get_earnings_dates.return_value=pd.DataFrame({"Event Type":["Meeting"]},index=pd.DatetimeIndex(["2027-01-01T16:00:00-05:00"]))
                self.assertEqual(earnings("TEST",Path(tmp),self.config,self.sessions,"2026-11-24",now)["status"],"unknown")

    def test_circuit_breaker_prevents_unbounded_provider_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=deepcopy(self.config)
            config["data"].update({"batch_size":1,"attempts":1,"max_failed_batches":1,"pause_seconds":0})
            download=Mock(return_value=pd.DataFrame())
            data=DailyData(Path(tmp),config,self.sessions,"2026-11-24",downloader=download)
            results=list(data.fetch(["ONE","TWO","THREE"]))
            self.assertEqual([q.status for q in results],["fetch_failed","not_attempted","not_attempted"])
            self.assertEqual(download.call_count,1)

    def test_runner_zero_candidates_has_headers_and_failure_preserves_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            spy=Quote("SPY","ok",self.adjusted(self.spy),metadata={"sha256":"spy"})
            no_signal=self.adjusted(self.confirmed)
            no_signal.loc[:,"RawClose"]=1.0
            provider=Mock()
            provider.fetch.side_effect=lambda names: iter([spy] if names==["SPY"] else [Quote("TEST","ok",no_signal)] if names else [])
            universe=pd.DataFrame({"Ticker":["TEST"],"Name":["Test Common Stock"],"Exchange":["NYSE"]})
            with patch("pullback_app.DailyData",return_value=provider),patch("pullback_app.daily_universe",return_value=universe),patch("pullback_app.utc_now",return_value=datetime(2026,11,25,12,tzinfo=timezone.utc)),patch("builtins.print"):
                summary=run(Path(__file__).with_name("pullback_config.json"),root)
            self.assertTrue(summary["quality_passed"])
            self.assertTrue(pd.read_csv(Path(summary["output_dir"])/"selected.csv").empty)
            saved=(root/"history/pullback/signals.csv").read_bytes()
            latest=(root/"output/pullback/latest_success.json").read_bytes()
            provider.fetch.side_effect=lambda names:iter([spy] if names==["SPY"] else [Quote("TEST","cache_miss",source="cache")])
            with patch("pullback_app.DailyData",return_value=provider),patch("pullback_app.daily_universe",return_value=universe),patch("pullback_app.utc_now",return_value=datetime(2026,11,25,12,1,tzinfo=timezone.utc)),patch("builtins.print"):
                failure=run(Path(__file__).with_name("pullback_config.json"),root)
            self.assertFalse(failure["quality_passed"])
            self.assertEqual(failure["cache_hits"],0)
            self.assertEqual((root/"history/pullback/signals.csv").read_bytes(),saved)
            self.assertEqual((root/"output/pullback/latest_success.json").read_bytes(),latest)

    def test_runner_writes_main_review_and_first_observation_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            spy=Quote("SPY","ok",self.adjusted(self.spy))
            stock=self.adjusted(self.confirmed)
            provider=Mock()
            provider.fetch.side_effect=lambda names:iter([spy] if names==["SPY"] else [Quote(n,"ok",stock) for n in names])
            provider.cached.return_value=Quote("TEST","ok",stock)
            provider.as_of="2026-11-24"
            universe=pd.DataFrame({"Ticker":["TEST","REVIEW"],"Name":["Test","Review"],"Exchange":["NYSE","NYSE"]})
            event=lambda ticker,*a,**k:{"status":"outside_holding" if ticker=="TEST" else "unknown","next_date":"2027-01-01" if ticker=="TEST" else ""}
            with patch("pullback_app.DailyData",return_value=provider),patch("pullback_app.daily_universe",return_value=universe),patch("pullback_app.earnings",side_effect=event),patch("pullback_app.utc_now",return_value=datetime(2026,11,25,12,tzinfo=timezone.utc)),patch("builtins.print"):
                summary=run(Path(__file__).with_name("pullback_config.json"),root)
            self.assertEqual(summary["selected_main"],1)
            self.assertEqual(summary["review_count"],1)
            output=Path(summary["output_dir"])
            self.assertEqual(pd.read_csv(output/"main.csv")["Ticker"].tolist(),["TEST"])
            self.assertEqual(pd.read_csv(output/"review.csv")["Ticker"].tolist(),["REVIEW"])
            history=read_signals(root/"history/pullback")
            self.assertTrue(history["FirstSeenAtUTC"].eq("2026-11-25T12:00:00+00:00").all())
            self.assertEqual(set(pd.read_csv(root/"history/pullback/outcomes.csv")["Stage"]),{"confirmed"})

    def test_completed_observations_survive_provider_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            complete=pd.DataFrame([{"SignalID":"first","Horizon":5,"Status":"complete","NetReturnPct":2.0}])
            save_observations(root,complete)
            missing=pd.DataFrame([{"SignalID":"first","Horizon":5,"Status":"missing_prices","NetReturnPct":np.nan}])
            retained=save_observations(root,missing)
            self.assertTrue(retained["Status"].eq("complete").all())
            saved=pd.read_csv(root/"outcomes.csv")
            self.assertEqual(len(saved),1)
            self.assertEqual(saved.iloc[0]["Status"],"complete")
            self.assertEqual(saved.iloc[0]["NetReturnPct"],2.0)

    def test_editable_default_cannot_validate_its_own_typo(self):
        config=deepcopy(self.config)
        config["rules"]["misspelled_threshold"]=1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"default.json"
            write_json(path,config)
            with patch("pullback_settings.DEFAULT_CONFIG",path),self.assertRaises(ValueError):
                load_config(path)

    def test_offline_universe_never_fetches_and_daily_filters_plural_warrants(self):
        frame=pd.DataFrame({"Ticker":["KEEP","WARRANT"],"Name":["Example Common Stock","Example Warrants"],"Exchange":["NYSE","NYSE"]})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            with patch("pullback_data.fetch_us_equity_universe") as fetch:
                with self.assertRaises(RuntimeError):
                    daily_universe(path,self.config,datetime(2026,11,25,tzinfo=timezone.utc),offline=True)
                fetch.assert_not_called()
            write_csv(path/"universe.csv",frame)
            with patch("pullback_data.fetch_us_equity_universe",return_value=frame) as fetch:
                result=daily_universe(path,self.config,datetime(2026,11,25,tzinfo=timezone.utc),offline=True)
                self.assertEqual(result["Ticker"].tolist(),["KEEP"])
                self.assertFalse(fetch.call_args.kwargs["refresh"])


if __name__=="__main__":
    unittest.main()
