"""Bounded, offline recent-session trial. Never writes live history or fetches data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from pullback_app import CANDIDATE_COLUMNS, assign_bucket
from pullback_data import DailyData, Sessions, prepare, utc_now, write_csv, write_json
from pullback_history import append_signals, identify, read_signals
from pullback_screener import analyse_daily
from pullback_settings import config_hash, load_config

LIMITATIONS = ("現在の銘柄一覧・取得時点の調整済み株価を使う事後の価格条件確認です。"
               "過去の決算予定は再現していないため全件を決算不明として扱い、主候補へ選出しません。"
               "検出時点の反発確率や売買成績を検証するバックテストではありません。")


def transitions(signals: pd.DataFrame) -> pd.DataFrame:
    columns = ["Ticker", "SetupID", "FormingDate", "ConfirmedDate", "FormingScore", "ConfirmedScore"]
    rows = []
    for setup_id, group in signals.groupby("SetupID"):
        forming = group.loc[group["Stage"].eq("forming")].sort_values("MarketDate")
        confirmed = group.loc[group["Stage"].eq("confirmed")].sort_values("MarketDate")
        if not forming.empty and not confirmed.empty and forming.iloc[0]["MarketDate"] < confirmed.iloc[0]["MarketDate"]:
            a, b = forming.iloc[0], confirmed.iloc[0]
            rows.append(dict(zip(columns, [a["Ticker"], setup_id, a["MarketDate"], b["MarketDate"], a["Score"], b["Score"]])))
    return pd.DataFrame(rows, columns=columns)


def trial(root: Path, count: int = 6) -> dict:
    if not 2 <= count <= 10:
        raise ValueError("Recent trial is bounded to 2–10 sessions, including the latest session")
    root = Path(root)
    pointer = json.loads((root / "output/pullback/latest_success.json").read_text(encoding="utf-8"))
    source_run = Path(pointer["run"])
    config = load_config(source_run / "config.json")
    fingerprint, sessions, now = config_hash(config), Sessions(), utc_now()
    if fingerprint != pointer["config_hash"]:
        raise ValueError("Run configuration does not match latest_success")
    as_of = pointer["as_of"]
    dates = sessions.between(pd.Timestamp(as_of) - pd.Timedelta(days=40), as_of)[-count:]
    days = [day.date().isoformat() for day in dates]
    output = root / "output/pullback/recent" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "source_hashes.json", {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ["pullback_recent.py", "pullback_screener.py", "pullback_data.py", "pullback_history.py"]})
    universe = pd.read_csv(source_run / "universe.csv", keep_default_na=False)
    write_csv(output / "universe.csv", universe)
    data = DailyData(root / "cache/daily", config, sessions, as_of, offline=True)
    spy = data.cached("SPY")
    if spy is None or spy.status != "ok":
        raise RuntimeError("A valid latest-session SPY cache is required")
    records, statuses = [], []
    started = time.perf_counter()
    print(f"Recent trial: {output}\nDates: {days}; {len(universe)} tickers; offline only", flush=True)
    for number, info in enumerate(universe.to_dict("records"), start=1):
        ticker = info["Ticker"]
        cached = data.cached(ticker)
        raw = pd.read_csv(data.root / f"{ticker}.csv", index_col="Date", parse_dates=True) if cached is not None else pd.DataFrame()
        for day in days:
            quote = prepare(ticker, raw, day, sessions, config, "retrospective_cache")
            status = {"Ticker": ticker, "MarketDate": day, "DataStatus": quote.status,
                      "Decision": "not_analysed", "ReasonCodes": "", "RawScore": ""}
            if quote.status == "ok":
                result = analyse_daily(ticker, quote.frame, spy.frame, config, day)
                status.update(Decision=result["status"], ReasonCodes=",".join(result["reason_codes"]), RawScore=result.get("raw_score", ""))
                if result["status"] == "signal":
                    record = assign_bucket(result["record"], {"status": "unknown", "next_date": "", "source": "not_reconstructed",
                        "reason": "Historical earnings schedules unavailable; retrospective price-only trial"})
                    record.update(Name=info["Name"], Exchange=info["Exchange"], ObservedAtUTC=utc_now().isoformat(),
                                  PriceFetchedAtUTC=cached.metadata["fetched_at_utc"], PriceSHA256=cached.metadata["sha256"])
                    records.append(identify(record, fingerprint))
                    print(f"{day} {ticker}: {record['Stage']} {record['Score']:.2f}", flush=True)
            statuses.append(status)
        if number % 250 == 0:
            print(f"Recent progress {number}/{len(universe)}; {time.perf_counter() - started:.1f}s", flush=True)
    frame = pd.DataFrame(records, columns=CANDIDATE_COLUMNS)
    write_csv(output / "all_candidates.csv", frame)
    write_csv(output / "confirmed.csv", frame.loc[frame["Stage"].eq("confirmed")])
    status_frame = pd.DataFrame(statuses)
    write_csv(output / "data_status.csv", status_frame)
    # Chronological appends exercise the same immutable, per-episode stage IDs,
    # but actual observation timestamps remain today. No historical fills/returns.
    ordered = sorted(records, key=lambda row: (row["MarketDate"], row["Ticker"]))
    history_root = output / "history"
    signals, added = append_signals(ordered, history_root, config, fingerprint, now.isoformat())
    before = (history_root / "signals.csv").read_bytes()
    _, repeated_added = append_signals(ordered, history_root, config, fingerprint, now.isoformat())
    after = (history_root / "signals.csv").read_bytes()
    linked = transitions(read_signals(history_root))
    write_csv(output / "transitions.csv", linked)
    per_day = []
    for day in days:
        daily_status = status_frame.loc[status_frame["MarketDate"].eq(day)]
        daily = frame.loc[frame["MarketDate"].eq(day)]
        valid = daily_status["DataStatus"].isin(["ok", "insufficient_history"])
        per_day.append({"date": day, "target_count": len(universe), "price_success_count": int(valid.sum()),
                        "price_success_rate": float(valid.mean()), "analysable_count": int(daily_status["DataStatus"].eq("ok").sum()),
                        "confirmed": int(daily["Stage"].eq("confirmed").sum()), "forming": int(daily["Stage"].eq("forming").sum()),
                        "data_status_counts": dict(Counter(daily_status["DataStatus"]))})
    summary = {"mode": "retrospective_price_only", "source_run": str(source_run), "source_as_of": as_of,
               "dates": days, "config_hash": fingerprint, "finished_at_utc": utc_now().isoformat(), "limitations": LIMITATIONS,
               "per_day": per_day, "candidate_observations": len(frame), "confirmed_observations": int(frame["Stage"].eq("confirmed").sum()),
               "signals_added": added, "repeat_added": repeated_added, "repeat_bytes_equal": before == after,
               "transition_count": len(linked), "elapsed_seconds": round(time.perf_counter() - started, 3), "output_dir": str(output.resolve())}
    write_json(output / "trial_status.json", summary)
    write_json(root / "output/pullback/latest_recent.json", {"run": str(output.resolve())})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sessions", type=int, default=6)
    args = parser.parse_args()
    trial(args.root, args.sessions)
