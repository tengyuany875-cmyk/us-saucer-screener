"""Episode/stage snapshots and forward observations, isolated from weekly history."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from pullback_data import DailyData, Sessions, write_csv

SIGNAL_COLUMNS = ["SignalID", "SetupID", "StrategyVersion", "ConfigHash", "Ticker", "MarketDate", "Stage", "Bucket",
                  "FirstSeenAtUTC", "Score", "ScoreBreakdown", "Reasons", "EarningsStatus", "EarningsNextDate", "Selected", "Horizons", "CostBpsPerSide"]
OUTCOME_COLUMNS = ["SignalID", "Ticker", "Stage", "Bucket", "SignalDate", "Horizon", "EntryDate", "ExitDate", "Status",
                   "AdjustedEntryOpen", "AdjustedExitClose", "GrossReturnPct", "NetReturnPct", "UpdatedAtUTC"]


def _id(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def identify(record: dict, fingerprint: str) -> dict:
    record = record.copy()
    record["SetupID"] = _id(fingerprint, record["Ticker"], record["SetupPeakDate"])
    record["SignalID"] = _id(record["SetupID"], record["Stage"])
    return record


def read_signals(root: Path) -> pd.DataFrame:
    path = Path(root) / "signals.csv"
    return pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=SIGNAL_COLUMNS)


def append_signals(records: list[dict], root: Path, config: dict, fingerprint: str, now: str) -> tuple[pd.DataFrame, int]:
    history = read_signals(root)
    rows = []
    for record in records:
        row = {key: record.get(key, "") for key in SIGNAL_COLUMNS}
        row.update({"StrategyVersion": config["strategy_version"], "ConfigHash": fingerprint,
                    "FirstSeenAtUTC": record.get("ObservedAtUTC", now), "Horizons": json.dumps(config["tracking"]["horizons"]),
                    "CostBpsPerSide": config["tracking"]["cost_bps_per_side"]})
        rows.append(row)
    added = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    updated = pd.concat([history, added], ignore_index=True).drop_duplicates("SignalID", keep="first")
    write_csv(Path(root) / "signals.csv", updated)
    return updated, len(updated) - len(history)


def observe(signals: pd.DataFrame, data: DailyData, sessions: Sessions, now: str) -> pd.DataFrame:
    rows, frames = [], {}
    for signal in signals.to_dict("records"):
        ticker = signal["Ticker"]
        if ticker not in frames:
            quote = data.cached(ticker)
            frames[ticker] = quote.frame if quote is not None and quote.status == "ok" else pd.DataFrame()
        frame = frames[ticker]
        horizons = json.loads(signal["Horizons"])
        future = sessions.following(signal["MarketDate"], max(horizons))
        entry_date = future[0]
        entry_open_time = sessions.calendar.session_open(entry_date)
        timely = pd.Timestamp(signal["FirstSeenAtUTC"]) < entry_open_time
        for horizon in horizons:
            exit_date = future[horizon - 1]
            row = {"SignalID": signal["SignalID"], "Ticker": ticker, "Stage": signal["Stage"], "Bucket": signal["Bucket"],
                   "SignalDate": signal["MarketDate"], "Horizon": horizon, "EntryDate": entry_date.date().isoformat(),
                   "ExitDate": exit_date.date().isoformat(), "Status": "pending", "UpdatedAtUTC": now}
            if not timely:
                row["Status"] = "late_observation"
            elif exit_date <= pd.Timestamp(data.as_of):
                required = future[:horizon]
                if frame.empty or not required.isin(frame.index).all():
                    row["Status"] = "missing_prices"
                else:
                    entry, exit_price = float(frame.loc[entry_date, "Open"]), float(frame.loc[exit_date, "Close"])
                    fee = float(signal["CostBpsPerSide"]) / 10000
                    row.update({"Status": "complete", "AdjustedEntryOpen": entry, "AdjustedExitClose": exit_price,
                                "GrossReturnPct": (exit_price / entry - 1) * 100,
                                "NetReturnPct": (exit_price * (1 - fee) / (entry * (1 + fee)) - 1) * 100})
            rows.append(row)
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def save_observations(root: Path, observations: pd.DataFrame) -> pd.DataFrame:
    # Keep the first completed vintage even after provider revisions or when an
    # old episode eventually leaves the rolling price cache.
    path = Path(root) / "outcomes.csv"
    if path.exists():
        existing = pd.read_csv(path)
        completed = existing.loc[existing["Status"].eq("complete")]
        observations = pd.concat([completed, observations], ignore_index=True).drop_duplicates(["SignalID", "Horizon"], keep="first")
    write_csv(path, observations)
    return observations
