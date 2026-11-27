"""Daily-only cache and quality checks; reuse the existing universe/extractor."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf

from app import extract_ticker_frame
from market_caps import EXCLUDED_SECURITY
from pullback_earnings_archive import archive_snapshot
from universe import fetch_us_equity_universe

OHLC = ["Open", "High", "Low", "Close"]
RAW_COLUMNS = [*OHLC, "Adj Close", "Volume"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=index, encoding="utf-8", index_label="Date" if index else None)
    temp.replace(path)


class Sessions:
    def __init__(self):
        self.calendar = xcals.get_calendar("XNYS")

    def last_completed(self, now: datetime, delay_minutes: float) -> str:
        stamp = pd.Timestamp(now)
        if stamp.tzinfo is None:
            raise ValueError("now must be timezone aware")
        local_day = stamp.tz_convert("America/New_York").date().isoformat()
        session = self.calendar.date_to_session(local_day, direction="previous")
        if stamp < self.calendar.session_close(session) + pd.Timedelta(minutes=delay_minutes):
            session = self.calendar.previous_session(session)
        return session.date().isoformat()

    def between(self, start, end) -> pd.DatetimeIndex:
        return self.calendar.sessions_in_range(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())).tz_localize(None)

    def following(self, day: str, count: int) -> pd.DatetimeIndex:
        start = pd.Timestamp(day) + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=max(45, count * 3 + 14))
        sessions = self.between(start, end)[:count]
        if len(sessions) < count:
            raise ValueError("Calendar does not cover tracking horizon")
        return sessions


@dataclass
class Quote:
    ticker: str
    status: str
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = "network"
    message: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def price_success(self) -> bool:
        return self.status in {"ok", "insufficient_history"}


def prepare(ticker: str, raw: pd.DataFrame, as_of: str, sessions: Sessions, config: dict, source="network") -> Quote:
    if raw.empty or not set(RAW_COLUMNS).issubset(raw.columns):
        return Quote(ticker, "fetch_failed", source=source, message="Empty result or missing OHLCV/Adj Close")
    data = raw.copy()
    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    if index.tz is not None:
        index = index.tz_convert("America/New_York").tz_localize(None)
    data.index = index.normalize()
    data = data.loc[data.index <= pd.Timestamp(as_of)].sort_index()
    data = data.dropna(how="all", subset=RAW_COLUMNS)
    if data.empty:
        return Quote(ticker, "fetch_failed", source=source, message="No completed session data")
    if data.index.has_duplicates:
        return Quote(ticker, "invalid_data", source=source, message="Duplicate session")
    if data.index[-1] != pd.Timestamp(as_of):
        return Quote(ticker, "stale", source=source, message=f"Latest bar {data.index[-1].date()}; expected {as_of}")
    for col in RAW_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    values = data[RAW_COLUMNS].to_numpy(dtype=float)
    valid = (np.isfinite(values).all() and (data[[*OHLC, "Adj Close"]] > 0).all().all()
             and data["Volume"].ge(0).all() and data["Volume"].iloc[-1] > 0
             and data["High"].ge(data[["Open", "Close", "Low"]].max(axis=1) - 1e-6).all()
             and data["Low"].le(data[["Open", "Close", "High"]].min(axis=1) + 1e-6).all())
    if not valid:
        return Quote(ticker, "invalid_data", source=source, message="Invalid, nonfinite, or halted OHLCV")
    all_sessions = sessions.between(data.index[0], as_of)
    if not data.index.isin(all_sessions).all():
        return Quote(ticker, "invalid_data", source=source, message="Non-session date in daily prices")
    minimum = config["data"]["minimum_history"]
    if len(data) < minimum:
        return Quote(ticker, "insufficient_history", source=source, message=f"{len(data)} bars; need {minimum}")
    expected = all_sessions[-minimum:]
    if len(expected) != minimum or not expected.isin(data.index).all():
        return Quote(ticker, "missing_sessions", source=source, message="Gap in recent required sessions; no forward-fill")
    factor = data["Adj Close"] / data["Close"]
    adjusted = data[RAW_COLUMNS].copy()
    adjusted["RawClose"], adjusted["RawOpen"] = data["Close"], data["Open"]
    for column in OHLC:
        adjusted[column] = data[column] * factor
    return Quote(ticker, "ok", adjusted, source)


class DailyData:
    def __init__(self, cache_dir: Path, config: dict, sessions: Sessions, as_of: str, offline=False, downloader=None):
        self.root, self.config, self.sessions, self.as_of, self.offline = Path(cache_dir), config, sessions, as_of, offline
        self.download = downloader or yf.download
        self.root.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(self.root / "yfinance"))

    def _paths(self, ticker: str) -> tuple[Path, Path]:
        if not re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", ticker):
            raise ValueError(f"Invalid ticker: {ticker}")
        return self.root / f"{ticker}.csv", self.root / f"{ticker}.json"

    def cached(self, ticker: str) -> Quote | None:
        path, meta_path = self._paths(ticker)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["schema"] != 1 or meta["as_of"] != self.as_of or meta["history_calendar_days"] != self.config["data"]["history_calendar_days"]:
                return None
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
                return None
            quote = prepare(ticker, pd.read_csv(path, index_col="Date", parse_dates=True), self.as_of, self.sessions, self.config, "cache")
            quote.metadata = meta
            return quote if quote.price_success else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def fetch(self, tickers: list[str]):
        pending = []
        for ticker in dict.fromkeys(tickers):
            cached = self.cached(ticker)
            if cached is not None:
                yield cached
            elif self.offline:
                yield Quote(ticker, "cache_miss", source="cache", message="No valid cache for this completed session")
            else:
                pending.append(ticker)
        d, failed_batches = self.config["data"], 0
        for start in range(0, len(pending), d["batch_size"]):
            batch = pending[start:start + d["batch_size"]]
            if failed_batches >= d["max_failed_batches"]:
                for ticker in batch:
                    yield Quote(ticker, "not_attempted", message="Circuit breaker: consecutive complete batch failures")
                continue
            print(f"Daily prices {start + 1}-{start + len(batch)}/{len(pending)}", flush=True)
            retry, found, last_error = batch[:], {}, ""
            for attempt in range(d["attempts"]):
                try:
                    downloaded = self.download(tickers=retry, start=(pd.Timestamp(self.as_of) - pd.Timedelta(days=d["history_calendar_days"])).date().isoformat(),
                        end=(pd.Timestamp(self.as_of) + pd.Timedelta(days=1)).date().isoformat(), interval="1d", auto_adjust=False, actions=True,
                        group_by="ticker", threads=d["threads"], progress=False, timeout=d["timeout_seconds"], prepost=False, ignore_tz=True)
                    for ticker in retry:
                        frame = extract_ticker_frame(downloaded, ticker, len(retry))
                        if not frame.empty and "Close" in frame and frame["Close"].notna().any():
                            found[ticker] = frame
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                retry = [ticker for ticker in retry if ticker not in found]
                if not retry:
                    break
                if attempt + 1 < d["attempts"]:
                    time.sleep(d["retry_seconds"] * (attempt + 1))
            failed_batches = failed_batches + 1 if not found else 0
            for ticker in batch:
                if ticker not in found:
                    yield Quote(ticker, "fetch_failed", message=last_error or "Provider returned no prices after bounded retries")
                    continue
                quote = prepare(ticker, found[ticker], self.as_of, self.sessions, self.config)
                if quote.price_success:
                    path, meta_path = self._paths(ticker)
                    # Cache a complete daily window, never merge differently adjusted vintages.
                    raw = found[ticker].copy()
                    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
                    raw = raw.loc[raw.index <= pd.Timestamp(self.as_of)]
                    write_csv(path, raw, index=True)
                    quote.metadata = {"schema": 1, "as_of": self.as_of, "fetched_at_utc": utc_now().isoformat(), "source": "Yahoo Finance via yfinance",
                        "history_calendar_days": d["history_calendar_days"], "auto_adjust": False, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    write_json(meta_path, quote.metadata)
                yield quote
            time.sleep(d["pause_seconds"])


def daily_universe(cache_dir: Path, config: dict, now: datetime, offline=False) -> pd.DataFrame:
    path = Path(cache_dir) / "universe.csv"
    if offline and not path.exists():
        raise RuntimeError("Offline mode requires an existing daily universe cache")
    refresh = not offline and (not path.exists() or now.timestamp() - path.stat().st_mtime > config["data"]["universe_cache_days"] * 86400)
    frame = fetch_us_equity_universe(path, refresh=refresh)
    # Daily-only supplement fixes plural security names without altering weekly behavior.
    return frame.loc[~frame["Name"].fillna("").map(lambda name: bool(EXCLUDED_SECURITY.search(name)))].reset_index(drop=True)


def classify_earnings(dates: list[str], known_at: datetime, sessions: Sessions, as_of: str, holding: int) -> dict:
    window_start = sessions.calendar.session_close(as_of)
    deadline = sessions.calendar.session_close(sessions.following(as_of, holding)[-1])
    future = []
    for value in dates:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            # No time supplied: conservatively treat the entire ET date as uncertain.
            stamp = stamp.tz_localize("America/New_York")
            if pd.Timestamp(as_of).date() <= stamp.date() <= deadline.tz_convert("America/New_York").date():
                return {"status": "within_holding", "next_date": stamp.isoformat(), "reason": "Time unknown inside signal/holding dates"}
        # A release since the signal close remains risky before the next daily
        # bar, including a release that occurred before this morning's scan.
        if stamp >= window_start:
            future.append(stamp)
    if not future:
        return {"status": "unknown", "next_date": "", "reason": "No future earnings report date supplied"}
    next_date = min(future)
    return {"status": "within_holding" if next_date <= deadline else "outside_holding", "next_date": next_date.isoformat(), "reason": "Provider schedule, not a guarantee"}


def earnings(ticker: str, cache_dir: Path, config: dict, sessions: Sessions, as_of: str, now: datetime, offline=False) -> dict:
    path = Path(cache_dir) / "earnings" / f"{ticker}.json"
    dates, cache_hit, fetched = [], False, now.isoformat()
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        archive_snapshot(path.parent / "snapshots", ticker, saved)
        age = (now - datetime.fromisoformat(saved["fetched_at_utc"])).total_seconds()
        if saved["as_of"] == as_of and 0 <= age < config["data"]["earnings_cache_hours"] * 3600:
            dates, cache_hit, fetched = saved["dates"], True, saved["fetched_at_utc"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    error = ""
    if not cache_hit and not offline:
        try:
            frame = yf.Ticker(ticker).get_earnings_dates(limit=12)
            if frame is not None and not frame.empty:
                # Calls/meetings are not interchangeable with earnings reports.
                if "Event Type" in frame:
                    frame = frame.loc[frame["Event Type"].astype(str).str.lower().isin(["earnings", "2"])]
                dates = [pd.Timestamp(index).isoformat() for index in frame.index]
            snapshot = {"as_of": as_of, "dates": dates, "fetched_at_utc": fetched,
                        "observed_at_utc": utc_now().isoformat()}
            archive_snapshot(path.parent / "snapshots", ticker, snapshot)
            write_json(path, snapshot)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    result = classify_earnings(dates, now, sessions, as_of, config["rules"]["holding_sessions"])
    result.update({"fetched_at_utc": fetched if cache_hit or not offline else "", "source": "cache" if cache_hit else ("unavailable" if offline else "Yahoo Finance"), "error": error})
    return result
