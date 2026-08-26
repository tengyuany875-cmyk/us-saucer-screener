from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
import yfinance as yf


RESULTS_PATH = Path("output/results.csv")
CANDIDATES_PATH = Path("history/candidates.csv")
TRACKING_PATH = Path("history/weekly_tracking.csv")
TRACK_BATCH_SIZE = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()



def extract_ticker_frame(downloaded: pd.DataFrame, ticker: str, batch_size: int) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()

    if not isinstance(downloaded.columns, pd.MultiIndex):
        return downloaded.copy() if batch_size == 1 else pd.DataFrame()

    level0 = downloaded.columns.get_level_values(0)
    level1 = downloaded.columns.get_level_values(1)

    if ticker in level0:
        return downloaded[ticker].copy()
    if ticker in level1:
        return downloaded.xs(ticker, level=1, axis=1).copy()

    return pd.DataFrame()

def update_candidate_history(results: pd.DataFrame) -> tuple[int, int]:
    history = pd.read_csv(CANDIDATES_PATH)

    known = set(history["Ticker"].astype(str))
    new_rows = results[~results["Ticker"].astype(str).isin(known)].copy()

    if new_rows.empty:
        return len(history), 0

    new_rows.insert(0, "HistoryType", "first_seen")
    new_rows.insert(1, "FirstSeenMarketDate", new_rows["LastDate"])
    new_rows.insert(2, "RecordedAtUTC", utc_now())

    updated = pd.concat([history, new_rows], ignore_index=True)
    updated.to_csv(CANDIDATES_PATH, index=False)

    return len(updated), len(new_rows)



def fetch_latest_weekly_prices(tickers: list[str], market_date: str) -> dict:
    prices = {}
    cutoff = pd.Timestamp(market_date).date()

    for start in range(0, len(tickers), TRACK_BATCH_SIZE):
        batch = tickers[start:start + TRACK_BATCH_SIZE]
        print(f"追跡価格取得: {batch[0]} ～ {batch[-1]}")

        data = yf.download(
            batch,
            period="2mo",
            interval="1wk",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        for ticker in batch:
            frame = extract_ticker_frame(data, ticker, len(batch))
            if frame.empty:
                continue

            frame = frame[frame.index.date <= cutoff].dropna(how="all")
            if frame.empty:
                continue

            row = frame.iloc[-1]
            prices[ticker] = {
                "DataDate": frame.index[-1].date().isoformat(),
                "Price": row.get("Close"),
                "WeeklyHigh": row.get("High"),
                "WeeklyLow": row.get("Low"),
            }

        time.sleep(0.5)

    return prices

def update_weekly_tracking(results: pd.DataFrame) -> tuple[int, int]:
    tracking = pd.read_csv(TRACKING_PATH)

    market_dates = results["LastDate"].dropna().astype(str).unique()
    if len(market_dates) != 1:
        raise ValueError(
            f"results.csv の LastDate が1種類ではありません: {market_dates.tolist()}"
        )

    market_date = market_dates[0]

    already = tracking["MarketDate"].astype(str).eq(market_date)
    if already.any():
        print(f"SKIP: {market_date} の週次追跡データは既に存在します")
        return len(tracking), 0

    candidate_history = pd.read_csv(CANDIDATES_PATH)
    all_tickers = sorted(
        candidate_history["Ticker"].dropna().astype(str).unique()
    )
    print(f"追跡対象銘柄数: {len(all_tickers)}")

    weekly_prices = fetch_latest_weekly_prices(all_tickers, market_date)

    current = results.set_index("Ticker")
    first_seen = candidate_history.drop_duplicates("Ticker").set_index("Ticker")
    rows = []

    for ticker in all_tickers:
        in_screener = ticker in current.index
        row = current.loc[ticker] if in_screener else None
        price = weekly_prices.get(ticker, {})

        rows.append({
            "MarketDate": market_date,
            "RecordedAtUTC": utc_now(),
            "Ticker": ticker,
            "Exchange": row["Exchange"] if in_screener else first_seen.loc[ticker]["Exchange"],
            "DataDate": price.get("DataDate", market_date),
            "Price": price.get("Price", row["Price"] if in_screener else pd.NA),
            "WeeklyHigh": price.get("WeeklyHigh", pd.NA),
            "WeeklyLow": price.get("WeeklyLow", pd.NA),
            "Score": row["Score"] if in_screener else pd.NA,
            "Stage": row["Stage"] if in_screener else "",
            "Distance52wHighPct": row["Distance52wHighPct"] if in_screener else pd.NA,
            "VolumeRatio4wVs12w": row["VolumeRatio4wVs12w"] if in_screener else pd.NA,
            "RecoveryPct": row["RecoveryPct"] if in_screener else pd.NA,
            "BottomWeeks": row["BottomWeeks"] if in_screener else pd.NA,
            "InScreener": in_screener,
        })

    snapshot = pd.DataFrame(rows)

    updated = pd.concat([tracking, snapshot], ignore_index=True)
    updated.to_csv(TRACKING_PATH, index=False)

    return len(updated), len(snapshot)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} がありません")

    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"{CANDIDATES_PATH} がありません")

    if not TRACKING_PATH.exists():
        raise FileNotFoundError(f"{TRACKING_PATH} がありません")

    results = pd.read_csv(RESULTS_PATH)

    candidate_total, candidate_added = update_candidate_history(results)
    tracking_total, tracking_added = update_weekly_tracking(results)

    print("OK: 履歴更新完了")
    print(f"候補履歴: {candidate_total}件 (+{candidate_added})")
    print(f"週次追跡: {tracking_total}件 (+{tracking_added})")


if __name__ == "__main__":
    main()
