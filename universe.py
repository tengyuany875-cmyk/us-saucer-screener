from __future__ import annotations

from io import StringIO
from pathlib import Path
import re

import pandas as pd
import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

EXCLUDE_NAME_PATTERN = re.compile(
    r"\b(warrant|right|unit|preferred|notes?|bonds?|debentures?|etf|fund|"
    r"income shares|closed-end|when issued|contingent value|acquisition corp)\b",
    flags=re.IGNORECASE,
)


def _download_pipe_file(url: str) -> pd.DataFrame:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 stock-screener/1.0"},
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), sep="|")
    first = frame.columns[0]
    return frame[~frame[first].astype(str).str.startswith("File Creation Time")]


def _normalise_yahoo_symbol(symbol: str) -> str | None:
    value = str(symbol).strip().upper().replace(".", "-")
    if not value or value == "NAN" or len(value) > 8:
        return None
    if any(char in value for char in ["$", "^", "/", "~", " "]):
        return None
    return value


def _quality_filter(frame: pd.DataFrame, symbol_col: str) -> pd.DataFrame:
    result = frame.copy()
    if "Test Issue" in result.columns:
        result = result[result["Test Issue"].fillna("N").eq("N")]
    if "ETF" in result.columns:
        result = result[result["ETF"].fillna("N").eq("N")]
    if "Financial Status" in result.columns:
        result = result[result["Financial Status"].fillna("N").isin(["N", ""])]
    if "Security Name" in result.columns:
        result = result[
            ~result["Security Name"].fillna("").str.contains(EXCLUDE_NAME_PATTERN, regex=True)
        ]
    result["Yahoo Symbol"] = result[symbol_col].map(_normalise_yahoo_symbol)
    return result.dropna(subset=["Yahoo Symbol"])


def fetch_us_equity_universe(
    cache_path: str | Path = "cache/tickers.csv",
    refresh: bool = False,
) -> pd.DataFrame:
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return pd.read_csv(cache).drop_duplicates("Ticker").reset_index(drop=True)

    nasdaq = _quality_filter(_download_pipe_file(NASDAQ_LISTED_URL), "Symbol")
    nasdaq_out = pd.DataFrame({
        "Ticker": nasdaq["Yahoo Symbol"],
        "Name": nasdaq["Security Name"],
        "Exchange": "NASDAQ",
    })

    other = _quality_filter(_download_pipe_file(OTHER_LISTED_URL), "ACT Symbol")
    exchange_map = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe", "V": "IEX"}
    other_out = pd.DataFrame({
        "Ticker": other["Yahoo Symbol"],
        "Name": other["Security Name"],
        "Exchange": other["Exchange"].map(exchange_map).fillna(other["Exchange"]),
    })

    combined = pd.concat([nasdaq_out, other_out], ignore_index=True)
    combined = combined.drop_duplicates("Ticker").sort_values("Ticker").reset_index(drop=True)
    combined.to_csv(cache, index=False)
    return combined
