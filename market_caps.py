"""Market-cap reference ranks for US-listed ordinary shares."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import re
from urllib.request import Request, urlopen


SOURCE_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true"
MINIMUM_REFERENCE_SIZE = 1_000
EXCLUDED_SECURITY = re.compile(
    r"\b(warrants?|rights?|units?|preferred|notes?|bonds?|debentures?|"
    r"etfs?|funds?|closed[- ]end|income shares|when[- ]issued|contingent value)\b",
    re.IGNORECASE,
)


def _number(value: object) -> Decimal | None:
    try:
        number = Decimal(str(value).replace(",", "").replace("$", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() and number > 0 else None


def _symbol(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    symbol = value.strip().upper().replace(".", "-").replace("/", "-")
    return symbol if re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", symbol) else None


def build_snapshot(payload: dict, fetched_at: str) -> dict:
    """Rank the full reference list, before matching any screener candidates."""
    if not isinstance(payload, dict):
        raise ValueError("Nasdaq market-cap response is not an object")
    status = payload.get("status") or {}
    if not isinstance(status, dict) or status.get("rCode") != 200:
        raise ValueError("Nasdaq returned an unsuccessful market-cap response")
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise ValueError("Nasdaq market-cap rows are missing")

    caps = {}
    missing = 0
    for row in data["rows"]:
        if not isinstance(row, dict):
            raise ValueError("Invalid market-cap row")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Market-cap security name is missing")
        ticker = _symbol(row.get("symbol"))
        if not ticker or EXCLUDED_SECURITY.search(name):
            continue
        value = _number(row.get("marketCap"))
        if value is None:
            missing += 1
            continue
        if ticker in caps and caps[ticker] != value:
            raise ValueError(f"Conflicting market caps for {ticker}")
        caps[ticker] = value

    if len(caps) < MINIMUM_REFERENCE_SIZE:
        raise ValueError(f"Market-cap reference list is too small: {len(caps)}")

    entries = {}
    previous_cap = None
    rank = 0
    for position, (ticker, value) in enumerate(
        sorted(caps.items(), key=lambda item: (-item[1], item[0])), start=1
    ):
        if value != previous_cap:
            rank = position
        entries[ticker] = {"market_cap_usd": str(value), "rank": rank}
        previous_cap = value

    return {
        "source": SOURCE_URL,
        "fetched_at_utc": fetched_at,
        "source_as_of": data.get("asOf"),
        "ranked_count": len(entries),
        "missing_count": missing,
        "entries": entries,
    }


def fetch_snapshot() -> dict:
    request = Request(SOURCE_URL, headers={
        "Accept": "application/json", "User-Agent": "us-saucer-screener/1.0",
    })
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return build_snapshot(payload, datetime.now(timezone.utc).isoformat())


def add_market_caps(results, snapshot: dict):
    enriched = results.copy()
    entries = snapshot["entries"]
    matched = results["Ticker"].map(lambda ticker: entries.get(_symbol(ticker), {}))
    enriched["MarketCapUSD"] = matched.map(lambda entry: entry.get("market_cap_usd"))
    enriched["USMarketCapRank"] = matched.map(lambda entry: entry.get("rank"))
    enriched["MarketCapFetchedAtUTC"] = snapshot["fetched_at_utc"]
    enriched["MarketCapReferenceCount"] = snapshot["ranked_count"]
    return enriched


def format_market_cap(value: object) -> str:
    number = _number(value)
    if number is None:
        return "不明"
    oku = number / Decimal("100000000")
    if oku >= 10:
        return f"{oku:,.0f}億ドル"
    if oku >= 1:
        return f"{oku:.1f}億ドル"
    return f"{oku:.2f}億ドル" if oku >= Decimal("0.01") else "0.01億ドル未満"


def format_market_rank(value: object) -> str:
    number = _number(value)
    if number is None or number != number.to_integral_value():
        return "不明"
    return f"参考{int(number):,}位"
