from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, ScreenerConfig


def _log_slope(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or np.any(array <= 0) or np.any(~np.isfinite(array)):
        return float("nan")
    return float(np.polyfit(np.arange(len(array)), np.log(array), 1)[0])


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b and np.isfinite(b) else float("nan")


def analyse_ticker(
    ticker: str,
    history: pd.DataFrame,
    config: ScreenerConfig = CONFIG,
) -> dict[str, Any] | None:
    if not {"Close", "Volume"}.issubset(history.columns):
        return None

    data = history.copy().sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")
    data = data.dropna(subset=["Close", "Volume"])
    data = data[data["Close"] > 0]
    if len(data) < config.minimum_history_weeks:
        return None

    close, volume = data["Close"], data["Volume"]
    latest = float(close.iloc[-1])
    dollar_volume = float((close.tail(8) * volume.tail(8)).mean())
    if latest < config.minimum_price or dollar_volume < config.minimum_average_dollar_volume:
        return None

    sma10 = float(close.rolling(10).mean().iloc[-1])
    sma40 = float(close.rolling(40).mean().iloc[-1])
    sma200w = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    d10 = _ratio(latest, sma10) - 1
    d40 = _ratio(latest, sma40) - 1
    d200 = _ratio(latest, sma200w) - 1 if np.isfinite(sma200w) else np.nan
    if d40 < -config.max_below_sma40:
        return None
    if np.isfinite(d200) and d200 < -config.max_below_sma200w:
        return None

    window = close.tail(config.pattern_weeks)
    n = len(window)
    a, b = n // 3, 2 * n // 3
    left, middle, right = window.iloc[:a], window.iloc[a:b], window.iloc[b:]
    left_slope, mid_slope, right_slope = map(_log_slope, [left, middle, right])

    bottom = float(window.min())
    bottom_pos = int(np.argmin(window.to_numpy()))
    left_rim = float(left.max())
    drawdown = 1 - _ratio(bottom, left_rim)
    rim_range = left_rim - bottom
    recovery = (latest - bottom) / rim_range if rim_range > 0 else np.nan
    bottom_weeks = int((window <= bottom * (1 + config.bottom_band)).sum())
    curvature = float(np.polyfit(np.linspace(-1, 1, n), np.log(window.to_numpy()), 2)[0])

    mid_vol = float(middle.pct_change().dropna().std())
    outer = pd.concat([left.pct_change().dropna(), right.pct_change().dropna()])
    outer_vol = float(outer.std())
    compression = _ratio(mid_vol, outer_vol)

    recent_volume = float(volume.tail(4).mean())
    prior_volume = float(volume.iloc[-16:-4].mean())
    volume_ratio = _ratio(recent_volume, prior_volume)
    high52 = float(close.tail(52).max())
    d52 = _ratio(latest, high52) - 1

    pattern_ok = (
        np.isfinite(left_slope) and np.isfinite(mid_slope) and np.isfinite(right_slope)
        and left_slope < -0.001
        and abs(mid_slope) < 0.008
        and right_slope > 0.001
        and config.minimum_drawdown <= drawdown <= config.maximum_drawdown
        and int(n * 0.20) <= bottom_pos <= int(n * 0.80)
        and bottom_weeks >= config.minimum_bottom_weeks
        and curvature > 0
    )
    if not pattern_ok:
        return None

    score = 0.0
    score += min(6, max(0, (-left_slope - 0.001) / 0.008 * 6))
    score += min(6, max(0, (0.008 - abs(mid_slope)) / 0.008 * 6))
    score += min(6, max(0, (right_slope - 0.001) / 0.008 * 6))
    score += min(7, max(0, curvature / 0.08 * 7))
    score += min(5, bottom_weeks / 8 * 5)
    if np.isfinite(compression):
        score += min(5, max(0, (1.4 - compression) * 5))

    score += 7 if d10 >= 0 else max(0, 7 + d10 * 70)
    score += min(10, max(0, (d40 + 0.10) / 0.20 * 10))
    score += min(4, max(0, (d200 + 0.30) / 0.60 * 4)) if np.isfinite(d200) else 2
    if np.isfinite(recovery):
        score += min(4, max(0, recovery / 0.90 * 4))

    if np.isfinite(volume_ratio):
        score += min(10, max(0, (volume_ratio - 0.75) / 0.75 * 10))
    score += min(10, max(0, (np.log10(max(dollar_volume, 1)) - 6.5) / 2 * 10))

    score += 8 if 0.15 <= drawdown <= 0.45 else max(0, 8 - abs(drawdown - 0.30) * 20)
    score += 6 if -0.35 <= d52 <= -0.05 else max(0, 6 - abs(d52 + 0.20) * 15)
    if np.isfinite(recovery):
        score += 6 if 0.55 <= recovery <= 1.15 else max(0, 6 - abs(recovery - 0.85) * 8)

    score = round(min(score, 100), 1)
    if score < config.minimum_score:
        return None

    stage = "ブレイク接近" if recovery >= 0.85 and d10 >= 0 else (
        "右肩上がり確認" if recovery >= 0.60 else "底固め中"
    )

    return {
        "Ticker": ticker,
        "Score": score,
        "Stage": stage,
        "Price": round(latest, 2),
        "DrawdownToBottomPct": round(drawdown * 100, 1),
        "RecoveryPct": round(recovery * 100, 1),
        "BottomWeeks": bottom_weeks,
        "VolumeRatio4wVs12w": round(volume_ratio, 2),
        "AverageDollarVolume": round(dollar_volume),
        "DistanceSMA10Pct": round(d10 * 100, 1),
        "DistanceSMA40Pct": round(d40 * 100, 1),
        "DistanceSMA200wPct": round(d200 * 100, 1) if np.isfinite(d200) else np.nan,
        "Distance52wHighPct": round(d52 * 100, 1),
        "LastDate": str(data.index[-1].date()),
    }


def sort_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(
        ["Score", "VolumeRatio4wVs12w"], ascending=[False, False]
    ).reset_index(drop=True)
