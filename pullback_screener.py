"""Pure, as-of daily pullback classification. No fetching, files, or future bars."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


def wilder(series: pd.Series, period: int) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.first_valid_index()
    if valid is None:
        return out
    start = series.index.get_loc(valid)
    if len(series) - start < period:
        return out
    seed = start + period - 1
    out.iloc[seed] = series.iloc[start:seed + 1].mean()
    for pos in range(seed + 1, len(series)):
        out.iloc[pos] = (out.iloc[pos - 1] * (period - 1) + series.iloc[pos]) / period
    return out


def indicators(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = frame.copy()
    i = config["indicators"]
    close = data["Close"]
    data["EMA"] = close.ewm(span=i["ema"], adjust=False, min_periods=i["ema"]).mean()
    data["SMAFast"] = close.rolling(i["sma_fast"]).mean()
    data["SMASlow"] = close.rolling(i["sma_slow"]).mean()
    data["FastSlope"] = data["SMAFast"] / data["SMAFast"].shift(i["slope_days"]) - 1
    change = close.diff()
    gain, loss = wilder(change.clip(lower=0), i["rsi"]), wilder(-change.clip(upper=0), i["rsi"])
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    data["RSI"] = rsi.mask((loss == 0) & (gain > 0), 100).mask((gain == 0) & (loss > 0), 0).mask((gain == 0) & (loss == 0), 50)
    previous = close.shift()
    true_range = pd.concat([data["High"] - data["Low"], (data["High"] - previous).abs(), (data["Low"] - previous).abs()], axis=1).max(axis=1)
    data["ATR"] = wilder(true_range, i["atr"])
    # RawClose is kept separate from dividend-adjusted indicator prices.
    data["DollarVolume"] = (data.get("RawClose", close) * data["Volume"]).rolling(i["liquidity_days"]).mean()
    data["PriorVolume"] = data["Volume"].shift().rolling(i["liquidity_days"]).mean()
    return data


def _base_failure(data: pd.DataFrame, spy: pd.DataFrame, config: dict) -> list[str]:
    row, r, n = data.iloc[-1], config["rules"], config["indicators"]["relative_strength_days"]
    required = ["Close", "EMA", "SMAFast", "SMASlow", "FastSlope", "ATR", "RSI", "DollarVolume"]
    if not all(math.isfinite(float(row[key])) for key in required) or row["ATR"] <= 0:
        return ["indicator_unavailable"]
    failures = []
    raw_price = float(row.get("RawClose", row["Close"]))
    if raw_price < r["minimum_price"]:
        failures.append("price")
    if row["DollarVolume"] < r["minimum_dollar_volume"]:
        failures.append("liquidity")
    if not row["Close"] > row["SMAFast"] > row["SMASlow"] or row["FastSlope"] <= 0:
        failures.append("trend")
    if not r["atr_pct_min"] <= row["ATR"] / row["Close"] <= r["atr_pct_max"]:
        failures.append("volatility")
    if len(data) <= n or not data.index[-n - 1:].isin(spy.index).all():
        failures.append("benchmark_alignment")
    else:
        aligned = spy.reindex(data.index)
        excess = (row["Close"] / data["Close"].iloc[-n - 1]) / (aligned["Close"].iloc[-1] / aligned["Close"].iloc[-n - 1]) - 1
        if not math.isfinite(excess) or excess <= r["minimum_relative_strength"]:
            failures.append("relative_strength")
    return failures


def _setup(data: pd.DataFrame, spy: pd.DataFrame, config: dict) -> tuple[dict | None, list[str]]:
    failures = _base_failure(data, spy, config)
    if failures:
        return None, failures
    r = config["rules"]
    highs = data["High"].iloc[-r["peak_lookback"] - 1:-1]
    peak_value = highs.max()
    peak_date = highs[highs.eq(peak_value)].index[-1]
    pos = data.index.get_loc(peak_date)
    age = len(data) - 1 - pos
    if not r["pullback_days_min"] <= age <= r["pullback_days_max"]:
        return None, ["pullback_duration"]
    atr = float(data["ATR"].iloc[pos])
    if not math.isfinite(atr) or atr <= 0:
        return None, ["setup_atr"]
    pullback = data.iloc[pos + 1:]
    low = float(pullback["Low"].min())
    depth_atr, drawdown = (peak_value - low) / atr, 1 - low / peak_value
    if not r["depth_atr_min"] <= depth_atr <= r["depth_atr_max"] or not r["drawdown_min"] <= drawdown <= r["drawdown_max"]:
        failures.append("pullback_depth")
    distance = abs(float(data["Close"].iloc[-1] - data["EMA"].iloc[-1])) / atr
    if distance > r["support_distance_atr"]:
        failures.append("support_distance")
    if not pullback["Close"].gt(pullback["SMAFast"]).all():
        failures.append("support_broken")
    rsi_min = float(pullback["RSI"].min())
    if not r["rsi_min"] <= rsi_min <= r["rsi_max"]:
        failures.append("pullback_rsi")
    if failures:
        return None, failures
    return {"peak_date": peak_date.date().isoformat(), "peak": float(peak_value), "low": low, "atr": atr,
            "age": age, "depth_atr": depth_atr, "drawdown": drawdown, "support_distance": distance,
            "rsi_min": rsi_min}, []


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _triangle(value: float, low: float, ideal: float, high: float) -> float:
    if value <= ideal:
        return _clip((value - low) / (ideal - low)) if ideal > low else 1.0
    return _clip((high - value) / (high - ideal)) if high > ideal else 1.0


def analyse_daily(ticker: str, history: pd.DataFrame, benchmark: pd.DataFrame, config: dict, as_of: str) -> dict:
    """Input OHLC are consistently adjusted; RawClose supplies nominal-price/ADV gates."""
    cutoff = pd.Timestamp(as_of)
    frame = history.loc[history.index <= cutoff].copy()
    spy_frame = benchmark.loc[benchmark.index <= cutoff].copy()
    if len(frame) < config["data"]["minimum_history"] or spy_frame.empty:
        return {"status": "no_signal", "reason_codes": ["insufficient_history"]}
    if frame.index[-1] != cutoff or spy_frame.index[-1] != cutoff:
        return {"status": "no_signal", "reason_codes": ["stale"]}
    data, spy = indicators(frame, config), spy_frame.copy()
    spy["SMAFast"] = spy["Close"].rolling(config["indicators"]["sma_fast"]).mean()
    failures = _base_failure(data, spy, config)
    if failures:
        return {"status": "no_signal", "reason_codes": failures}
    r, s, row = config["rules"], config["scoring"], data.iloc[-1]
    previous = data.iloc[-2]
    price = float(row["Close"])
    location = (price - row["Low"]) / (row["High"] - row["Low"]) if row["High"] > row["Low"] else 0.0
    prior_setup = _setup(data.iloc[:-1], spy, config)[0] if len(data) - 1 >= config["data"]["minimum_history"] else None
    confirmation = (prior_setup is not None and price > previous["High"] and price > row["EMA"]
                    and price > row["Open"] and location >= r["close_location_min"])
    if confirmation:
        setup, stage = prior_setup, "confirmed"
    else:
        setup, failures = _setup(data, spy, config)
        if setup is None:
            return {"status": "no_signal", "reason_codes": failures}
        stage = "forming"
    atr = setup["atr"]
    extension = (price - row["EMA"]) / atr
    stop = setup["low"] - r["stop_buffer_atr"] * atr
    risk = price - stop
    reward_risk = (setup["peak"] - price) / risk if risk > 0 else -1
    if extension > r["maximum_extension_atr"] or not r["risk_atr_min"] <= risk / atr <= r["risk_atr_max"] or reward_risk < r["minimum_reward_risk"]:
        return {"status": "no_signal", "reason_codes": ["entry_risk_or_headroom"]}
    n = config["indicators"]["relative_strength_days"]
    aligned = spy.reindex(data.index)
    excess = (price / data["Close"].iloc[-n - 1]) / (aligned["Close"].iloc[-1] / aligned["Close"].iloc[-n - 1]) - 1
    w, baseline = s["weights"], s["baseline_fraction"]
    def quality(key: str, strength: float) -> float:
        return w[key] * (baseline + (1 - baseline) * _clip(strength))
    pullback_fit = (_triangle(setup["depth_atr"], r["depth_atr_min"], r["depth_atr_ideal"], r["depth_atr_max"])
                    + _triangle(setup["age"], r["pullback_days_min"], r["pullback_days_ideal"], r["pullback_days_max"])
                    + 1 - _clip(setup["support_distance"] / r["support_distance_atr"])) / 3
    breakdown = {
        "trend": quality("trend", (_clip(row["FastSlope"] / s["trend_slope_full"]) + _clip((row["SMAFast"] / row["SMASlow"] - 1) / s["trend_spread_full"])) / 2),
        "relative_strength": quality("relative_strength", excess / s["relative_strength_full"]),
        "pullback": quality("pullback", pullback_fit),
        "liquidity": quality("liquidity", math.log(row["DollarVolume"] / r["minimum_dollar_volume"]) / math.log(s["liquidity_full"] / r["minimum_dollar_volume"])),
        "reward_risk": quality("reward_risk", (reward_risk - r["minimum_reward_risk"]) / (s["reward_risk_full"] - r["minimum_reward_risk"])),
        "rebound": 0.0,
    }
    volume_ratio = float(row["Volume"] / row["PriorVolume"]) if row["PriorVolume"] > 0 else 0.0
    if stage == "confirmed":
        rw = s["rebound_weights"]
        breakdown["rebound"] = (rw["breakout"] * (baseline + (1 - baseline) * _clip((price - previous["High"]) / atr / s["breakout_atr_full"]))
            + rw["close_location"] * (baseline + (1 - baseline) * _clip((location - r["close_location_min"]) / (1 - r["close_location_min"])))
            + rw["rsi"] * _clip((row["RSI"] - previous["RSI"]) / s["rsi_increase_full"])
            + rw["volume"] * _clip((volume_ratio - s["rebound_volume_min"]) / (s["rebound_volume_full"] - s["rebound_volume_min"])))
    # Select using unrounded values; CSV components add up exactly at two decimals.
    score = sum(breakdown.values())
    threshold = r["confirmed_score_min"] if stage == "confirmed" else r["forming_score_min"]
    if score < threshold:
        return {"status": "no_signal", "reason_codes": ["score"], "raw_score": score, "stage": stage}
    components = {key: round(float(value), 2) for key, value in breakdown.items()}
    i = config["indicators"]
    reasons = [f"終値>{i['sma_fast']}日線>{i['sma_slow']}日線、{i['sma_fast']}日線上昇率 {row['FastSlope'] * 100:.2f}%",
               f"SPY比{i['relative_strength_days']}営業日リターン {excess * 100:+.2f}%",
               f"{setup['age']}営業日の調整、深さ {setup['depth_atr']:.2f} ATR / {setup['drawdown'] * 100:.2f}%",
               f"調整中RSI安値 {setup['rsi_min']:.1f}、高値までの余地/想定リスク {reward_risk:.2f}"]
    if stage == "confirmed":
        reasons.append(f"前日高値突破・{i['ema']}日EMA上・陽線、終値位置 {location * 100:.1f}%、出来高比 {volume_ratio:.2f}")
    else:
        reasons.append(f"{i['ema']}日EMA付近で押し目形成中。反発確認条件は未成立")
    record = {
        "Ticker": ticker, "MarketDate": as_of, "Stage": stage, "StageJA": "反発確認済み" if stage == "confirmed" else "押し目形成中",
        "Score": round(sum(components.values()), 2), "Price": float(row.get("RawClose", price)), "AdjustedClose": price,
        "SetupPeakDate": setup["peak_date"], "SetupHigh": setup["peak"], "SetupLow": setup["low"], "SetupATR": atr,
        "PullbackDays": setup["age"], "DepthATR": setup["depth_atr"], "DrawdownPct": setup["drawdown"] * 100,
        "EMA": float(row["EMA"]), "SMAFast": float(row["SMAFast"]), "SMASlow": float(row["SMASlow"]),
        "RSI": float(row["RSI"]), "ATR": float(row["ATR"]), "ATRPercent": float(row["ATR"] / price * 100),
        "RelativeStrengthPct": float(excess * 100), "AverageDollarVolume": float(row["DollarVolume"]),
        "VolumeRatio": volume_ratio, "CloseLocation": float(location), "ReferenceStop": stop,
        "RewardRisk": reward_risk, "SPYAboveSMA": bool(spy["Close"].iloc[-1] > spy["SMAFast"].iloc[-1]),
        "Reasons": " / ".join(reasons), "ScoreBreakdown": json.dumps(components, ensure_ascii=False, sort_keys=True),
    }
    record.update({f"Score_{key}": value for key, value in components.items()})
    return {"status": "signal", "record": record, "reason_codes": []}
