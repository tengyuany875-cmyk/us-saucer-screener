"""Validated, fingerprinted settings for the isolated Daily Pullback experiment."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("pullback_config.json")

# Keys live in code, while values live in JSON: a typo in the editable default
# must not silently become its own validation schema.
SCHEMA_KEYS = {
    "data": "history_calendar_days minimum_history batch_size threads attempts max_failed_batches pause_seconds retry_seconds timeout_seconds close_delay_minutes minimum_success_rate universe_cache_days earnings_cache_hours",
    "indicators": "ema sma_fast sma_slow slope_days rsi atr relative_strength_days liquidity_days",
    "rules": "minimum_price minimum_dollar_volume atr_pct_min atr_pct_max minimum_relative_strength peak_lookback pullback_days_min pullback_days_max pullback_days_ideal depth_atr_min depth_atr_max depth_atr_ideal drawdown_min drawdown_max support_distance_atr rsi_min rsi_max close_location_min maximum_extension_atr stop_buffer_atr risk_atr_min risk_atr_max minimum_reward_risk confirmed_score_min forming_score_min maximum_selected holding_sessions",
    "scoring": "baseline_fraction trend_slope_full trend_spread_full relative_strength_full liquidity_full reward_risk_full breakout_atr_full rsi_increase_full rebound_volume_min rebound_volume_full",
    "tracking": "cost_bps_per_side",
}


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    template = {section: {key: 0 for key in keys.split()} for section, keys in SCHEMA_KEYS.items()}
    template["strategy_version"] = ""
    template["scoring"]["weights"] = dict.fromkeys("trend relative_strength pullback liquidity reward_risk rebound".split(), 0)
    template["scoring"]["rebound_weights"] = dict.fromkeys("breakout close_location rsi volume".split(), 0)
    template["tracking"]["horizons"] = []

    def validate_keys(value, expected, prefix=""):
        if isinstance(expected, dict):
            if not isinstance(value, dict) or set(value) != set(expected):
                raise ValueError(f"Unknown or missing config keys: {prefix}")
            for key in expected:
                validate_keys(value[key], expected[key], f"{prefix}.{key}")
        elif isinstance(expected, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid nonnegative number: {prefix}")
    validate_keys(config, template)
    d, i, r, s, tr = (config[key] for key in ("data", "indicators", "rules", "scoring", "tracking"))
    integer_groups = [(d, ["history_calendar_days", "minimum_history", "batch_size", "threads", "attempts", "max_failed_batches", "timeout_seconds", "universe_cache_days"]),
                      (i, list(i)), (r, ["peak_lookback", "pullback_days_min", "pullback_days_max", "pullback_days_ideal", "maximum_selected", "holding_sessions"])]
    for values, keys in integer_groups:
        if any(not isinstance(values[k], int) or isinstance(values[k], bool) or values[k] < 1 for k in keys):
            raise ValueError("Periods, counts, and attempts must be positive integers")
    if not isinstance(config["strategy_version"], str) or not config["strategy_version"].strip():
        raise ValueError("strategy_version is required")
    if d["minimum_history"] < max(i["sma_slow"], i["sma_fast"] + i["slope_days"], i["relative_strength_days"] + 1, i["rsi"] + 1, i["atr"], i["ema"], i["liquidity_days"] + 1, r["peak_lookback"] + i["atr"]):
        raise ValueError("minimum_history is shorter than indicator warmup")
    if d["history_calendar_days"] <= d["minimum_history"] or r["minimum_dollar_volume"] <= 0:
        raise ValueError("History window or liquidity divisor is invalid")
    if not 0 < d["minimum_success_rate"] <= 1 or not 0 < s["baseline_fraction"] < 1:
        raise ValueError("Invalid rate")
    for low, high in [("atr_pct_min", "atr_pct_max"), ("pullback_days_min", "pullback_days_max"), ("depth_atr_min", "depth_atr_max"), ("drawdown_min", "drawdown_max"), ("rsi_min", "rsi_max"), ("risk_atr_min", "risk_atr_max")]:
        if not r[low] < r[high]:
            raise ValueError(f"Expected {low} < {high}")
    if not r["pullback_days_min"] <= r["pullback_days_ideal"] <= r["pullback_days_max"] < r["peak_lookback"]:
        raise ValueError("Invalid pullback duration")
    if not r["depth_atr_min"] < r["depth_atr_ideal"] < r["depth_atr_max"]:
        raise ValueError("Invalid ideal depth")
    if sum(s["weights"].values()) != 100 or sum(s["rebound_weights"].values()) != s["weights"]["rebound"]:
        raise ValueError("Weights must total 100, with consistent rebound weights")
    if not 0 <= r["close_location_min"] < 1 or r["maximum_selected"] > 20 or not 0 <= r["forming_score_min"] <= 100 - s["weights"]["rebound"] or not 0 <= r["confirmed_score_min"] <= 100:
        raise ValueError("Invalid selection thresholds")
    if s["reward_risk_full"] <= r["minimum_reward_risk"] or s["liquidity_full"] <= r["minimum_dollar_volume"] or s["rebound_volume_full"] <= s["rebound_volume_min"]:
        raise ValueError("Scoring scale must exceed its minimum")
    if any(s[k] <= 0 for k in ("trend_slope_full", "trend_spread_full", "relative_strength_full", "breakout_atr_full", "rsi_increase_full")) or r["support_distance_atr"] <= 0:
        raise ValueError("Scoring divisors must be positive")
    if not isinstance(tr["horizons"], list) or not tr["horizons"] or any(not isinstance(h, int) or isinstance(h, bool) or h < 1 for h in tr["horizons"]) or len(set(tr["horizons"])) != len(tr["horizons"]) or tr["cost_bps_per_side"] >= 10000:
        raise ValueError("Invalid tracking settings")
    return config
