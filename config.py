from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenerConfig:
    pattern_weeks: int = 30
    minimum_history_weeks: int = 80
    minimum_price: float = 3.0
    minimum_average_dollar_volume: float = 5_000_000.0
    max_below_sma40: float = 0.10
    max_below_sma200w: float = 0.30
    minimum_drawdown: float = 0.08
    maximum_drawdown: float = 0.65
    minimum_bottom_weeks: int = 4
    bottom_band: float = 0.08
    minimum_score: float = 60.0
    top_n: int = 20
    history_period: str = "10y"
    interval: str = "1wk"
    batch_size: int = 50
    request_pause_seconds: float = 1.0


CONFIG = ScreenerConfig()
