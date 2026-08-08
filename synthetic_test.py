import numpy as np
import pandas as pd

from screener import analyse_ticker

rng = np.random.default_rng(42)
base = np.linspace(30, 70, 240)
x = np.linspace(-1, 1, 30)
saucer = 58 + 13 * (x ** 2) + np.linspace(0, 4, 30)
prices = np.concatenate([base, saucer]) * (1 + rng.normal(0, 0.006, 270))
volume = np.full(270, 1_200_000.0)
volume[-4:] *= 1.45
index = pd.date_range("2021-01-01", periods=270, freq="W-FRI")
frame = pd.DataFrame({"Close": prices, "Volume": volume}, index=index)
result = analyse_ticker("TEST", frame)
assert result is not None, "人工ソーサーを検出できませんでした"
print("Synthetic test passed")
print(result)
