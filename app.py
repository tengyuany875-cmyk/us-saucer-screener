from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import html
import time

import pandas as pd
import yfinance as yf

from config import CONFIG
from screener import analyse_ticker, sort_results
from universe import fetch_us_equity_universe


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


def download_batch(tickers: list[str]) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        period=CONFIG.history_period,
        interval=CONFIG.interval,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
        timeout=30,
    )


def render_html(results: pd.DataFrame, output_path: Path, total_count: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if results.empty:
        body = "<p>条件に合う候補はありませんでした。</p>"
    else:
        columns = [
            "Ticker", "Score", "Stage", "Price", "DrawdownToBottomPct",
            "RecoveryPct", "BottomWeeks", "VolumeRatio4wVs12w",
            "DistanceSMA40Pct", "DistanceSMA200wPct", "Distance52wHighPct", "LastDate"
        ]
        labels = {
            "Ticker": "銘柄", "Score": "点数", "Stage": "段階", "Price": "終値",
            "DrawdownToBottomPct": "底までの下落%", "RecoveryPct": "回復率%",
            "BottomWeeks": "底値圏週数", "VolumeRatio4wVs12w": "出来高比",
            "DistanceSMA40Pct": "40週線乖離%", "DistanceSMA200wPct": "200週線乖離%",
            "Distance52wHighPct": "52週高値乖離%", "LastDate": "基準日"
        }
        rows = []
        for row_index, (_, row) in enumerate(results.iterrows()):
            ticker = html.escape(str(row["Ticker"]))
            exchange = html.escape(str(row["Exchange"]))
            cells = [f'<td><a href="https://www.tradingview.com/chart/?symbol={exchange}%3A{ticker}" target="_blank">{ticker}</a> <a href="https://finance.yahoo.com/quote/{ticker}" target="_blank">Yahoo</a></td>']
            for col in columns[1:]:
                value = "" if pd.isna(row[col]) else html.escape(str(row[col]))
                cells.append(f'<td data-label="{labels[col]}">{value}</td>')
            row_style = "" if row_index < 20 else ' style="display:none"'
            score = "" if pd.isna(row["Score"]) else str(row["Score"])
            stage = html.escape(str(row["Stage"]))
            exchange_attr = html.escape(str(row["Exchange"]))
            volume_ratio = "" if pd.isna(row["VolumeRatio4wVs12w"]) else str(row["VolumeRatio4wVs12w"])
            recovery = "" if pd.isna(row["RecoveryPct"]) else str(row["RecoveryPct"])
            bottom_weeks = "" if pd.isna(row["BottomWeeks"]) else str(row["BottomWeeks"])
            rows.append(
                f'<tr data-row-index="{row_index}" data-score="{score}" '
                f'data-stage="{stage}" data-exchange="{exchange_attr}" '
                f'data-volume="{volume_ratio}" data-recovery="{recovery}" '
                f'data-bottom-weeks="{bottom_weeks}"{row_style}>'
                + "".join(cells) + "</tr>"
            )
        headers = "".join(f"<th>{labels[c]}</th>" for c in columns)
        body = f"""
<div class="controls">
  <label>並び替え
    <select id="sort-select" onchange="applyControls()">
      <option value="score">点数が高い順</option>
      <option value="volume">出来高比が高い順</option>
      <option value="recovery">回復率が高い順</option>
      <option value="bottom-weeks">底値圏週数が多い順</option>
    </select>
  </label>

  <label>段階
    <select id="stage-filter" onchange="applyControls()">
      <option value="all">すべて</option>
      <option value="ブレイク接近">ブレイク接近</option>
      <option value="右肩上がり確認">右肩上がり確認</option>
      <option value="底固め中">底固め中</option>
    </select>
  </label>

  <label>最低点数
    <select id="score-filter" onchange="applyControls()">
      <option value="0">指定なし</option>
      <option value="90">90点以上</option>
      <option value="80">80点以上</option>
      <option value="70">70点以上</option>
      <option value="60">60点以上</option>
    </select>
  </label>

  <label>取引所
    <select id="exchange-filter" onchange="applyControls()">
      <option value="all">すべて</option>
      <option value="NASDAQ">NASDAQ</option>
      <option value="NYSE">NYSE</option>
    </select>
  </label>
</div>

<div class="limit-buttons">
  <button type="button" onclick="setLimit(20)">20件</button>
  <button type="button" onclick="setLimit(50)">50件</button>
  <button type="button" onclick="setLimit(100)">100件</button>
  <button type="button" onclick="setLimit('all')">全件</button>
</div>

<div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>

<script>
let currentLimit = 20;

function numberValue(row, name) {{
  const value = parseFloat(row.dataset[name]);
  return Number.isFinite(value) ? value : -Infinity;
}}

function setLimit(limit) {{
  currentLimit = limit;
  applyControls();
}}

function applyControls() {{
  const tbody = document.querySelector("tbody");
  const rows = Array.from(tbody.querySelectorAll("tr"));

  const sortBy = document.getElementById("sort-select").value;
  const stage = document.getElementById("stage-filter").value;
  const minScore = parseFloat(document.getElementById("score-filter").value);
  const exchange = document.getElementById("exchange-filter").value;

  const sortKeys = {{
    "score": "score",
    "volume": "volume",
    "recovery": "recovery",
    "bottom-weeks": "bottomWeeks"
  }};

  const key = sortKeys[sortBy];

  rows.sort((a, b) => {{
    return numberValue(b, key) - numberValue(a, key);
  }});

  rows.forEach(row => tbody.appendChild(row));

  let matched = 0;
  let shown = 0;

  rows.forEach(row => {{
    const score = numberValue(row, "score");

    const matchesStage =
      stage === "all" || row.dataset.stage === stage;

    const matchesScore = score >= minScore;

    const matchesExchange =
      exchange === "all" || row.dataset.exchange === exchange;

    const matches =
      matchesStage && matchesScore && matchesExchange;

    if (matches) {{
      matched += 1;

      if (currentLimit === "all" || shown < currentLimit) {{
        row.style.display = "";
        shown += 1;
      }} else {{
        row.style.display = "none";
      }}
    }} else {{
      row.style.display = "none";
    }}
  }});

  document.getElementById("matched-count").textContent = matched;
  document.getElementById("visible-count").textContent = shown;
}}

applyControls();
</script>
"""

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>米国株ソーサーボトム候補</title>
<style>
:root{{color-scheme:light dark}}body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:18px;line-height:1.5;background:Canvas;color:CanvasText}}main{{max-width:1500px;margin:auto}}h1{{font-size:1.35rem;margin-bottom:4px}}.meta{{font-size:.9rem;margin-bottom:12px}}.generated{{opacity:.65}}.count-badges{{display:inline-flex;gap:6px;margin-left:8px}}.count-badges strong{{padding:3px 8px;border-radius:999px;background:#e6f0ff;color:#173b70;font-size:.9rem}}.controls{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-bottom:8px}}.controls label{{font-size:.85rem;font-weight:600}}.controls select{{display:block;width:100%;margin-top:2px}}.table-wrap{{overflow-x:auto;border:1px solid #8885;border-radius:10px}}table{{border-collapse:collapse;width:100%;min-width:1050px}}th,td{{padding:10px 8px;border-bottom:1px solid #8884;text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:Canvas}}th:first-child,td:first-child,th:nth-child(3),td:nth-child(3){{text-align:left}}tbody tr:hover{{background:#8882}}a{{font-weight:700}}.note{{margin-top:14px;padding:12px;border:1px solid #8885;border-radius:10px}}@media(max-width:700px){{body{{padding:6px}}.meta{{margin-bottom:10px}}.generated{{display:block;margin-bottom:5px}}.count-badges{{margin-left:0}}.controls{{grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 8px}}.controls label{{font-size:.82rem}}.controls select{{font-size:.9rem}}.table-wrap{{border:0;overflow:visible}}table,thead,tbody,th{{display:block;min-width:0}}thead{{display:none}}tr{{display:flex;flex-wrap:wrap;border:1px solid #8885;border-radius:9px;margin-bottom:5px;padding:4px}}td{{display:block;width:50%;box-sizing:border-box;border:0;padding:1px 4px;font-size:.87rem;line-height:1.25;text-align:right!important}}td::before{{content:attr(data-label);float:left;opacity:.68}}td:first-child{{width:100%;font-size:1.05rem;line-height:1.3;padding:1px 4px 2px}}}}
</style></head><body><main><h1>米国株・週足ソーサーボトム候補</h1>
<div class="meta">
<span class="generated">生成日時: {generated}</span>
<span class="count-badges">
<strong>条件通過 {total_count}件</strong>
<strong>絞り込み該当 <span id="matched-count">{len(results)}</span>件</strong>
<strong>表示 <span id="visible-count">{min(20, len(results))}</span>件</strong>
</span>
</div>
{body}
<div class="note">候補抽出用です。決算、業績、希薄化、ニュースを確認してから投資判断してください。40週線はおおむね200取引日線に相当します。</div>
</main></body></html>'''
    output_path.write_text(page, encoding="utf-8")


def run_scan(mode: str, refresh_universe: bool) -> pd.DataFrame:
    universe = fetch_us_equity_universe(refresh=refresh_universe)
    limit = {"quick": 250, "medium": 1000, "full": None}[mode]
    if limit:
        universe = universe.head(limit)

    tickers = universe["Ticker"].tolist()
    name_map = universe.set_index("Ticker")["Name"].to_dict()
    exchange_map = universe.set_index("Ticker")["Exchange"].to_dict()
    results, failures = [], []
    total_batches = (len(tickers) + CONFIG.batch_size - 1) // CONFIG.batch_size

    for batch_no, start in enumerate(range(0, len(tickers), CONFIG.batch_size), start=1):
        batch = tickers[start:start + CONFIG.batch_size]
        print(f"[{batch_no}/{total_batches}] {batch[0]} ～ {batch[-1]} を取得中")
        try:
            downloaded = download_batch(batch)
        except Exception as exc:
            print(f"  取得失敗: {exc}")
            failures.extend(batch)
            time.sleep(CONFIG.request_pause_seconds * 2)
            continue

        for ticker in batch:
            try:
                result = analyse_ticker(ticker, extract_ticker_frame(downloaded, ticker, len(batch)))
            except Exception as exc:
                print(f"  {ticker}: 判定エラー {exc}")
                failures.append(ticker)
                continue
            if result:
                result["Name"] = name_map.get(ticker, "")
                result["Exchange"] = exchange_map.get(ticker, "")
                results.append(result)
        time.sleep(CONFIG.request_pause_seconds)

    print(f"\n条件通過総数: {len(results)}")
    all_results = sort_results(results)
    final = all_results.head(CONFIG.top_n)
    Path("output").mkdir(exist_ok=True)
    all_results.to_csv("output/results.csv", index=False)
    render_html(all_results, Path("output/results.html"), len(results))
    if failures:
        pd.Series(sorted(set(failures)), name="Ticker").to_csv("output/failures.csv", index=False)
    print(f"\n候補数: {len(final)}")
    print("出力: output/results.csv / output/results.html")
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="米国株の週足ソーサーボトム候補を抽出します。")
    parser.add_argument("--mode", choices=["quick", "medium", "full"], default="quick")
    parser.add_argument("--refresh-universe", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_scan(args.mode, args.refresh_universe)
