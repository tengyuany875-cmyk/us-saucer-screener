from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import html
import time
from urllib.parse import quote

import pandas as pd
import yfinance as yf

from config import CONFIG
from market_caps import add_market_caps, fetch_snapshot, format_market_cap, format_market_rank
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

    current_market_date = "—"
    previous_count = None
    candidate_delta = None

    if not results.empty and "LastDate" in results.columns:
        market_dates = sorted(results["LastDate"].dropna().astype(str).unique())
        if market_dates:
            current_market_date = market_dates[-1]

    history_path = Path("history/weekly_tracking.csv")
    if current_market_date != "—" and history_path.exists():
        try:
            tracking = pd.read_csv(history_path)
            tracking["MarketDate"] = tracking["MarketDate"].astype(str)

            previous_dates = sorted(
                d for d in tracking["MarketDate"].dropna().unique()
                if d < current_market_date
            )

            if previous_dates:
                previous_date = previous_dates[-1]
                previous = tracking[
                    tracking["MarketDate"].eq(previous_date)
                ]
                previous_count = int(
                    previous["InScreener"]
                    .astype(str)
                    .str.lower()
                    .eq("true")
                    .sum()
                )
                candidate_delta = total_count - previous_count
        except Exception:
            pass

    delta_text = "—" if candidate_delta is None else f"{candidate_delta:+d}"
    if results.empty:
        body = "<p>条件に合う候補はありませんでした。</p>"
    else:
        detail_columns = [
            "DrawdownToBottomPct", "RecoveryPct", "BottomWeeks", "VolumeRatio4wVs12w",
            "DistanceSMA40Pct", "DistanceSMA200wPct", "Distance52wHighPct", "LastDate"
        ]
        labels = {
            "DrawdownToBottomPct": "底までの下落%", "RecoveryPct": "回復率%",
            "BottomWeeks": "底値圏週数", "VolumeRatio4wVs12w": "出来高比",
            "DistanceSMA40Pct": "40週線乖離%", "DistanceSMA200wPct": "200週線乖離%",
            "Distance52wHighPct": "52週高値乖離%", "LastDate": "基準日"
        }
        cards = []
        for row_index, (_, row) in enumerate(results.iterrows()):
            ticker = html.escape(str(row["Ticker"]))
            exchange = html.escape(str(row["Exchange"]))
            details = []
            for col in detail_columns:
                value = "—" if pd.isna(row[col]) else html.escape(str(row[col]))
                details.append(f'<div><dt>{labels[col]}</dt><dd>{value}</dd></div>')
            hidden = "" if row_index < 20 else " hidden"
            score = "" if pd.isna(row["Score"]) else str(row["Score"])
            stage = html.escape(str(row["Stage"]))
            stage_class = {
                "ブレイク接近": "near-breakout",
                "右肩上がり確認": "recovering",
                "底固め中": "forming",
            }.get(str(row["Stage"]), "forming")
            price = "—" if pd.isna(row["Price"]) else f'${float(row["Price"]):,.2f}'
            market_cap = format_market_cap(row.get("MarketCapUSD"))
            market_rank = format_market_rank(row.get("USMarketCapRank"))
            cap_context = html.escape(
                "Nasdaq掲載の米国上場普通株（ETF・優先株等を除外）の銘柄単位の参考順位。"
                f'比較対象: {row.get("MarketCapReferenceCount", "不明")}銘柄。'
                f'取得日時（UTC）: {row.get("MarketCapFetchedAtUTC", "不明")}。'
                "米国企業以外の米国上場株も含みます。株価の基準日とは異なる場合があります。"
            )
            chart_symbol = quote(f'{row["Exchange"]}:{row["Ticker"]}', safe="")
            yahoo_symbol = quote(str(row["Ticker"]), safe="")
            volume_ratio = "" if pd.isna(row["VolumeRatio4wVs12w"]) else str(row["VolumeRatio4wVs12w"])
            recovery = "" if pd.isna(row["RecoveryPct"]) else str(row["RecoveryPct"])
            bottom_weeks = "" if pd.isna(row["BottomWeeks"]) else str(row["BottomWeeks"])
            cards.append(
                f'<article class="candidate" aria-labelledby="ticker-{row_index}" '
                f'data-row-index="{row_index}" data-score="{score}" '
                f'data-stage="{stage}" data-exchange="{exchange}" '
                f'data-volume="{volume_ratio}" data-recovery="{recovery}" '
                f'data-bottom-weeks="{bottom_weeks}"{hidden}>'
                f'<div class="candidate-heading"><div><h2 id="ticker-{row_index}">{ticker}</h2>'
                f'<span class="exchange">{exchange}</span></div>'
                f'<div class="score"><span class="score-label">点数</span>'
                f'<strong>{html.escape(score)}</strong><span class="score-unit"> / 100</span></div></div>'
                f'<div class="candidate-overview"><span class="stage {stage_class}">{stage}</span>'
                f'<div class="price"><span>株価</span><strong>{price}</strong></div></div>'
                f'<p class="market-cap-line" title="{cap_context}">'
                f'<span>時価総額：{market_cap}</span><span aria-hidden="true">｜</span>'
                f'<span>米国株内：{market_rank}</span></p>'
                f'<div class="candidate-actions"><a class="chart-link" '
                f'href="https://www.tradingview.com/chart/?symbol={chart_symbol}" '
                f'target="_blank" rel="noopener noreferrer" '
                f'aria-label="{ticker}のチャートをTradingViewで開く（新しいタブ）">'
                f'チャートを見る <span aria-hidden="true">↗</span></a>'
                f'<details class="candidate-details"><summary '
                f'aria-label="{ticker}の詳細"><span>詳細</span><span class="chevron" aria-hidden="true"></span>'
                f'</summary><div class="detail-body"><dl>{"".join(details)}</dl>'
                f'<a class="yahoo-link" href="https://finance.yahoo.com/quote/{yahoo_symbol}" '
                f'target="_blank" rel="noopener noreferrer" '
                f'aria-label="{ticker}をYahoo Financeで開く（新しいタブ）">'
                f'Yahoo Financeで見る <span aria-hidden="true">↗</span></a>'
                f'</div></details></div></article>'
            )
        body = f"""
<details class="filter-panel" id="filter-panel">
<summary><span>絞り込み・並び替え</span><span class="filter-status">
<span id="active-filter-count" hidden></span><span id="sort-label">点数順</span>
<span class="chevron" aria-hidden="true"></span></span></summary>
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

<label class="limit-control">表示件数
  <select id="limit-select" onchange="applyControls()">
    <option value="20">20件</option>
    <option value="50">50件</option>
    <option value="100">100件</option>
    <option value="all">全件</option>
  </select>
</label>
</details>

<div class="list-heading"><span>候補一覧</span><span role="status" aria-live="polite" aria-atomic="true">
該当 <strong id="matched-count">{len(results)}</strong>件<span class="count-divider">/</span>
表示 <strong id="visible-count">{min(20, len(results))}</strong>件</span></div>
<p class="empty-state" id="filter-empty" hidden>条件に合う候補はありません。絞り込み条件を変更してください。</p>
<div class="candidate-list" id="candidate-list">
{"".join(cards)}
</div>

<script>
function numberValue(row, name) {{
  const value = parseFloat(row.dataset[name]);
  return Number.isFinite(value) ? value : -Infinity;
}}

function applyControls() {{
  const list = document.getElementById("candidate-list");
  const rows = Array.from(list.querySelectorAll(".candidate"));

  const sortBy = document.getElementById("sort-select").value;
  const stage = document.getElementById("stage-filter").value;
  const minScore = parseFloat(document.getElementById("score-filter").value);
  const exchange = document.getElementById("exchange-filter").value;
  const limit = document.getElementById("limit-select").value;

  const sortKeys = {{
    "score": "score",
    "volume": "volume",
    "recovery": "recovery",
    "bottom-weeks": "bottomWeeks"
  }};

  const key = sortKeys[sortBy];

  rows.sort((a, b) => {{
    return numberValue(b, key) - numberValue(a, key)
      || numberValue(a, "rowIndex") - numberValue(b, "rowIndex");
  }});

  rows.forEach(row => list.appendChild(row));

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

      if (limit === "all" || shown < Number(limit)) {{
        row.hidden = false;
        shown += 1;
      }} else {{
        row.hidden = true;
      }}
    }} else {{
      row.hidden = true;
    }}
  }});

  document.getElementById("matched-count").textContent = matched;
  document.getElementById("visible-count").textContent = shown;
  document.getElementById("filter-empty").hidden = matched !== 0;

  const activeCount = Number(stage !== "all") + Number(minScore > 0) + Number(exchange !== "all");
  const badge = document.getElementById("active-filter-count");
  badge.hidden = activeCount === 0;
  badge.textContent = `${{activeCount}}条件`;
  document.getElementById("sort-label").textContent = {{
    "score": "点数順", "volume": "出来高比順", "recovery": "回復率順", "bottom-weeks": "底値圏週数順"
  }}[sortBy];
}}

applyControls();
</script>
"""

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>米国株ソーサーボトム候補</title>
<style>
:root {{
  color-scheme: light dark;
  --page: #f5f7f8; --surface: #fff; --ink: #192a32; --muted: #5a6b73;
  --line: #dce4e7; --soft: #edf2f4; --accent: #156753; --accent-soft: #e8f4ed;
}}
* {{ box-sizing: border-box; }}
[hidden] {{ display: none !important; }}
body {{ margin: 0; padding: 26px 16px; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", "Yu Gothic", sans-serif; line-height: 1.5; }}
main {{ max-width: 920px; margin: auto; }}
.page-heading {{ margin-bottom: 20px; }}
.eyebrow {{ margin: 0 0 4px; color: var(--muted); font-size: 13px; font-weight: 600; }}
h1 {{ font-size: 24px; line-height: 1.45; letter-spacing: -.025em; margin: 0 0 12px; }}
.market-date {{ color: var(--muted); font-size: 13px; margin: 0 0 7px; }}
.update-status {{ display: flex; align-items: baseline; gap: 12px; font-size: 13px; color: var(--muted); }}
.total-count {{ color: var(--ink); }}
.total-count strong {{ font-size: 26px; letter-spacing: -.03em; margin-right: 3px; }}
.filter-panel {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }}
summary {{ list-style: none; cursor: pointer; touch-action: manipulation; }}
summary::-webkit-details-marker {{ display: none; }}
.filter-panel > summary {{ min-height: 52px; padding: 12px 14px; display: flex;
  align-items: center; justify-content: space-between; gap: 8px; font-size: 14px; font-weight: 600; }}
.filter-status {{ display: flex; align-items: center; justify-content: flex-end; gap: 8px;
  color: var(--muted); font-size: 12px; font-weight: 400; }}
#active-filter-count {{ background: var(--accent-soft); color: var(--accent); padding: 2px 7px; border-radius: 6px; white-space: nowrap; }}
.chevron {{ width: 7px; height: 7px; border-right: 1.5px solid currentColor;
  border-bottom: 1.5px solid currentColor; transform: rotate(45deg); margin: -3px 2px 0 3px; flex-shrink: 0; }}
details[open] > summary .chevron {{ transform: rotate(225deg); margin-top: 3px; }}
.controls {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 4px 14px 14px; }}
.controls label, .limit-control {{ color: var(--muted); font-size: 12px; }}
select {{ display: block; width: 100%; min-width: 0; min-height: 44px; margin-top: 5px;
  padding: 8px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); color: var(--ink); font: inherit; font-size: 16px; }}
.limit-control {{ display: block; margin: 0 14px 14px; max-width: 180px; }}
.list-heading {{ display: flex; align-items: center; justify-content: space-between; gap: 8px;
  margin: 20px 2px 10px; font-size: 13px; font-weight: 600; }}
.list-heading [role="status"] {{ color: var(--muted); font-size: 12px; font-weight: 400; }}
.list-heading strong {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
.count-divider {{ padding: 0 6px; color: var(--line); }}
.candidate-list {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; align-items: start; }}
.candidate {{ min-width: 0; padding: 16px; border: 1px solid var(--line); border-radius: 14px; background: var(--surface); }}
.candidate-heading {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
h2 {{ margin: 0; font-size: 23px; line-height: 1.15; letter-spacing: -.02em; }}
.exchange {{ color: var(--muted); font-size: 11px; }}
.score {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
.score-label {{ display: block; color: var(--muted); font-size: 11px; line-height: 1.2; }}
.score strong {{ color: var(--accent); font-size: 26px; font-weight: 700; letter-spacing: -.04em; }}
.score-unit {{ color: var(--muted); font-size: 11px; }}
.candidate-overview {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 12px 0; }}
.stage {{ display: inline-block; border-radius: 6px; padding: 4px 8px; font-size: 12px; font-weight: 600; }}
.near-breakout {{ background: var(--accent-soft); color: var(--accent); }}
.recovering {{ background: #edf1fc; color: #425a93; }}
.forming {{ background: var(--soft); color: var(--muted); }}
.price {{ display: flex; align-items: baseline; gap: 6px; white-space: nowrap; }}
.price span {{ color: var(--muted); font-size: 11px; }}
.price strong {{ font-size: 17px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.market-cap-line {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 3px;
  margin: -2px 0 12px; color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
.market-cap-line > span {{ white-space: nowrap; }}
.candidate-actions {{ position: relative; }}
a {{ color: var(--accent); text-underline-offset: 3px; }}
.chart-link {{ position: absolute; left: 0; top: 0; right: 92px; min-height: 44px;
  display: flex; align-items: center; justify-content: center; gap: 12px; border-radius: 8px;
  background: var(--accent); color: #fff; text-decoration: none; font-size: 14px; font-weight: 600; touch-action: manipulation; }}
.candidate-details > summary {{ display: flex; align-items: center; justify-content: center; gap: 8px;
  margin-left: auto; width: 80px; min-height: 44px; border: 1px solid var(--line); border-radius: 8px; font-size: 13px; }}
.candidate-details[open] > summary {{ background: var(--soft); }}
.detail-body {{ margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); }}
dl {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 12px; margin: 0; }}
dt {{ color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }}
dd {{ margin: 3px 0 0; font-size: 15px; font-variant-numeric: tabular-nums; }}
.yahoo-link {{ display: flex; align-items: center; gap: 8px; min-height: 44px; margin-top: 12px; font-size: 13px; }}
.empty-state {{ padding: 20px 16px; border: 1px solid var(--line); border-radius: 12px;
  background: var(--surface); color: var(--muted); font-size: 14px; }}
.note {{ color: var(--muted); font-size: 11px; line-height: 1.8; margin: 24px 2px 8px; }}
.generated {{ color: var(--muted); font-size: 10px; margin: 0 2px; }}
a:focus-visible, summary:focus-visible, select:focus-visible {{ outline: 3px solid var(--accent); outline-offset: 3px; }}
@media (hover: hover) {{ .chart-link:hover {{ filter: brightness(1.1); }} summary:hover {{ color: var(--accent); }} }}
@media (max-width: 700px) {{
  body {{ padding: 22px 14px; }}
  main {{ max-width: 520px; }}
  h1 {{ font-size: 22px; }}
  .candidate-list {{ grid-template-columns: minmax(0, 1fr); gap: 10px; }}
}}
@media (max-width: 350px) {{
  body {{ padding-left: 10px; padding-right: 10px; }}
  .candidate {{ padding: 14px; }}
  #sort-label {{ display: none; }}
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --page: #111b20; --surface: #1b282e; --ink: #edf3f5; --muted: #a6b8c0;
    --line: #35474f; --soft: #26373f; --accent: #8bdfbc; --accent-soft: #243e35; }}
  .chart-link {{ background: #21694f; color: #fff; }}
  .recovering {{ background: #29364c; color: #b8caff; }}
}}
</style></head><body><main>
<header class="page-heading"><p class="eyebrow">米国株スクリーナー</p>
<h1>週足ソーサーボトム候補</h1>
<p class="market-date">データ基準日 <time>{current_market_date}</time></p>
<div class="update-status"><span class="total-count">候補 <strong>{total_count}</strong>件</span>
<span>前回比 {delta_text}</span></div></header>
{body}
<div class="note">候補抽出用です。決算、業績、希薄化、ニュースを確認してから投資判断してください。40週線はおおむね200取引日線に相当します。</div>
<p class="generated">ページ生成日時: {generated}</p>
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
    if not all_results.empty:
        try:
            all_results = add_market_caps(all_results, fetch_snapshot())
        except Exception as exc:
            print(f"時価総額の取得に失敗したため、時価総額・順位は不明と表示します: {exc}")
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
