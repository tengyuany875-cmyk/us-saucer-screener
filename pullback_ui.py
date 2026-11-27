"""Shared daily renderer and local preview; detection rules stay untouched."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd

from market_caps import fetch_snapshot, format_market_cap
from pullback_data import utc_now, write_json
from pullback_settings import config_hash
from site_navigation import NAV_CSS, add_navigation, nav

EXTRA_CSS = NAV_CSS + """
.local-label { margin: 0 2px 8px; font-size: 11px; color: var(--muted); }
.local-note { margin: 0 0 18px; padding: 10px 12px; font-size: 12px; border-left: 3px solid var(--line); color: var(--muted); }
.market-context { margin: 10px 0 0; color: var(--muted); font-size: 12px; }
.data-details { margin: 14px 0 20px; font-size: 12px; color: var(--muted); }
.data-details > summary, .review-section > summary { display: flex; align-items: center; justify-content: space-between;
  gap: 10px; min-height: 44px; padding: 10px 14px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
.data-details .detail-body { margin: 0; padding: 10px 14px; border-top: none; }
.data-details p { margin: 6px 0; }
.review-section { margin-top: 18px; font-size: 13px; }
.section-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin: 24px 2px 4px; }
.section-heading h2 { font-size: 17px; line-height: 1.5; letter-spacing: 0; }
.section-heading span { color: var(--muted); font-size: 13px; white-space: nowrap; }
.section-description { margin: 0 2px 10px; color: var(--muted); font-size: 12px; }
.section-empty { margin: 0; padding: 13px 14px; border: 1px dashed var(--line); border-radius: 10px; font-size: 13px; color: var(--muted); }
.candidate .company-name { margin: 3px 0 0; max-width: 220px; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
.review-badge { display: inline-block; margin: 0 0 8px; padding: 3px 7px; border-radius: 6px; font-size: 11px; background: var(--soft); color: var(--ink); }
.confirmed { color: var(--accent); background: var(--accent-soft); }
.candidate .detail-body { font-size: 12px; overflow-wrap: anywhere; }
.detail-body h3 { font-size: 13px; margin: 18px 0 8px; }
.detail-body p { margin: 8px 0; }
.score-breakdown { gap: 10px 12px; }
.score-breakdown dd { font-size: 13px; }
.archive-link { display: flex; align-items: center; min-height: 44px; font-size: 13px; }
.archive-date { margin: 0 0 10px; font-size: 12px; color: var(--muted); }
.overflow-section { margin-top: 20px; }
.overflow-section > summary { min-height: 44px; padding: 8px 2px; font-size: 13px; }
.history-note { font-size: 12px; color: var(--muted); }
@media (max-width:350px) { .category-nav a { font-size: 12px; } .candidate .company-name { max-width: 160px; } }
"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def number(value, digits=2, suffix="") -> str:
    try:
        parsed = float(value)
        return f"{parsed:,.{digits}f}{suffix}" if math.isfinite(parsed) else "不明"
    except (ValueError, TypeError):
        return "不明"


def time_label(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M JST")
    except (ValueError, TypeError):
        return "不明"


def weekly_preview(source: str) -> str:
    if "<main>" not in source or "</style>" not in source:
        raise ValueError("Weekly snapshot does not have the expected page structure")
    copied = add_navigation(source).replace("</style>", EXTRA_CSS + "</style>", 1)
    copied = copied.replace("<main>", '<main><p class="local-label">ローカル試作</p>', 1)
    return copied.replace('<!-- categories:end -->','<!-- categories:end --><p class="local-note">週足は既存の保存済み結果です。表示されているデータ基準日をご確認ください。</p>',1)


def card(row: dict, config: dict, caps: dict | None, unique: str, historical=False) -> str:
    ticker = str(row["Ticker"])
    stage = row["Stage"]
    weights = config["scoring"]["weights"]
    maximum = sum(weights.values()) - (weights["rebound"] if stage == "forming" else 0)
    entry = (caps or {}).get("entries", {}).get(ticker, {})
    cap = format_market_cap(entry.get("market_cap_usd"))
    cap_title = "Nasdaq取得: " + time_label((caps or {}).get("fetched_at_utc", "")) + "。株価基準日と異なる場合があります。"
    chart_symbol = quote(f'{row["Exchange"]}:{ticker}', safe="")
    link = f"https://www.tradingview.com/chart/?symbol={chart_symbol}&interval=D"
    stage_label = "反発確認済み" if stage == "confirmed" else "押し目形成中"
    badge = '<p class="review-badge">決算不明・要確認</p>' if row.get("Bucket") == "review" else ""
    if row.get("Bucket") == "earnings_excluded":
        badge = '<p class="review-badge">保有想定期間に決算・選出対象外</p>'
    if historical:
        badge = '<p class="review-badge">過去日の価格条件のみ・決算未再現</p>'
    score_labels = {"trend": "トレンド", "relative_strength": "相対強度", "pullback": "押し目の形", "liquidity": "流動性", "reward_risk": "リスクリワード", "rebound": "反発"}
    breakdown = json.loads(row["ScoreBreakdown"])
    scores = "".join(f'<div><dt>{label}</dt><dd>{number(breakdown[key])} / {weights[key]}</dd></div>'
                     for key, label in score_labels.items() if stage == "confirmed" or key != "rebound")
    event = {"outside_holding": "想定保有期間の外", "within_holding": "想定保有期間内", "unknown": "不明"}.get(row.get("EarningsStatus"), "不明")
    event_date = row.get("EarningsNextDate") or "不明"
    return f'''<article class="candidate" aria-labelledby="ticker-{unique}" data-ticker="{esc(ticker)}" data-stage="{esc(stage)}" data-bucket="{esc(row.get('Bucket', ''))}">
<div class="candidate-heading"><div><h2 id="ticker-{unique}">{esc(ticker)}</h2><span class="exchange">{esc(row['Exchange'])}</span></div>
<div class="score"><span class="score-label">点数</span><strong>{number(row['Score'])}</strong><span class="score-unit"> / {maximum}</span></div></div>
<div class="candidate-overview"><span class="stage {esc(stage)}">{stage_label}</span><div class="price"><span>株価</span><strong>${number(row['Price'])}</strong></div></div>
{badge}<p class="market-cap-line" title="{esc(cap_title)}">時価総額：{esc(cap)}</p>
<div class="candidate-actions"><a class="chart-link" href="{esc(link)}" target="_blank" rel="noopener noreferrer" aria-label="{esc(ticker)}の日足をTradingViewで開く（新しいタブ）">チャートを見る <span aria-hidden="true">↗</span></a>
<details class="candidate-details"><summary aria-label="{esc(ticker)}の詳細"><span>詳細</span><span class="chevron" aria-hidden="true"></span></summary>
<div class="detail-body"><p class="company-name">{esc(row.get('Name', ''))}</p>
<dl><div><dt>高値からの調整幅</dt><dd>{number(row['DrawdownPct'])}%</dd></div><div><dt>想定リスクリワード</dt><dd>1 : {number(row['RewardRisk'])}</dd></div></dl>
<h3>検出理由</h3><p>{esc(row['Reasons'])}</p><h3>スコア内訳</h3><dl class="score-breakdown">{scores}</dl>
<p>点数は反発確率ではありません。形成中は共通条件のみ{sum(weights.values()) - weights['rebound']}点満点です。</p>
<h3>判定時点の参考値</h3><dl><div><dt>調整期間</dt><dd>{number(row['PullbackDays'], 0)}営業日</dd></div><div><dt>高値の日付</dt><dd>{esc(row['SetupPeakDate'])}</dd></div>
<div><dt>想定ストップ（調整後価格）</dt><dd>${number(row['ReferenceStop'])}</dd></div><div><dt>調整安値までの深さ</dt><dd>{number(row['DepthATR'])} ATR</dd></div></dl>
<p>調整幅は高値から調整安値まで。リスクリワードはリスク1に対する高値までの上値余地で、約定を保証する値ではありません。</p>
<h3>決算予定</h3><p>{esc(event)} / {esc(event_date)}</p><p>確認日時：{esc(time_label(row.get('EarningsFetchedAtUTC', '')))}</p>
<p>データ基準日：{esc(row['MarketDate'])}</p>
<a class="yahoo-link" href="https://finance.yahoo.com/quote/{quote(ticker, safe='')}" target="_blank" rel="noopener noreferrer">Yahoo Financeで見る ↗</a></div></details></div></article>'''


def section(key: str, title: str, description: str, rows: list[dict], config: dict, caps: dict | None, historical=False) -> str:
    rows = sorted(rows, key=lambda row: (-float(row["Score"]), row["Ticker"]))
    contents = ('<div class="candidate-list">' + "".join(card(row, config, caps, f"{key}-{index}", historical) for index, row in enumerate(rows)) + '</div>') if rows else '<p class="section-empty">今回、該当する銘柄はありません。</p>'
    return f'<section aria-labelledby="section-{esc(key)}"><div class="section-heading"><h2 id="section-{esc(key)}">{esc(title)}</h2><span>{len(rows)}件</span></div><p class="section-description">{esc(description)}</p>{contents}</section>'


def page(title: str, css: str, contents: str, *, local=True) -> str:
    label = '<p class="local-label">ローカル試作</p>' if local else ''
    return f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)} | 米国株スクリーナー</title><style>{css}\n{EXTRA_CSS}</style></head><body><main>{label}{nav('daily')}{contents}</main></body></html>'''


def render_daily(summary: dict, records: list[dict], config: dict, css: str, caps: dict | None, recent: dict | None = None, history_counts: dict | None = None, *, local=True) -> str:
    selected = [row for row in records if str(row.get("Selected")).lower() == "true"]
    main = [r for r in selected if r["Bucket"] == "main"]
    watch = [r for r in selected if r["Bucket"] == "watch"]
    review = [r for r in records if r["Bucket"] == "review"]
    market = esc(summary["market_regime"])
    content = f'''<header class="page-heading"><p class="eyebrow">米国株スクリーナー</p><h1>日足押し目候補</h1><p class="market-date">データ基準日 <time>{esc(summary['as_of'])}</time>（米国）</p>
<div class="update-status"><span class="total-count">主候補 <strong>{len(main)}</strong>件</span><span>監視 {len(watch)}件</span></div>
<p class="market-context">{market} · 取得成功率 {summary['price_success_rate']:.2%}</p></header>
<details class="data-details"><summary><span>更新情報・市場環境</span><span class="chevron" aria-hidden="true"></span></summary><div class="detail-body">
<p>更新 <time datetime="{esc(summary['finished_at_utc'])}">{esc(time_label(summary['finished_at_utc']))}</time></p>
<p>{market}。市場環境として記録しており、一律の除外条件にはしていません。</p>
<p>価格取得成功率 {summary['price_success_rate']:.2%}（{summary['price_success_count']:,} / {summary['target_count']:,}銘柄）</p>
<p>判定可能：{summary['analysable_count']:,}銘柄。正常な最新価格がある履歴不足の銘柄は取得成功に含みます。</p><p>処理時間：{summary['elapsed_seconds']:.1f}秒</p><p>時価総額の取得：{esc(time_label((caps or {}).get('fetched_at_utc', '')))}（Nasdaq）</p>'''
    if history_counts:
        content += f'<p>追跡記録：形成中 {history_counts.get("forming", 0)}件 / 反発確認済み {history_counts.get("confirmed", 0)}件（押し目・状態ごとの初回）</p>'
    if summary.get('update_policy'):
        content += f'<p>{esc(summary["update_policy"])}</p>'
    content += '</div></details>'
    if summary.get("scope") != "full_universe":
        content += '<p class="local-note">一部銘柄を対象にした試験結果です。</p>'
    if main:
        content += section("main", "反発確認済み", "主候補", main, config, caps)
    if watch:
        content += section("watch", "押し目形成中", "監視候補。反発は未確認です。", watch, config, caps)
    if not main and not watch:
        content += '<p class="empty-state">今回は候補がありません。</p>'
    if review:
        content += f'<details class="review-section"><summary><span>決算不明・要確認 {len(review)}件</span><span class="chevron" aria-hidden="true"></span></summary>'
        content += section("review", "決算不明・要確認", "決算予定を確認できないため、選出対象外です。", review, config, caps) + '</details>'
    excluded = [r for r in records if r["Bucket"] == "earnings_excluded"]
    other = [r for r in records if r["Bucket"] in {"main", "watch"} and str(r.get("Selected")).lower() != "true"]
    if excluded or other:
        content += '<details class="overflow-section"><summary>選出枠以外の検出結果を見る</summary>'
        if excluded:
            content += section("excluded", "決算期間内・選出対象外", "次の5営業日の保有想定期間に決算予定があります。", excluded, config, caps)
        if other:
            content += section("other", "選出上限の対象外", "条件は通過していますが、最大20銘柄の選出枠には入っていません。", other, config, caps)
        content += '</details>'
    if recent:
        content += '<a class="archive-link" href="recent.html">過去の検出例を見る →</a>'
    content += '<p class="note">主候補・監視候補は合計最大20銘柄。点数は反発確率ではありません。</p>'
    return page("日足押し目候補", css, content, local=local)


def render_recent(summary: dict, records: list[dict], linked: list[dict], config: dict, css: str, caps: dict | None, *, local=True) -> str:
    content = '<header class="page-heading"><p class="eyebrow">過去日の試験判定</p><h1>反発確認済みの検出例</h1></header>'
    content += '<p class="local-note">過去日の価格条件を確認した試験です。決算予定は未再現のため、主候補には含めません。</p>'
    content += f'<p class="archive-date">対象：{esc(summary["dates"][0])} 〜 {esc(summary["dates"][-1])} / {len(summary["dates"])}営業日。時価総額は現在取得した参考値です。</p>'
    content += f'<details class="overflow-section"><summary>過去日試験の前提</summary><p class="history-note">{esc(summary["limitations"])}</p></details>'
    if linked:
        content += '<details class="overflow-section"><summary>形成中から反発確認済みへの移行を見る</summary>'
        weights = config["scoring"]["weights"]
        full, forming_max = sum(weights.values()), sum(weights.values()) - weights["rebound"]
        for index, transition in enumerate(linked):
            content += f'<p class="history-note">{esc(transition["Ticker"])}：{esc(transition["FormingDate"])} 形成中 {number(transition["FormingScore"])} / {forming_max}点 → {esc(transition["ConfirmedDate"])} 反発確認済み {number(transition["ConfirmedScore"])} / {full}点</p>'
            previous = [row for row in records if row["SetupID"] == transition["SetupID"] and row["Stage"] == "forming" and row["MarketDate"] == transition["FormingDate"]]
            if previous:
                content += '<div class="candidate-list">' + card(previous[0], config, caps, f"transition-{index}", historical=True) + '</div>'
        content += '</details>'
    for day in reversed(summary["dates"]):
        daily = [row for row in records if row["MarketDate"] == day and row["Stage"] == "confirmed"]
        if daily:
            content += section(day, day, "当日の引けまでの価格条件を満たした例。過去の主候補としての選出ではありません。", daily, config, caps, historical=True)
    if not any(row["Stage"] == "confirmed" for row in records):
        content += '<p class="section-empty">この期間には反発確認済みの検出例はありませんでした。条件は変更していません。</p>'
    content += '<a class="archive-link" href="daily.html">最新の確定日の候補に戻る →</a>'
    return page("過去日の反発確認例", css, content, local=local)


def build(root: Path, refresh_caps=False) -> dict:
    root = Path(root)
    pointer = json.loads((root / "output/pullback/latest_success.json").read_text(encoding="utf-8"))
    run = Path(pointer["run"])
    summary = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if not summary.get("quality_passed") or config_hash(config) != pointer["config_hash"] or summary["as_of"] != pointer["as_of"]:
        raise ValueError("Only a consistent, quality-passed run can become the local UI")
    weekly = (root / "docs/index.html").read_text(encoding="utf-8")
    style = re.search(r"<style>(.*?)</style>", weekly, flags=re.S)
    if style is None:
        raise ValueError("Existing weekly CSS is unavailable")
    output = root / "output/local-ui"
    output.mkdir(parents=True, exist_ok=True)
    caps_path = root / "output/pullback/ui_market_caps.json"
    caps, cap_error = None, ""
    if refresh_caps:
        try:
            caps = fetch_snapshot()
            write_json(caps_path, caps)
        except Exception as exc:
            cap_error = f"{type(exc).__name__}: {exc}"
    elif caps_path.exists():
        saved = json.loads(caps_path.read_text(encoding="utf-8"))
        age = (utc_now() - datetime.fromisoformat(saved["fetched_at_utc"])).total_seconds()
        if 0 <= age < 86400:
            caps = saved
    # A failed refresh never silently falls back to an old market-cap vintage.
    records = pd.read_csv(run / "all_candidates.csv", keep_default_na=False).to_dict("records")
    recent, recent_records, linked = None, [], []
    recent_pointer = root / "output/pullback/latest_recent.json"
    if recent_pointer.exists():
        recent_run = Path(json.loads(recent_pointer.read_text(encoding="utf-8"))["run"])
        candidate = json.loads((recent_run / "trial_status.json").read_text(encoding="utf-8"))
        if candidate["source_as_of"] == summary["as_of"] and candidate["config_hash"] == pointer["config_hash"]:
            recent = candidate
            recent_records = pd.read_csv(recent_run / "all_candidates.csv", keep_default_na=False).to_dict("records")
            linked = pd.read_csv(recent_run / "transitions.csv", keep_default_na=False).to_dict("records")
    history_path = root / "history/pullback/signals.csv"
    counts = pd.read_csv(history_path)["Stage"].value_counts().to_dict() if history_path.exists() else {}
    (output / "index.html").write_text(weekly_preview(weekly), encoding="utf-8")
    (output / "daily.html").write_text(render_daily(summary, records, config, style[1], caps, recent, counts), encoding="utf-8")
    (output / "recent.html").write_text(render_recent(recent, recent_records, linked, config, style[1], caps) if recent else page("過去日の試験", style[1], '<h1>過去日の試験結果はまだありません</h1><a class="archive-link" href="daily.html">最新の候補へ戻る →</a>'), encoding="utf-8")
    manifest = {"as_of": summary["as_of"], "run": str(run), "built_at_utc": utc_now().isoformat(), "config_hash": pointer["config_hash"],
                "market_caps_fetched_at_utc": (caps or {}).get("fetched_at_utc"), "market_caps_error": cap_error,
                "recent_trial": recent and recent["output_dir"], "weekly_source": "docs/index.html (saved snapshot)",
                "weekly_source_sha256": hashlib.sha256(weekly.encode()).hexdigest(), "output_dir": str(output.resolve())}
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--refresh-caps", action="store_true", help="Fetch optional display data once; does not affect scores")
    args = parser.parse_args()
    build(args.root, args.refresh_caps)
