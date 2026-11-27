"""Daily detection and tracking; publication is handled separately after quality passes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import pandas as pd

from pullback_data import DailyData, Sessions, daily_universe, earnings, utc_now, write_csv, write_json
from pullback_history import append_signals, identify, observe, read_signals, save_observations
from pullback_screener import analyse_daily, indicators
from pullback_settings import DEFAULT_CONFIG, config_hash, load_config
from pullback_earnings_archive import persist_archives

CANDIDATE_COLUMNS = ["Ticker", "Name", "Exchange", "MarketDate", "Stage", "StageJA", "Bucket", "Selected", "Score", "Price", "AdjustedClose",
    "SetupPeakDate", "SetupHigh", "SetupLow", "SetupATR", "PullbackDays", "DepthATR", "DrawdownPct", "EMA", "SMAFast", "SMASlow", "RSI", "ATR",
    "ATRPercent", "RelativeStrengthPct", "AverageDollarVolume", "VolumeRatio", "CloseLocation", "ReferenceStop", "RewardRisk", "SPYAboveSMA",
    "Reasons", "ScoreBreakdown", "Score_trend", "Score_relative_strength", "Score_pullback", "Score_liquidity", "Score_reward_risk", "Score_rebound",
    "EarningsStatus", "EarningsNextDate", "EarningsFetchedAtUTC", "EarningsSource", "EarningsMessage", "SetupID", "SignalID", "ObservedAtUTC", "PriceFetchedAtUTC", "PriceSHA256"]


def assign_bucket(record: dict, event: dict) -> dict:
    row = record.copy()
    row.update({"EarningsStatus": event["status"], "EarningsNextDate": event["next_date"], "EarningsFetchedAtUTC": event.get("fetched_at_utc", ""),
                "EarningsSource": event.get("source", ""), "EarningsMessage": event.get("error") or event.get("reason", "")})
    if event["status"] == "unknown":
        row["Bucket"] = "review"
    elif event["status"] == "within_holding":
        row["Bucket"] = "earnings_excluded"
    else:
        row["Bucket"] = "main" if row["Stage"] == "confirmed" else "watch"
    row["Selected"] = False
    return row


def select(records: list[dict], limit: int) -> list[dict]:
    eligible = [row for row in records if row["Bucket"] in {"main", "watch"}]
    ranked = sorted(eligible, key=lambda row: (0 if row["Bucket"] == "main" else 1, -row["Score"], -row["AverageDollarVolume"], row["Ticker"]))
    chosen = {row["SignalID"] for row in ranked[:limit]}
    for row in records:
        row["Selected"] = row["SignalID"] in chosen
    return ranked[:limit]


def quality_passed(statuses: list[dict], config: dict, benchmark_ok: bool) -> bool:
    return bool(statuses) and benchmark_ok and sum(row["PriceSuccess"] for row in statuses) / len(statuses) >= config["data"]["minimum_success_rate"]


def report_text(summary: dict, records: list[dict]) -> str:
    lines = ["# Daily Pullback 実行結果", "", f"- 基準日: {summary['as_of']}", f"- 設定: {summary['strategy_version']} / {summary['config_hash']}",
             f"- 対象: {summary['target_count']}銘柄 / 有効な最新価格: {summary['price_success_count']} ({summary['price_success_rate']:.2%})",
             f"- 判定可能: {summary['analysable_count']} / キャッシュ: {summary['cache_hits']} / 処理: {summary['elapsed_seconds']:.2f}秒",
             f"- SPY環境: {summary['market_regime']}（除外条件には使用しない）",
             f"- 品質基準: {'合格' if summary['quality_passed'] else '未達・結果は診断用、履歴へ昇格しない'}",
             f"- 主候補: {summary['selected_main']} / 監視候補: {summary['selected_watch']} / 決算不明の要確認: {summary['review_count']} / 決算期間内除外: {summary['earnings_excluded_count']}",
             "", "今回は検出・追跡基盤の試験です。反発確率・将来の収益性・実際の約定を検証した結果ではありません。", "",
             "## 検出一覧（未選出・要確認・決算除外も含む）", "", "|銘柄|状態|分類|選出|点数|株価|決算予定|", "|---|---|---|---|---:|---:|---|"]
    for row in sorted(records, key=lambda row: (row["Bucket"], -row["Score"], row["Ticker"])):
        lines.append(f"|{row['Ticker']}|{row['StageJA']}|{row['Bucket']}|{row['Selected']}|{row['Score']:.2f}|{row['Price']:.2f}|{row['EarningsNextDate'] or '不明'}|")
    for row in records:
        lines += ["", f"## {row['Ticker']} — {row['StageJA']} / {row['Bucket']}", "", row["Reasons"], "",
                  f"採点内訳: {row['ScoreBreakdown']}", f"決算: {row['EarningsStatus']} / {row['EarningsNextDate'] or '不明'} / {row['EarningsMessage']}"]
    lines += ["", "## 診断", "", f"取得状態: {json.dumps(summary['data_status_counts'], ensure_ascii=False)}",
              f"不一致理由: {json.dumps(summary['rejection_counts'], ensure_ascii=False)}", "",
              "- 選出枠は主候補＋監視候補で合計最大20。決算不明は別ファイルで確認する。",
              "- 初期閾値は実データの件数に合わせて調整していない。",
              "- 新規シグナルの成績は将来の営業日が確定するまで pending。形成中・確認済み・決算区分を別々に集計可能。",
              "- この実行は検出と追跡を行い、公開は別処理の品質確認後に行う。売買注文は行わない。", ""]
    return "\n".join(lines)


def run(config_path: Path, root: Path, as_of: str | None = None, tickers: list[str] | None = None, limit: int | None = None, offline=False) -> dict:
    started, now = time.perf_counter(), utc_now()
    config, sessions = load_config(config_path), Sessions()
    fingerprint = config_hash(config)
    latest = sessions.last_completed(now, config["data"]["close_delay_minutes"])
    as_of = as_of or latest
    if as_of != latest:
        raise ValueError(f"Live trial requires the latest completed session {latest}; historical calendars/earnings are not point-in-time")
    root = Path(root)
    cache_root, history_root = root / "cache/daily", root / "history/pullback"
    output = root / "output/pullback/runs" / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{fingerprint}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    source_names = ["pullback_app.py", "pullback_data.py", "pullback_screener.py", "pullback_history.py", "pullback_settings.py", "pullback_earnings_archive.py", "app.py", "universe.py", "market_caps.py"]
    write_json(output / "source_hashes.json", {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names})
    print(f"Run: {output}\nAs of: {as_of}; config: {fingerprint}", flush=True)
    data = DailyData(cache_root, config, sessions, as_of, offline=offline)
    spy = next(data.fetch(["SPY"]))
    if spy.status != "ok":
        write_json(output / "run_status.json", {"status": "benchmark_failed", "as_of": as_of, "message": spy.message, "data_status": spy.status})
        raise RuntimeError(f"SPY unavailable: {spy.status}: {spy.message}; history unchanged")
    market = indicators(spy.frame, config)
    regime = f"SPYは{config['indicators']['sma_fast']}日線より上" if market["Close"].iloc[-1] > market["SMAFast"].iloc[-1] else f"SPYは{config['indicators']['sma_fast']}日線以下"
    universe = daily_universe(cache_root, config, now, offline=offline)
    if tickers:
        wanted = set(tickers)
        missing = wanted - set(universe["Ticker"])
        if missing:
            raise ValueError(f"Tickers outside the ordinary-share universe: {sorted(missing)}")
        universe = universe.loc[universe["Ticker"].isin(wanted)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        universe = universe.head(limit)
    if universe.empty:
        raise ValueError("Empty universe")
    write_csv(output / "universe.csv", universe)
    lookup = universe.set_index("Ticker")
    records, statuses, rejection_counts = [], [], Counter()
    target_tickers = universe["Ticker"].tolist()
    for quote in data.fetch(target_tickers):
        status = {"Ticker": quote.ticker, "DataStatus": quote.status, "Source": quote.source, "PriceSuccess": quote.price_success,
                  "Message": quote.message, "PriceFetchedAtUTC": quote.metadata.get("fetched_at_utc", ""), "PriceSHA256": quote.metadata.get("sha256", ""),
                  "Decision": "not_analysed", "ReasonCodes": "", "RawScore": ""}
        if quote.status == "ok":
            result = analyse_daily(quote.ticker, quote.frame, spy.frame, config, as_of)
            status.update({"Decision": result["status"], "ReasonCodes": ",".join(result["reason_codes"]), "RawScore": result.get("raw_score", "")})
            rejection_counts.update(result["reason_codes"])
            if result["status"] == "signal":
                record = result["record"]
                record.update({"Name": lookup.loc[quote.ticker, "Name"], "Exchange": lookup.loc[quote.ticker, "Exchange"],
                               "PriceFetchedAtUTC": status["PriceFetchedAtUTC"], "PriceSHA256": status["PriceSHA256"]})
                event = earnings(quote.ticker, cache_root, config, sessions, as_of, utc_now(), offline=offline)
                record["ObservedAtUTC"] = utc_now().isoformat()
                records.append(identify(assign_bucket(record, event), fingerprint))
                print(f"Signal {quote.ticker}: {record['Stage']} {record['Score']:.2f}; earnings {event['status']}", flush=True)
        statuses.append(status)
    selected = select(records, config["rules"]["maximum_selected"])
    healthy = quality_passed(statuses, config, spy.status == "ok")
    added, observation_counts = 0, {}
    if healthy:
        persist_archives(cache_root / 'earnings/snapshots', history_root / 'earnings')
        old = read_signals(history_root)
        extra = sorted(set(old.get("Ticker", [])) - set(target_tickers))
        # Track older episodes even if they leave today's universe. Separate from scan coverage.
        extra_status = [{"Ticker": q.ticker, "Status": q.status} for q in data.fetch(extra)]
        write_json(output / "tracking_fetch_status.json", extra_status)
        signals, added = append_signals(records, history_root, config, fingerprint, now.isoformat())
        observations = observe(signals, data, sessions, utc_now().isoformat())
        observations = save_observations(history_root, observations)
        observation_counts = observations["Status"].value_counts().to_dict()
    frame = pd.DataFrame(records, columns=CANDIDATE_COLUMNS)
    write_csv(output / "all_candidates.csv", frame)
    write_csv(output / "selected.csv", pd.DataFrame(selected, columns=CANDIDATE_COLUMNS))
    write_csv(output / "main.csv", frame.loc[frame["Bucket"].eq("main") & frame["Selected"].eq(True)])
    write_csv(output / "watch.csv", frame.loc[frame["Bucket"].eq("watch") & frame["Selected"].eq(True)])
    write_csv(output / "review.csv", frame.loc[frame["Bucket"].eq("review")])
    write_csv(output / "earnings_excluded.csv", frame.loc[frame["Bucket"].eq("earnings_excluded")])
    write_csv(output / "data_status.csv", pd.DataFrame(statuses))
    success = sum(row["PriceSuccess"] for row in statuses)
    summary = {"status": "complete" if healthy else "quality_failed", "as_of": as_of, "started_at_utc": now.isoformat(), "finished_at_utc": utc_now().isoformat(),
        "strategy_version": config["strategy_version"], "config_hash": fingerprint, "scope": "subset" if tickers or limit is not None else "full_universe",
        "target_count": len(statuses), "price_success_count": success, "price_success_rate": success / len(statuses),
        "analysable_count": sum(row["DataStatus"] == "ok" for row in statuses), "cache_hits": sum(row["Source"] == "cache" and row["PriceSuccess"] for row in statuses),
        "data_status_counts": dict(Counter(row["DataStatus"] for row in statuses)), "rejection_counts": dict(rejection_counts),
        "benchmark": {"status": spy.status, "fetched_at_utc": spy.metadata.get("fetched_at_utc"), "sha256": spy.metadata.get("sha256")},
        "market_regime": regime, "quality_passed": healthy, "selected_main": sum(row["Bucket"] == "main" for row in selected),
        "selected_watch": sum(row["Bucket"] == "watch" for row in selected), "review_count": sum(row["Bucket"] == "review" for row in records),
        "earnings_excluded_count": sum(row["Bucket"] == "earnings_excluded" for row in records), "signals_added": added,
        "observation_status_counts": observation_counts, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "output_dir": str(output.resolve()), "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "yfinance", "exchange_calendars"]}}
    write_json(output / "run_status.json", summary)
    (output / "report.md").write_text(report_text(summary, records), encoding="utf-8")
    if healthy:
        write_json(root / "output/pullback/latest_success.json", {"run": str(output.resolve()), "as_of": as_of, "config_hash": fingerprint})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Daily Pullback scan and independent tracking (publication is separate)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--as-of")
    parser.add_argument("--tickers", help="Comma-separated ordinary-share tickers; explicitly reported as a subset")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offline", action="store_true", help="Use this completed session's valid caches only")
    args = parser.parse_args()
    result = run(args.config, args.root, args.as_of, [s.strip().upper() for s in args.tickers.split(",")] if args.tickers else None, args.limit, args.offline)
    raise SystemExit(0 if result["quality_passed"] else 2)


if __name__ == "__main__":
    main()
