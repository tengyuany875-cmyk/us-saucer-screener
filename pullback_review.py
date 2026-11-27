"""Offline follow-up diagnostics and a prepriced opening-order experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import pandas as pd

from pullback_backtest import metrics
from pullback_backtest_data import HistoricalData
from pullback_backtest_engine import Series, evaluate_window
from pullback_data import Sessions, classify_earnings, utc_now, write_csv, write_json
from pullback_earnings_archive import import_cache, latest_known, load_snapshots
from pullback_screener import _base_failure, _setup


PROTOCOL = {
    'version': 'baseline-review-1', 'years': 5, 'horizon': 5,
    'conditions': 'original detector, thresholds and weights unchanged',
    'transition': 'next session after first forming observation; not every later opportunity',
    'entry_experiment': 'opening auction limit priced at signal close: (setup high + minimum RR * stop)/(1 + minimum RR)',
    'fill': 'next regular open <= precomputed limit; otherwise cancel at open, no intraday fill',
    'units': 'split-normalized raw Open for fill; no dividend adjustment to limit',
    'gap_down': 'fills below the reference stop remain included; no hindsight lower-price veto',
    'exit': 'unchanged fifth session close, entry day counts as 1; no stop/target order',
    'cost': 'unchanged 10 bps per side, same baseline returns for filled orders',
    'earnings': 'only response-received snapshots <= signal close and age <= existing TTL; others unknown',
    'limits': 'exploratory price-only study, current universe, ideal auction liquidity, no optimization or portfolio model',
}


def opening_limit(row, minimum_rr):
    """Uses only the signal snapshot, with the existing minimum RR."""
    high, stop, rr = float(row['SetupHigh']), float(row['ReferenceStop']), float(minimum_rr)
    if not all(math.isfinite(value) for value in [high, stop, rr]) or not high > stop > 0 or rr <= 0:
        raise ValueError('Invalid reference prices or minimum RR')
    return (high + rr * stop) / (1 + rr)


def opening_fill(row, stock, sessions, as_of, minimum_rr):
    limit = opening_limit(row, minimum_rr)
    day = sessions.following(row['MarketDate'], 1)[0]
    result = {'EpisodeStageID': row['EpisodeStageID'], 'LimitAtSignalUnits': limit,
              'OrderDate': str(day.date()), 'FillStatus': 'pending_open'}
    if day > pd.Timestamp(as_of):
        return result
    s, t = stock.dates.get_indexer([pd.Timestamp(row['MarketDate']), day])
    if min(s, t) < 0:
        return {**result, 'FillStatus': 'missing_open'}
    # A shared split unit cancels later splits. Cash dividends do not change the
    # nominal limit. Do not inspect next-day High/Low/Close to decide the fill.
    entry = float(stock.raw.Open.iloc[t] * stock.future_split[s])
    if not math.isfinite(entry) or entry <= 0:
        return {**result, 'FillStatus': 'missing_open'}
    risk = entry - float(row['ReferenceStop'])
    return {**result, 'FillStatus': 'filled' if entry <= limit else 'not_filled',
            'EntryOpenAtSignalUnits': entry, 'EntryAtOrBelowStop': risk <= 0,
            'EntryRewardRisk': (float(row['SetupHigh']) - entry) / risk if risk > 0 else None}


def next_session_diagnostic(row, stock, spy, sessions, as_of):
    day = sessions.following(row['MarketDate'], 1)[0]
    result = {'EpisodeStageID': row['EpisodeStageID'], 'Ticker': row['Ticker'],
              'MarketDate': row['MarketDate'], 'FollowupDate': str(day.date())}
    if day > pd.Timestamp(as_of):
        return {**result, 'Outcome': 'pending'}
    t, b = stock.dates.get_indexer([day])[0], spy.dates.get_indexer([day])[0]
    if t < 0 or b < 0 or not stock.quality(t) or not spy.quality(b):
        return {**result, 'Outcome': 'quality_unavailable'}
    data, benchmark = stock.window(t), spy.window(b)
    config = stock.config
    base = _base_failure(data, benchmark, config)
    setup, prior_failures = _setup(data.iloc[:-1], benchmark, config)
    current, previous = data.iloc[-1], data.iloc[-2]
    location = (current.Close-current.Low)/(current.High-current.Low) if current.High > current.Low else 0.
    checks = {'previous_high': bool(current.Close > previous.High), 'ema': bool(current.Close > current.EMA),
              'positive_candle': bool(current.Close > current.Open),
              'close_location': bool(location >= config['rules']['close_location_min'])}
    evaluated = evaluate_window(row['Ticker'], data, benchmark, config, str(day.date()))
    result.update(BaseFailures='|'.join(base), PriorSetupFailures='|'.join(prior_failures),
                  ReboundFailures='|'.join(key for key, passed in checks.items() if not passed),
                  DetectorReason='|'.join(evaluated['reason_codes']),
                  DetectedStage=evaluated.get('record', {}).get('Stage', ''),
                  PriorSetupValid=setup is not None)
    if base:
        outcome = 'base_conditions'
    elif setup is None:
        outcome = 'prior_setup'
    elif not all(checks.values()):
        outcome = 'rebound_conditions'
    elif evaluated['status'] == 'signal':
        if evaluated['record']['Stage'] != 'confirmed':
            raise AssertionError('Confirmation diagnostic disagrees with detector')
        outcome = 'confirmed'
    elif evaluated['reason_codes'] == ['entry_risk_or_headroom']:
        outcome = 'entry_risk_or_headroom'
    elif evaluated['reason_codes'] == ['score']:
        outcome = 'score'
    else:
        raise AssertionError(f'Unclassified confirmation result: {evaluated}')
    result['Outcome'] = outcome
    if not base and setup and all(checks.values()):
        rules = config['rules']
        stop = setup['low'] - rules['stop_buffer_atr'] * setup['atr']
        risk = current.Close - stop
        rr = (setup['peak'] - current.Close) / risk if risk > 0 else -1.
        result.update(ExtensionATR=float((current.Close-current.EMA)/setup['atr']),
                      RiskATR=float(risk/setup['atr']), RewardRisk=float(rr),
                      RawScore=evaluated.get('raw_score'),
                      SamePeak=setup['peak_date'] == row['SetupPeakDate'])
    return result


def known_earnings(row, snapshots, sessions, config):
    decision = sessions.calendar.session_close(row['MarketDate']).isoformat()
    item = latest_known(snapshots, decision, config['data']['earnings_cache_hours'])
    result = {'EpisodeStageID': row['EpisodeStageID'], 'Ticker': row['Ticker'],
              'MarketDate': row['MarketDate'], 'DecisionAtUTC': decision, 'KnownSnapshot': item is not None,
              'Status': 'unknown', 'ObservedAtUTC': ''}
    if item:
        classified = classify_earnings(item['dates'], pd.Timestamp(decision), sessions, row['MarketDate'], config['rules']['holding_sessions'])
        result.update(Status=classified['status'], ObservedAtUTC=item['observed_at_utc'])
    return result


def observed_schedules(root, ticker):
    root = Path(root)
    stores = [root/'cache/daily/earnings/snapshots',root/'history/pullback/earnings']
    unique = {json.dumps(item,sort_keys=True):item for store in stores for item in load_snapshots(store,ticker)}
    return list(unique.values())


def compare_orders(events, orders):
    combined = events.merge(orders, on='EpisodeStageID', validate='one_to_one')
    results = []
    for stage, group in combined.groupby('Stage'):
        for model in ['market_open', 'rr_limit_open']:
            unknown = group.FillStatus.isin(['missing_open','pending_open'])
            fills = group.loc[~unknown] if model == 'market_open' else group.loc[group.FillStatus.eq('filled')]
            results.append({'Stage': stage, 'Model': model, 'Orders': len(group), 'Filled': len(fills),
                            'NotFilled': 0 if model == 'market_open' else int(group.FillStatus.eq('not_filled').sum()),
                            'UnknownOpen': int(unknown.sum()),
                            **metrics(fills)})
    return pd.DataFrame(results), combined


def run(root, scope='full'):
    root = Path(root)
    source = Path(json.loads((root/'output/pullback/backtest/latest_full.json').read_text(encoding='utf-8'))['run'])
    baseline_protocol = json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    hashes = json.loads((source/'source_hashes.json').read_text(encoding='utf-8'))
    for name, digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Baseline source changed: {name}')
    config = json.loads((source/'baseline_config.json').read_text(encoding='utf-8'))
    as_of = baseline_protocol['as_of']
    first = (pd.Timestamp(as_of)-pd.DateOffset(years=PROTOCOL['years'])).date().isoformat()
    signals = pd.read_csv(source/'signals.csv', keep_default_na=False)
    signals = signals.loc[signals.Strategy.eq('baseline') & signals.MarketDate.ge(first)].copy()
    if scope == 'pilot':
        universe = pd.read_csv(source/'universe.csv')
        import numpy as np
        tickers = universe.iloc[np.linspace(0,len(universe)-1,7,dtype=int)].Ticker.tolist()+['AAPL','SPNT']
        signals = signals.loc[signals.Ticker.isin(tickers)].copy()
    now, began = utc_now(), time.perf_counter()
    output = root/'output/pullback/review'/f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{scope}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', {**PROTOCOL, 'source': str(source), 'as_of': as_of, 'scope': scope})
    original = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                for name in ['protocol.json','signals.csv','trades.csv','metrics.csv','run_status.json']}
    archive = root/'cache/daily/earnings/snapshots'
    imported = import_cache(archive.parent)
    data = HistoricalData(root/'cache/backtest', baseline_protocol['download_start'], as_of, config, offline=True)
    sessions = Sessions()
    raw_spy, _ = data.cached('SPY')
    spy = Series('SPY', raw_spy, sessions, config)
    snapshots_expected = pd.read_csv(source/'data_status.csv').set_index('Ticker').sha256.to_dict()
    spy_manifest = json.loads((source/'benchmark_snapshot.json').read_text(encoding='utf-8'))
    if data.cached('SPY')[1]['sha256'] != spy_manifest['sha256']:
        raise ValueError('SPY snapshot differs from baseline')
    orders, followups, coverage = [], [], []
    for index, (ticker, group) in enumerate(signals.groupby('Ticker', sort=True), 1):
        cached = data.cached(ticker)
        if cached is None or cached[1]['sha256'] != snapshots_expected[ticker]:
            raise ValueError(f'Missing/changed baseline snapshot: {ticker}')
        stock = Series(ticker, cached[0], sessions, config)
        observed = observed_schedules(root, ticker)
        for row in group.to_dict('records'):
            orders.append(opening_fill(row, stock, sessions, as_of, config['rules']['minimum_reward_risk']))
            coverage.append(known_earnings(row, observed, sessions, config))
            if row['Stage'] == 'forming':
                followups.append(next_session_diagnostic(row, stock, spy, sessions, as_of))
        if index % 100 == 0:
            print(f'Reviewed {index} tickers; {len(orders)} signals; {time.perf_counter()-began:.1f}s', flush=True)
    columns = ['EpisodeStageID','Horizon','Status','NetReturnPct','SPYNetReturnPct','ExcessReturnPct','MFEPercent','MAEPercent']
    trades = pd.read_csv(source/'trades.csv', usecols=columns)
    trades = trades.loc[trades.Horizon.eq(PROTOCOL['horizon'])]
    events = signals.merge(trades, on='EpisodeStageID', validate='one_to_one')
    comparison, combined = compare_orders(events, pd.DataFrame(orders))
    followups = pd.DataFrame(followups)
    coverage = pd.DataFrame(coverage)
    failures = Counter()
    fields = ['BaseFailures', 'PriorSetupFailures', 'ReboundFailures']
    for row in followups.reindex(columns=fields).fillna('').to_dict('records'):
        for field in ['BaseFailures', 'PriorSetupFailures', 'ReboundFailures']:
            for code in row.get(field, '').split('|'):
                if code:
                    failures[(field,code)] += 1
    for name, frame in [('orders.csv',combined), ('comparison.csv',comparison), ('followups.csv',followups),
                        ('earnings_coverage.csv',coverage),
                        ('gate_failures.csv',pd.DataFrame([{'Gate':gate,'Reason':reason,'Count':count} for (gate,reason),count in failures.items()]))]:
        write_csv(output/name, frame)
    unchanged = all(hashlib.sha256((source/name).read_bytes()).hexdigest() == digest for name,digest in original.items())
    if not unchanged or len(combined) != len(signals) or combined.EpisodeStageID.duplicated().any():
        raise AssertionError('Original evidence or signal identities changed')
    summary = {'scope':scope,'as_of':as_of,'signals':len(signals),'forming_followups':len(followups),
               'followup_outcomes':followups.Outcome.value_counts().to_dict(), 'legacy_cache_files_preserved':imported,
               'known_earnings_snapshots':int(coverage.KnownSnapshot.sum()),'earnings_status':coverage.Status.value_counts().to_dict(),
               'baseline_artifacts_unchanged':unchanged,'elapsed_seconds':time.perf_counter()-began,'output_dir':str(output.resolve())}
    write_json(output/'summary.json', summary)
    own_sources = ['pullback_review.py','pullback_earnings_archive.py','pullback_data.py']
    write_json(output/'manifest.json', {'baseline_artifacts':original,
               'sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in own_sources},
               'baseline_sources':hashes})
    stage_labels = {'confirmed':'反発確認済み','forming':'押し目形成中'}
    model_labels = {'market_open':'翌日始値','rr_limit_open':'事前上限付き・寄付指値'}
    outcome_labels = {'rebound_conditions':'反発条件が未成立','entry_risk_or_headroom':'反発条件成立後にリスク・上値余地で除外',
                      'base_conditions':'当日の基礎条件で除外','prior_setup':'前日の形成条件が未成立','score':'反発条件成立後に点数不足',
                      'confirmed':'反発確認済みを検出','pending':'翌営業日が未到来','quality_unavailable':'必要データが不十分'}
    lines = ['# 日足押し目・検証修正の確認', '', f'保存済み実データ {first}〜{as_of}、{len(signals):,}シグナル。スコープ：{scope}。',
             '', '検出条件・配点は原本と同一。元の成績は上書きしていない。翌日の始値を見てから同じ始値で買う判断はせず、前日確定情報で上限価格を固定する。',
             '', '## 5営業日成績（片道0.1%）', '', '|状態|注文|対象|決済数|平均%|PF|', '|---|---|---:|---:|---:|---:|']
    for item in comparison.to_dict('records'):
        mean = f"{item['MeanPct']:.3f}" if pd.notna(item['MeanPct']) else '不明'
        pf = f"{item['ProfitFactor']:.3f}" if pd.notna(item['ProfitFactor']) else '不明'
        lines.append(f"|{stage_labels[item['Stage']]}|{model_labels[item['Model']]}|{item['Orders']}|{item['Complete']}|{mean}|{pf}|")
    lines += ['', '寄付指値モデルは既存RR下限1.5から上限価格を計算。始値が上限を超えたら見送り、日中の約定は仮定しない。',
              '下方向ギャップで参考ストップを割る取引も、注文が成立するなら除外しない。ストップ決済は追加していない。取引所の注文受理・値刻み・寄付流動性は未再現であり、一般的な寄付指値モデルの参考比較。',
              '', '## 形成中の翌営業日の判定', '', '初回形成中シグナルだけを母集団とする。遅れて反発した例や、形成中の採点を通らず確認済みになった例は含めない。以下は相互排他的な最初の障害。個別条件の重複した未達数はgate_failures.csvに保存。', '']
    lines += [f'- {outcome_labels[key]}: {value:,}件' for key,value in summary['followup_outcomes'].items()]
    lines += ['', '## 決算の再現可能性', '', f"受信時刻が確認でき、判定時刻以前かつ既存6時間TTL内の予定：{summary['known_earnings_snapshots']:,} / {len(signals):,}件。",
              f'旧キャッシュ{imported}ファイルを別途保存したが、旧取得時刻はリクエスト開始時刻のため、厳密な過去判定には使わない。今後はレスポンス受信時刻付きの予定を追記保存する。',
              '発表日と予定の取得時刻を混同しない。過去5年の決算ゲート込み成績は未再現。未知の予定を安全と扱わず、主候補の好成績を示す結果とも解釈しない。',
              '', '## 制約', '', '価格条件の事後診断・追加仮説であり、独立した検証期間ではない。現在の銘柄一覧による生存者バイアス、企業再編、日足の寄付価格近似、欠損決済の偏りは残る。スコア・閾値・ライブの選出や売買ルールは変更していない。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(root/'output/pullback/review'/f'latest_{scope}.json', {'run':str(output.resolve())})
    print(json.dumps(summary,ensure_ascii=False,indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--scope',choices=['pilot','full'],default='pilot')
    args = parser.parse_args()
    run(args.root,args.scope)
