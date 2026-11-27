"""Frozen baseline event study; no live detector/config/history modifications."""
from __future__ import annotations
import argparse
from collections import Counter
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from pullback_backtest_data import HistoricalData
from pullback_backtest_engine import Series, evaluate_window, outcome, simple_signal
from pullback_data import Sessions, utc_now, write_csv, write_json
from pullback_settings import config_hash, load_config

PROTOCOL = {
    'version':'baseline-event-study-1', 'years':[5,10], 'horizons':[3,5,10], 'primary_horizon':5,
    'cost_bps_per_side':10, 'top_n':[5,10,20], 'warmup_calendar_days':760,
    'entry':'signal next NYSE session regular open', 'exit':'Nth session close, entry day counts as 1',
    'duplicates':'first per strategy/ticker/peak_date/stage; pre-period signals seed duplicate state',
    'ranking':'daily first episode-stage observations, score desc / dollar volume desc / ticker asc; simple uses dollar volume',
    'earnings':'historical schedule unavailable; price-only stage study, not historical main picks',
    'universe':'frozen current ordinary-share universe; excludes delisted stocks and prior constituents',
    'adjustment':'current-vintage adjusted OHLC; undo future split nominal-price normalization; daily prefix reanchored to signal units',
    'returns':'dividend-adjusted OHLC proxy, fixed horizons, no stop/target or portfolio capital constraints',
    'simple':'same price/ADV/trend gates; EMA-ATR <= Close <= EMA, RSI 40..55; confirmed next day close>priorHigh, EMA, Open',
    'score_bands':{'forming':[60,65,70,75.00001], 'confirmed':[75,80,85,90,95,100.00001]},
    'optimization':'none; protocol fixed before pilot/full performance',
}
BASELINE_HASHES = {'pullback_config.json':'527d2862b3478bed6c01b84ead22510012b996e6b9ee23120096079a37ef58cd',
                   'pullback_screener.py':'ec674eb977efe96d69b369a988fc3abe61300692deed851af12a4fba151a6d7b'}


def init_worker(spy_raw,config):
    global WORKER_SPY,WORKER_CONFIG,WORKER_SESSIONS
    WORKER_CONFIG,WORKER_SESSIONS=config,Sessions()
    WORKER_SPY=Series('SPY',spy_raw,WORKER_SESSIONS,config)


def process_stock(ticker,raw,meta,source,first,last,validate=False,spy=None,sessions=None,config=None):
    spy=spy if spy is not None else WORKER_SPY
    sessions=sessions if sessions is not None else WORKER_SESSIONS
    config=config if config is not None else WORKER_CONFIG
    item={'Ticker':ticker,'Source':source,**meta}
    records,outcomes,counts=[],[],{}
    try:
        stock=Series(ticker,raw,sessions,config)
        records,counts=scan(ticker,stock,spy,sessions,first,last,validate=validate)
        item.update(Status='available',**counts)
        for row in records:
            for horizon in PROTOCOL['horizons']:
                outcomes.append(outcome(row,stock,spy,sessions,last,horizon,PROTOCOL['cost_bps_per_side']))
    except (ValueError,KeyError,IndexError) as exc:
        item.update(Status='unavailable',Error=f'{type(exc).__name__}: {exc}')
    return records,outcomes,counts,item


def processed_results(data,universe,scope,first,last,spy,sessions,config):
    if scope=='pilot':
        for ticker,raw,meta,source in data.fetch(universe.Ticker.tolist()):
            yield process_stock(ticker,raw,meta,source,first,last,True,spy,sessions,config)
        return
    # Four bounded CPU workers; network fetching stays in the parent with the
    # unchanged four-thread batch settings. Results are consumed in input order.
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(spy.raw,config)) as pool:
        pending=deque()
        for ticker,raw,meta,source in data.fetch(universe.Ticker.tolist()):
            pending.append(pool.submit(process_stock,ticker,raw,meta,source,first,last))
            if len(pending)>=8:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def source_hashes():
    names = [*BASELINE_HASHES,'pullback_backtest.py','pullback_backtest_engine.py','pullback_backtest_data.py']
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def equal_results(a,b):
    if a['status'] != b['status'] or a['reason_codes'] != b['reason_codes']:
        return False
    if a['status'] != 'signal':
        return True
    for key,value in a['record'].items():
        other = b['record'][key]
        if isinstance(value,(float,int)) and not isinstance(value,bool):
            if not np.isclose(value,other,rtol=1e-9,atol=1e-8):
                return False
        elif value != other:
            return False
    return True


def scan(ticker, stock, spy, sessions, first, last, validate=False):
    baseline, simple = stock.candidates(spy)
    records, diagnostics, seen = [], Counter(), set()
    # Start a full setup window earlier, so an episode first observed before the
    # evaluation boundary is not counted again on the first evaluation day.
    scan_start = pd.Timestamp(first)-pd.Timedelta(days=40)
    positions = np.flatnonzero((stock.dates >= scan_start) & (stock.dates <= pd.Timestamp(last)))
    compare_positions = set(positions[::max(1,len(positions)//24)]) if validate else set()
    for t in positions:
        day = str(stock.dates[t].date())
        if not stock.quality(t):
            diagnostics['quality_ineligible'] += 1
            continue
        spy_pos = spy.dates.get_indexer([stock.dates[t]])[0]
        if spy_pos < 0 or not spy.quality(spy_pos):
            diagnostics['benchmark_ineligible'] += 1
            continue
        diagnostics['analysable_days'] += int(day >= first)
        diagnostics[f'eligible_days_{day[:4]}'] += int(day >= first)
        if not baseline[t] and not simple[t] and t not in compare_positions:
            continue
        frame = stock.window(t)
        benchmark = spy.window(spy_pos)
        result = evaluate_window(ticker,frame,benchmark,stock.config,day) if baseline[t] or validate else {'status':'no_signal','reason_codes':[]}
        diagnostics['evaluated_prefixes'] += 1
        if validate:
            reference = evaluate_window(ticker,frame,benchmark,stock.config,day,reference=True)
            if not equal_results(result,reference):
                raise AssertionError(f'Fast/reference mismatch {ticker} {day}: {result} vs {reference}')
            if not baseline[t] and result['status'] == 'signal':
                raise AssertionError(f'Prefilter lost reference signal {ticker} {day}')
            diagnostics['reference_comparisons'] += 1
        simple_row = simple_signal(ticker,frame,stock.config,day) if simple[t] else None
        candidates = []
        if result['status'] == 'signal':
            candidates.append(('baseline',result['record']))
        if simple_row:
            candidates.append(('simple',simple_row))
        for strategy,row in candidates:
            key = (strategy,row['SetupPeakDate'],row['Stage'])
            if day >= first:
                diagnostics[f"raw_{strategy}_{row['Stage']}"] += 1
            if key in seen:
                diagnostics[f'duplicates_{strategy}'] += int(day >= first)
                continue
            seen.add(key)
            if day < first:
                continue
            idx = spy.full.iloc[spy_pos]
            regime = 'above_50' if idx.Close > idx.SMAFast else 'below_50'
            records.append({**row,'Strategy':strategy,'MarketRegime':regime,'Year':int(day[:4]),
                            'EpisodeStageID':hashlib.sha256(f'{ticker}|{key}'.encode()).hexdigest()[:24],
                            'EarningsStatus':'not_reconstructed'})
    return records,dict(diagnostics)


def metrics(frame):
    complete = frame.loc[frame.Status.eq('complete')]
    result = {'Signals':int(frame.EpisodeStageID.nunique()),'Complete':len(complete),'Pending':int(frame.Status.eq('pending').sum()),
              'Missing':int((~frame.Status.isin(['complete','pending'])).sum())}
    if complete.empty:
        return {**result,**dict.fromkeys(['MeanPct','MedianPct','WinRatePct','ProfitFactor','ExcessMeanPct','MFEMeanPct','MAEMeanPct'],None)}
    net = complete.NetReturnPct
    gains,losses = float(net[net>0].sum()),float(-net[net<0].sum())
    return {**result,'MeanPct':float(net.mean()),'MedianPct':float(net.median()),'WinRatePct':float(net.gt(0).mean()*100),
            'ProfitFactor':gains/losses if losses>0 else None,'NoLosses':losses==0 and gains>0,
            'ExcessMeanPct':float(complete.ExcessReturnPct.mean()),'MFEMeanPct':float(complete.MFEPercent.mean()),
            'MAEMeanPct':float(complete.MAEPercent.mean()),'WorstMAEPct':float(complete.MAEPercent.min()),
            'BestMFEPct':float(complete.MFEPercent.max()),'SPYMeanPct':float(complete.SPYNetReturnPct.mean())}


def aggregate(trades, period_start, period_end):
    rows = []
    years = (pd.Timestamp(period_end)-pd.Timestamp(period_start)).days/365.2425
    for keys,group in trades.groupby(['Strategy','Stage','Horizon','TopN'],dropna=False):
        base = dict(zip(['Strategy','Stage','Horizon','TopN'],keys))
        rows.append({**base,'Dimension':'all','Group':'all',**metrics(group),'SignalsPerYear':group.EpisodeStageID.nunique()/years})
        for dimension,column in [('year','Year'),('regime','MarketRegime'),('score','ScoreBand')]:
            if dimension=='score' and base['Strategy']!='baseline':
                continue
            for label,part in group.groupby(column,dropna=False):
                rows.append({**base,'Dimension':dimension,'Group':str(label),**metrics(part)})
    return pd.DataFrame(rows)


def run(root,scope,offline=False):
    root = Path(root)
    config,sessions,now = load_config(),Sessions(),utc_now()
    hashes = source_hashes()
    if any(hashes[name]!=expected for name,expected in BASELINE_HASHES.items()):
        raise ValueError('Frozen baseline source/config has changed')
    latest = json.loads((root/'output/pullback/latest_success.json').read_text(encoding='utf-8'))
    as_of = latest['as_of']
    universe = pd.read_csv(Path(latest['run'])/'universe.csv',keep_default_na=False)
    if scope=='pilot':
        # Fixed alphabetical sampling plus the known transition for correctness,
        # not an attempt to select stocks with profitable past returns.
        chosen = universe.iloc[np.linspace(0,len(universe)-1,7,dtype=int)].Ticker.tolist()+['AAPL','SPNT']
        universe = universe.loc[universe.Ticker.isin(chosen)]
    else:
        gate = json.loads((root/'output/pullback/backtest/pilot_validation.json').read_text(encoding='utf-8'))
        if not gate['passed'] or gate['source_hashes']!=hashes:
            raise ValueError('A passing small pilot with identical source is required before full expansion')
    earliest = (pd.Timestamp(as_of)-pd.DateOffset(years=max(PROTOCOL['years']))).date().isoformat()
    start = (pd.Timestamp(earliest)-pd.Timedelta(days=PROTOCOL['warmup_calendar_days'])).date().isoformat()
    output = root/'output/pullback/backtest'/f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{scope}"
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',{**PROTOCOL,'as_of':as_of,'download_start':start,'evaluation_start_10y':earliest})
    write_json(output/'baseline_config.json',config)
    write_json(output/'source_hashes.json',hashes)
    write_csv(output/'universe.csv',universe)
    print(f'Backtest {scope}: {output}; {len(universe)} tickers; {start}..{as_of}',flush=True)
    data = HistoricalData(root/'cache/backtest',start,as_of,config,offline)
    ticker,raw,meta,source = next(data.fetch(['SPY']))
    if raw.empty:
        raise RuntimeError('Historical SPY unavailable')
    spy = Series('SPY',raw,sessions,config)
    all_records,all_outcomes,statuses,diagnostics = [],[],[],Counter()
    began = time.perf_counter()
    for index,(records,outcomes,counts,item) in enumerate(processed_results(data,universe,scope,earliest,as_of,spy,sessions,config),start=1):
        diagnostics.update(counts)
        all_records+=records
        all_outcomes+=outcomes
        statuses.append(item)
        if index%25==0 or index==len(universe):
            print(f'Processed {index}/{len(universe)}; signals {len(all_records)}; {time.perf_counter()-began:.1f}s',flush=True)
            write_csv(output/'data_status.csv',pd.DataFrame(statuses))
            write_csv(output/'signals_checkpoint.csv',pd.DataFrame(all_records))
    signals = pd.DataFrame(all_records)
    if signals.empty:
        raise RuntimeError('No signals found; see saved data status, no evaluation published')
    signals = signals.sort_values(['MarketDate','Strategy','Stage','Score','AverageDollarVolume','Ticker'],ascending=[True,True,True,False,False,True],na_position='last')
    signals['DailyRank'] = signals.groupby(['MarketDate','Strategy','Stage']).cumcount()+1
    def band(row):
        if row.Strategy!='baseline':
            return 'not_scored'
        bounds = PROTOCOL['score_bands'][row.Stage]
        for low,high in zip(bounds,bounds[1:]):
            if low <= row.Score < high:
                return f'{low:g}–{min(high,100):g}'
        return 'outside_display_band'
    signals['ScoreBand'] = signals.apply(band,axis=1)
    trades = pd.DataFrame(all_outcomes).merge(signals[['EpisodeStageID','DailyRank','ScoreBand']],on='EpisodeStageID',validate='many_to_one')
    write_csv(output/'signals.csv',signals)
    write_csv(output/'trades.csv',trades)
    reports = []
    for years in PROTOCOL['years']:
        first = (pd.Timestamp(as_of)-pd.DateOffset(years=years)).date().isoformat()
        sub = trades.loc[trades.MarketDate.ge(first)]
        # Ranking happens before reading any future return. Missing exits remain
        # in the selection, rather than being replaced by lower-ranked winners.
        variations = [sub.assign(TopN='all')]+[sub.loc[sub.DailyRank.le(n)].assign(TopN=str(n)) for n in PROTOCOL['top_n']]
        aggregate_frame = aggregate(pd.concat(variations,ignore_index=True),first,as_of)
        aggregate_frame.insert(0,'PeriodYears',years)
        reports.append(aggregate_frame)
    results = pd.concat(reports,ignore_index=True)
    write_csv(output/'metrics.csv',results)
    summary = {'scope':scope,'as_of':as_of,'started_at_utc':now.isoformat(),'finished_at_utc':utc_now().isoformat(),
               'elapsed_seconds':round(time.perf_counter()-began,3),'config_hash':config_hash(config),'target_count':len(universe),
               'available_count':sum(s['Status']=='available' for s in statuses),'unique_signals':len(signals),'diagnostics':dict(diagnostics),
               'trade_status_counts':trades.Status.value_counts().to_dict(),'output_dir':str(output.resolve())}
    write_json(output/'run_status.json',summary)
    if scope=='pilot':
        write_json(root/'output/pullback/backtest/pilot_validation.json',{'passed':True,'source_hashes':hashes,'reference_comparisons':diagnostics['reference_comparisons'],'run':str(output.resolve())})
    write_json(root/f'output/pullback/backtest/latest_{scope}.json',{'run':str(output.resolve())})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--scope',choices=['pilot','full'],default='pilot')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    run(args.root,args.scope,args.offline)
