"""Publish validated daily snapshots and preserve weekly navigation on regeneration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from site_navigation import add_navigation

SUMMARY_FIELDS = ['as_of','finished_at_utc','market_regime','price_success_rate','price_success_count',
                  'target_count','analysable_count','elapsed_seconds','scope','quality_passed',
                  'selected_main','selected_watch','review_count','earnings_excluded_count']
CARD_FIELDS = ['Ticker','Name','Exchange','MarketDate','Stage','Bucket','Selected','Score','Price',
               'DrawdownPct','RewardRisk','ScoreBreakdown','Reasons','PullbackDays','SetupPeakDate',
               'ReferenceStop','DepthATR','EarningsStatus','EarningsNextDate','EarningsFetchedAtUTC','SetupID']
TRANSITION_FIELDS = ['Ticker','SetupID','FormingDate','ConfirmedDate','FormingScore','ConfirmedScore']
UPDATE_POLICY = '米国市場終了後、日本時間の火〜土曜7:30頃に自動更新。休場日は更新せず、取得品質が未達の日は前回の正常結果を維持します。'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def projected(row, fields):
    return {key:row[key] for key in fields if key in row}


def save(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text,encoding='utf-8')
    temporary.replace(path)


def weekly(root, source=None):
    root = Path(root)
    source = Path(source) if source else root/'docs/index.html'
    result = add_navigation(source.read_text(encoding='utf-8'))
    save(root/'docs/index.html',result)
    return result


def needs_update(root, now=None):
    from pullback_data import Sessions
    from pullback_settings import load_config, config_hash
    root = Path(root)
    config = load_config(root/'pullback_config.json')
    latest = Sessions().last_completed(now or datetime.now(timezone.utc),config['data']['close_delay_minutes'])
    path = root/'data/pullback/latest.json'
    if not path.exists():
        return True
    saved = read_json(path)
    return saved['summary']['as_of'] != latest or saved['config_hash'] != config_hash(config)


def extract(root, now=None, refresh_caps=False):
    import pandas as pd
    from market_caps import fetch_snapshot
    from pullback_data import Sessions
    from pullback_settings import config_hash, load_config
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    pointer = read_json(root/'output/pullback/latest_success.json')
    run = Path(pointer['run'])
    summary, config = read_json(run/'run_status.json'),load_config(run/'config.json')
    fingerprint = config_hash(config)
    latest = Sessions().last_completed(now,config['data']['close_delay_minutes'])
    if (not summary.get('quality_passed') or summary.get('status') != 'complete' or summary.get('scope') != 'full_universe'
        or summary['as_of'] != latest or pointer['as_of'] != latest or pointer['config_hash'] != fingerprint
        or config_hash(load_config(root/'pullback_config.json')) != fingerprint):
        raise ValueError('Publication requires a current, complete, full-universe, quality-passed run with matching config')
    status = pd.read_csv(run/'data_status.csv',keep_default_na=False)
    successes = int(status.PriceSuccess.astype(str).str.lower().eq('true').sum())
    if (len(status) != summary['target_count'] or not len(status) or status.Ticker.duplicated().any()
        or successes != summary['price_success_count'] or successes/len(status) < config['data']['minimum_success_rate']
        or abs(successes/len(status)-summary['price_success_rate']) > 1e-9
        or int(status.DataStatus.eq('ok').sum()) != summary['analysable_count']):
        raise ValueError('Publication coverage does not match the saved quality report')
    records = pd.read_csv(run/'all_candidates.csv',keep_default_na=False).to_dict('records')
    tickers = [row['Ticker'] for row in records]
    if (len(set(tickers)) != len(tickers) or set(tickers) != set(status.loc[status.Decision.eq('signal'),'Ticker'])
        or not set(tickers).issubset(set(status.loc[status.DataStatus.eq('ok'),'Ticker']))):
        raise ValueError('Published candidates do not match the per-ticker detector decisions')
    selected = [r for r in records if str(r.get('Selected')).lower() == 'true']
    if (len(selected) > config['rules']['maximum_selected']
        or any(r['MarketDate'] != latest or r['Bucket'] not in {'main','watch','review','earnings_excluded'} for r in records)
        or any(r['Bucket'] not in {'main','watch'} for r in selected)
        or sum(r['Bucket']=='main' for r in selected) != summary['selected_main']
        or sum(r['Bucket']=='watch' for r in selected) != summary['selected_watch']
        or sum(r['Bucket']=='review' for r in records) != summary['review_count']
        or sum(r['Bucket']=='earnings_excluded' for r in records) != summary['earnings_excluded_count']):
        raise ValueError('Candidate buckets disagree with the scan report')
    recent = None
    recent_pointer = root/'output/pullback/latest_recent.json'
    if recent_pointer.exists():
        recent_run = Path(read_json(recent_pointer)['run'])
        trial = read_json(recent_run/'trial_status.json')
        if trial['config_hash'] == fingerprint and trial['dates'] and max(trial['dates']) <= latest:
            recent = {'summary':projected(trial,['dates','limitations']),
                      'records':[projected(r,CARD_FIELDS) for r in pd.read_csv(recent_run/'all_candidates.csv',keep_default_na=False).to_dict('records')],
                      'linked':[projected(r,TRANSITION_FIELDS) for r in pd.read_csv(recent_run/'transitions.csv',keep_default_na=False).to_dict('records')]}
    previous = root/'data/pullback/latest.json'
    if recent is None and previous.exists():
        previous = read_json(previous)
        if previous['config_hash'] == fingerprint:
            recent = previous.get('recent')
    cap_path, caps, cap_error = root/'output/pullback/ui_market_caps.json',None,''
    if refresh_caps:
        try:
            caps = fetch_snapshot()
            save(cap_path,json.dumps(caps,ensure_ascii=False,indent=2,allow_nan=False))
        except Exception as exc:
            cap_error = f'{type(exc).__name__}: {exc}'
    elif cap_path.exists():
        saved = read_json(cap_path)
        age = (now-datetime.fromisoformat(saved['fetched_at_utc'])).total_seconds()
        if 0 <= age < 86400:
            caps = saved
    if caps:
        wanted = {r['Ticker'] for r in records+(recent or {}).get('records',[])}
        caps = {'fetched_at_utc':caps['fetched_at_utc'],'entries':{k:v for k,v in caps['entries'].items() if k in wanted}}
    history = root/'history/pullback/signals.csv'
    counts = pd.read_csv(history).Stage.value_counts().to_dict() if history.exists() else {}
    public_summary = {**projected(summary,SUMMARY_FIELDS),'update_policy':UPDATE_POLICY}
    return {'schema':1,'config_hash':fingerprint,'config':config,'summary':public_summary,
            'records':[projected(r,CARD_FIELDS) for r in records],'recent':recent,'caps':caps,
            'history_counts':counts,'market_caps_error':'unavailable' if cap_error else ''}


def daily(root, now=None, refresh_caps=False):
    from pullback_ui import render_daily, render_recent, page
    root = Path(root)
    payload = extract(root,now,refresh_caps)
    weekly_html = (root/'docs/index.html').read_text(encoding='utf-8')
    style = re.search(r'<style>(.*?)</style>',weekly_html,flags=re.S)
    if style is None:
        raise ValueError('Weekly style is unavailable')
    recent, config, caps = payload['recent'],payload['config'],payload['caps']
    rendered = render_daily(payload['summary'],payload['records'],config,style[1],caps,
                            recent and recent['summary'],payload['history_counts'],local=False)
    historical = render_recent(recent['summary'],recent['records'],recent['linked'],config,style[1],caps,local=False) if recent else page(
        '過去日の検出例',style[1],'<h1>過去日の検出例はまだありません</h1><a href="daily.html">日足の候補へ戻る →</a>',local=False)
    encoded = json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)
    # Complete validation and render all content before changing public files.
    save(root/'data/pullback/latest.json',encoded)
    save(root/'docs/daily.html',rendered)
    save(root/'docs/recent.html',historical)
    result = {'as_of':payload['summary']['as_of'],'config_hash':payload['config_hash'],
              'daily_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'market_caps_error':payload['market_caps_error']}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['weekly','daily','needs-update'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--refresh-caps',action='store_true')
    args = parser.parse_args()
    if args.mode == 'weekly':
        weekly(args.root,args.source)
    elif args.mode == 'daily':
        daily(args.root,refresh_caps=args.refresh_caps)
    else:
        print('true' if needs_update(args.root) else 'false')
