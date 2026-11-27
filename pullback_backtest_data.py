"""Isolated, resumable historical snapshots. No live cache/history writes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import time

import pandas as pd
import yfinance as yf

from app import extract_ticker_frame
from pullback_data import utc_now, write_json


class HistoricalData:
    def __init__(self, root: Path, start: str, end: str, config: dict, offline=False):
        self.root = Path(root) / f"{start}_{end}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.start, self.end, self.config, self.offline = start, end, config, offline
        yf.set_tz_cache_location(str(self.root / 'yfinance'))

    def paths(self, ticker):
        if not re.fullmatch(r'[A-Z][A-Z0-9-]{0,9}', ticker):
            raise ValueError(ticker)
        return self.root / f'{ticker}.csv.gz', self.root / f'{ticker}.json'

    def cached(self, ticker):
        csv, meta_path = self.paths(ticker)
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta['start'] != self.start or meta['end'] != self.end or meta['schema'] != 1:
                return None
            if hashlib.sha256(csv.read_bytes()).hexdigest() != meta['sha256']:
                return None
            raw = pd.read_csv(csv, index_col='Date', parse_dates=True)
            return raw, meta
        except (OSError, ValueError, KeyError):
            return None

    def fetch(self, tickers):
        pending = []
        for ticker in dict.fromkeys(tickers):
            cached = self.cached(ticker)
            if cached is not None:
                yield ticker, cached[0], cached[1], 'cache'
            elif self.offline:
                yield ticker, pd.DataFrame(), {'error':'no valid historical snapshot'}, 'cache_miss'
            else:
                pending.append(ticker)
        d, failed = self.config['data'], 0
        for pos in range(0, len(pending), d['batch_size']):
            batch = pending[pos:pos + d['batch_size']]
            if failed >= d['max_failed_batches']:
                for ticker in batch:
                    yield ticker, pd.DataFrame(), {'error':'consecutive empty batch circuit breaker'}, 'not_attempted'
                continue
            print(f'History {pos+1}-{pos+len(batch)}/{len(pending)}', flush=True)
            found, retry, error = {}, batch[:], ''
            for attempt in range(d['attempts']):
                try:
                    frame = yf.download(tickers=retry, start=self.start, end=(pd.Timestamp(self.end)+pd.Timedelta(days=1)).date().isoformat(),
                        interval='1d', auto_adjust=False, actions=True, repair=False, group_by='ticker', threads=d['threads'],
                        progress=False, prepost=False, ignore_tz=True, timeout=d['timeout_seconds'])
                    for ticker in retry:
                        part = extract_ticker_frame(frame, ticker, len(retry))
                        if not part.empty and 'Close' in part and part['Close'].notna().any():
                            part = part.dropna(how='all', subset=['Open','High','Low','Close','Volume'])
                            part.index = pd.DatetimeIndex(part.index).tz_localize(None).normalize()
                            part = part.loc[(part.index >= self.start) & (part.index <= self.end)]
                            if not part.empty:
                                found[ticker] = part
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                retry = [ticker for ticker in retry if ticker not in found]
                if not retry:
                    break
                if attempt + 1 < d['attempts']:
                    time.sleep(d['retry_seconds'])
            failed = failed + 1 if not found else 0
            for ticker in batch:
                if ticker not in found:
                    yield ticker, pd.DataFrame(), {'error':error or 'provider returned no history'}, 'fetch_failed'
                    continue
                raw = found[ticker]
                csv, meta_path = self.paths(ticker)
                temp = csv.with_suffix('.tmp')
                raw.to_csv(temp, index_label='Date', compression='gzip')
                temp.replace(csv)
                meta = {'schema':1, 'start':self.start, 'end':self.end, 'fetched_at_utc':utc_now().isoformat(),
                        'source':'Yahoo Finance via yfinance', 'auto_adjust':False, 'repair':False, 'actions':True,
                        'rows':len(raw), 'first_bar':str(raw.index.min().date()), 'last_bar':str(raw.index.max().date()),
                        'sha256':hashlib.sha256(csv.read_bytes()).hexdigest()}
                write_json(meta_path, meta)
                yield ticker, raw, meta, 'network'
            time.sleep(d['pause_seconds'])
