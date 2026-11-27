"""Immutable observed schedules, separate from reported earnings event dates."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def aware(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError('An explicit timezone is required')
    return stamp.astimezone(timezone.utc)


def archive_snapshot(directory: Path, ticker: str, payload: dict) -> Path:
    if not re.fullmatch(r'[A-Z][A-Z0-9-]{0,9}', ticker):
        raise ValueError('Invalid earnings ticker')
    aware(payload['fetched_at_utc'])
    observed = payload.get('observed_at_utc')
    if observed:
        aware(observed)
    if not isinstance(payload['dates'], list) or not all(isinstance(d, str) for d in payload['dates']):
        raise ValueError('Invalid earnings dates')
    snapshot = {'schema': 1, 'ticker': ticker, 'as_of': payload['as_of'],
                'dates': payload['dates'], 'fetched_at_utc': payload['fetched_at_utc'],
                'observed_at_utc': observed or None, 'source': 'Yahoo Finance via yfinance',
                'timestamp_basis': 'response_received' if observed else 'legacy_fetch_start_unverified'}
    body = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    path = Path(directory) / ticker / (hashlib.sha256(body.encode()).hexdigest() + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(body)
    except FileExistsError:
        if path.read_text(encoding='utf-8') != body:
            raise ValueError('Earnings archive content mismatch')
    return path


def load_snapshots(directory: Path, ticker: str) -> list[dict]:
    if not re.fullmatch(r'[A-Z][A-Z0-9-]{0,9}', ticker):
        raise ValueError('Invalid earnings ticker')
    result = []
    for path in sorted((Path(directory) / ticker).glob('*.json')):
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != path.stem:
            raise ValueError(f'Earnings archive hash mismatch: {path.name}')
        item = json.loads(body)
        if item['schema'] != 1 or item['ticker'] != ticker:
            raise ValueError('Earnings archive schema mismatch')
        result.append(item)
    return result


def latest_known(snapshots: list[dict], decision_at: str, max_age_hours: float) -> dict | None:
    """Never promote event dates or legacy request-start times to observed times."""
    cutoff = aware(decision_at)
    eligible = [item for item in snapshots if item.get('observed_at_utc')
                and item.get('timestamp_basis') == 'response_received'
                and 0 <= (cutoff - aware(item['observed_at_utc'])).total_seconds() <= max_age_hours * 3600]
    if not eligible:
        return None
    newest = max(aware(item['observed_at_utc']) for item in eligible)
    latest = [item for item in eligible if aware(item['observed_at_utc']) == newest]
    if len({tuple(item['dates']) for item in latest}) > 1:
        raise ValueError('Conflicting earnings observations at the same time')
    return latest[0]


def import_cache(cache_dir: Path) -> int:
    """Retain existing files without inventing a historical observation time."""
    count = 0
    for path in sorted(Path(cache_dir).glob('*.json')):
        archive_snapshot(Path(cache_dir) / 'snapshots', path.stem, json.loads(path.read_text(encoding='utf-8')))
        count += 1
    return count


def persist_archives(cache_dir: Path, history_dir: Path) -> int:
    """Promote verified observations after price quality passes, for durable history."""
    count = 0
    for folder in sorted(Path(cache_dir).iterdir()) if Path(cache_dir).exists() else []:
        if folder.is_dir():
            for snapshot in load_snapshots(cache_dir, folder.name):
                archive_snapshot(history_dir, folder.name, snapshot)
                count += 1
    return count
