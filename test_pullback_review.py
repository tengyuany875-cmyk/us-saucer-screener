from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pullback_backtest_engine import Series, evaluate_window
from pullback_data import Sessions, earnings
from pullback_earnings_archive import archive_snapshot, import_cache, latest_known, load_snapshots, persist_archives
from pullback_review import compare_orders, known_earnings, next_session_diagnostic, observed_schedules, opening_fill, opening_limit
from pullback_settings import load_config
from test_pullback import market_frames


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions, cls.config = Sessions(), load_config()

    def row(self):
        return {'EpisodeStageID':'episode', 'Ticker':'TEST', 'MarketDate':'2026-11-24',
                'SetupHigh':110.,'ReferenceStop':90.,'AdjustedClose':95.,'Price':95.}

    def stock(self, opening=98., split=1.):
        raw=pd.DataFrame({'Open':[95./split,opening/split], 'Close':[95./split,99./split]},
                         index=pd.to_datetime(['2026-11-24','2026-11-25']))
        return SimpleNamespace(raw=raw, dates=raw.index, future_split=np.array([split,1.]))

    def test_limit_is_precomputed_and_open_only_controls_fill(self):
        row=self.row()
        self.assertEqual(opening_limit(row,1.5),98.)
        row['FutureReturn']=1000
        self.assertEqual(opening_limit(row,1.5),98.)
        self.assertEqual(opening_fill(row,self.stock(98),self.sessions,'2026-11-25',1.5)['FillStatus'],'filled')
        self.assertEqual(opening_fill(row,self.stock(98.01),self.sessions,'2026-11-25',1.5)['FillStatus'],'not_filled')
        changed=self.stock(97)
        changed.raw.loc['2026-11-25','Close']=1e9
        self.assertEqual(opening_fill(row,changed,self.sessions,'2026-11-25',1.5)['FillStatus'],'filled')

    def test_downward_gap_below_stop_is_not_removed_after_observing_open(self):
        result=opening_fill(self.row(),self.stock(80),self.sessions,'2026-11-25',1.5)
        self.assertEqual(result['FillStatus'],'filled')
        self.assertTrue(result['EntryAtOrBelowStop'])
        self.assertIsNone(result['EntryRewardRisk'])

    def test_split_units_do_not_change_fill_and_dividend_adjustment_is_not_used(self):
        row=self.row()
        a=opening_fill(row,self.stock(98),self.sessions,'2026-11-25',1.5)
        b=opening_fill(row,self.stock(98,20),self.sessions,'2026-11-25',1.5)
        self.assertEqual(a,b)
        adjusted=self.stock(98.01)
        adjusted.raw['Adj Close']=adjusted.raw.Close*.1
        self.assertEqual(opening_fill(row,adjusted,self.sessions,'2026-11-25',1.5)['FillStatus'],'not_filled')

    def test_pending_and_missing_open_are_never_counted_as_nonfills(self):
        self.assertEqual(opening_fill(self.row(),self.stock(),self.sessions,'2026-11-24',1.5)['FillStatus'],'pending_open')
        self.assertEqual(opening_fill(self.row(),self.stock(np.nan),self.sessions,'2026-11-25',1.5)['FillStatus'],'missing_open')
        with self.assertRaises(ValueError):
            opening_limit({**self.row(),'ReferenceStop':120.},1.5)

    def test_comparison_reports_unfilled_and_pending_separately_without_zero_returns(self):
        events=pd.DataFrame([
            {'EpisodeStageID':str(i),'Stage':'confirmed','Status':status,'NetReturnPct':net,
             'SPYNetReturnPct':0.,'ExcessReturnPct':net,'MFEPercent':5.,'MAEPercent':-5.}
            for i,status,net in [(0,'complete',2.),(1,'complete',-4.),(2,'pending',np.nan)]])
        orders=pd.DataFrame([{'EpisodeStageID':'0','FillStatus':'filled'},
                             {'EpisodeStageID':'1','FillStatus':'not_filled'},
                             {'EpisodeStageID':'2','FillStatus':'filled'}])
        comparison,_=compare_orders(events,orders)
        baseline,limit=comparison.to_dict('records')
        self.assertEqual(baseline['MeanPct'],-1.)
        self.assertEqual(limit['Orders'],3)
        self.assertEqual(limit['Filled'],2)
        self.assertEqual(limit['Complete'],1)
        self.assertEqual(limit['Pending'],1)
        self.assertEqual(limit['NotFilled'],1)
        self.assertEqual(limit['Missing'],0)
        self.assertEqual(limit['MeanPct'],2.)

    def payload(self, observed='2026-11-24T20:00:00+00:00'):
        return {'as_of':'2026-11-24','fetched_at_utc':'2026-11-24T19:59:00+00:00',
                'observed_at_utc':observed,'dates':['2027-02-01T16:00:00-05:00']}

    def test_schedule_observation_is_immutable_and_asof_not_event_date(self):
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)
            first=archive_snapshot(archive,'TEST',self.payload())
            self.assertEqual(first,archive_snapshot(archive,'TEST',self.payload()))
            second={**self.payload('2026-11-25T12:00:00+00:00'),'dates':['2026-11-25T16:00:00-05:00']}
            archive_snapshot(archive,'TEST',second)
            snapshots=load_snapshots(archive,'TEST')
            self.assertEqual(len(snapshots),2)
            self.assertEqual(latest_known(snapshots,'2026-11-24T21:00:00+00:00',6)['dates'],self.payload()['dates'])
            self.assertIsNone(latest_known(snapshots,'2026-11-24T19:00:00+00:00',6))
            self.assertIsNone(latest_known(snapshots,'2026-11-25T03:00:00+00:00',6))
            first.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):
                load_snapshots(archive,'TEST')

    def test_legacy_cache_cannot_invent_a_received_timestamp(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'TEST.json'
            payload=self.payload()
            del payload['observed_at_utc']
            path.write_text(json.dumps(payload),encoding='utf-8')
            self.assertEqual(import_cache(Path(temp)),1)
            original=path.read_bytes()
            snapshots=load_snapshots(Path(temp)/'snapshots','TEST')
            self.assertIsNone(latest_known(snapshots,'2026-11-24T21:00:00+00:00',6))
            self.assertEqual(path.read_bytes(),original)

    def test_earnings_archive_survives_cache_replacement_in_separate_history(self):
        with tempfile.TemporaryDirectory() as temp:
            cache,history=Path(temp)/'cache',Path(temp)/'history'
            first=archive_snapshot(cache,'TEST',self.payload())
            original=first.read_bytes()
            self.assertEqual(persist_archives(cache,history),1)
            self.assertEqual(persist_archives(cache,history),1)
            self.assertEqual(len(load_snapshots(history,'TEST')),1)
            self.assertEqual(first.read_bytes(),original)
            self.assertEqual((history/'TEST'/first.name).read_bytes(),original)

    def test_replay_reads_durable_history_even_after_cache_observation_is_lost(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cache=root/'cache/daily/earnings/snapshots'
            history=root/'history/pullback/earnings'
            first=archive_snapshot(cache,'TEST',self.payload())
            persist_archives(cache,history)
            self.assertEqual(len(observed_schedules(root,'TEST')),1)
            first.unlink()
            observations=observed_schedules(root,'TEST')
            self.assertEqual(len(observations),1)
            self.assertEqual(known_earnings(self.row(),observations,self.sessions,self.config)['Status'],'outside_holding')

    def test_earnings_gate_uses_only_received_schedule_and_unknown_stays_unknown(self):
        row=self.row()
        with tempfile.TemporaryDirectory() as temp:
            archive_snapshot(Path(temp),'TEST',self.payload())
            known=known_earnings(row,load_snapshots(Path(temp),'TEST'),self.sessions,self.config)
            self.assertEqual(known['Status'],'outside_holding')
            unknown=known_earnings(row,[],self.sessions,self.config)
            self.assertFalse(unknown['KnownSnapshot'])
            self.assertEqual(unknown['Status'],'unknown')

    def test_live_fetch_keeps_response_received_snapshot_for_future_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            now=datetime(2026,11,25,12,tzinfo=timezone.utc)
            received=datetime(2026,11,25,12,0,2,tzinfo=timezone.utc)
            with patch('pullback_data.yf.Ticker') as ticker, patch('pullback_data.utc_now',return_value=received):
                ticker.return_value.get_earnings_dates.return_value=pd.DataFrame({'Event Type':['Earnings']},
                    index=pd.DatetimeIndex(['2027-02-01T16:00:00-05:00']))
                result=earnings('TEST',Path(temp),self.config,self.sessions,'2026-11-24',now)
            self.assertEqual(result['status'],'outside_holding')
            snapshots=load_snapshots(Path(temp)/'earnings/snapshots','TEST')
            self.assertEqual(len(snapshots),1)
            self.assertIsNone(latest_known(snapshots,now.isoformat(),6))
            self.assertIsNotNone(latest_known(snapshots,received.isoformat(),6))

    def test_followup_classification_agrees_with_detector_and_ignores_later_bars(self):
        _, raw, spy_raw=market_frames()
        def series(ticker,frame):
            frame=frame.copy();frame['Stock Splits']=0.
            return Series(ticker,frame,self.sessions,self.config)
        stock,spy=series('TEST',raw),series('SPY',spy_raw)
        t=len(raw)-2
        day=str(raw.index[t].date())
        forming=evaluate_window('TEST',stock.window(t),spy.window(t),self.config,day)['record']
        forming['EpisodeStageID']='episode'
        result=next_session_diagnostic(forming,stock,spy,self.sessions,'2026-11-25')
        self.assertEqual(result['Outcome'],'confirmed')
        self.assertEqual(result['DetectedStage'],'confirmed')
        self.assertTrue(result['SamePeak'])
        future=raw.iloc[-1:].copy()
        future.index=self.sessions.following(str(raw.index[-1].date()),1)
        future.loc[:,'Close']=np.nan
        after=series('TEST',pd.concat([raw,future]))
        self.assertEqual(result,next_session_diagnostic(forming,after,spy,self.sessions,'2026-11-25'))


if __name__ == '__main__':
    unittest.main()
