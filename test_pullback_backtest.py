from __future__ import annotations
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from pullback_backtest import PROTOCOL, equal_results, metrics, scan
from pullback_backtest_engine import Series, array_wilder, evaluate_window, outcome
from pullback_data import Sessions
from pullback_screener import indicators, wilder
from pullback_settings import load_config
from test_pullback import market_frames


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.sessions=load_config(),Sessions()

    def series(self,ticker,raw):
        raw=raw.copy()
        if 'Stock Splits' not in raw:
            raw['Stock Splits']=0.
        return Series(ticker,raw,self.sessions,self.config)

    def test_array_wilder_matches_original_recurrence_exactly(self):
        for values in [[np.nan,1.,2.,3.,9.], [np.nan]*20, [0.]*1000, [*np.random.default_rng(2).normal(size=3000)]]:
            sample=pd.Series(values)
            pd.testing.assert_series_equal(array_wilder(sample,14),wilder(sample,14))

    def test_original_detector_and_accelerated_body_agree_on_both_states(self):
        formed,full,spy_raw=market_frames()
        stock,spy=self.series('TEST',full),self.series('SPY',spy_raw)
        for t,stage in [(len(full)-2,'forming'),(len(full)-1,'confirmed')]:
            day=str(full.index[t].date())
            frame,benchmark=stock.window(t),spy.window(t)
            fast=evaluate_window('TEST',frame,benchmark,self.config,day)
            original=evaluate_window('TEST',frame,benchmark,self.config,day,reference=True)
            self.assertTrue(equal_results(fast,original))
            self.assertEqual(fast['record']['Stage'],stage)
            self.assertTrue(stock.candidates(spy)[0][t])

    def test_760_day_seeds_match_reference_over_long_history(self):
        dates=self.sessions.between('2018-01-01','2026-11-25')
        rng=np.random.default_rng(42)
        close=100*np.exp(np.cumsum(rng.normal(.0003,.01,len(dates))))
        raw=pd.DataFrame({'Open':close*.998,'High':close*1.012,'Low':close*.988,'Close':close,'Adj Close':close*.97,'Volume':2e6},index=dates)
        stock=self.series('TEST',raw)
        for t in [300,600,1000,len(raw)-1]:
            frame=stock.window(t)
            ref=indicators(frame,self.config)
            for column in ['EMA','ATR','RSI','SMAFast','SMASlow','FastSlope','DollarVolume','PriorVolume']:
                np.testing.assert_allclose(frame[column].tail(40),ref[column].tail(40),rtol=1e-10,atol=1e-8,err_msg=column)

    def test_later_splits_and_dividends_do_not_change_asof_units_or_decision(self):
        _,raw,spy_raw=market_frames()
        raw=raw.copy(); raw['Stock Splits']=0.
        stock=self.series('TEST',raw)
        spy=self.series('SPY',spy_raw)
        day=str(raw.index[-1].date())
        original=stock.window(len(raw)-1)
        later=raw.iloc[-1:].copy()
        later.index=self.sessions.following(day,1)
        current=pd.concat([raw,later])
        current.loc[current.index<=day,['Open','High','Low','Close','Adj Close']]/=20
        current.loc[current.index<=day,'Volume']*=20
        current.loc[current.index[-1],'Stock Splits']=20
        current.loc[current.index<=day,'Adj Close']*=.9  # future dividend adjustment
        rebuilt=self.series('TEST',current).window(len(raw)-1)
        for col in ['RawClose','Volume','EMA','RSI','ATR','DollarVolume','PriorVolume']:
            np.testing.assert_allclose(original[col].tail(30),rebuilt[col].tail(30),rtol=1e-10,atol=1e-8)
        a=evaluate_window('TEST',original,spy.window(len(raw)-1),self.config,day)
        b=evaluate_window('TEST',rebuilt,spy.window(len(raw)-1),self.config,day)
        self.assertTrue(equal_results(a,b))

    def test_future_bad_bar_and_future_prices_never_affect_past_quality(self):
        _,raw,_=market_frames()
        old=self.series('TEST',raw)
        later=raw.iloc[-1:].copy()
        later.index=self.sessions.following(str(raw.index[-1].date()),1)
        later.loc[:,'Close']=np.nan
        combined=self.series('TEST',pd.concat([raw,later]))
        self.assertEqual(old.quality(len(raw)-1),combined.quality(len(raw)-1))
        self.assertFalse(combined.quality(len(raw)))
        pd.testing.assert_frame_equal(old.window(len(raw)-1),combined.window(len(raw)-1))

    def test_entry_horizons_costs_and_excursions_use_regular_sessions(self):
        dates=self.sessions.between('2026-11-24','2026-12-10')
        close=np.full(len(dates),100.)
        raw=pd.DataFrame({'Open':100.,'High':110.,'Low':90.,'Close':close,'Adj Close':close,'Volume':1e6},index=dates)
        raw.loc[dates[3],['Close','Adj Close']]=103
        raw.loc[dates[5],['Close','Adj Close']]=106
        stock=self.series('TEST',raw)
        spy_raw=raw.copy();spy_raw[['Close','Adj Close']]=100.
        spy=self.series('SPY',spy_raw)
        row={'MarketDate':'2026-11-24','Ticker':'TEST'}
        result=outcome(row,stock,spy,self.sessions,'2026-12-10',5,10)
        self.assertEqual(result['EntryDate'],'2026-11-25')
        self.assertEqual(result['ExitDate'],'2026-12-02')
        self.assertAlmostEqual(result['NetReturnPct'],(106*.999/(100*1.001)-1)*100)
        self.assertAlmostEqual(result['MFEPercent'],10.)
        self.assertAlmostEqual(result['MAEPercent'],-10.)
        three=outcome(row,stock,spy,self.sessions,'2026-12-10',3,10)
        self.assertEqual(three['ExitDate'],'2026-11-30')
        ten=outcome(row,stock,spy,self.sessions,'2026-12-10',10,10)
        self.assertEqual(ten['ExitDate'],'2026-12-09')
        pending=outcome(row,stock,spy,self.sessions,'2026-11-25',5,10)
        self.assertEqual(pending['Status'],'pending')
        self.assertNotIn('NetReturnPct',pending)

    def test_missing_or_halted_exit_is_not_zero_return(self):
        dates=self.sessions.between('2026-11-24','2026-12-04')
        raw=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Adj Close':100.,'Volume':1e6},index=dates)
        spy=self.series('SPY',raw)
        row={'MarketDate':'2026-11-24','Ticker':'TEST'}
        stock=self.series('TEST',raw.drop(pd.Timestamp('2026-12-02')))
        result=outcome(row,stock,spy,self.sessions,'2026-12-04',5,10)
        self.assertEqual(result['Status'],'missing_prices')
        self.assertNotIn('NetReturnPct',result)
        raw.loc['2026-12-02','Volume']=0
        halted=outcome(row,self.series('TEST',raw),spy,self.sessions,'2026-12-04',5,10)
        self.assertEqual(halted['Status'],'not_tradable')

    def test_episode_dedup_keeps_stages_separate_and_first_snapshot(self):
        _,raw,spy_raw=market_frames()
        stock,spy=self.series('TEST',raw),self.series('SPY',spy_raw)
        records,counts=scan('TEST',stock,spy,self.sessions,'2026-11-18','2026-11-24',validate=True)
        baseline=[row for row in records if row['Strategy']=='baseline']
        self.assertEqual([row['Stage'] for row in baseline],['forming','confirmed'])
        self.assertEqual(len({row['EpisodeStageID'] for row in baseline}),2)
        self.assertEqual(len({row['SetupPeakDate'] for row in baseline}),1)
        self.assertGreater(counts['reference_comparisons'],0)

    def test_consecutive_forming_days_count_only_the_first(self):
        formed,_,spy_raw=market_frames()
        more=self.sessions.following(str(formed.index[-1].date()),2)
        extension=pd.DataFrame({'Open':101.,'High':102.4,'Low':99.75,'Close':101.1,'Adj Close':101.1,'Volume':3e6},index=more)
        raw=pd.concat([formed,extension])
        spy_extra=spy_raw.iloc[-1:].copy();spy_extra.index=more[-1:]
        spy_raw=pd.concat([spy_raw,spy_extra])
        records,counts=scan('TEST',self.series('TEST',raw),self.series('SPY',spy_raw),self.sessions,'2026-11-18','2026-11-25',validate=True)
        baseline=[row for row in records if row['Strategy']=='baseline' and row['Stage']=='forming']
        self.assertEqual(len(baseline),1)
        self.assertEqual(baseline[0]['MarketDate'],'2026-11-23')
        self.assertGreaterEqual(counts['duplicates_baseline'],2)

    def test_metrics_include_cost_losses_and_unfinished_counts(self):
        rows=[]
        for n,value in enumerate([2.,-1.,0.]):
            rows.append(dict(EpisodeStageID=str(n),Status='complete',NetReturnPct=value,ExcessReturnPct=value-.1,MFEPercent=3.,MAEPercent=-2.,SPYNetReturnPct=.1))
        rows.append(dict(EpisodeStageID='p',Status='pending'))
        result=metrics(pd.DataFrame(rows))
        self.assertEqual(result['Complete'],3)
        self.assertEqual(result['Pending'],1)
        self.assertAlmostEqual(result['ProfitFactor'],2.)
        self.assertAlmostEqual(result['WinRatePct'],100/3)
        self.assertEqual(PROTOCOL['primary_horizon'],5)
        self.assertEqual(PROTOCOL['horizons'],[3,5,10])
        self.assertEqual(PROTOCOL['cost_bps_per_side'],10)


if __name__=='__main__':
    unittest.main()
