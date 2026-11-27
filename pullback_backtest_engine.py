"""Causal prefix calculations and unchanged detector body for baseline evaluation."""
from __future__ import annotations
import types

import numpy as np
import pandas as pd

from pullback_screener import analyse_daily, indicators

PRICE_COLUMNS = ['Open','High','Low','Close']


def array_wilder(series, period):
    """Same seed and recurrence, using arrays to avoid per-cell pandas access."""
    values=series.to_numpy(float)
    out=np.full(len(values),np.nan)
    valid=np.flatnonzero(~np.isnan(values))
    if not len(valid) or len(values)-valid[0]<period:
        return pd.Series(out,index=series.index)
    start=int(valid[0]); seed=start+period-1
    out[seed]=series.iloc[start:seed+1].mean()
    for pos in range(seed+1,len(values)):
        out[pos]=(out[pos-1]*(period-1)+values[pos])/period
    return pd.Series(out,index=series.index)


class Series:
    def __init__(self, ticker, raw, sessions, config):
        required = {*PRICE_COLUMNS, 'Adj Close','Volume','Stock Splits'}
        if not required.issubset(raw.columns):
            raise ValueError('OHLCV, Adj Close and split history are required')
        self.ticker, self.config = ticker, config
        raw = raw.copy().sort_index()
        duplicates = raw.index[raw.index.duplicated(keep=False)].unique()
        raw = raw.loc[~raw.index.duplicated(keep='first')]
        if not len(raw):
            raise ValueError('Empty history')
        self.raw = raw
        self.dates = pd.DatetimeIndex(raw.index)
        all_sessions = sessions.between(self.dates[0], self.dates[-1])
        numeric = raw[list(required)].apply(pd.to_numeric, errors='coerce')
        a = numeric[[*PRICE_COLUMNS,'Adj Close','Volume']].to_numpy(float)
        valid = (np.isfinite(a).all(axis=1) & (a[:,:5] > 0).all(axis=1) & (a[:,5] >= 0)
                 & numeric.High.ge(numeric[['Open','Low','Close']].max(axis=1)-1e-6).to_numpy()
                 & numeric.Low.le(numeric[['Open','High','Close']].min(axis=1)+1e-6).to_numpy()
                 & ~self.dates.isin(duplicates) & self.dates.isin(all_sessions))
        splits = numeric['Stock Splits'].fillna(0).to_numpy(float)
        valid &= np.isfinite(splits) & (splits >= 0)
        # Undo later split normalization for the nominal-price gate. Scaling an
        # entire as-of window uses only one factor; volume scales inversely.
        split_events = np.where(splits > 0, splits, 1.)
        self.future_split = np.r_[np.cumprod(split_events[:0:-1])[::-1], 1.]
        self.factor = (numeric['Adj Close']/numeric.Close).to_numpy(float)
        clean = numeric.replace([np.inf,-np.inf],np.nan).fillna(0.)
        adjusted = clean[[*PRICE_COLUMNS,'Volume']].copy()
        factor = np.where(np.isfinite(self.factor),self.factor,1.)
        for col in PRICE_COLUMNS:
            adjusted[col] *= factor
        adjusted['RawClose'], adjusted['RawOpen'] = clean.Close, clean.Open
        self.base = adjusted
        accelerated=types.FunctionType(indicators.__code__,{**indicators.__globals__,'wilder':array_wilder},indicators.__name__)
        self.full = accelerated(adjusted, config)
        self.full_ema = adjusted.Close.ewm(span=config['indicators']['ema'],adjust=False).mean().to_numpy()
        self.bad_prefix = np.r_[0, np.cumsum(~valid)]
        self.starts = self.dates.searchsorted(self.dates - pd.Timedelta(days=config['data']['history_calendar_days']), side='left')
        self.calendar_positions = all_sessions.get_indexer(self.dates)
        self.valid = valid
        self.volume = clean.Volume.to_numpy(float)
        self.close = adjusted.Close.to_numpy(float)
        self.high = adjusted.High.to_numpy(float)
        self.low = adjusted.Low.to_numpy(float)
        delta = np.diff(self.close, prepend=np.nan)
        self.gain = np.maximum(delta,0)
        self.loss = np.maximum(-delta,0)
        previous = np.r_[np.nan,self.close[:-1]]
        self.tr = np.nanmax(np.c_[self.high-self.low,np.abs(self.high-previous),np.abs(self.low-previous)],axis=1)
        # Exactly the same Wilder seed/recurrence; full-prefix values are the
        # anchor for a linear correction to each 760-calendar-day window.
        self.wgain = array_wilder(pd.Series(self.gain),config['indicators']['rsi']).to_numpy()
        self.wloss = array_wilder(pd.Series(self.loss),config['indicators']['rsi']).to_numpy()

    def quality(self, t):
        l = self.starts[t]
        minimum = self.config['data']['minimum_history']
        return (t-l+1 >= minimum and self.volume[t] > 0 and self.bad_prefix[t+1] == self.bad_prefix[l]
                and self.calendar_positions[t] >= 0 and self.calendar_positions[t-minimum+1] >= 0
                and self.calendar_positions[t]-self.calendar_positions[t-minimum+1] == minimum-1)

    def window(self, t):
        """760-day window with the same seeds as indicators(prefix), no future rows."""
        l, i = int(self.starts[t]), self.config['indicators']
        frame = self.full.iloc[l:t+1].copy()
        n = len(frame)
        positions = np.arange(l,t+1)
        # EMA's true full seed is Close[0], even when min_periods hides it.
        full_ema = self.full_ema
        ema = full_ema[l:t+1] + (1-2/(i['ema']+1))**(positions-l)*(self.close[l]-full_ema[l])
        ema[:i['ema']-1] = np.nan
        frame['EMA'] = ema
        p = i['rsi']
        gain, loss = np.full(n,np.nan), np.full(n,np.nan)
        if n > p:
            seed = l+p
            power = (1-1/p)**(positions[p:]-seed)
            gain[p:] = self.wgain[seed:t+1]+power*(np.mean(self.gain[l+1:seed+1])-self.wgain[seed])
            loss[p:] = self.wloss[seed:t+1]+power*(np.mean(self.loss[l+1:seed+1])-self.wloss[seed])
        with np.errstate(divide='ignore',invalid='ignore'):
            rsi = 100-100/(1+gain/loss)
        rsi[(loss==0)&(gain>0)] = 100
        rsi[(gain==0)&(loss>0)] = 0
        rsi[(gain==0)&(loss==0)] = 50
        frame['RSI'] = rsi
        p = i['atr']
        atr = np.full(n,np.nan)
        if n >= p:
            seed = l+p-1
            seed_value = (np.sum(self.tr[l+1:seed+1])+self.high[l]-self.low[l])/p
            atr[p-1:] = self.full.ATR.to_numpy()[seed:t+1] + (1-1/p)**(positions[p-1:]-seed)*(seed_value-self.full.ATR.iloc[seed])
        frame['ATR'] = atr
        # Other indicators have finite rolling support; remove their warmup.
        for column, period in [('SMAFast',i['sma_fast']),('SMASlow',i['sma_slow']),('DollarVolume',i['liquidity_days']),('PriorVolume',i['liquidity_days']+1),('FastSlope',i['sma_fast']+i['slope_days'])]:
            frame.iloc[:period-1,frame.columns.get_loc(column)] = np.nan
        scale = self.future_split[t]/self.factor[t]
        for col in [*PRICE_COLUMNS,'EMA','SMAFast','SMASlow','ATR']:
            frame[col] *= scale
        frame['RawClose'] *= self.future_split[t]
        frame['RawOpen'] *= self.future_split[t]
        frame['Volume'] /= self.future_split[t]
        frame['PriorVolume'] /= self.future_split[t]
        return frame

    def candidates(self, spy):
        """Broad, vectorized necessary conditions; the detector makes decisions."""
        f, r, i = self.full,self.config['rules'],self.config['indicators']
        positions=np.arange(len(f))
        starts=self.starts
        ema=self.full_ema+(1-2/(i['ema']+1))**(positions-starts)*(self.close[starts]-self.full_ema[starts])
        offsets=np.arange(r['peak_lookback']+2)
        indexes=np.maximum(positions[:,None]-offsets[None,:],0)
        p=i['atr']; seeds=np.minimum(starts+p-1,len(f)-1)
        prefix=np.r_[0.,np.cumsum(self.tr)]
        seed_value=(prefix[seeds+1]-prefix[starts+1]+self.high[starts]-self.low[starts])/p
        atr_full=f.ATR.to_numpy()
        atrs=atr_full[indexes]+(1-1/p)**(indexes-seeds[:,None])*(seed_value-atr_full[seeds])[:,None]
        atrs[indexes<seeds[:,None]]=np.nan
        # These rows are only a necessary-condition filter, never accepted by
        # themselves. All final decisions use the unchanged detector body.
        max_atr=np.max(np.where(np.isnan(atrs),-np.inf,atrs),axis=1)
        p=i['rsi']; seeds=np.minimum(starts+p,len(f)-1)
        rsi_parts=[]
        for raw,full in [(self.gain,self.wgain),(self.loss,self.wloss)]:
            prefix=np.r_[0.,np.cumsum(np.nan_to_num(raw))]
            seed_value=(prefix[seeds+1]-prefix[starts+1])/p
            values=full[indexes]+(1-1/p)**(indexes-seeds[:,None])*(seed_value-full[seeds])[:,None]
            values[indexes<seeds[:,None]]=np.nan
            rsi_parts.append(values)
        gain,loss=rsi_parts
        with np.errstate(divide='ignore',invalid='ignore'):
            rsi=100-100/(1+gain/loss)
        rsi[(loss==0)&(gain>0)]=100; rsi[(gain==0)&(loss>0)]=0; rsi[(gain==0)&(loss==0)]=50
        tolerance = np.maximum(1.,np.abs(self.close))*1e-7
        base = ((self.raw.Close.to_numpy()*self.future_split >= r['minimum_price']-1e-9)
                & f.DollarVolume.ge(r['minimum_dollar_volume']*(1-1e-12)).to_numpy()
                & ((f.Close-f.SMAFast).to_numpy()>-tolerance) & ((f.SMAFast-f.SMASlow).to_numpy()>-tolerance) & f.FastSlope.gt(-1e-12).to_numpy())
        near = self.close-ema <= max_atr*max(r['maximum_extension_atr'],r['support_distance_atr'])+tolerance
        near &= ema-self.close <= max_atr*r['support_distance_atr']+tolerance
        n = i['relative_strength_days']
        spy_close = spy.full.Close.reindex(self.dates)
        rs = (f.Close/f.Close.shift(n))/(spy_close/spy_close.shift(n))-1
        # Missing older dates are dealt with by quality/reference checks.
        baseline = base & near & rs.gt(r['minimum_relative_strength']-1e-10).to_numpy()
        look=r['peak_lookback']; age=np.zeros(len(f),dtype=int)
        if len(f)>look:
            windows=np.lib.stride_tricks.sliding_window_view(self.high,look)
            age[look:]=1+np.argmax(windows[:-1,::-1],axis=1)
        ages_ok=(age>=r['pullback_days_min'])&(age<=r['pullback_days_max'])
        baseline &= ages_ok | np.r_[False,ages_ok[:-1]]
        baseline &= np.any((rsi[:,:r['pullback_days_max']+1]>=r['rsi_min']-1e-7)&(rsi[:,:r['pullback_days_max']+1]<=r['rsi_max']+1e-7),axis=1)
        baseline &= (atrs[:,0]/self.close >= r['atr_pct_min']-1e-10)&(atrs[:,0]/self.close<=r['atr_pct_max']+1e-10)
        location=(self.close-self.low)/np.maximum(self.high-self.low,1e-100)
        bounce=(self.close>np.r_[np.inf,self.high[:-1]])&(self.close>ema)&(self.close>self.base.Open.to_numpy())&(location>=r['close_location_min']-1e-10)
        baseline &= (np.abs(self.close-ema)<=max_atr*r['support_distance_atr']+tolerance)|bounce
        def simple_form(offset):
            idx=indexes[:,offset]
            before_ema=self.full_ema[idx]+(1-2/(i['ema']+1))**(idx-starts)*(self.close[starts]-self.full_ema[starts])
            return ((self.close[idx]-before_ema<=tolerance)&(before_ema-self.close[idx]-atrs[:,offset]<=tolerance)
                    &(rsi[:,offset]>=r['rsi_min']-1e-7)&(rsi[:,offset]<=r['rsi_max']+1e-7))
        simple = base & (simple_form(0) | simple_form(1))
        return baseline, simple

    def possible(self, t, spy):
        """Conservative common-gate prefilter; final rules stay in detector code."""
        r, i = self.config['rules'],self.config['indicators']
        if not self.quality(t):
            return False
        row = self.full.iloc[t]
        if (self.raw.Close.iloc[t]*self.future_split[t] < r['minimum_price'] or row.DollarVolume < r['minimum_dollar_volume']
            or not row.Close > row.SMAFast > row.SMASlow or row.FastSlope <= 0):
            return False
        n = i['relative_strength_days']
        # spy is indexed on the same calendar, never forward-filled.
        segment = spy.reindex(self.dates[t-n:t+1])
        if segment.Close.isna().any():
            return False
        excess = (self.close[t]/self.close[t-n])/(segment.Close.iloc[-1]/segment.Close.iloc[0])-1
        if excess <= r['minimum_relative_strength']:
            return False
        # Both stages must have a recent peak, today or one session earlier.
        look = r['peak_lookback']
        for end in [t,t-1]:
            highs = self.high[end-look:end]
            peak = end-look+len(highs)-1-np.argmax(highs[::-1])
            if r['pullback_days_min'] <= end-peak <= r['pullback_days_max']:
                return True
        return False


def evaluate_window(ticker, frame, spy, config, day, reference=False):
    # A new function object runs the original, unchanged detector bytecode.
    # Only indicator calculation is supplied from a verified causal window.
    if reference:
        return analyse_daily(ticker,frame,spy,config,day)
    def supplied_indicators(received, settings):
        return frame.loc[received.index].copy()
    globals_copy = {**analyse_daily.__globals__, 'indicators':supplied_indicators}
    evaluator = types.FunctionType(analyse_daily.__code__,globals_copy,analyse_daily.__name__)
    return evaluator(ticker,frame,spy,config,day)


def simple_signal(ticker, frame, config, day):
    r = config['rules']
    def formed(part):
        row = part.iloc[-1]
        return (row.RawClose >= r['minimum_price'] and row.DollarVolume >= r['minimum_dollar_volume']
                and row.Close > row.SMAFast > row.SMASlow and row.FastSlope > 0
                and row.EMA-row.ATR <= row.Close <= row.EMA and r['rsi_min'] <= row.RSI <= r['rsi_max'])
    row, previous = frame.iloc[-1], frame.iloc[-2]
    confirmed = (len(frame)-1 >= config['data']['minimum_history'] and formed(frame.iloc[:-1]) and row.RawClose >= r['minimum_price'] and row.DollarVolume >= r['minimum_dollar_volume']
                 and row.Close > row.SMAFast > row.SMASlow and row.FastSlope > 0
                 and row.Close > previous.High and row.Close > row.EMA and row.Close > row.Open)
    if not confirmed and not formed(frame):
        return None
    end = -1 if confirmed else len(frame)
    prior = frame.iloc[:end] if confirmed else frame
    highs = prior.High.iloc[-r['peak_lookback']-1:-1]
    peak_day = highs[highs.eq(highs.max())].index[-1]
    return {'Ticker':ticker,'MarketDate':day,'Stage':'confirmed' if confirmed else 'forming','Score':None,
            'AverageDollarVolume':float(row.DollarVolume),'SetupPeakDate':str(peak_day.date()),'Price':float(row.RawClose),
            'SPYAboveSMA':None, 'ScoreBreakdown':'{}'}


def outcome(row, stock, spy, sessions, as_of, horizon, cost_bps):
    dates = sessions.following(row['MarketDate'],horizon)
    result = {**row,'Horizon':horizon,'EntryDate':str(dates[0].date()),'ExitDate':str(dates[-1].date()),'Status':'pending'}
    if dates[-1] > pd.Timestamp(as_of):
        return result
    a, b = stock.full.reindex(dates), spy.full.reindex(dates)
    if (a[['Open','High','Low','Close']].isna().any().any() or b[['Open','High','Low','Close']].isna().any().any()
        or not np.isfinite(a[['Open','High','Low','Close']].to_numpy()).all() or not np.isfinite(b[['Open','High','Low','Close']].to_numpy()).all()
        or not stock.valid[stock.dates.get_indexer(dates)].all() or not spy.valid[spy.dates.get_indexer(dates)].all()):
        result['Status'] = 'missing_prices'
        return result
    entry, exit_price = float(a.Open.iloc[0]),float(a.Close.iloc[-1])
    spy_entry, spy_exit = float(b.Open.iloc[0]),float(b.Close.iloc[-1])
    if min(entry,exit_price,spy_entry,spy_exit) <= 0:
        result['Status'] = 'invalid_prices'
        return result
    if (a.Volume.le(0).any() or b.Volume.le(0).any()):
        result['Status'] = 'not_tradable'
        return result
    fee = cost_bps/10000
    net = (exit_price*(1-fee)/(entry*(1+fee))-1)*100
    spy_net = (spy_exit*(1-fee)/(spy_entry*(1+fee))-1)*100
    result.update(Status='complete',GrossReturnPct=(exit_price/entry-1)*100,NetReturnPct=net,SPYNetReturnPct=spy_net,
                  ExcessReturnPct=net-spy_net,MFEPercent=max(0.,float(a.High.max()/entry-1)*100),
                  MAEPercent=min(0.,float(a.Low.min()/entry-1)*100),AdjustedEntryOpen=entry,AdjustedExitClose=exit_price)
    return result
