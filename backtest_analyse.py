"""Read completed backtest artifacts; produce auditable descriptive reports."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd

LABELS={'baseline':'現行価格条件','simple':'単純条件','forming':'形成中','confirmed':'反発確認済み','outside_display_band':'丸め境界の別帯',
        'above_50':'SPY>SMA50','below_50':'SPY≤SMA50'}


def md(frame,columns):
    lines=['|'+'|'.join(label for key,label in columns)+'|','|'+'|'.join('---' for _ in columns)+'|']
    for row in frame.to_dict('records'):
        values=[]
        for key,label in columns:
            value=row.get(key)
            if value is None or (isinstance(value,float) and not math.isfinite(value)):
                text='—'
            elif key in ['Signals','Complete','Pending','Missing','Horizon','Year']:
                text=f'{int(value):,}'
            elif isinstance(value,(float,np.floating)):
                text=f'{value:.3f}'
            else:
                text=LABELS.get(str(value),str(value))
            values.append(text.replace('|','/'))
        lines.append('|'+'|'.join(values)+'|')
    return '\n'.join(lines)


def bootstrap(group,calendar,first,seed=42):
    complete=group.loc[group.Status.eq('complete')].copy()
    calendar=calendar.loc[calendar.ge(first)].reset_index(drop=True)
    positions=pd.Series(np.arange(len(calendar)),index=calendar)
    complete['Block']=complete.MarketDate.map(positions)//20
    assert complete.Block.notna().all(),'Signal date not in benchmark calendar'
    block_count=math.ceil(len(calendar)/20)
    sums=complete.groupby('Block')[['NetReturnPct','ExcessReturnPct']].sum().reindex(range(block_count),fill_value=0).to_numpy()
    counts=complete.groupby('Block').size().reindex(range(block_count),fill_value=0).to_numpy()
    if len(complete)<20 or np.count_nonzero(counts)<5:
        return {'NetLow':None,'NetHigh':None,'ExcessLow':None,'ExcessHigh':None}
    rng=np.random.default_rng(seed)
    sampled=rng.integers(0,block_count,(2000,block_count))
    denominator=counts[sampled].sum(axis=1)
    values=sums[sampled].sum(axis=1)/np.maximum(1,denominator)[:,None]
    values=values[denominator>0]
    quantiles=np.quantile(values,[.025,.975],axis=0)
    return dict(zip(['NetLow','NetHigh','ExcessLow','ExcessHigh'],[float(quantiles[0,0]),float(quantiles[1,0]),float(quantiles[0,1]),float(quantiles[1,1])]))


def analyse(root,run=None):
    root=Path(root)
    run=Path(run) if run else Path(json.loads((root/'output/pullback/backtest/latest_full.json').read_text(encoding='utf-8'))['run'])
    status=json.loads((run/'run_status.json').read_text(encoding='utf-8'))
    protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    fingerprints=json.loads((run/'source_hashes.json').read_text(encoding='utf-8'))
    assert status['scope'] in {'full','pilot'} and status['target_count']==len(pd.read_csv(run/'universe.csv'))
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in fingerprints.items())
    results=pd.read_csv(run/'metrics.csv')
    signals=pd.read_csv(run/'signals.csv')
    trades=pd.read_csv(run/'trades.csv')
    data_status=pd.read_csv(run/'data_status.csv')
    as_of=status['as_of']
    first=(pd.Timestamp(as_of)-pd.DateOffset(years=5)).date().isoformat()
    primary=results.loc[results.PeriodYears.eq(5)&results.Dimension.eq('all')&results.TopN.eq('all')&results.Horizon.eq(5)]
    write_checks={
        'all_targets_processed':len(data_status)==status['target_count'],
        'unique_episode_stages':not signals.EpisodeStageID.duplicated().any(),
        'unique_outcome_keys':not trades.duplicated(['EpisodeStageID','Horizon']).any(),
        'all_signals_have_all_horizons':len(trades)==len(signals)*3 and trades.groupby('EpisodeStageID').Horizon.nunique().eq(3).all(),
        'no_results_from_unavailable_assets':not signals.Ticker.isin(data_status.loc[data_status.Status.ne('available'),'Ticker']).any(),
        'only_three_horizons':sorted(trades.Horizon.unique().tolist())==[3,5,10],
        'no_historical_earnings_gate_claim':signals.EarningsStatus.eq('not_reconstructed').all(),
        'entry_after_signal':trades.EntryDate.gt(trades.MarketDate).all(),
        'closed_not_after_asof':trades.loc[trades.Status.eq('complete'),'ExitDate'].le(as_of).all(),
    }
    complete=trades.loc[trades.Status.eq('complete')]
    cache_root=root/f"cache/backtest/{protocol['download_start']}_{as_of}"
    snapshot_count=0
    snapshot_ok=True
    for row in data_status.loc[data_status.sha256.notna()].to_dict('records'):
        path=cache_root/f"{row['Ticker']}.csv.gz"
        snapshot_ok &= path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        snapshot_count+=1
    benchmark_meta=json.loads((cache_root/'SPY.json').read_text(encoding='utf-8'))
    snapshot_ok &= hashlib.sha256((cache_root/'SPY.csv.gz').read_bytes()).hexdigest()==benchmark_meta['sha256']
    write_checks['all_source_snapshot_hashes_match']=bool(snapshot_ok)
    (run/'benchmark_snapshot.json').write_text(json.dumps(benchmark_meta,ensure_ascii=False,indent=2),encoding='utf-8')
    recomputed=(complete.AdjustedExitClose*.999/(complete.AdjustedEntryOpen*1.001)-1)*100
    write_checks['cost_formula']=bool(np.allclose(recomputed,complete.NetReturnPct,rtol=1e-10,atol=1e-9))
    ranked=signals.sort_values(['MarketDate','Strategy','Stage','Score','AverageDollarVolume','Ticker'],ascending=[True,True,True,False,False,True],na_position='last').copy()
    ranked['CheckRank']=ranked.groupby(['MarketDate','Strategy','Stage']).cumcount()+1
    write_checks['ranking_only_signal_fields']=bool((ranked.CheckRank==ranked.DailyRank).all())
    common=complete.groupby('EpisodeStageID').Horizon.nunique()
    common_ids=common[common.eq(3)].index
    risk=complete.loc[complete.EpisodeStageID.isin(common_ids)].pivot(index='EpisodeStageID',columns='Horizon',values=['MFEPercent','MAEPercent'])
    write_checks['excursions_monotonic']=bool(((risk['MFEPercent'][3]<=risk['MFEPercent'][5]+1e-9)&(risk['MFEPercent'][5]<=risk['MFEPercent'][10]+1e-9)
        &(risk['MAEPercent'][3]>=risk['MAEPercent'][5]-1e-9)&(risk['MAEPercent'][5]>=risk['MAEPercent'][10]-1e-9)).all())
    write_checks={key:bool(value) for key,value in write_checks.items()}
    assert all(write_checks.values()),write_checks
    (run/'verification.json').write_text(json.dumps(write_checks,indent=2),encoding='utf-8')

    calendar=pd.read_csv(root/f"cache/backtest/{protocol['download_start']}_{as_of}/SPY.csv.gz",usecols=['Date']).Date
    intervals=[]
    five=trades.loc[trades.MarketDate.ge(first)&trades.Horizon.eq(5)]
    for (strategy,stage),group in five.groupby(['Strategy','Stage']):
        intervals.append({'Strategy':strategy,'Stage':stage,**bootstrap(group,calendar,first)})
    intervals=pd.DataFrame(intervals)
    intervals.to_csv(run/'confidence_intervals.csv',index=False)
    primary=primary.merge(intervals,on=['Strategy','Stage'],validate='one_to_one')
    primary.to_csv(run/'primary_5y_5d.csv',index=False)
    # Same fully observed episodes for all three horizons, as a sensitivity table.
    same=trades.loc[trades.EpisodeStageID.isin(common_ids)&trades.MarketDate.ge(first)]
    common_rows=[]
    for keys,group in same.groupby(['Strategy','Stage','Horizon']):
        common_rows.append(dict(zip(['Strategy','Stage','Horizon'],keys))|{'Complete':len(group),'MeanPct':float(group.NetReturnPct.mean()),'MedianPct':float(group.NetReturnPct.median()),'ExcessMeanPct':float(group.ExcessReturnPct.mean())})
    common_frame=pd.DataFrame(common_rows)
    common_frame.to_csv(run/'common_cohort.csv',index=False)
    extreme=five.loc[five.Status.eq('complete')].sort_values('NetReturnPct')
    pd.concat([extreme.head(10),extreme.tail(10)]).to_csv(run/'extreme_trades.csv',index=False)
    coverage=[]
    for year in range(2016,2027):
        key=f'eligible_days_{year}'
        eligible=int(data_status[key].fillna(0).sum()) if key in data_status else 0
        n=int(data_status[key].fillna(0).gt(0).sum()) if key in data_status else 0
        coverage.append({'Year':year,'TickersWithEligibleDays':n,'EligibleTickerDays':eligible})
    pd.DataFrame(coverage).to_csv(run/'yearly_coverage.csv',index=False)
    # Diagnostic only: reassess the signal-close reference RR at the actual
    # next open. Never use this information to select or remove any trade.
    gap_rows=[]
    baseline_closed=five.loc[five.Strategy.eq('baseline')&five.Status.eq('complete')]
    for ticker,group in baseline_closed.groupby('Ticker'):
        prices=pd.read_csv(root/f"cache/backtest/{protocol['download_start']}_{as_of}/{ticker}.csv.gz",usecols=['Date','Open','Close'],index_col='Date')
        for row in group.to_dict('records'):
            ratio=float(prices.loc[row['EntryDate'],'Open']/prices.loc[row['MarketDate'],'Close'])
            entry=row['Price']*ratio
            distance=entry-row['ReferenceStop']
            gap_rows.append({'Ticker':ticker,'MarketDate':row['MarketDate'],'Stage':row['Stage'],'Score':row['Score'],
                'EntryGapPct':(ratio-1)*100,'EntryRewardRisk':(row['SetupHigh']-entry)/distance if distance>0 else np.nan,
                'AtOrBelowReferenceStop':distance<=0,'NetReturnPct':row['NetReturnPct']})
    gap_frame=pd.DataFrame(gap_rows)
    gap_frame.to_csv(run/'entry_gap_diagnostics.csv',index=False)
    gap_summary=[]
    for stage,group in gap_frame.groupby('Stage'):
        returns=baseline_closed.loc[baseline_closed.Stage.eq(stage)]
        correlation=returns.Score.rank().corr(returns.NetReturnPct.rank()) if len(returns)>2 else np.nan
        gap_summary.append({'Stage':stage,'Complete':len(group),'MedianGapPct':group.EntryGapPct.median(),
            'GapAbove1Pct':group.EntryGapPct.ge(1).mean()*100,
            'EntryRRBelowMinimumPct':group.EntryRewardRisk.lt(1.5).mean()*100,
            'EntryAtOrBelowStop':int(group.AtOrBelowReferenceStop.sum()),'ScoreRankCorrelation':correlation})
    gap_summary=pd.DataFrame(gap_summary)
    gap_summary.to_csv(run/'entry_gap_summary.csv',index=False)
    paired=baseline_closed.loc[baseline_closed.Stage.eq('forming')].merge(
        baseline_closed.loc[baseline_closed.Stage.eq('confirmed')],on=['Ticker','SetupPeakDate'],
        suffixes=('_forming','_confirmed'),validate='one_to_one')
    paired=paired.loc[paired.MarketDate_confirmed.gt(paired.MarketDate_forming)].copy()
    paired['ConfirmedMinusFormingPct']=paired.NetReturnPct_confirmed-paired.NetReturnPct_forming
    paired.to_csv(run/'paired_stages.csv',index=False)
    paired_text=(f"両状態が同じ押し目で検出され、両方の5日決済が観測できた組は{len(paired):,}件。"
                 f"確認済みのネット損益−形成中のネット損益の平均は{paired.ConfirmedMinusFormingPct.mean():.3f}pp、"
                 f"中央値は{paired.ConfirmedMinusFormingPct.median():.3f}pp。") if len(paired) else '両状態の5日成績を比較できる同一押し目の組は0件。'

    columns=[('Strategy','条件'),('Stage','状態'),('Signals','検出数'),('Complete','決済可能'),('MeanPct','平均%'),('MedianPct','中央値%'),('WinRatePct','勝率%'),('ProfitFactor','PF'),('ExcessMeanPct','SPY超過pp')]
    pilot_note='（小規模試験・性能判断には使わない）' if status['scope']=='pilot' else ''
    lines=['# Daily Pullback ベースライン・バックテスト結果'+pilot_note,'',f"基準日：{as_of}。対象は現在の普通株ユニバース{status['target_count']:,}銘柄。検出条件・閾値・配点はv0.1のまま（ConfigHash {status['config_hash']}）。",'',
           '**現在の銘柄一覧と取得時点の株価を使った、決算除外前の価格条件の評価です。過去の実際の主候補や、資金配分込みの運用成績ではありません。**','',
           '## 主評価：5年間・5営業日・コスト控除後','',md(primary,columns),'',
           'SPY超過は同じ購入日・決済日・片道0.1%コストのSPYとの差（パーセントポイント）で、β調整アルファではない。平均・中央値・勝率・PFは決済可能例について計算し、検出数には未確定/欠損も含める。PFは同額投資の正の損益率合計÷負の損益率合計の絶対値。負けがない/決済例がない区分のPFは未定義（—）。','',
           '### 主評価の不確実性','',md(primary,[('Strategy','条件'),('Stage','状態'),('NetLow','平均95%下限%'),('NetHigh','上限%'),('ExcessLow','SPY超過95%下限pp'),('ExcessHigh','上限pp')]),'',
           '20営業日ごとの時間ブロックを単位に2,000回再抽出（seed42）。同じ日の銘柄を一緒に再抽出して相関をある程度保つ。検出20件未満/検出ブロック5未満は区間を出さない。定常性や将来の優位性を保証する区間ではない。','',
           '## 3・5・10営業日での比較','']
    horizons=results.loc[results.PeriodYears.eq(5)&results.Dimension.eq('all')&results.TopN.eq('all')]
    lines += [md(horizons,[('Strategy','条件'),('Stage','状態'),('Horizon','保有日数'),('Complete','決済可能'),('MeanPct','平均%'),('MedianPct','中央値%'),('WinRatePct','勝率%'),('ProfitFactor','PF'),('ExcessMeanPct','SPY超過pp')]),'',
              '各期間で未確定件数が異なる影響を避けるため、3期間すべて決済可能な同じ押し目だけの比較も common_cohort.csv に保存した。','',
              '## 最大含み益・最大含み損','',md(primary,[('Strategy','条件'),('Stage','状態'),('MFEMeanPct','平均MFE%'),('MAEMeanPct','平均MAE%'),('BestMFEPct','最大MFE%'),('WorstMAEPct','最悪MAE%'),('Pending','未確定'),('Missing','欠損等')]),'',
              'MFE/MAEは購入日の始値に対する保有中High最大/Low最小。コスト前、0を境界とする。ストップや利確価格で売れたという意味ではない。','',
              '## 翌日始値へのギャップとスコアの診断','',
              md(gap_summary,[('Stage','状態'),('Complete','決済可能'),('MedianGapPct','始値ギャップ中央値%'),('GapAbove1Pct','+1%以上の割合%'),('EntryRRBelowMinimumPct','始値でRR1.5未満の割合%'),('EntryAtOrBelowStop','参考ストップ以下の始値数'),('ScoreRankCorrelation','スコア/5日損益の順位相関')]),'',
              '現行の想定RRはシグナル終値基準。ここでは実際の翌営業日始値に置き換えた場合を事後に確認しただけで、取引の除外や設定変更には使っていない。スコアの順位相関も探索的な参考値で、相関から因果や最適閾値を判断しない。','',
              paired_text+' この比較には後に確認済みになった押し目だけが入る。形成後に反発しなかった例を含まないため、確認を待つことの無条件の効果ではない。','',
              '## 検出頻度・連続検出の重複排除','']
    frequencies=results.loc[results.Dimension.eq('all')&results.TopN.eq('all')&results.Horizon.eq(5)]
    lines += [md(frequencies,[('PeriodYears','年数'),('Strategy','条件'),('Stage','状態'),('Signals','初回検出数'),('SignalsPerYear','年換算件数')]),'',
              '戦略・ティッカー・高値日・状態ごとの初回だけを数える。形成中と確認済みが同じ押し目なら各状態の評価には1件ずつ入るが、両者を合わせた独立標本としては扱わない。','',
              '## 上位5・10・20の比較（5年間・5営業日）','']
    tops=results.loc[results.PeriodYears.eq(5)&results.Dimension.eq('all')&results.Horizon.eq(5)]
    tops=tops.assign(Order=tops.TopN.map({'5':5,'10':10,'20':20,'all':1000})).sort_values(['Strategy','Stage','Order'])
    lines += [md(tops,[('Strategy','条件'),('Stage','状態'),('TopN','日別状態別上位'),('Complete','決済可能'),('MeanPct','平均%'),('WinRatePct','勝率%'),('ProfitFactor','PF'),('ExcessMeanPct','SPY超過pp')]),'',
              '新規シグナルのスコア順。単純条件は採点がないため売買代金順。既存UIの両状態合算20枠や、重複保有・資金不足を含む運用再現とは異なる。','',
              '## スコア帯別（現行価格条件・5年間・5営業日）','']
    bands=results.loc[results.PeriodYears.eq(5)&results.Dimension.eq('score')&results.TopN.eq('all')&results.Horizon.eq(5)]
    lines += [md(bands,[('Stage','状態'),('Group','点数帯'),('Complete','決済可能'),('MeanPct','平均%'),('MedianPct','中央値%'),('WinRatePct','勝率%'),('ProfitFactor','PF'),('ExcessMeanPct','SPY超過pp')]),'',
              '形成中は75点満点、確認済みは100点満点。帯は下限以上・上限未満（最後の帯は満点を含む）。未丸め合計で閾値判定した後の表示点を使う。少数例の高得点帯を、そのまま最適閾値と判断しない。','',
              '## 年別（5年間・5営業日）','']
    annual=results.loc[results.PeriodYears.eq(5)&results.Dimension.eq('year')&results.TopN.eq('all')&results.Horizon.eq(5)]
    lines += [md(annual,[('Group','年'),('Strategy','条件'),('Stage','状態'),('Complete','決済可能'),('MeanPct','平均%'),('WinRatePct','勝率%'),('ProfitFactor','PF'),('ExcessMeanPct','SPY超過pp')]),'',
              '2021年は11月26日以降、2026年は11月25日までの端数期間。判定可能銘柄数・銘柄日は年によって異なる。10年間の年別品質は yearly_coverage.csv に保存した。','',
              '## 市場環境別（5年間・5営業日）','']
    regimes=results.loc[results.PeriodYears.eq(5)&results.Dimension.eq('regime')&results.TopN.eq('all')&results.Horizon.eq(5)]
    lines += [md(regimes,[('Group','SPY環境'),('Strategy','条件'),('Stage','状態'),('Complete','決済可能'),('MeanPct','平均%'),('WinRatePct','勝率%'),('ProfitFactor','PF'),('ExcessMeanPct','SPY超過pp')]),'',
              'SPYの当日終値と50日線で分類。検出時に除外・減点はしていない。','',
              '## 補助評価：10年間・5営業日','']
    ten=results.loc[results.PeriodYears.eq(10)&results.Dimension.eq('all')&results.TopN.eq('all')&results.Horizon.eq(5)]
    lines += [md(ten,columns),'','10年の保有期間別・年別・スコア帯・上位N・市場環境別も metrics.csv に収録。10年と5年は期間が重複しているため、独立した外部検証ではない。','',
              '## データ品質と検証','',
              f"長期履歴を評価器で読み込めた銘柄：{status['available_count']:,}/{status['target_count']:,}。これは全銘柄に全期間が揃うという意味ではない。IPO前・252本未満・不正/欠損の判定日は除外している。",'',
              f"全期間の判定可能銘柄日：{status['diagnostics']['analysable_days']:,}。全戦略・全保有期間の決済状態：{json.dumps(status['trade_status_counts'],ensure_ascii=False)}。",'',
              f"処理時間：{status['elapsed_seconds']:.1f}秒。銘柄原本{snapshot_count:,}件とSPYのSHA256も照合した。詳細は data_status.csv と run_status.json。",'',
              '小規模試験、元検出器との照合、分散/逐次処理の一致、未来分割/配当・未来価格の非参照、約定日・コスト・欠損・重複のテストを実施した。数値照合は絶対1e-8/相対1e-9以内、判定状態・理由・表示点の一致を確認した。全銘柄の全日を元処理で二重計算したわけではない。完成成果物について、重複キー、選出順位、コスト再計算、決済日、MFE/MAEの期間単調性を再検証し verification.json に保存した。','',
              '## 解釈の限界','',
              '- 現在の適格普通株一覧を過去へ当てはめており、上場廃止・過去の上場構成・当時の財務状態フィルターを再現していない。生存者バイアスが残る。',
              '- 過去の決算予定の公表履歴がないため、決算ゲート込みの主候補の成績は不明。現在の予定で過去を再分類していない。',
              '- 提供元の遡及訂正・ティッカー再利用/企業再編の完全な復元はできない。Yahooの[調整後価格仕様](https://help.yahoo.com/kb/SLN28256.html)に基づく取得時点の系列を使う。',
              '- 決済不能例は損益を捏造せず欠損として残す。その除外による偏りは残る。極端な損益例は extreme_trades.csv で確認できる。',
              '- スコア帯・上位N・年・環境の比較は探索的。多重比較補正・独立期間での追試を経ず、良い区分だけを採用すると過剰適合する。今回は閾値変更を行っていない。',
              '- 単純条件とはRSIの使い方などが異なり、厳密な包含関係ではない。集団同士の記述比較であり、個々のフィルターの因果効果ではない。',
              '- 同じ押し目の識別は既存仕様の高値日による操作的定義。異なる高値日の取引は保有が重なる場合もあり、完全な独立標本とはいえない。',
              '- 同時保有、資金制約、発注滑り、ストップ/利確、税は含まない。MFEや平均損益を、そのまま実現可能な運用利益と扱わない。','',
              '## 分析・改善候補','',
              '以下の分析は全銘柄の出力を確認後に追記する。検出設定は変更しない。','']
    report='\n'.join(lines)
    executive=run/'executive_summary.md'
    interpretation=run/'interpretation.md'
    if executive.exists():
        report=report.replace('## 主評価：5年間・5営業日・コスト控除後',executive.read_text(encoding='utf-8')+'\n\n## 主評価：5年間・5営業日・コスト控除後',1)
    if interpretation.exists():
        report=report.replace('以下の分析は全銘柄の出力を確認後に追記する。検出設定は変更しない。',interpretation.read_text(encoding='utf-8'),1)
    (run/'report.md').write_text(report,encoding='utf-8')
    manifest={'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'run':str(run.resolve()),
              'metrics_sha256':hashlib.sha256((run/'metrics.csv').read_bytes()).hexdigest(),'trades_sha256':hashlib.sha256((run/'trades.csv').read_bytes()).hexdigest(),
              'bootstrap':{'block_sessions':20,'draws':2000,'seed':42},'checks':write_checks,
              'python':platform.python_version(),'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','yfinance','exchange_calendars']}}
    manifest['report_sha256']=hashlib.sha256((run/'report.md').read_bytes()).hexdigest()
    manifest['reviewed_notes']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [executive,interpretation] if path.exists()}
    (run/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(primary.to_string(index=False))
    print(f'Report: {run / "report.md"}')
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    analyse(args.root,args.run)
