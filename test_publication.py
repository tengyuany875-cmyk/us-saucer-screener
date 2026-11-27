from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from publish_site import daily, extract, needs_update, weekly
from pullback_data import write_json, write_csv
from pullback_settings import config_hash, load_config
from site_navigation import add_navigation


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.now = datetime(2026,11,25,12,tzinfo=timezone.utc)

    def fixture(self, root):
        run = root/'output/pullback/runs/test'
        (root/'docs').mkdir()
        (root/'docs/index.html').write_text('<html><style>:root{--line:#ddd}</style><main><p>weekly</p><script>keep()</script></main></html>',encoding='utf-8')
        write_json(root/'pullback_config.json',self.config)
        write_json(run/'config.json',self.config)
        summary = {'status':'complete','as_of':'2026-11-24','finished_at_utc':'2026-11-25T11:00:00+00:00',
                   'quality_passed':True,'scope':'full_universe','target_count':100,'price_success_count':98,
                   'price_success_rate':.98,'analysable_count':95,'elapsed_seconds':20.,'market_regime':'SPYは50日線以下',
                   'selected_main':0,'selected_watch':0,'review_count':0,'earnings_excluded_count':0,
                   'output_dir':'C:\\private\\local\\path','packages':{'example':'1'}}
        write_json(run/'run_status.json',summary)
        write_json(root/'output/pullback/latest_success.json',{'run':str(run),'as_of':summary['as_of'],'config_hash':config_hash(self.config)})
        status = pd.DataFrame({'Ticker':[f'TEST{i}' for i in range(100)],'Decision':['no_signal']*100,'PriceSuccess':[True]*98+[False]*2,
                               'DataStatus':['ok']*95+['insufficient_history']*3+['fetch_failed']*2})
        write_csv(run/'data_status.csv',status)
        write_csv(run/'all_candidates.csv',pd.DataFrame(columns=['Ticker','MarketDate','Selected','Bucket']))
        return run,summary

    def test_quality_passed_zero_publishes_and_does_not_leak_runtime_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.fixture(root)
            daily(root,self.now)
            page=(root/'docs/daily.html').read_text(encoding='utf-8')
            snapshot=(root/'data/pullback/latest.json').read_text(encoding='utf-8')
            self.assertIn('今回は候補がありません',page)
            self.assertNotIn('ローカル試作',page)
            self.assertIn('98.00%',page)
            self.assertIn('日本時間の火〜土曜7:30頃',page)
            self.assertNotIn(str(root),snapshot)
            self.assertNotIn('C:\\private',snapshot)
            self.assertNotIn('packages',snapshot)

    def test_failed_subset_stale_or_mismatched_config_never_overwrites_public_output(self):
        for change in [{'quality_passed':False},{'scope':'subset'},{'as_of':'2026-11-23'}]:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                root=Path(temp); run,summary=self.fixture(root)
                public=root/'docs/daily.html'; public.write_text('previous good result',encoding='utf-8')
                write_json(run/'run_status.json',{**summary,**change})
                with self.assertRaises(ValueError): daily(root,self.now)
                self.assertEqual(public.read_text(encoding='utf-8'),'previous good result')
                self.assertFalse((root/'data/pullback/latest.json').exists())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); run,summary=self.fixture(root)
            other=json.loads(json.dumps(self.config));other['rules']['maximum_selected']=10
            write_json(root/'pullback_config.json',other)
            with self.assertRaises(ValueError): daily(root,self.now)

    def test_summary_cannot_hide_provider_failure_or_invalid_candidate_buckets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,summary=self.fixture(root)
            status=pd.read_csv(run/'data_status.csv');status['PriceSuccess']=False
            write_csv(run/'data_status.csv',status)
            with self.assertRaises(ValueError): extract(root,self.now)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,summary=self.fixture(root)
            status=pd.read_csv(run/'data_status.csv');status.loc[0,'Decision']='signal'
            write_csv(run/'data_status.csv',status)
            with self.assertRaises(ValueError): extract(root,self.now)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,summary=self.fixture(root)
            write_csv(run/'all_candidates.csv',pd.DataFrame([{'Ticker':'TEST','MarketDate':'2026-11-24','Selected':True,'Bucket':'review'}]))
            with self.assertRaises(ValueError): extract(root,self.now)

    def test_navigation_is_idempotent_and_weekly_regeneration_keeps_original_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            original=(root/'docs/index.html').read_text(encoding='utf-8')
            result=weekly(root)
            self.assertEqual(result.count('スクリーナーのカテゴリー'),1)
            self.assertIn('<script>keep()</script>',result)
            self.assertIn('<p>weekly</p>',result)
            self.assertEqual(add_navigation(result),result)
            self.assertEqual(weekly(root),result)
            self.assertNotEqual(original,result)

    def test_market_cap_failure_is_optional_and_never_uses_stale_vintage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            with patch('market_caps.fetch_snapshot',side_effect=RuntimeError('private diagnostic')):
                snapshot=extract(root,self.now,refresh_caps=True)
            self.assertIsNone(snapshot['caps'])
            self.assertEqual(snapshot['market_caps_error'],'unavailable')
            self.assertNotIn('private diagnostic',json.dumps(snapshot))

    def test_session_check_skips_holidays_and_rechecks_config_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            self.assertTrue(needs_update(root,self.now))
            daily(root,self.now)
            self.assertFalse(needs_update(root,self.now))
            self.assertTrue(needs_update(root,datetime(2026,11,27,12,tzinfo=timezone.utc)))
            # Thanksgiving Nov 26: a Nov 25 snapshot is still the last completed session.
            saved=json.loads((root/'data/pullback/latest.json').read_text(encoding='utf-8'))
            saved['summary']['as_of']='2026-11-25'
            write_json(root/'data/pullback/latest.json',saved)
            self.assertFalse(needs_update(root,datetime(2026,11,27,12,tzinfo=timezone.utc)))


if __name__ == '__main__':
    unittest.main()
