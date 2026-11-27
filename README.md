# 米国株スクリーナー

米国上場の普通株候補を対象に、週足で丸い底を形成し始めた銘柄を抽出します。

[公開サイト](https://tengyuany875-cmyk.github.io/us-saucer-screener/) のカテゴリーから「週足ソーサーボトム」と「日足押し目」を切り替えられます。

日足押し目は、上昇トレンド中の調整を検出し、反発確認済みを主候補、形成中を監視候補として合計最大20銘柄を表示します。決算予定不明は別枠、想定保有期間中に決算がある銘柄は選出対象外です。SPYの50日線割れは市場環境として表示し、一律の除外条件にはしません。各カードの「詳細」で検出理由・スコア内訳・調整幅・想定リスクリワードを確認できます。

検出条件や点数の高さだけで収益性が確認されたわけではありません。価格条件のベースライン検証では明確な優位性を示せておらず、過去の決算予定と当時の全銘柄一覧を再現できない制約があります。

候補カードには時価総額（米ドルを億ドル単位に丸めた値）と、米国株内の参考順位を表示します。
参考順位はNasdaqの公開一覧からETF・優先株・ワラント等を除き、時価総額が確認できた普通株を銘柄単位で比較した順位です。外国企業の米国上場株も含みます。同額は同順位とし、取得できない値は「不明」と表示します。
時価総額はスキャン時に別途取得するため、週足株価と基準日が異なる場合があります。取得日時と比較対象銘柄数は表示行の補足情報に保存します。取得失敗時も既存の候補抽出結果は出力します。

## 初期版の判定項目

- 左側が下落、中央が横ばい、右側が上昇
- 30週の価格形状が上向きの二次曲線
- 底値圏に4週以上滞在
- 底までの下落率が8～65%
- 40週移動平均線（約200取引日線）から大きく崩れていない
- 200週線が存在する場合は、大きく崩れていない
- 直近出来高と流動性
- 52週高値との距離
- 回復率とブレイク接近度

週足は100点で採点し、CSV・HTMLに全候補を出力します。画面の初期表示は20件です。

## 自動更新

|カテゴリー|実行予定|更新内容|
|---|---|---|
|週足ソーサーボトム|毎週日曜 9:00 JST|従来の週足スキャン・追跡・公開|
|日足押し目|火〜土曜 7:30 JST頃（前日の22:30 UTC）|米国市場終了後の確定日足・独立した追跡・公開|

GitHub Actionsの実行時刻は遅れることがあります。休場等で新しい確定日がなければ日足の取得を省きます。日足はSPY取得成功・全銘柄対象・取得成功率95%以上・設定/基準日の整合を公開前に確認し、不合格時には最後の正常な公開結果を残します。画面のデータ基準日と更新日時を確認してください。

日足は `Daily Pullback` ワークフローから手動でも起動でき、`force` で同じ基準日を再取得できます。検出条件・閾値・配点は `pullback_config.json` で管理します。詳細は [DAILY_PULLBACK_SPEC.md](DAILY_PULLBACK_SPEC.md)、価格条件の検証方法は [BACKTEST_SPEC.md](BACKTEST_SPEC.md) を参照してください。

日足の公開用スナップショットは `data/pullback/latest.json`、通常の追跡と決算予定の観測履歴は `history/pullback/` に保存します。株価キャッシュ・ローカル実行ディレクトリ・バックテストの大量データはGitへ含めません。

## 日足のローカル実行

週足とは別のPython 3.12環境を使います。Windows PowerShellの場合：

```powershell
python -m venv .venv-pullback
.\.venv-pullback\Scripts\python.exe -m pip install -r requirements-pullback.lock.txt
.\.venv-pullback\Scripts\python.exe -m unittest discover -v
.\.venv-pullback\Scripts\python.exe pullback_app.py
.\.venv-pullback\Scripts\python.exe publish_site.py daily --refresh-caps
```

公開用HTMLは `docs/daily.html`・`docs/recent.html` に生成します。週足再生成時もカテゴリー切り替えを保持します。ローカル試作用は従来どおり `pullback_ui.py` と `preview_local.cjs` を使用できます。

## CodeSandboxでの実行

1. CodeSandboxのPythonテンプレートを開きます。
2. このフォルダ内のファイルをアップロードします。
3. ターミナルで以下を実行します。

```bash
pip install -r requirements.txt
python synthetic_test.py
python app.py --mode quick --refresh-universe
```

結果は以下に出ます。

- `output/results.csv`
- `output/results.html`

HTMLをブラウザで見る場合：

```bash
python -m http.server 8000 -d output
```

CodeSandboxのポート8000のプレビューを開いてください。

## 実行モード

```bash
python app.py --mode quick   # 250銘柄
python app.py --mode medium  # 1000銘柄
python app.py --mode full    # 全銘柄
```

無料のYahoo Financeデータは欠損や取得制限が起きることがあります。重要な数値は証券会社や企業IRでも確認してください。
