# 米国株・週足ソーサーボトム スクリーナー MVP

米国上場の普通株候補を対象に、週足で丸い底を形成し始めた銘柄を抽出します。

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

最終的に100点で採点し、上位20銘柄をCSVとスマホ対応HTMLへ出力します。

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

## 次に追加する機能

1. 実際の候補を見ながら誤検出を減らす
2. 日足200MAを週足40MAとは別に厳密計算
3. 決算日・時価総額・業種を追加
4. GitHub Actionsで週次自動実行
5. 通知機能
6. 抽出後1・3・6か月のバックテスト

無料のYahoo Financeデータは欠損や取得制限が起きることがあります。重要な数値は証券会社や企業IRでも確認してください。
