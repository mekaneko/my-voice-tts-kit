# SETUP — 自分の声TTSを作る全手順

Style-Bert-VITS2（JP-Extra）を自分の声で fine-tune し、専用アプリで読み上げられるようにするまでの手順です。想定環境は macOS（Apple Silicon）。所要期間の目安は、録音が半日、学習が1〜2日（CPU学習のため）です。

> **大原則：声データは外に出さない。** 録音・学習・推論のすべてを自分のマシンの中で完結させます。外部から取得するのはコードと公開モデル（ベースモデル等）のみです。

## 0. 必要なもの

- Mac（Apple Silicon、RAM 16GB以上推奨）
- マイク（ダイナミックマイク推奨）
- Homebrew、Python 3.10
- ffmpeg（MP3変換と音声処理）: `brew install ffmpeg`

## 1. Style-Bert-VITS2 のセットアップ

```bash
git clone https://github.com/litagin02/Style-Bert-VITS2.git style-bert-vits2
cd style-bert-vits2
python3.10 -m venv ../sbv2-venv
../sbv2-venv/bin/pip install -r requirements.txt
../sbv2-venv/bin/pip install "transformers==4.46.3" pywebview
```

- ⚠️ transformers は **4.46.3 固定**（5.x では torch が無効化される事象を確認済み）

必要なモデル資産（BERT、SLM、JP-Extraの事前学習モデル）を SBV2 の手順に従ってダウンロードし、`bert/`、`slm/`、`model_assets/` 等の所定位置に配置します。

## 2. 台本を用意する

- 450文前後が目安。日常会話・感情表現・朗読調をバランスよく
- 形式は recorder-app の `script_sample.csv`（number, filename, text, category, hint）
- 感情文の hint には「会話の7割の強さで」など、演じ方の目安を書いておくと録音が安定します

## 3. 録音する（recorder-app/）

1. `script.csv` と `config.json`（話者名を設定）をプロジェクトフォルダに配置
2. 録音アプリを起動し、台本に沿って1文ずつ録音
3. マイクは口から5〜10cm、ピーク −12〜−6 dB を目安に
4. 録音後、アプリが `recordings/` に連番WAV、対応表CSV、filelist を自動生成します

## 4. 前処理とQC

1. 録音WAVを 44.1kHz 化し前後の無音をトリミング → `dataset/<話者名>/wav/`
2. `filelist.txt` を用意（形式: `wav/001.wav|<話者名>|JP|<本文>`）
3. tools の `qc_asr.py` で全件を Whisper 文字起こしし、台本と照合。読み間違いがあれば録り直す
4. SBV2 の前処理（resample → preprocess_text → bert_gen → style_gen）を実行し、`Data/<話者名>/` に学習データを生成

## 5. 学習する（training/）

1. **まず暗記テスト**：16文だけの小データで数百ステップ回し、学習済み文を再現できるか確認する（training/README.md 参照）。**MPSでは学習が成立しないことを確認済みのため、CPU学習スクリプトを使う**
2. 本番：`train_ms_jp_extra_cpu.py` で学習（batch 4、LR 2e-4、100 epochs 目安）。`caffeinate` でスリープを防ぐ
3. 進捗は1,000ステップごとにチェックポイント生成＋未学習文の試聴＋Whisper照合で評価する。「ロスが下がっているのに読めない」場合はバックエンドを疑う

## 6. アプリで使う（tts-app/）

1. 完成モデル（safetensors）が `style-bert-vits2/model_assets/<話者名>/` に書き出されていることを確認
2. tts-app の手順で配置・起動
3. テキストを入力して「読み上げ」。英単語や固有名詞の読みは読み辞書タブで固定していく

## トラブルシューティング

| 症状 | 疑うこと |
|---|---|
| ロスは下がるのに音声がノイズ | 学習バックエンド（MPS）の不調。暗記テストで切り分け、CPUへ切替 |
| 学習が途中で止まる | スリープ／蓋閉じ。caffeinate とAC接続を確認 |
| transformers を入れたら torch が使えない | transformers のバージョン。4.46.3 に固定 |
| 英単語の読みが変 | モデルの限界ではなく辞書の出番。読み辞書に登録する |
