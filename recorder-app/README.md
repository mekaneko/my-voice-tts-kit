# 専用録音アプリ「TTSボイス収録」

台本CSVを1文ずつ表示し、読み上げを録音して連番WAVで自動保存するアプリです。台本と話者名は差し替え可能で、他の話者でもそのまま使えます。

## 仕様

- 録音形式: 48kHz / モノラル / 16bit WAV
- 文番号＝ファイル名（001.wav〜）で自動保存。撮り直しは同じ番号に上書き
- 録音停止後に自動再生、ピーク音量などの警告表示
- 録音のたびに台本対応表（CSV）と学習用ファイルリスト（filelist.txt）を自動生成
- pywebview 製のデスクトップアプリ（macOS想定）

## 使い方

1. `script.csv` を用意する（形式は `script_sample.csv` を参照。列は number, filename, text, category, hint）
2. `config_template.json` を `config.json` にコピーし、`speaker_name` を自分の話者名にする
3. 依存を入れて起動する

```bash
python3 -m venv venv
./venv/bin/pip install pywebview sounddevice soundfile numpy
./venv/bin/python recorder_app.py
```

## 録音のコツ（実測に基づく）

- マイクはダイナミックマイク推奨。口から5〜10cm、ピークが −12〜−6 dB になるようゲイン調整
- 感情文は「普段の会話の7割くらいの強さ」で自然に。音量ではなく抑揚で表現する
- 文数は450文前後、実声で20分程度が一つの目安（本家の運用実績）
