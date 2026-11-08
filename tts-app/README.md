# 推論アプリ「俺ボイスTTS」

学習済みモデルを使って「テキスト入力 → 自分の声で合成 → 試聴 → MP3保存」をする専用アプリです（pywebview製、すべてローカル処理）。

## 機能

- メイン画面：テキスト入力（500文字まで）、読み上げ試聴、MP3保存、合成履歴（最新50件・再生／MP3保存／削除）
- 読み辞書タブ：表記→読みの置き換え辞書。英単語や固有名詞の読みを固定できる（例: Twitter → ツイッター）。合成前に自動適用され、その場で保存される
- 起動時に `style-bert-vits2/model_assets/<話者名>/` の最新モデル（最大ステップの safetensors）を自動読み込み

## 配置と起動

アプリ一式は、Style-Bert-VITS2 のプロジェクトフォルダの **1つ上の階層** に置く想定です：

```
my-voice-tts/
├── style-bert-vits2/        # SBV2本体（model_assets/<話者名>/ に完成モデル）
├── tts_app.py               # 本アプリ
├── tts_ui.html
├── tts_dict.json            # 初回起動時に自動生成
└── 俺ボイスTTS.app/          # launcher + Info.plist から組み立て
```

```bash
./sbv2-venv/bin/pip install pywebview
./sbv2-venv/bin/python tts_app.py
```

`.app` にする場合：`俺ボイスTTS.app/Contents/MacOS/launcher` に `launcher` を、`Contents/Info.plist` に `Info.plist` を配置し、launcher に実行権限を付けます。アイコンは `icon_1024.png` から iconset を作り `iconutil -c icns` で `AppIcon.icns` 化して `Contents/Resources/` に置きます。

MP3変換には ffmpeg が必要です（`brew install ffmpeg`）。

## 注意

- 話者名やモデル配置が異なる場合は、`tts_app.py` の `ASSETS` パスを自分の構成に合わせてください
- 合成はCPU実行です（1文あたり約1〜数秒）
