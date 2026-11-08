#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俺ボイスTTS - 自分の声で読み上げる専用アプリ (Style-Bert-VITS2 / ローカル完結)
テキスト入力 → 合成 → 試聴 → MP3保存。読み辞書（表記→読み）付き。
合成履歴は最新50件を保持し、再生・MP3保存・削除ができる。
モデルも音声もこのPCの外に出ない。
"""
import base64
import io
import json
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import soundfile as sf
import webview

BASE = Path(__file__).resolve().parent  # ~/my-voice-tts
REPO = BASE / "style-bert-vits2"
ASSETS = REPO / "model_assets" / "myvoice"
DICT_JSON = BASE / "tts_dict.json"
HISTORY_JSON = BASE / "tts_history.json"
OUT_DIR = BASE / "outputs"
HISTORY_MAX = 50
sys.path.insert(0, str(REPO))

FFMPEG_CANDIDATES = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]

DEFAULT_DICT = [
    {"word": "Twitter", "reading": "ツイッター"},
    {"word": "GitHub", "reading": "ギットハブ"},
    {"word": "YouTube", "reading": "ユーチューブ"},
    {"word": "Google", "reading": "グーグル"},
    {"word": "iPhone", "reading": "アイフォーン"},
]


def find_ffmpeg():
    for c in FFMPEG_CANDIDATES:
        if Path(c).exists():
            return c
    return "ffmpeg"


def find_model():
    """完成モデルの書き出し (safetensors, 最大ステップ) を優先。無ければ最新チェックポイント。"""
    def step_of(p):
        m = re.search(r"_s(\d+)\.safetensors$", p.name)
        return int(m.group(1)) if m else -1

    cands = sorted(ASSETS.glob("*.safetensors"), key=step_of)
    if cands:
        return cands[-1]
    gdir = REPO / "Data" / "myvoice" / "models"
    gs = sorted(gdir.glob("G_*.pth"),
                key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    return gs[-1] if gs else None


# ---- dictionary ----

def load_dict():
    if DICT_JSON.exists():
        try:
            data = json.loads(DICT_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [d for d in data if d.get("word") and d.get("reading")]
        except Exception:
            pass
    save_dict(DEFAULT_DICT)
    return list(DEFAULT_DICT)


def save_dict(entries):
    DICT_JSON.write_text(json.dumps(entries, ensure_ascii=False, indent=2),
                         encoding="utf-8")


def apply_dict(text, entries):
    applied = text
    for e in sorted(entries, key=lambda d: -len(d["word"])):
        applied = applied.replace(e["word"], e["reading"])
    return applied


# ---- history ----

def load_history():
    if HISTORY_JSON.exists():
        try:
            data = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [h for h in data if h.get("wav") and h.get("text")]
        except Exception:
            pass
    return []


def save_history(entries):
    HISTORY_JSON.write_text(json.dumps(entries, ensure_ascii=False, indent=2),
                            encoding="utf-8")


class Api:
    def __init__(self):
        self.model = None
        self.model_name = ""
        self.load_error = ""
        self.lock = threading.Lock()
        self.entries = load_dict()
        self.history = load_history()
        self.window = None

    # ---- model ----
    def load_model(self):
        print("LOAD: importing TTSModel", flush=True)
        try:
            from style_bert_vits2.tts_model import TTSModel
            print("LOAD: TTSModel imported", flush=True)
            mp = find_model()
            if mp is None:
                self.load_error = "モデルファイルが見つかりません"
                print("LOAD: no model file", flush=True)
                return
            print("LOAD: model file:", mp.name, flush=True)
            self.model = TTSModel(
                model_path=mp,
                config_path=ASSETS / "config.json",
                style_vec_path=ASSETS / "style_vectors.npy",
                device="cpu",
            )
            self.model_name = mp.name
            print("LOAD: model ready:", self.model_name, flush=True)
        except Exception as e:  # noqa: BLE001
            self.load_error = f"{type(e).__name__}: {e}"
            print("LOAD: error:", self.load_error, flush=True)

    def status(self):
        if self.model is not None:
            return {"ready": True, "model": self.model_name, "error": ""}
        return {"ready": False, "model": "", "error": self.load_error}

    # ---- synthesis ----
    def synthesize(self, text):
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "テキストが空です"}
        if self.model is None:
            return {"ok": False, "error": "モデル読み込み中です。少し待ってください"}
        if len(text) > 500:
            return {"ok": False, "error": "一度に読み上げられるのは500文字までです"}
        applied = apply_dict(text, self.entries)
        try:
            with self.lock:
                sr, audio = self.model.infer(text=applied)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"合成に失敗しました: {e}"}
        OUT_DIR.mkdir(exist_ok=True)
        now = datetime.now()
        wav_path = OUT_DIR / f"voice_{now.strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(str(wav_path), audio, sr)
        entry = {
            "time": now.strftime("%Y-%m-%d %H:%M"),
            "text": text,
            "applied": applied,
            "wav": wav_path.name,
            "sec": round(len(audio) / sr, 2),
        }
        self._push_history(entry)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "ok": True,
            "audio": "data:audio/wav;base64," + b64,
            "applied": applied,
            "sec": entry["sec"],
            "entry": entry,
        }

    def _push_history(self, entry):
        self.history.insert(0, entry)
        while len(self.history) > HISTORY_MAX:
            old = self.history.pop()
            try:
                (OUT_DIR / old["wav"]).unlink(missing_ok=True)
            except Exception:
                pass
        save_history(self.history)

    def get_history(self):
        return {"entries": self.history}

    def _history_wav_path(self, wav_name):
        name = Path(str(wav_name)).name
        for h in self.history:
            if h["wav"] == name:
                p = OUT_DIR / name
                return p if p.exists() else None
        return None

    def play_history(self, wav_name):
        p = self._history_wav_path(wav_name)
        if p is None:
            return {"ok": False, "error": "履歴の音声が見つかりません"}
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return {"ok": True, "audio": "data:audio/wav;base64," + b64}

    def delete_history(self, wav_name):
        name = Path(str(wav_name)).name
        self.history = [h for h in self.history if h["wav"] != name]
        save_history(self.history)
        try:
            (OUT_DIR / name).unlink(missing_ok=True)
        except Exception:
            pass
        return {"ok": True, "entries": self.history}

    def save_mp3(self, wav_name=None):
        if wav_name:
            src = self._history_wav_path(wav_name)
            if src is None:
                return {"ok": False, "error": "履歴の音声が見つかりません"}
        else:
            if not self.history:
                return {"ok": False, "error": "先に読み上げてください"}
            src = OUT_DIR / self.history[0]["wav"]
            if not src.exists():
                return {"ok": False, "error": "音声ファイルが見つかりません"}
        suggested = src.with_suffix(".mp3").name
        try:
            dlg = getattr(webview, "FileDialog", None)
            save_const = dlg.SAVE if dlg is not None else webview.SAVE_DIALOG
            result = self.window.create_file_dialog(
                save_const,
                directory=str(Path.home() / "Desktop"),
                save_filename=suggested,
                file_types=("MP3 (*.mp3)",),
            )
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"保存ダイアログを開けませんでした: {e}"}
        if not result:
            return {"ok": False, "error": "キャンセルしました"}
        dest = result[0] if isinstance(result, (list, tuple)) else result
        if not str(dest).lower().endswith(".mp3"):
            dest = str(dest) + ".mp3"
        try:
            subprocess.run(
                [find_ffmpeg(), "-y", "-i", str(src),
                 "-codec:a", "libmp3lame", "-q:a", "2", str(dest)],
                check=True, capture_output=True,
            )
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"MP3変換に失敗しました: {e}"}
        return {"ok": True, "path": str(dest)}

    # ---- dictionary ----
    def get_dict(self):
        return {"entries": self.entries}

    def set_dict(self, entries):
        clean = []
        for e in entries or []:
            w = str(e.get("word", "")).strip()
            r = str(e.get("reading", "")).strip()
            if w and r:
                clean.append({"word": w, "reading": r})
        self.entries = clean
        save_dict(clean)
        return {"ok": True, "entries": clean}


def main():
    api = Api()
    print("MAIN: starting load thread", flush=True)
    threading.Thread(target=api.load_model, daemon=True).start()
    print("MAIN: creating window", flush=True)
    window = webview.create_window(
        "俺ボイスTTS",
        url="file://" + str(BASE / "tts_ui.html"),
        js_api=api,
        width=780,
        height=680,
        min_size=(640, 540),
    )
    api.window = window
    print("MAIN: starting webview", flush=True)
    webview.start()


if __name__ == "__main__":
    main()
