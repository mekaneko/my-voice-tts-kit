#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTSボイス収録 - 台本読み上げ録音アプリ (pywebview)
台本CSV (number,filename,text,category,hint) を読み、1文ずつ録音して
連番WAVで自動保存し、台本対応表CSVとStyle-Bert-VITS2用filelistを自動生成する。
台本・話者名は差し替え可能（将来の配布・他話者利用を想定した汎用設計）。
音声はこのPCの外に出ない。
"""
import csv, json, os, sys, threading, time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import webview

BASE = Path(__file__).resolve().parent
REC_DIR = BASE / "recordings"
SCRIPT_CSV = BASE / "script.csv"
CONFIG_JSON = BASE / "config.json"
TABLE_CSV = BASE / "台本対応表.csv"
FILELIST_TXT = BASE / "filelist.txt"
SAMPLE_RATE = 48000

DEFAULT_CONFIG = {
    "speaker_name": "myvoice",
    "language": "JP",
    "input_device": None,   # device index or None=default
    "output_device": None,
    "last_number": 1,
    "auto_play": True,
}

def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_JSON.exists():
        try:
            cfg.update(json.loads(CONFIG_JSON.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg

def save_config(cfg):
    CONFIG_JSON.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

def load_script():
    rows = []
    with SCRIPT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "number": int(r["number"]),
                "filename": r["filename"],
                "text": r["text"],
                "category": r.get("category", ""),
                "hint": r.get("hint", ""),
            })
    rows.sort(key=lambda x: x["number"])
    return rows

def wav_info(path):
    try:
        info = sf.info(str(path))
        data, _ = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        peak_db = round(20 * np.log10(peak), 1) if peak > 0 else -99.0
        duration = round(info.frames / info.samplerate, 2)
        # noise floor: RMS of the quietest 10% of 50ms frames
        noise_db = -99.0
        if data.size:
            fl = max(1, int(info.samplerate * 0.05))
            n = (len(data) // fl) * fl
            if n:
                frames = data[:n].reshape(-1, fl)
                rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
                db = 20 * np.log10(rms + 1e-9)
                noise_db = round(float(np.percentile(db, 10)), 1)
        warnings = []
        clipped = bool(data.size and np.mean(np.abs(data) >= 0.999) > 0.001)
        if peak_db > -1.0 or clipped:
            warnings.append("うるさすぎ（歪みの可能性）")
        if peak_db <= -90.0:
            warnings.append("音が入っていません（マイク・入力選択を確認）")
        elif peak_db < -30.0:
            warnings.append("小さすぎ（ゲイン不足かも）")
        if duration >= 1.0 and noise_db > -48.0:
            warnings.append(f"ノイズ多め（ノイズフロア {noise_db}dB）")
        if duration < 0.7:
            warnings.append("短すぎ（録り直し推奨）")
        return {"duration": duration, "peak_db": peak_db,
                "noise_db": noise_db, "warnings": warnings,
                "samplerate": info.samplerate}
    except Exception:
        return None

LAST_INFOS = {}

def regenerate_tables(rows, cfg):
    """録音済みファイルを走査し、対応表CSVとfilelistを再生成する。"""
    REC_DIR.mkdir(exist_ok=True)
    done = 0
    with TABLE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["number", "filename", "text", "category", "recorded",
                    "duration_sec", "peak_db", "noise_db", "warnings",
                    "sample_rate", "speaker"])
        flines = []
        for r in rows:
            p = REC_DIR / r["filename"]
            info = wav_info(p) if p.exists() else None
            if info:
                done += 1
                LAST_INFOS[r["number"]] = info
                w.writerow([r["number"], r["filename"], r["text"], r["category"], "yes",
                            info["duration"], info["peak_db"], info["noise_db"],
                            " / ".join(info["warnings"]), info["samplerate"],
                            cfg["speaker_name"]])
                flines.append(f"recordings/{r['filename']}|{cfg['speaker_name']}|{cfg['language']}|{r['text']}")
            else:
                LAST_INFOS.pop(r["number"], None)
                w.writerow([r["number"], r["filename"], r["text"], r["category"], "no",
                            "", "", "", "", "", cfg["speaker_name"]])
    FILELIST_TXT.write_text("\n".join(flines) + ("\n" if flines else ""), encoding="utf-8")
    return done

class Recorder:
    def __init__(self):
        self.stream = None
        self.frames = []
        self.recording = False
        self.level = 0.0
        self.start_time = 0.0
        self.lock = threading.Lock()

    def _cb(self, indata, frames, time_info, status):
        with self.lock:
            self.frames.append(indata.copy())
        self.level = float(np.max(np.abs(indata))) if indata.size else 0.0

    def start(self, device):
        with self.lock:
            self.frames = []
        self.level = 0.0
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                     dtype="float32", device=device,
                                     callback=self._cb)
        self.stream.start()
        self.recording = True
        self.start_time = time.time()

    def stop(self):
        if self.stream:
            self.stream.stop(); self.stream.close(); self.stream = None
        self.recording = False
        with self.lock:
            frames = list(self.frames); self.frames = []
        if not frames:
            return None
        return np.concatenate(frames, axis=0).reshape(-1)

def log(msg):
    try:
        with (BASE / "logs" / "recorder_app.log").open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except Exception:
        pass

class Api:
    def __init__(self):
        self.cfg = load_config()
        self.cfg.pop("auto_advance", None)
        self.rows = load_script()
        self.by_num = {r["number"]: r for r in self.rows}
        self.rec = Recorder()
        self.current = self.cfg.get("last_number") or 1
        if self.current not in self.by_num:
            self.current = self.rows[0]["number"]
        REC_DIR.mkdir(exist_ok=True)
        regenerate_tables(self.rows, self.cfg)

    # --- state ---
    def get_state(self):
        recorded = {r["number"]: (REC_DIR / r["filename"]).exists() for r in self.rows}
        cur = self.by_num.get(self.current, self.rows[0])
        info = wav_info(REC_DIR / cur["filename"]) if recorded[cur["number"]] else None
        return {
            "total": len(self.rows),
            "done": sum(1 for v in recorded.values() if v),
            "warnings": {n: i["warnings"] for n, i in LAST_INFOS.items() if i.get("warnings")},
            "recorded": recorded,
            "current": cur,
            "current_info": info,
            "config": self.cfg,
            "recording": self.rec.recording,
        }

    def get_devices(self):
        ins, outs = [], []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                ins.append({"index": i, "name": d["name"]})
            if d["max_output_channels"] > 0:
                outs.append({"index": i, "name": d["name"]})
        return {"inputs": ins, "outputs": outs}

    def set_config(self, patch):
        self.cfg.update(patch or {})
        save_config(self.cfg)
        regenerate_tables(self.rows, self.cfg)
        return {"ok": True}

    def jump(self, number):
        n = int(number)
        if n in self.by_num:
            self.current = n
            self.cfg["last_number"] = n
            save_config(self.cfg)
        return self.get_state()

    def jump_next_unrecorded(self):
        for r in self.rows:
            if not (REC_DIR / r["filename"]).exists():
                return self.jump(r["number"])
        return self.get_state()

    # --- recording ---
    def start_recording(self):
        log("start_recording called")
        if self.rec.recording:
            return {"ok": False, "error": "already recording"}
        try:
            self.rec.start(self.cfg.get("input_device"))
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def stop_recording(self):
        log("stop_recording called")
        if not self.rec.recording:
            return {"ok": False, "error": "not recording"}
        data = self.rec.stop()
        if data is None or len(data) == 0:
            return {"ok": False, "error": "no audio captured"}
        cur = self.by_num.get(self.current, self.rows[0])
        path = REC_DIR / cur["filename"]
        sf.write(str(path), data, SAMPLE_RATE, subtype="PCM_16")
        log(f"saved {cur['filename']} {round(len(data)/SAMPLE_RATE,2)}s")
        done = regenerate_tables(self.rows, self.cfg)
        info = wav_info(path)
        # stay on the same sentence so the user can listen back immediately
        self.cfg["last_number"] = self.current
        save_config(self.cfg)
        state = self.get_state()
        state.update({"ok": True, "saved": cur["filename"], "info": info, "done": done})
        return state

    def get_level(self):
        return {"level": self.rec.level,
                "recording": self.rec.recording,
                "elapsed": round(time.time() - self.rec.start_time, 1) if self.rec.recording else 0}

    def play_current(self):
        cur = self.by_num.get(self.current, self.rows[0])
        path = REC_DIR / cur["filename"]
        if not path.exists():
            return {"ok": False, "error": "not recorded yet"}
        try:
            data, sr = sf.read(str(path), dtype="float32")
            sd.play(data, sr, device=self.cfg.get("output_device"))
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def stop_playback(self):
        sd.stop()
        return {"ok": True}

    def delete_current(self):
        cur = self.by_num.get(self.current, self.rows[0])
        path = REC_DIR / cur["filename"]
        if path.exists():
            path.unlink()
        regenerate_tables(self.rows, self.cfg)
        return self.get_state()

    def add_sentence(self, text):
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "文が空です"}
        if len(text) > 200:
            return {"ok": False, "error": "200文字までにしてください"}
        n = max(r["number"] for r in self.rows) + 1
        filename = f"{n:03d}.wav"
        with SCRIPT_CSV.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([n, filename, text, "追加", "自分の言葉で自然に"])
        self.rows = load_script()
        self.by_num = {r["number"]: r for r in self.rows}
        self.current = n
        self.cfg["last_number"] = n
        save_config(self.cfg)
        regenerate_tables(self.rows, self.cfg)
        state = self.get_state()
        state["ok"] = True
        state["added"] = n
        return state

    def regenerate(self):
        done = regenerate_tables(self.rows, self.cfg)
        return {"ok": True, "done": done}

def main():
    api = Api()
    html = (BASE / "recorder_ui.html").read_text(encoding="utf-8")
    window = webview.create_window("TTSボイス収録", html=html, js_api=api,
                                   width=980, height=760, min_size=(760, 560))
    webview.start()

if __name__ == "__main__":
    main()
