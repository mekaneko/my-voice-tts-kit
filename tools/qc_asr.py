import csv, difflib, pathlib, re, sys
import mlx_whisper

BASE = pathlib.Path(__file__).resolve().parent.parent  # ~/my-voice-tts
WAV = BASE / "dataset/wanwan/wav"
rows = {int(r["number"]): r for r in csv.DictReader(open(BASE / "台本対応表.csv", encoding="utf-8"))}

def norm(s):
    return re.sub(r"[\s、。！？!?,.・「」『』（）()\-ー~〜…]", "", s)

out = open(BASE / "qc_report.csv", "w", newline="", encoding="utf-8")
w = csv.writer(out)
w.writerow(["number", "ratio", "script", "asr"])
flags = []
for n in sorted(rows):
    p = WAV / f"{n:03d}.wav"
    r = mlx_whisper.transcribe(str(p), path_or_hf_repo="mlx-community/whisper-small-mlx",
                               language="ja", verbose=False)
    asr = r["text"].strip()
    ratio = difflib.SequenceMatcher(None, norm(rows[n]["text"]), norm(asr)).ratio()
    w.writerow([n, round(ratio, 3), rows[n]["text"], asr])
    out.flush()
    if ratio < 0.75:
        flags.append((n, round(ratio, 2)))
    if n % 50 == 0:
        print("done", n, flush=True)
print("FLAGS:", flags)
print("QC DONE")
