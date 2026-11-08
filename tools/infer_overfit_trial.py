import sys
from pathlib import Path

import soundfile as sf

REPO = Path.home() / "my-voice-tts" / "style-bert-vits2"
sys.path.insert(0, str(REPO))

from style_bert_vits2.tts_model import TTSModel

assets = REPO / "model_assets" / "overfit"
model = TTSModel(
    model_path=assets / "overfit_e101_s400.safetensors",
    config_path=assets / "config.json",
    style_vec_path=assets / "style_vectors.npy",
    device="cpu",
)
sr, audio = model.infer(text="おはようございます、今日もいい天気ですね。")
out = Path.home() / "my-voice-tts" / "trial_overfit.wav"
sf.write(str(out), audio, sr)
print("WROTE", out, "sr=", sr, "samples=", len(audio), "sec=", len(audio) / sr)
