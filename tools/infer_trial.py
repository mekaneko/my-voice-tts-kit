import sys
from pathlib import Path

import soundfile as sf

REPO = Path.home() / "my-voice-tts" / "style-bert-vits2"
sys.path.insert(0, str(REPO))

from style_bert_vits2.tts_model import TTSModel

assets = REPO / "model_assets" / "myvoice"
model = TTSModel(
    model_path=assets / "myvoice_e9_s2000.safetensors",
    config_path=assets / "config.json",
    style_vec_path=assets / "style_vectors.npy",
    device="cpu",
)
sr, audio = model.infer(text="私はツイッターをやります")
out = Path.home() / "my-voice-tts" / "trial_step2000.wav"
sf.write(str(out), audio, sr)
print("WROTE", out, "sr=", sr, "samples=", len(audio), "sec=", len(audio) / sr)
