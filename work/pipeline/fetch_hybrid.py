"""Download the hybrid (deployed) Neodragon pipeline weights.

fp16 variants are used where the repo offers them, which is what CLAUDE.md's
minimal-subset accounting assumes. The plain .safetensors twins of those files
are skipped.
"""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download

LOCAL = Path(__file__).resolve().parents[1] / "models" / "neodragon"
REPO = "Qualcomm-AI-Research/Neodragon"

ALLOW = [
    "text_encoder/*", "text_encoder_2/*", "text_encoder_3/*",
    "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*",
    "context_adapter/*",
    "diffusion_transformer_320p/*",
    "causal_video_vae/*",
    "ssd_1b_unet/*",
    "ssd_1b_vae/*",
    "ssd_1b_text_encoder/*", "ssd_1b_text_encoder_2/*",
    "ssd_1b_tokenizer/*", "ssd_1b_tokenizer_2/*",
    "*.json",
]
# prefer the fp16 twins for the SSD1B pieces that have them
IGNORE = [
    "ssd_1b_vae/diffusion_pytorch_model.safetensors",
    "ssd_1b_text_encoder/model.safetensors",
    "ssd_1b_text_encoder_2/model.safetensors",
    "*_multistep_t2v/*",
]


def main():
    p = snapshot_download(REPO, allow_patterns=ALLOW, ignore_patterns=IGNORE,
                          local_dir=str(LOCAL), max_workers=4)
    print("OK", p)
    tot = 0
    for r, _d, fs in os.walk(LOCAL):
        for n in fs:
            if ".cache" in r:
                continue
            tot += (Path(r) / n).stat().st_size
    print(f"total on disk: {tot/1e9:.2f} GB")


if __name__ == "__main__":
    main()
