"""Download official Wav2Lip checkpoint from HuggingFace."""
from huggingface_hub import hf_hub_download
import os, shutil

target_dir = "/mnt/d/HeyGem ONNX/HeyGem-Linux-Python-Hack-RTX-50/pretrain_models"
cache_dir = os.path.join(target_dir, ".cache")

repos = [
    "crispyman/Wav2Lip",
    "Rudrabha/Wav2Lip",
    "numz/wav2lip_gan",
    "YOURAKU/wav2lip",
]

for repo in repos:
    try:
        path = hf_hub_download(
            repo_id=repo,
            filename="wav2lip_gan.pth",
            cache_dir=cache_dir,
        )
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"OK {repo}: {path} ({size_mb:.1f}MB)")
        # If different from current, back up old and copy new
        current = os.path.join(target_dir, "wav2lip_gan.pth")
        if os.path.abspath(path) != os.path.abspath(current):
            if os.path.getsize(path) != os.path.getsize(current):
                backup = current + ".bak"
                shutil.move(current, backup)
                shutil.copy(path, current)
                print(f"  Replaced! Old backed up to {backup}")
        break
    except Exception as e:
        print(f"FAIL {repo}: {e}")
