"""
scripts/download_wan_distilled.py

Automated provisioner for Wan-Animate-2 (1.3B Flash Distilled) weights.
Downloads the distilled model weights into storage/models/wan-animate-2-distilled/.
Explicitly skips the 14B model as requested by user.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is not installed. Install with: pip install huggingface_hub")
    sys.exit(1)


def get_target_dir() -> Path:
    # Check if running inside Docker container
    if os.path.exists("/app/storage/models"):
        target = Path("/app/storage/models/wan-animate-2-distilled")
    else:
        root_dir = Path(__file__).resolve().parent.parent
        target = root_dir / "storage" / "models" / "wan-animate-2-distilled"

    target.mkdir(parents=True, exist_ok=True)
    return target


def download_weights():
    target_dir = get_target_dir()
    print(f"🎬 Provisioning Wan-Animate-2 (1.3B Distilled) Weights into: {target_dir}")

    # 1. Distilled Transformer Weights (~2.83GB)
    distill_file = target_dir / "model_iter6000.pt"
    if distill_file.exists() and distill_file.stat().st_size > 1024 * 1024 * 100:
        print(f"✅ Distilled weights already present: {distill_file.name} ({distill_file.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print("📥 Downloading 1.3B distilled flow-matching weights from lightx2v/Wan2.1-T2V-1.3B-Distill-Models...")
        cached_path = hf_hub_download(
            repo_id="lightx2v/Wan2.1-T2V-1.3B-Distill-Models",
            filename="model_iter6000.pt",
            local_dir=str(target_dir),
        )
        print(f"✅ Downloaded: {cached_path}")

    # 2. VAE Checkpoint (~507MB)
    vae_file = target_dir / "Wan2.1_VAE.pth"
    if vae_file.exists() and vae_file.stat().st_size > 1024 * 1024 * 50:
        print(f"✅ VAE weights already present: {vae_file.name} ({vae_file.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print("📥 Downloading 1.3B VAE checkpoint from Wan-AI/Wan2.1-VACE-1.3B...")
        cached_vae = hf_hub_download(
            repo_id="Wan-AI/Wan2.1-VACE-1.3B",
            filename="Wan2.1_VAE.pth",
            local_dir=str(target_dir),
        )
        print(f"✅ Downloaded VAE: {cached_vae}")

    # Verify presence
    files = list(target_dir.glob("*"))
    print(f"\n🎉 Wan-Animate-2 Distilled Provisioning Complete! Files in {target_dir}:")
    for f in files:
        if f.is_file():
            print(f"  • {f.name}: {f.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    download_weights()
