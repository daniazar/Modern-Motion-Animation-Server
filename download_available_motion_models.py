"""
scripts/download_available_motion_models.py

Automated batch downloader for all available open-source motion model weights:
  1. SCAIL-2 (zai-org/SCAIL-2)
  2. MimicMotion (Tencent/MimicMotion)
  3. EchoMimicV3 (BadToBest/EchoMimic)
  4. Moore-AnimateAnyone (camenduru/AnimateAnyone)
  5. Champ (fudan-generative-ai/champ)

EXPLICITLY EXCLUDES:
  - Wan-Animate-2 14B (by user command)
  - Closed research models (Alibaba AnimateAnyone / AnimateAnyone 2)
  - Unreleased paper models without HF checkpoints (EMOSH, HOMA, LiveAnimate)
"""

import os
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def get_base_models_dir() -> Path:
    if os.path.exists("/app/storage/models"):
        return Path("/app/storage/models")
    root_dir = Path(__file__).resolve().parent.parent
    p = root_dir / "storage" / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


MODELS_TO_DOWNLOAD = [
    {
        "model_id": "scail-2",
        "name": "SCAIL-2",
        "repo_id": "zai-org/SCAIL-2",
        "files": [
            {"filename": "model/bias-aware-dpo-lora.pt", "min_mb": 500},
        ],
    },
    {
        "model_id": "mimicmotion",
        "name": "MimicMotion",
        "repo_id": "Tencent/MimicMotion",
        "files": [
            {"filename": "MimicMotion_1-1.pth", "min_mb": 1000},
        ],
    },
    {
        "model_id": "echomimicv3",
        "name": "EchoMimicV3",
        "repo_id": "BadToBest/EchoMimic",
        "files": [
            {"filename": "denoising_unet_acc.pth", "min_mb": 1000},
        ],
    },
    {
        "model_id": "moore-animateanyone",
        "name": "Moore-AnimateAnyone",
        "repo_id": "camenduru/AnimateAnyone",
        "files": [
            {"filename": "denoising_unet.pth", "min_mb": 1000},
            {"filename": "pose_guider.pth", "min_mb": 100},
        ],
    },
    {
        "model_id": "champ",
        "name": "Champ",
        "repo_id": "fudan-generative-ai/champ",
        "files": [
            {"filename": "champ/denoising_unet.pth", "min_mb": 1000},
        ],
    },
    {
        "model_id": "ltx-ripple",
        "name": "LTX-2.5 Ripple (FFAF IC-LoRA)",
        "repo_id": "WepeNerd/LTX-Ripple",
        "files": [
            {"filename": "LTX25_Ripple_v11.safetensors", "min_mb": 500},
        ],
    },
]


def download_all():
    base_dir = get_base_models_dir()
    print("=" * 70)
    print(f"🚀 Batch Model Weight Provisioner")
    print(f"📂 Target base directory: {base_dir}")
    print(f"🚫 Wan-Animate-2 14B: EXCLUDED (Testing 1.3B distilled first)")
    print(f"🔒 Closed / Unreleased models: EXCLUDED")
    print("=" * 70)

    for spec in MODELS_TO_DOWNLOAD:
        model_id = spec["model_id"]
        model_name = spec["name"]
        repo_id = spec["repo_id"]
        target_dir = base_dir / model_id
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📦 [{model_name}] (ID: {model_id}) from HuggingFace repo '{repo_id}'")
        print(f"   Destination: {target_dir}")

        for file_info in spec["files"]:
            fname = file_info["filename"]
            min_bytes = file_info["min_mb"] * 1024 * 1024
            local_name = os.path.basename(fname)
            dest_file = target_dir / local_name

            # Also check if file exists in subfolder or root
            alt_dest = target_dir / fname
            if (dest_file.exists() and dest_file.stat().st_size > min_bytes) or (alt_dest.exists() and alt_dest.stat().st_size > min_bytes):
                size_mb = (dest_file.stat().st_size if dest_file.exists() else alt_dest.stat().st_size) / (1024 * 1024)
                print(f"   ✅ Already present: {local_name} ({size_mb:.1f} MB)")
                continue

            print(f"   📥 Downloading {fname}...")
            t0 = time.time()
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    local_dir=str(target_dir),
                )
                elapsed = time.time() - t0
                file_size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
                print(f"   ✨ Downloaded {local_name} ({file_size_mb:.1f} MB in {elapsed:.1f}s)")
            except Exception as e:
                print(f"   ⚠️ Failed to download {fname}: {e}")

    print("\n" + "=" * 70)
    print("🎉 Model Weight Provisioning Run Finished!")
    print("Checking discovered model weights in storage/models:")
    for d in sorted(base_dir.iterdir()):
        if d.is_dir():
            files = [f for f in d.rglob("*") if f.is_file() and any(f.suffix.lower() in [".pth", ".pt", ".safetensors", ".bin"] for _ in [1])]
            total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
            print(f"  • {d.name}: {len(files)} weight file(s) ({total_mb:.1f} MB total)")
    print("=" * 70)


if __name__ == "__main__":
    download_all()
