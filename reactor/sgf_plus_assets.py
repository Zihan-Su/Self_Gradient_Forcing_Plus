"""Resolve deployment configuration and prepare pinned public assets once."""

import subprocess
from pathlib import Path

import yaml
from reactor_runtime import get_weights_path

from sgf_plus_model import Settings

SOURCE_REVISION = "14cda9bb35f87000fbc11138a5170002d213effe"
BASE_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
SGF_REVISION = "c11e68083d8904226ac18ad7874bd300649f7679"


def prepare(config_path: Path) -> Settings:
    """Validate the source already installed by the build and download weights."""
    from huggingface_hub import snapshot_download

    config = yaml.safe_load(config_path.read_text())
    source = Path(config["source_path"]).expanduser().resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Expected upstream source {SOURCE_REVISION}, got {revision}")
    weights = get_weights_path().resolve()
    snapshot_download(
        "Wan-AI/Wan2.1-T2V-1.3B",
        revision=BASE_REVISION,
        local_dir=weights / "wan_models/Wan2.1-T2V-1.3B",
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "google/umt5-xxl/*",
        ],
    )
    snapshot_download(
        "ZihanSu/Self_Gradient_Forcing_Plus",
        revision=SGF_REVISION,
        local_dir=weights / "hf_weights",
        allow_patterns=["chunkwise/model.pt"],
    )
    return Settings(
        source=source,
        weights=weights,
        output_latents=int(config.get("output_latents", 963)),
    )
