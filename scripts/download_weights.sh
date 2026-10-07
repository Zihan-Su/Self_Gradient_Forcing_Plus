#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

HF_CLI="${HF_CLI:-hf}"
SGF_REPO="JunhaoZhuang/Self_Gradient_Forcing"
SGF_PLUS_REPO="ZihanSu/Self_Gradient_Forcing_Plus"

"$HF_CLI" download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
"$HF_CLI" download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B

for variant in chunkwise framewise; do
  "$HF_CLI" download "$SGF_REPO" "init/$variant/ar_diffusion.pt" --local-dir checkpoints
done
"$HF_CLI" download "$SGF_REPO" vidprom_filtered_extended.txt --local-dir prompts

"$HF_CLI" download "$SGF_PLUS_REPO" \
  chunkwise/model.pt framewise/model.pt \
  --local-dir hf_weights
