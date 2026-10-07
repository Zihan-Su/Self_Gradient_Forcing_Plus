#!/usr/bin/env bash
# Simple inference launcher for Self Gradient Forcing.
#
# Usage:
#   bash scripts/infer_self_gradient_forcing.sh [framewise|chunkwise] [checkpoint.pt] [prompts.txt]
#
# Examples:
#   bash scripts/infer_self_gradient_forcing.sh framewise
#   bash scripts/infer_self_gradient_forcing.sh chunkwise
#   bash scripts/infer_self_gradient_forcing.sh framewise logs/sgf/checkpoint_model_001000/model.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

VARIANT="${1:-framewise}"
case "$VARIANT" in
    framewise|chunkwise) ;;
    *)
        echo "Usage: bash scripts/infer_self_gradient_forcing.sh [framewise|chunkwise] [checkpoint.pt] [prompts.txt]" >&2
        exit 1
        ;;
esac
[[ $# -gt 0 ]] && shift

CONFIG="${CONFIG:-configs/self_gradient_forcing_${VARIANT}.yaml}"
CKPT="${1:-checkpoints/${VARIANT}/ar/model.pt}"
[[ $# -gt 0 ]] && shift
PROMPTS="${1:-prompts/test_prompt.txt}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
SEED="${SEED:-0}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-963}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/self_gradient_forcing}"

if [[ "$VARIANT" == "framewise" ]]; then
    KV_CACHE_SINK="${KV_CACHE_SINK:-4}"
    KV_CACHE_FIFO_FRAMES="${KV_CACHE_FIFO_FRAMES:-16}"
    KV_CACHE_CURRENT_FRAMES="${KV_CACHE_CURRENT_FRAMES:-1}"
else
    KV_CACHE_SINK="${KV_CACHE_SINK:-3}"
    KV_CACHE_FIFO_FRAMES="${KV_CACHE_FIFO_FRAMES:-6}"
    KV_CACHE_CURRENT_FRAMES="${KV_CACHE_CURRENT_FRAMES:-3}"
fi
KV_CACHE_MAX_FRAMES=$((KV_CACHE_SINK + KV_CACHE_FIFO_FRAMES + KV_CACHE_CURRENT_FRAMES))

USE_EMA="${USE_EMA:-1}"
EMA_ARGS=()
[[ "$USE_EMA" == "1" ]] && EMA_ARGS=(--use_ema)

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$PROMPTS" ]] || { echo "ERROR: prompt file not found: $PROMPTS" >&2; exit 1; }

GPU_COUNT="$($PYTHON_BIN - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
if (( GPU_COUNT >= 8 )); then
    RUN_GPUS=8
else
    RUN_GPUS=1
fi

promptset="$(basename "$PROMPTS" .txt)"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${OUTPUT_ROOT}/${VARIANT}_${promptset}_frames${NUM_OUTPUT_FRAMES}}"
mkdir -p "$OUTPUT_FOLDER"

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

cat <<INFO
==========================================
Self Gradient Forcing inference
Variant:           $VARIANT
Config:            $CONFIG
Checkpoint:        $CKPT
Prompts:           $PROMPTS
Visible GPUs:      $GPU_COUNT
Launch GPUs:       $RUN_GPUS
Output frames:     $NUM_OUTPUT_FRAMES latent frames
KV cache:          sink=$KV_CACHE_SINK, fifo=$KV_CACHE_FIFO_FRAMES, current=$KV_CACHE_CURRENT_FRAMES
Output:            $OUTPUT_FOLDER
==========================================
INFO

COMMON_ARGS=(
    inference.py
    --config_path "$CONFIG"
    --checkpoint_path "$CKPT"
    --data_path "$PROMPTS"
    --output_folder "$OUTPUT_FOLDER"
    --num_output_frames "$NUM_OUTPUT_FRAMES"
    --seed "$SEED"
    --long_video
    --kv_cache_max_frames "$KV_CACHE_MAX_FRAMES"
    --kv_cache_sink "$KV_CACHE_SINK"
    --kv_cache_train_frames 21
    --kv_cache_position_mode top_aligned
    "${EMA_ARGS[@]}"
)

if (( RUN_GPUS > 1 )); then
    unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK LOCAL_WORLD_SIZE GROUP_RANK
    "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node="$RUN_GPUS" "${COMMON_ARGS[@]}" \
        2>&1 | tee "$OUTPUT_FOLDER/infer.log"
else
    "$PYTHON_BIN" "${COMMON_ARGS[@]}" 2>&1 | tee "$OUTPUT_FOLDER/infer.log"
fi

echo
echo "[done] Videos written to: $OUTPUT_FOLDER"
find "$OUTPUT_FOLDER" -maxdepth 1 -name '*.mp4' -print | sort
