#!/usr/bin/env bash
# Distributed training launcher for Self Gradient Forcing.
# Run the same command on every node. Without NNODES/NODE_RANK/MASTER_ADDR,
# nodes discover each other through .rendezvous on the shared project directory.

set -euo pipefail

SGF_VARIANT="${SGF_VARIANT:?SGF_VARIANT must be framewise or chunkwise}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

DEFAULT_CONFIG="configs/self_gradient_forcing_${SGF_VARIANT}.yaml"
if [[ $# -gt 0 && "$1" == *.yaml ]]; then
    CONFIG="$1"
    shift
else
    CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
fi

DEFAULT_LOGDIR="logs/$(basename "${CONFIG%.yaml}")"
if [[ $# -gt 0 ]]; then
    LOGDIR="$1"
    shift
else
    LOGDIR="${LOGDIR:-$DEFAULT_LOGDIR}"
fi
EXTRA_ARGS=("$@")

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

TORCHRUN="${TORCHRUN:-torchrun}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDZV_ID="${RDZV_ID:-self_gradient_forcing_${SGF_VARIANT}}"

if [[ -n "${NNODES:-}" && -n "${NODE_RANK:-}" && -n "${MASTER_ADDR:-}" ]]; then
    echo "[topology] Manual: NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR"
elif [[ -n "${WORLD_SIZE:-}" && -n "${RANK:-}" && -n "${MASTER_ADDR:-}" ]]; then
    NNODES="$WORLD_SIZE"
    NODE_RANK="$RANK"
    echo "[topology] Scheduler: NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR"
else
    GATHER_WINDOW="${GATHER_WINDOW:-60}"
    SYNC_DIR="$PROJECT_ROOT/.rendezvous/$RDZV_ID"
    MY_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [[ -n "$MY_IP" ]] || { echo "ERROR: cannot determine local IP" >&2; exit 1; }

    mkdir -p "$SYNC_DIR"
    find "$SYNC_DIR" -maxdepth 1 -type f -mmin +15 -delete 2>/dev/null || true
    : > "$SYNC_DIR/$MY_IP"
    trap 'rm -f "$SYNC_DIR/$MY_IP"' EXIT

    echo "[topology] Registered $MY_IP; gathering peers for ${GATHER_WINDOW}s ..."
    while :; do
        oldest="$(find "$SYNC_DIR" -maxdepth 1 -type f -printf '%T@\n' 2>/dev/null | sort -n | head -1 | cut -d. -f1)"
        [[ -n "$oldest" ]] || { sleep 1; continue; }
        now="$(date +%s)"
        elapsed=$(( now - oldest ))
        n="$(find "$SYNC_DIR" -maxdepth 1 -type f | wc -l)"
        (( elapsed >= GATHER_WINDOW )) && break
        printf '\r[topology] %d node(s), %ds/%ds elapsed ...   ' "$n" "$elapsed" "$GATHER_WINDOW"
        sleep 2
    done
    echo

    mapfile -t NODE_IPS < <(ls "$SYNC_DIR" | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n)
    NNODES="${#NODE_IPS[@]}"
    MASTER_ADDR="${NODE_IPS[0]}"
    NODE_RANK=""
    for i in "${!NODE_IPS[@]}"; do
        [[ "${NODE_IPS[$i]}" == "$MY_IP" ]] && NODE_RANK="$i" && break
    done
    [[ -n "$NODE_RANK" ]] || { echo "ERROR: own IP $MY_IP missing from registrations" >&2; exit 1; }
    echo "[topology] Auto-detected: NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR"
    echo "[topology] Node order: ${NODE_IPS[*]}"
fi

if [[ ! "$MASTER_ADDR" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "WARNING: MASTER_ADDR=$MASTER_ADDR is not an IPv4 address; static rendezvous may hang." >&2
fi

TOTAL_GPUS=$((NNODES * NUM_GPUS))
WANDB_ARGS=()
if [[ "${ENABLE_WANDB:-0}" != "1" ]]; then
    WANDB_ARGS+=("--disable-wandb")
fi

mkdir -p "$LOGDIR"

cat <<INFO
==========================================
Self Gradient Forcing training
Variant:    $SGF_VARIANT
Config:     $CONFIG
Logdir:     $LOGDIR
Node Rank:  $NODE_RANK / $NNODES nodes
GPUs/node:  $NUM_GPUS   Total: $TOTAL_GPUS
Master:     $MASTER_ADDR:$MASTER_PORT
wandb:      $([[ ${#WANDB_ARGS[@]} -gt 0 ]] && echo DISABLED || echo enabled)
==========================================
INFO

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

unset WORLD_SIZE RANK LOCAL_RANK GROUP_RANK LOCAL_WORLD_SIZE 2>/dev/null || true

"$TORCHRUN" \
    --nnodes="$NNODES" \
    --nproc_per_node="$NUM_GPUS" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config_path "$CONFIG" \
    --logdir "$LOGDIR" \
    "${WANDB_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOGDIR/train_node${NODE_RANK}_shell.log"
