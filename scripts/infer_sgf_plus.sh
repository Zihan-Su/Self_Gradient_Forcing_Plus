#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/infer_sgf_plus.sh framewise|chunkwise checkpoint.pt [prompts.txt]" >&2
    exit 2
fi
VARIANT="$1"
export CONFIG="${CONFIG:-configs/sgf_plus_${VARIANT}.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sgf_plus}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/infer_self_gradient_forcing.sh" "$@"
