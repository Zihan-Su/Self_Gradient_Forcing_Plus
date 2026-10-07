#!/usr/bin/env bash
set -euo pipefail
export SGF_VARIANT=chunkwise
export CONFIG="${CONFIG:-configs/sgf_plus_chunkwise.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_distributed_training_common.sh" "$@"
