#!/usr/bin/env bash
set -euo pipefail
export SGF_VARIANT=framewise
export CONFIG="${CONFIG:-configs/sgf_plus_framewise.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_distributed_training_common.sh" "$@"
