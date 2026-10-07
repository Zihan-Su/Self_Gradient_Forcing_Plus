#!/usr/bin/env bash
set -euo pipefail
export SGF_VARIANT=chunkwise
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/_distributed_training_common.sh" "$@"
