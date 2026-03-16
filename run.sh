#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
nanoplm_bin="${NANOPLM_BIN:-$repo_root/.venv/bin/nanoplm}"

if [[ ! -x "$python_bin" ]]; then
  echo "Expected Python at $python_bin" >&2
  echo "Create the virtualenv first, then run ./scripts/uv-sync-cuda.sh" >&2
  exit 1
fi

if [[ ! -x "$nanoplm_bin" ]]; then
  echo "Expected nanoplm CLI at $nanoplm_bin" >&2
  echo "Sync dependencies first with ./scripts/uv-sync-cuda.sh" >&2
  exit 1
fi

if ! "$python_bin" - <<'PY' >/dev/null 2>&1
import grouped_gemm  # noqa: F401
PY
then
  echo "grouped_gemm is missing; syncing CUDA dependencies first..."
  "$repo_root/scripts/uv-sync-cuda.sh"
fi

exec env TORCH_LOGS="recompiles" \
  torchrun --nproc-per-node=4 \
  "$nanoplm_bin" pretrain from-yaml --pure-torch 2>&1 | tee pretrain.log
