#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
nanoplm_bin="${NANOPLM_BIN:-$repo_root/.venv/bin/nanoplm}"

if [[ ! -x "$python_bin" ]]; then
  echo "Expected Python at $python_bin" >&2
  echo "Create the virtualenv first, then install the project and CUDA extras from README.md." >&2
  exit 1
fi

if [[ ! -x "$nanoplm_bin" ]]; then
  echo "Expected nanoplm CLI at $nanoplm_bin" >&2
  echo "Install the project first with 'uv pip install -e .'." >&2
  exit 1
fi

if ! "$python_bin" - <<'PY' >/dev/null 2>&1
import grouped_gemm  # noqa: F401
PY
then
  echo "grouped_gemm is missing." >&2
  echo "Install it manually with GROUPED_GEMM_CUTLASS=1 and --no-build-isolation as documented in README.md." >&2
  exit 1
fi

exec env TORCH_LOGS="recompiles" \
  torchrun --nproc-per-node=4 \
  "$nanoplm_bin" pretrain from-yaml --pure-torch 2>&1 | tee pretrain.log
