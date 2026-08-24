#!/bin/bash -l
# Login-node development wrapper.
#
# uv's default project environment is ./.venv — which on this repo is the GPU
# venv built by slurm/install_env.sh. Running `uv sync` or `uv run` against it
# re-resolves declared dependencies inside it and silently breaks the vLLM
# stack (this is how numpy reached 2.5 and took numba, and therefore every
# engine start, down with it).
#
# This wrapper points uv at a separate environment so that can't happen.
#
#     ./dev.sh pytest -q
#     ./dev.sh evollm run -c configs/mock_smoke.yaml
#     ./dev.sh python -c 'import evollm; print(evollm.__version__)'

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=.venv-dev

if [ $# -eq 0 ]; then
    echo "usage: ./dev.sh <command> [args...]   (e.g. ./dev.sh pytest -q)" >&2
    exit 2
fi

uv run --no-config "$@"
