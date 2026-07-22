#!/bin/bash -l
# ============================================================
# Build the evoLLM venv with vLLM, following the Isambard tutorial recipe
# (same as MARLLLM/slurm_scripts/install_vllm.sh, which is known-good here):
#   https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-inference/
#
# Must run on a GPU node: wheel resolution is CUDA-aware, so install happens
# inside the job for --torch-backend=auto to detect CUDA correctly.
#     sbatch slurm/install_env.sh
#
# Afterwards, do NOT `uv sync` against this venv — it will re-resolve torch
# from PyPI defaults and break vLLM. Ad-hoc additions: `uv pip install <pkg>`.
# ============================================================

#SBATCH --job-name=evollm_install
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM/slurm_logs/install_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM/slurm_logs/install_%j.err

set -euo pipefail

REPO=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM
VENV="$REPO/.venv"
cd "$REPO"
mkdir -p "$REPO/slurm_logs"

echo "================================================================"
echo "Rebuilding $VENV with vLLM (Isambard recipe)"
echo "Job ID : ${SLURM_JOB_ID:-interactive}"
echo "Node   : $(hostname -s)"
echo "Date   : $(date)"
echo "================================================================"

module purge
module load cudatoolkit
module load brics/nccl

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "[ERROR] uv not found in PATH"; exit 2; }
echo "[INFO] uv: $(uv --version)"

if [ -d "$VENV" ]; then
    echo "[INFO] removing existing $VENV"
    rm -rf "$VENV"
fi

uv venv "$VENV" --seed --python=3.12
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ""
echo "--- Step 1/2: vLLM stack (Isambard tutorial recipe) ---"
uv pip install -U "vllm[flashinfer]==0.15.1" \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/0.15.1/vllm

echo ""
echo "--- Step 2/2: evollm package (torch already pinned by vLLM) ---"
uv pip install -e . --group dev

echo ""
echo "--- Validation ---"
python - <<'PY'
import platform, torch, transformers, vllm, evollm
print(f"python       {platform.python_version()}")
print(f"torch        {torch.__version__}")
print(f"cuda (torch) {torch.version.cuda}")
print(f"transformers {transformers.__version__}")
print(f"vllm         {vllm.__version__}")
print(f"evollm       {evollm.__version__}")
print(f"cuda avail   {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}      {p.name} | {p.total_memory/1024**3:.1f} GiB")
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
PY

echo ""
echo "--- Mock end-to-end smoke (no GPU needed, tests the whole loop) ---"
python -m pytest tests -q

echo ""
echo "================================================================"
echo "Install complete. Activate with:  source $VENV/bin/activate"
echo "================================================================"
