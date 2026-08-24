#!/bin/bash -l
# §6 precheck: mate-handshake base rate under zero adapters (frozen base
# model). Selection cannot act until reproduction has occurred at least once;
# if random/base agents essentially never complete a handshake, the
# population dies before generation one and the main run yields nothing.

#SBATCH --job-name=evollm_precheck
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM/slurm_logs/precheck_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM/slurm_logs/precheck_%j.err

set -euo pipefail
REPO=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM
cd "$REPO"
source "$REPO/slurm/_env.sh"
source "$REPO/.venv/bin/activate"

CONFIG="${CONFIG:-configs/single_gpu.yaml}"
echo "config: $CONFIG"
evollm precheck-handshake -c "$CONFIG" --steps 20000 --trace 6000
