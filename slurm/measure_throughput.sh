#!/bin/bash -l
# §4.1 precheck: concurrent-adapter throughput sweep. Adapter residency
# (max_loras) and swap-in latency, not raw device memory, likely bind the
# population size per room — measure before fixing N.

#SBATCH --job-name=evollm_throughput
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=02:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM/slurm_logs/throughput_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM/slurm_logs/throughput_%j.err

set -euo pipefail
REPO=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM
cd "$REPO"
source "$REPO/slurm/_env.sh"
source "$REPO/.venv/bin/activate"

evollm measure-throughput -c configs/single_gpu.yaml \
    --sweep 1,2,4,8,16,32,64 --tokens 256
