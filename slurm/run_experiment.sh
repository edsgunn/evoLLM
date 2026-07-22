#!/bin/bash -l
# The main run: 4 rooms on a full GH200 node (§2.1). Run the two prechecks
# first (measure_throughput.sh, precheck_handshake.sh) — they are what make a
# negative result interpretable (§7).

#SBATCH --job-name=evollm_run
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM/slurm_logs/run_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM/slurm_logs/run_%j.err

set -euo pipefail
REPO=/lus/lfs1aip2/projects/a5l/egunn/projects/evoLLM
CONFIG="${CONFIG:-configs/node_4room.yaml}"
cd "$REPO"
module purge
module load cudatoolkit
module load brics/nccl
source "$REPO/.venv/bin/activate"

echo "config: $CONFIG  job: ${SLURM_JOB_ID:-interactive}  node: $(hostname -s)"
nvidia-smi -L || true

evollm run -c "$CONFIG" --name "${RUN_NAME:-node4room_${SLURM_JOB_ID:-dev}}"
