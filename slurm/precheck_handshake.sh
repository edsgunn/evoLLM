#!/bin/bash -l
# §6 precheck: mate-handshake base rate under zero adapters (frozen base
# model). Selection cannot act until reproduction has occurred at least once;
# if random/base agents essentially never complete a handshake, the
# population dies before generation one and the main run yields nothing.

#SBATCH --job-name=evollm_precheck
#SBATCH --nodes=1
# Four GPUs, matching slurm/run_experiment.sh. The base rate is only
# interpretable against a main run, so it must be measurable under the SAME
# world -- and the reference config declares one room per GPU. Requesting one
# GPU for a four-room config cost a whole 4-hour allocation: room gpu0 built
# fine, gpu1 asked NVML for a device that was not allocated, and the engine
# core died 31 seconds in while the job sat to its time limit.
#SBATCH --gpus=4
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
