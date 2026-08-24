# Shared Isambard AI Phase 2 environment. Source this from every job script:
#   source "$REPO/slurm/_env.sh"
#
# Nothing here is optional — each line fixes a specific way this cluster
# breaks vLLM. Keep it in sync with MARLLLM/slurm_scripts, which is where the
# known-good recipe lives.

module purge
module load cudatoolkit
module load brics/nccl
module load gcc-native/12.3

# Native-extension compiler selection. `module load cudatoolkit` puts the
# NVIDIA HPC SDK's nvc on PATH as the default CC. Triton JIT-compiles its
# kernel launcher at runtime with GCC-only flags, and nvc rejects them:
#     nvc-Error-Unknown switch: -Wno-psabi
# which surfaces as "Engine core initialization failed" during CUDA graph
# capture. /usr/bin/gcc is SUSE base 7.5 (too old), so pin gcc-14.
# nvcc still finds its own bundled host compiler via CUDA_HOME.
export CC=gcc-14
export CXX=g++-14

# Model cache — keep large HF downloads off the $HOME quota.
export HF_HOME="${HF_HOME:-/lus/lfs1aip2/projects/a5l/egunn/hf_cache}"
mkdir -p "$HF_HOME"

# Triton/inductor caches: node-local and per-job, so concurrent jobs on the
# same node cannot race each other's compiled artefacts.
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$TRITON_CACHE_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "[env] CC=$CC ($(command -v $CC))  CXX=$CXX"
echo "[env] HF_HOME=$HF_HOME  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
