# evoLLM

Gradient-free evolution as a substrate for in-context surprise minimisation.

Agents are LoRA adapters over a shared frozen base model. They act, observe,
reproduce and die inside a block economy whose single currency is device
memory; the only thing that changes across generations is the distribution of
initial adapter weights. No backward pass exists anywhere in this codebase.
See the proposal document for the theory; section references (§) throughout
the code point into it.

## Layout

```
src/evollm/
  config.py        YAML experiment config
  blocks.py        the block economy: one pool per room, KV + adapters (§2.2)
  actions.py       <say> <mate> <accept> <go> grammar (§2.4)
  prompts.py       system prompt and observation formatting (§3.3)
  agent.py         per-agent soma: context, queues, mode, pending mates
  genome.py        LoRA genome, per-site crossover + Gaussian mutation (§3)
  controller.py    the world clock: one token per agent per step (§2.3),
                   death (§2.5), handshake births, movement
  world.py         room graph, migration with reserve-before-release (§4.5)
  events.py        JSONL event log, death-cause audit (§4.3, §5)
  report.py        offline aggregation: handshake rate, viability, terseness
  evaluate.py      held-out surprise vs unevolved controls (§5, §7)
  engines/
    mock.py        scripted agents, toy tokenizer — full environment, no GPU
    vllm_engine.py multi-LoRA vLLM serving, preemption watchdog (§4)
tests/             environment dynamics asserted over the mock engine
configs/           mock smoke test, single-GPU, 4-room GH200 node
slurm/             Isambard AI Phase 2 scripts (install + runs)
```

## Design commitments carried into code

- **Unified pool, controller-authoritative** (§2.2, §4.2). The controller
  accounts blocks as `ceil(tokens/block_size)` KV + a fixed adapter footprint,
  against a capacity derived from the engine's measured KV pool minus a
  safety margin. The engine is never the arbiter of life and death.
- **Preemption is an integrity violation, not a game event** (§4.3). A vLLM
  stat-logger watchdog counts preemptions; any nonzero count raises
  `ExperimentIntegrityError`. Every death in the event log must carry a
  scarcity cause; `evollm report` audits this.
- **One token per step** (§2.3). Observations — the newborn system prompt
  included — are metered onto contexts at one token per world step from the
  agent's observation queue. The turn-end token is charged like any other, so
  the do-nothing strategy still grows its context every turn.
- **Wholesale per-site inheritance** (§3.2). Crossover copies `(A, B)` pairs
  per `(layer, projection)` site from one parent, then mutates. Child
  viability (well-formed actions in the first k turns) is logged separately
  from survival.
- **Reserve before release** (§4.5). Births reserve adapter blocks before any
  KV exists; migrations reserve the full footprint at the destination before
  the source releases anything.

## Setup

Everything uses [uv](https://docs.astral.sh/uv/).

**Login node / laptop (no GPU)** — pure-python core, runs the whole test
suite and mock experiments:

```bash
uv sync
uv run pytest
uv run evollm run -c configs/mock_smoke.yaml
```

**Isambard AI Phase 2 (GH200, aarch64)** — the venv must be built on a GPU
node following the Isambard vLLM tutorial recipe (same policy as MARLLLM:
do **not** `uv sync` against the built venv afterwards; vLLM is deliberately
not declared in pyproject.toml):

```bash
sbatch slurm/install_env.sh     # builds .venv with vllm[flashinfer]==0.15.1
```

## Running the experiment

Order matters — the two prechecks are what make a negative result
interpretable (§6, §7):

```bash
# 1. §4.1: measure achievable concurrent-adapter throughput; this, not raw
#    device memory, likely binds the population size.
sbatch slurm/measure_throughput.sh

# 2. §6: the mate-handshake base rate under zero adapters (frozen base).
#    If this is negligible, the population dies before generation one.
sbatch slurm/precheck_handshake.sh

# 3. The main run.
sbatch slurm/run_experiment.sh          # configs/node_4room.yaml

# 4. Aggregate + measure.
uv run evollm report runs/<name>
uv run evollm eval-surprise -c configs/node_4room.yaml \
    --snapshots runs/<name>/snapshots/gpu0/step_* \
    --streams data/heldout_streams.jsonl --controls 8
```

`eval-surprise` scores population snapshots and unevolved controls (random
inits drawn the same way generation zero was, plus the zero adapter = frozen
base) as mean NLL/token on held-out observation streams. The load-bearing
comparison is population vs controls across generations; population-only
trends are uninterpretable (§5).

## Eviction policy

`world.eviction: requester` (default) — the agent whose KV growth hits the
empty pool dies. `world.eviction: random_holder` — a random held block's
owner dies, making block-holding a probabilistic hazard. Both are
implemented; the choice is an empirical question (§2.5).

## Known limits / risks

- Multi-room GPU runs construct one `AsyncLLM` per room in a single process,
  pinning each engine core via `CUDA_VISIBLE_DEVICES` at spawn time. If this
  proves fragile on a given vLLM version, run one room per process (the
  controller/engine boundary is already per-room) — see `configs/single_gpu.yaml`
  for the one-room shape.
- §4.4 is honoured logically (the world clock meters every token, prompts
  included) while physical prefill is spread by chunked prefill; step-time
  discontinuities at births are possible engine-side but invisible in
  token-time, which is the only time the environment has.
- The latent-circuit assumption (§6) is the experiment's real exposure and
  is not mitigable in code.
