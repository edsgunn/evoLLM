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
- **Scarcity binds before the context ceiling** (§4.3). A room's capacity is
  rationed explicitly, *not* taken from the engine's profiled KV pool, and
  `World` enforces
  `(capacity_blocks - adapter_blocks) * block_size < max_model_len`. The
  engine's unused surplus is what guarantees it can never preempt; a capacity
  larger than that surplus is rejected.
- **The context window is not part of the experiment.** `max_model_len: auto`
  derives the ceiling from the largest room — one token past what any single
  agent could ever hold — so pool exhaustion is always what kills. Agents then
  live far beyond the base model's trained window (127k vs 32k in the shipped
  configs) and degrade there, which is intended: degradation is a selection
  pressure like any other, and nothing in the environment refers to it.
- **One action token per step; observations arrive by the utterance** (§2.3,
  `world.observation_absorption`). Generation is metered at one token per
  step. A queued utterance, by contrast, enters context whole in a single
  step, so being spoken to costs blocks in proportion to what was said.
  The turn-end token is charged like any other, so the do-nothing strategy
  still grows its context every turn.

  This is a deliberate departure from §2.3's literal one-token-per-step rule,
  forced by measurement. Under that rule every agent's context grew at exactly
  one token per step *regardless of behaviour* (`context == age`, exactly, for
  every agent in every GPU run). Speech therefore cost listeners nothing —
  contradicting §2.4's "speech consumes blocks in every listener" and §2.6's
  "an agent in a busy room ... dies sooner" — and lifetime was independent of
  what an agent did, leaving selection with almost no signal. Set
  `observation_absorption: token` to reproduce the original behaviour; a mock
  A/B at GPU proportions reaches 6 generations under `token` and 27 under
  `utterance`. The world clock stays uniform (every living agent advances
  exactly one step) and nothing refers to wall-clock time; the cost is that a
  step's engine work is no longer uniform, which §4.4 flagged — it varies
  wall-clock throughput, not world time.
- **Co-resident perception** (§2.4). An agent is always told who shares its
  room: in its system prompt, on arrival, when someone arrives or leaves, and
  as the return value of a failed `<mate>` — mirroring what a failed `<go>`
  already returns. `<mate>` requires a directed request to a co-located agent,
  which is impossible if co-residents cannot be perceived; without this, 72%
  of mate requests in the first GPU run were addressed to agents that were not
  there. The roster says who is present, never what to do about it, and every
  such observation costs the listener blocks like any other.
- **Refill, takeoff and checkpoints** (§6, §7). Measured reproduction rates sit
  far below replacement, so a room runs down to extinction — §7's
  "uninformative" outcome — before any lineage establishes. `refill` admits
  immigrants perturbed from the base model whenever a room drops below its
  floor, and the population random-walks until some lineage reproduces faster
  than it dies, after which refill stops firing by itself. It scores nobody and
  inspects no behaviour; it counts heads. What it changes is the question, from
  "can 16 founders bootstrap" to "given continuous variation, does a
  self-sustaining lineage emerge" — and the answer is read off
  `self-sufficiency = births / (births + refills)` in the report, rising toward
  1 for takeoff and flat for a population carried by immigration.

  A room that goes `takeoff_window_steps` needing no immigrant while still
  producing children is judged self-sustaining, and its population is
  **checkpointed immediately** — finding such a population is the expensive
  part of a run, and no later run should have to repeat the search:

  ```bash
  evollm run -c configs/single_gpu_tell.yaml \
      --seed-from runs/<name>/checkpoints/<room>/takeoff_00003000
  ```

  If it later needs immigrants again the lapse is logged as `takeoff_lost`, so
  a flickering takeoff is visible rather than being reported as durable.
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
suite and mock experiments. Use `./dev.sh`, never bare `uv run`:

```bash
./dev.sh pytest
./dev.sh evollm run -c configs/mock_smoke.yaml
```

> **Do not run `uv sync` or `uv run` in this directory.** uv's default project
> environment is `./.venv`, which here is the *GPU* venv. Both commands sync
> it against `pyproject.toml`, silently re-resolving dependencies inside the
> carefully-built vLLM stack — this is how numpy once reached 2.5, breaking
> numba and with it every engine start, with nothing failing until a GPU job
> ran. `dev.sh` sets `UV_PROJECT_ENVIRONMENT=.venv-dev` so uv touches a
> separate environment. To change the GPU venv deliberately, use
> `uv pip install --python .venv/bin/python <pkg>`, which modifies only what
> you name.

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

- Room size is now bounded by throughput, not by the context window. Attention
  cost per token grows with context length, so an agent late in a large room
  is much slower per token than a newborn. This costs GPU-hours but does not
  distort the physics: all world timing is in tokens (§2.3) and nothing refers
  to wall-clock. Raise `capacity_blocks` for longer lives and slower runs;
  the ceiling follows automatically.
- A room is still far smaller than a GPU (8000 of ~180k blocks), so §2.1's
  "rooms are GPUs" leaves most of each device idle. Several rooms per GPU
  sharing one engine is the natural fix — cheap to add (adapter ids are
  already globally unique; capacity is already per-room) but a deviation from
  §2.1 worth deciding deliberately.
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
