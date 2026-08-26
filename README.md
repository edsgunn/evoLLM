# evoLLM

Populations of LoRA agents evolving in a shared world, as a third
pre-training-like stage after pre-training and RL. No gradients, no reward
function, no task — agents live in a block economy whose only currency is
device memory, and whatever gets more children into the world propagates.

**Read [`VISION.md`](VISION.md) first if you want to know why any of this is
worth doing.** This document is the guide to the repository: where things are,
how to run them, and how the research process works.

---

## How things work around here

Four documents, each with one job. Keeping them separate is what stops the
project going stale, so it is worth knowing which is which before you edit any
of them.

| document | answers | changes |
|---|---|---|
| [`VISION.md`](VISION.md) | **Why.** What we are trying to produce, the questions Q1–Q8, the phase plan and its exit criteria. | rarely |
| [`STATE.md`](STATE.md) | **What we believe now.** Every finding, graded by confidence, with a citation. What is running. What we are blind to. | every run |
| [`PROCESS.md`](PROCESS.md) | **How we work.** The loop below, in detail, with the checklists. | rarely |
| `README.md` | **Where things are.** This file. | with the code |

The rule that keeps them from collapsing into each other: **no beliefs in
`VISION.md`, no vision in `STATE.md`.** A finding goes in `STATE.md` with a
pointer to the run that established it; the reason anyone cares goes in
`VISION.md`.

### The research loop

Findings flow one way. Nothing is ever written into a downstream document from
memory — only from the document upstream of it.

```
configs/<run>.yaml        why this run exists, written BEFORE it is submitted
        ↓  sbatch
runs/<run>/               events, genomes, traces — the raw record
        ↓  analyse
runs/<run>/NOTES.md       what this run means. The only place numbers and
                          argument live together.
        ↓
runs/TIMELINE.md          chronology: what we ran, when, and what came of it
        ↓
STATE.md                  what the project now believes
```

Two habits do most of the work:

**Every config carries its own justification.** Before a run is submitted, its
config header says what question it serves, what it differs from, and what we
expect to learn — including the numbers we expect to move. A run whose header
cannot be written is a run that should not be submitted. Finished configs get a
`STATUS:` line so nobody resubmits them by accident.

**Every run gets a `NOTES.md`.** Written from `runs/_TEMPLATE_NOTES.md`, with
the full core-metrics table against a named comparison run. Runs are the
scarcest resource here — twelve GPU-hours and a queue measured in days — and a
run whose interpretation exists only in a conversation has been half wasted.

`PROCESS.md` has the full checklists, including how to retract a finding when a
later run overturns it.

---

## Layout

```
VISION.md  STATE.md  PROCESS.md      the three documents above

src/evollm/
  config.py        YAML experiment config; every knob, with why it exists
  world.py         room graph, migration with reserve-before-release
  controller.py    the world clock: one token per agent per step, births,
                   deaths, movement, the block accounting
  blocks.py        the block economy: one pool per room, KV + adapters,
                   dependent holdings for parental investment
  agent.py         per-agent soma: context, queues, mode, token origins
  genome.py        LoRA genome; chromosomal/uniform crossover, mutation,
                   cheap per-site fingerprints
  actions.py       <say> <tell> <mate> <accept> <go> grammar and its variants
  prompts.py       system prompt and observation formatting
  events.py        JSONL event log and the death-cause audit
  report.py        offline aggregation from the event log
  evaluate.py      held-out surprise for population snapshots vs controls
  plots.py         lineage, occupancy and block figures
  engines/
    base.py        the engine interface: turns, tokens, logprobs, capacity
    mock.py        scripted agents, toy tokenizer — the whole world, no GPU
    vllm_engine.py multi-LoRA vLLM serving
  analysis/        see "Analysing a run" below
    metrics/       what each metric means, and which vision question it serves

tests/             environment dynamics asserted over the mock engine
configs/           one YAML per experiment arm, each with its rationale
runs/              one directory per run: events, genomes, NOTES.md
  TIMELINE.md      day by day, run by run
  _TEMPLATE_NOTES.md
archive/           superseded runs, each with a README saying why
slurm/             Isambard AI scripts (install, prechecks, runs)
```

`runs/` and `archive/` hold gigabytes of event logs and are not tracked — but
their **markdown is**, deliberately. The write-ups are the project's memory.

---

## Setup

Everything uses [uv](https://docs.astral.sh/uv/).

### Login node or laptop — no GPU needed

The core is pure Python and the mock engine implements the entire world, so the
full test suite and complete mock experiments run anywhere. Use `./dev.sh`:

```bash
./dev.sh pytest -q
./dev.sh evollm run -c configs/mock_smoke.yaml
```

> ### Never run `uv sync` or `uv run` in this directory
>
> uv's default project environment is `./.venv`, which here is the **GPU**
> venv. Both commands sync it against `pyproject.toml`, silently re-resolving
> dependencies inside a carefully built vLLM stack. That is how numpy once
> reached 2.5, taking numba and every engine start with it — with nothing
> failing until a GPU job ran hours later.
>
> `dev.sh` sets `UV_PROJECT_ENVIRONMENT=.venv-dev` so uv touches a separate
> environment. To change the GPU venv deliberately, name what you want:
> `uv pip install --python .venv/bin/python <pkg>`.

### Isambard AI Phase 2 (GH200, aarch64)

The GPU venv must be built on a GPU node following the Isambard vLLM recipe.
vLLM is deliberately **not** declared in `pyproject.toml`, which is what keeps
an accidental sync from reinstalling it:

```bash
sbatch slurm/install_env.sh     # builds .venv with vllm[flashinfer]
```

---

## Running an experiment

```bash
CONFIG=configs/node_4room_7b_chr001_evict.yaml sbatch slurm/run_experiment.sh
```

The run is named after the config plus the Slurm job id, so
`configs/node_4room_7b_mlp.yaml` submitted as job 6141167 writes to
`runs/node_4room_7b_mlp_6141167/`. That naming is what lets a directory be
traced back to the exact config and job that produced it.

Two prechecks exist and are worth running when the environment changes, because
they are what make a negative result interpretable:

```bash
sbatch slurm/measure_throughput.sh    # concurrent-adapter throughput
sbatch slurm/precheck_handshake.sh    # mate-handshake rate under zero adapters
```

The second is the base-rate control: it measures what the frozen base model
does with no genome at all. Without it, "the population improved" only ever
means "better than a randomly perturbed base model", which is a much weaker
claim than it sounds.

---

## Analysing a run

Four tools, in the order you would use them.

```bash
# 1. Did the run behave? Deaths, births, integrity audit.
./dev.sh evollm report runs/<run>

# 2. The core metrics table, against a comparison run. This is what goes in
#    NOTES.md — never compute a table row in a one-off script.
./dev.sh python -m evollm.analysis.core_metrics runs/<run> runs/<comparison>

# 3. Read what the agents actually wrote. Seconds, deterministic, no model.
./dev.sh evollm inspect-traces runs/<run> --out runs/<run>/TRACES.txt

# 4. Population analysis: lineages, strategies, genotype-phenotype association.
./dev.sh evollm analyse runs/<run> --out runs/<run>/ANALYSIS.txt \
    --traits-csv runs/<run>/traits.csv
```

`inspect-traces` deserves its place in the loop rather than at the end of it.
Every other metric counts turns against categories chosen in advance, so a
well-formed action that always fails scores as healthy behaviour. Both of the
worst bugs found so far had exactly that shape. Where a question genuinely
needs a reader, `--bundle` writes a stratified sample of turns out for one; it
sends nothing anywhere.

The `analysis` package is also usable piecewise when the suite does not ask
your question:

```python
from evollm.analysis import Pedigree, build_phenotypes, surprise_curve

ped = Pedigree.from_run(run)
pheno = build_phenotypes(run, pedigree=ped)      # 28 traits per agent
surprise_curve(run)                              # within-lifetime change
```

**`src/evollm/analysis/metrics/` is the reference for what every metric means**
— the maths, how to read it, how it has misled us before, and which of the
vision's questions it serves. Read it before quoting a number at anyone.

---

## Conventions and hard-won gotchas

Things that have cost us runs or results, in rough order of how expensive they
were.

**The reference configuration is `configs/node_4room_7b_chr001_evict.yaml`.**
New arms are read against it, and its `NOTES.md` carries the numbers to beat.

**The dataclass defaults in `config.py` are not the reference configuration.**
They are older, and diverge on at least eviction (`requester` vs
`random_holder`), tools, mutation rate and read policy. Always start from a
config file, never from `Config()`.

**Death must always be scarcity.** Every death in the event log carries a cause
and `evollm report` audits it. If an agent can die for any other reason, the
run's selection story is unattributable and the run is worthless.

**Preemption is survivable, aborting is not.** vLLM requeues a preempted
request with its generated tokens intact. Treating any preemption as an
integrity violation cost us four runs before we read the vLLM source. There is
now a budget (`engine.preemption_budget`).

**The pool must fit.** Capacity is derived from the engine's own measured KV
pool, minus model weights and a reserve. Overshooting is fatal at engine start,
not merely wasteful.

**Instrumentation must never be able to kill a run.** Surprise recording probes
its own GPU path at startup and disables *itself*, not the run, if it fails.
Anything added to the measurement layer should follow that pattern: losing a
metric costs an analysis, losing a run costs days of queue.

**Changing a default silently changes every queued job.** Jobs read their
config from disk when they start, and there may be four of them pending. New
behaviour goes behind a config flag whose default preserves what the queued
runs expect.

**Watch out for `pkill -f`** on this machine — the pattern can match the shell
running the command. Use bracket tricks or explicit PIDs.

**Around 250 `§` references in the code** point into an original proposal
document that is not in this repository and that encodes a superseded framing.
They are historical. Where they conflict with `VISION.md`, `VISION.md` wins.

---

## Where to start

- **Why does this project exist?** → [`VISION.md`](VISION.md)
- **What do we currently believe, and what is running?** → [`STATE.md`](STATE.md)
- **I want to add a run.** → [`PROCESS.md`](PROCESS.md), then write the config
  header before the config.
- **What does this metric mean?** → `src/evollm/analysis/metrics/README.md`
- **What happened and when?** → [`runs/TIMELINE.md`](runs/TIMELINE.md)
