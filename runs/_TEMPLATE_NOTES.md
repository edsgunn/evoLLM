# <run_name>_<jobid>

*<which family / what this arm changes>*

> Copy this file to `runs/<run>/NOTES.md` and fill it in. Delete this quote
> block. Every section is required; write "not applicable" or "not measured"
> rather than dropping one, so gaps stay visible.
>
> Theory for each metric is in `src/evollm/analysis/metrics/`. Do not restate it
> here — link to it and give the number.

| | |
|---|---|
| Slurm job | `` |
| Ended | date · TIMEOUT / COMPLETED / FAILED · elapsed |
| Steps reached | of `run.max_world_steps`, **per room** |
| Model | |
| Rooms | count × capacity, seeded n/room |
| Tools · read policy | |
| Genome | σ, crossover, chromosomes, mutation, target modules |
| Economy | eviction, refill, parental investment |
| **Differs from** | the one run this should be read against, and in what |

## What it was for

One paragraph. What question, and why that question was worth a machine.

## Headline

Two or three sentences. What a reader should take away if they read nothing
else.

## Core metrics

Reported for every run, so runs stay comparable. See the metrics documents for
what each means and how it misleads.

| metric | value | vs the comparison run |
|---|---|---|
| births · deaths · max generation | | |
| births per 1,000 room-steps | | |
| agents per room (mean, final) | | |
| mean context (mean, final) | | |
| **V_k** (variance in offspring) | | |
| **Ne** and **1/(2Ne)** | | |
| **context at death ÷ room mean** | | |
| canonical action rate | | |
| effective lineages (inverse Simpson) | | |
| h² and midparent/single ratio | | |
| parent→child strategy concordance (excess, z) | | |
| drift from base ‖ΔW‖ vs gen 0 | | |
| diversity between agents (relative) | | |
| placeholder share of move attempts | | |
| **within-lifetime change** (last fifth − first fifth, paired) | | |
| **observation surprise, paired within-agent change** | | |
| surprise: starting level vs generation | | |
| stuck agents (same turn 80%+ of life) | | |
| communication share (`tell`+`say` of all actions) | | |

## Trends across generations

Partial correlation with generation, **room identity and room population held
fixed**, permuted within (room × population decile). Uncontrolled trends are not
acceptable here — see `src/evollm/analysis/metrics/behaviour.md`.

| trait | partial r | p |
|---|---|---|

## What this settles

Bullets. Each one states a fact and the confidence it can bear
(*established* / *likely* / *provisional* / *suspected*), matching `STATE.md`'s
key. **Only claims this run can support on its own.**

## What it does not settle

The confounds, the things that moved together, the questions it raised. Be
explicit about what a reader might wrongly conclude.

## Truncation and caveats

Did it finish? What is biased as a result? Agents alive at the cutoff have no
death record and are excluded — say how many.

## Status

Superseded by / still the reference for / closed. Where the next question goes.

---
*Numbers recomputed from this run's own `events/*.jsonl` via `evollm analyse`;
machine output in `ANALYSIS.txt`, per-agent table in `traits.csv`.*
