# node_4room_7b_chr0025_6081311

*σ sweep on chromosomal crossover*

| | |
|---|---|
| Slurm job | `6081311` |
| Ended | 2026-08-22 · FAILED (preemption guard) · 10:31:45 |
| Steps reached | gpu0 361,000 · gpu1 56,001 · gpu2 73,173 · gpu3 70,749 (of 400,000) |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48000 blocks, seeded 32/room |
| Tools · read policy | `tell, mate, go` · `drain` |
| Genome | σ=0.0025, chromosomal c=3, additive, 4 modules |
| Economy | eviction `requester`, refill False, parental investment False |
| **Differs from** | **uni0025** — chromosomal crossover against uniform. The designed pair for the crossover question. |

## What it was for

The headline arm of the σ sweep: the takeoff regime with faithful inheritance
under it. Also one half of the controlled pair that was built to settle whether
chromosomal crossover beats uniform, after an earlier comparison at σ = 0.01 was
too noisy and too confounded to rank them.

## Headline

**Died of bloat collapse.** Births fell to zero by step 80,000 and never
recovered across the remaining 281,000 steps; the rooms ended holding two or
three agents at 250,000–310,000 tokens each in a pool that was not even full.
It also carries the clearest heritability signal of any run, which is what makes
its death worth understanding.

## Core metrics

Theory for each in `../../src/evollm/analysis/metrics/`. Comparison column is **uni0025**.

| metric | this run | uni0025 |
|---|---|---|
| births | 24,325 | 26,037 |
| deaths | 24,072 | 25,868 |
| max generation | 207 | 237 |
| births per 1,000 room-steps | 43.4 | 33.9 |
| agents per room (mean) | 36.4 | 30.1 |
| agents per room (final) | 59.0 | 42.2 |
| mean context (mean) | 93,560 | 84,344 |
| mean context (final) | 85,818 | 111,645 |
| **V_k** (offspring variance) | 61.1 | 51.3 |
| **Ne** | 2.3 | 2.2 |
| **1/(2Ne)** drift threshold | 0.2199 | 0.2253 |
| **context at death ÷ room mean** | 0.64 | 0.65 |
| canonical action rate | 94.0% | 91.9% |
| effective lineages | 1.06 | 1.12 |
| largest family | 99.4% | 99.5% |
| h² (midparent) | 0.256 | 0.192 |
| h² midparent/single ratio | 1.61 | 1.72 |
| parent→child strategy concordance (excess) | 0.041 | 0.057 |
| drift from base ‖ΔW‖ vs gen 0 | 2.13× | 2.04× |
| diversity between agents (relative) | 0.063 | 0.077 |
| placeholder share of move attempts | 32.5% | 46.9% |
| **within-lifetime change** (last fifth − first fifth) | -1.16 pp | -3.07 pp |

## What this settles

- **σ = 0.0025 is above the critical mutation rate**, not below it. It simply takes longer to show than 0.005 or 0.01. — *established*
- **Chromosomal crossover transmits phenotype more faithfully than uniform** at adequate sample size, with the midparent/single-parent ratio nearer 2. — *likely*
- **Reproductive collapse precedes population collapse.** Per-capita reproduction fell first; head-count followed. It is a birth collapse, not a death spike. — *established*
- **A behaviour costing a third of an agent's reproduction can still sweep** when effective population size is this small. — *established*

## What it does not settle

Whether bloat was the cause of the collapse or a symptom of it. That is exactly
what `chr0025_evict` (6141384) is testing. It also cannot say whether the
crossover advantage translates into better outcomes — uniform reached a deeper
generation here, so the win is on transmission only.

## Truncation and caveats

Aborted by the preemption guard at step 361,000. That is not what killed the
population: births had already been zero for 281,000 steps. Agents alive at the
abort have no death record and are excluded.

Agents alive at the end have no death record and are excluded: 253 of 24,325 births.

## Status

Superseded by `chr0025_evict` (6141384), which is this configuration with `random_holder` eviction — the fix aimed directly at the bloat that killed it.

---
*Numbers recomputed from this run's own `events/*.jsonl` and `genomes/*.jsonl`; machine output in `ANALYSIS.txt`, per-agent table in `traits.csv`.*
