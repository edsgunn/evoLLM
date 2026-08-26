# node_4room_7b_chr001_6081313

*σ sweep on chromosomal crossover*

| | |
|---|---|
| Slurm job | `6081313` |
| Ended | 2026-08-22 · TIMEOUT · 12:00:04 |
| Steps reached | gpu0 79,926 · gpu1 79,963 · gpu2 77,260 · gpu3 92,121 (of 400,000) |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48000 blocks, seeded 32/room |
| Tools · read policy | `tell, mate, go` · `drain` |
| Genome | σ=0.001, chromosomal c=3, additive, 4 modules |
| Economy | eviction `requester`, refill False, parental investment False |
| **Differs from** | **chr0025** — σ = 0.001 against 0.0025. Nothing else. |

## What it was for

Lower bracket of the σ sweep. The question was whether less mutation is
monotonically better below the threshold, or whether there is a floor past which
the population starves for variation — mutation is the only source of new
variation here and selection erodes what it feeds on.

## Headline

**The first run to sustain a takeoff to the end of its clock.** It improved on
reproduction, formation and move success simultaneously, and was still
accelerating when the 12-hour limit stopped it. It is also the run with the
*least* genetic diversity and the *weakest* parent-to-child transmission of the
five — which is the tension the project has not resolved.

## Core metrics

Theory for each in `../../src/evollm/analysis/metrics/`. Comparison column is **chr0025**.

| metric | this run | chr0025 |
|---|---|---|
| births | 27,240 | 24,325 |
| deaths | 26,951 | 24,072 |
| max generation | 223 | 207 |
| births per 1,000 room-steps | 82.7 | 43.4 |
| agents per room (mean) | 60.9 | 36.4 |
| agents per room (final) | 70.8 | 59.0 |
| mean context (mean) | 12,504 | 93,560 |
| mean context (final) | 10,600 | 85,818 |
| **V_k** (offspring variance) | 40.2 | 61.1 |
| **Ne** | 5.7 | 2.3 |
| **1/(2Ne)** drift threshold | 0.0873 | 0.2199 |
| **context at death ÷ room mean** | 0.63 | 0.64 |
| canonical action rate | 94.8% | 94.0% |
| effective lineages | 1.04 | 1.06 |
| largest family | 99.4% | 99.4% |
| h² (midparent) | 0.059 | 0.256 |
| h² midparent/single ratio | 51.98 | 1.61 |
| parent→child strategy concordance (excess) | 0.008 | 0.041 |
| drift from base ‖ΔW‖ vs gen 0 | 1.21× | 2.13× |
| diversity between agents (relative) | 0.018 | 0.063 |
| placeholder share of move attempts | 0.9% | 32.5% |
| **within-lifetime change** (last fifth − first fifth) | 0.50 pp | -1.16 pp |

## What this settles

- **σ = 0.001 sustains a population where 0.0025 does not.** — *likely*
- **A population can improve on reproduction, formation and efficacy at once.** Every other arm traded one for another. — *established*
- **The degenerate placeholder action is mutation-rate dependent.** It stays near zero here across every generation band while reaching a majority of move attempts at higher σ. — *likely*
- **Less mutation does not obviously starve variation at this rate.** No sign of the predicted floor. — *provisional*

## What it does not settle

Whether 0.001 is optimal or merely the least-bad point on a curve nobody has
sampled below. Nothing under 0.001 has ever run, so the floor could sit just
below. It also cannot separate σ's effect on health from its effect on
diversity: this run has the least of both problems and the least of the raw
material selection needs.

## Truncation and caveats

Stopped by the 12-hour limit, not by anything internal. Rooms ended at very
different depths (see per-room table) because they desynchronise; an apparent
birth-rate collapse at step 80,000 is three of four rooms simply ending, not a
population failure. The one room still running produced its highest birth count
in that window.

Agents alive at the end have no death record and are excluded: 289 of 27,240 births.

## Status

Superseded as the reference by `chr001_evict` (6127798), which is this configuration with `random_holder` eviction. Still the reference for what σ = 0.001 does under the original eviction rule.

---
*Numbers recomputed from this run's own `events/*.jsonl` and `genomes/*.jsonl`; machine output in `ANALYSIS.txt`, per-agent table in `traits.csv`.*
