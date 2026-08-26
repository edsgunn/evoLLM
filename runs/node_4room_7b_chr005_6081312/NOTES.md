# node_4room_7b_chr005_6081312

*σ sweep on chromosomal crossover*

| | |
|---|---|
| Slurm job | `6081312` |
| Ended | 2026-08-21 · COMPLETED · 05:06:43 |
| Steps reached | gpu0 400,000 · gpu1 400,000 · gpu2 400,000 · gpu3 400,000 (of 400,000) |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48000 blocks, seeded 32/room |
| Tools · read policy | `tell, mate, go` · `drain` |
| Genome | σ=0.005, chromosomal c=3, additive, 4 modules |
| Economy | eviction `requester`, refill False, parental investment False |
| **Differs from** | **chr0025** — σ = 0.005 against 0.0025. |

## What it was for

Upper bracket. Collapse was total at σ = 0.01 and takeoff looked clean at
0.0025, so the critical rate was thought to lie between them and 0.005 was the
midpoint — the likeliest place to observe the transition itself rather than one
of its two outcomes.

## Headline

**Comfortably above the threshold.** It produced roughly a tenth of the births
of its neighbours and reached generation 28 against 200+, with the highest
offspring variance and the lowest effective population size measured anywhere in
the project. There is no sign of a gradual transition — it simply fails.

## Core metrics

Theory for each in `../../src/evollm/analysis/metrics/`. Comparison column is **chr0025**.

| metric | this run | chr0025 |
|---|---|---|
| births | 2,456 | 24,325 |
| deaths | 2,450 | 24,072 |
| max generation | 28 | 207 |
| births per 1,000 room-steps | 1.5 | 43.4 |
| agents per room (mean) | 7.6 | 36.4 |
| agents per room (final) | 1.5 | 59.0 |
| mean context (mean) | 196,725 | 93,560 |
| mean context (final) | 390,807 | 85,818 |
| **V_k** (offspring variance) | 221.7 | 61.1 |
| **Ne** | 0.1 | 2.3 |
| **1/(2Ne)** drift threshold | 3.9321 | 0.2199 |
| **context at death ÷ room mean** | 0.68 | 0.64 |
| canonical action rate | 85.9% | 94.0% |
| effective lineages | 1.47 | 1.06 |
| largest family | 95.3% | 99.4% |
| h² (midparent) | 0.313 | 0.256 |
| h² midparent/single ratio | 5.44 | 1.61 |
| parent→child strategy concordance (excess) | -0.146 | 0.041 |
| drift from base ‖ΔW‖ vs gen 0 | 1.70× | 2.13× |
| diversity between agents (relative) | 0.114 | 0.063 |
| placeholder share of move attempts | 11.7% | 32.5% |
| **within-lifetime change** (last fifth − first fifth) | -1.44 pp | -1.16 pp |

## What this settles

- **σ = 0.005 is above the critical rate.** — *established*
- **The transition is sharp rather than gradual.** Nothing intermediate has been observed at any σ. — *likely*
- **Effective population size collapses along with everything else** at high mutation, so selection is weakest exactly where it would need to be strongest. — *likely*

## What it does not settle

Where the threshold actually is. This bracketed it to (0.0025, 0.005) at the
time; the later finding that 0.0025 also collapses moved the bracket down to
(0.001, 0.0025) and made this run's contribution purely confirmatory.

## Truncation and caveats

Completed its 400,000 steps in only five hours, which is itself the result: with
7.6 agents per room there was almost nothing to compute.

Agents alive at the end have no death record and are excluded: 6 of 2,456 births.

## Status

Closed. Confirms the upper bracket; superseded as the boundary estimate.

---
*Numbers recomputed from this run's own `events/*.jsonl` and `genomes/*.jsonl`; machine output in `ANALYSIS.txt`, per-agent table in `traits.csv`.*
