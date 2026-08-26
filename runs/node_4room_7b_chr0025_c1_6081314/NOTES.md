# node_4room_7b_chr0025_c1_6081314

*σ sweep on chromosomal crossover*

| | |
|---|---|
| Slurm job | `6081314` |
| Ended | 2026-08-21 · FAILED (preemption guard) · 06:47:26 |
| Steps reached | gpu0 35,398 · gpu1 32,413 · gpu2 38,869 · gpu3 59,691 (of 400,000) |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48000 blocks, seeded 32/room |
| Tools · read policy | `tell, mate, go` · `drain` |
| Genome | σ=0.0025, chromosomal c=1, additive, 4 modules |
| Economy | eviction `requester`, refill False, parental investment False |
| **Differs from** | **chr0025** — one chromosome instead of three — the tightest possible linkage, ~1 parent switch per child against ~3.8. |

## What it was for

The linkage dial. Making chromosomal crossover the default raised a question
uniform never posed: how many chromosomes? Three was a guess modelled on
early/middle/late layers with nothing behind it. One crossover point across the
whole genome is the tightest linkage available.

## Headline

**One chromosome is too few.** It loses genetic diversity faster and further
than any other surviving arm — and in absolute terms, not merely relative to a
growing norm. With a single crossover point the genome travels as one unit, so
recombination cannot break up a sweeping variant.

## Core metrics

Theory for each in `../../src/evollm/analysis/metrics/`. Comparison column is **chr0025**.

| metric | this run | chr0025 |
|---|---|---|
| births | 17,982 | 24,325 |
| deaths | 17,581 | 24,072 |
| max generation | 161 | 207 |
| births per 1,000 room-steps | 108.1 | 43.4 |
| agents per room (mean) | 68.5 | 36.4 |
| agents per room (final) | 98.8 | 59.0 |
| mean context (mean) | 12,262 | 93,560 |
| mean context (final) | 7,409 | 85,818 |
| **V_k** (offspring variance) | 51.7 | 61.1 |
| **Ne** | 5.1 | 2.3 |
| **1/(2Ne)** drift threshold | 0.0986 | 0.2199 |
| **context at death ÷ room mean** | 0.64 | 0.64 |
| canonical action rate | 93.5% | 94.0% |
| effective lineages | 1.10 | 1.06 |
| largest family | 99.1% | 99.4% |
| h² (midparent) | 0.117 | 0.256 |
| h² midparent/single ratio | 2.39 | 1.61 |
| parent→child strategy concordance (excess) | 0.080 | 0.041 |
| drift from base ‖ΔW‖ vs gen 0 | 1.82× | 2.13× |
| diversity between agents (relative) | 0.065 | 0.063 |
| placeholder share of move attempts | 17.2% | 32.5% |
| **within-lifetime change** (last fifth − first fifth) | -4.94 pp | -1.16 pp |

## What this settles

- **Tighter linkage is not monotonically better.** c = 1 underperforms c = 3 on births, generation depth and heritability. — *likely*
- **The linkage dial has an interior optimum**, or at least is not maximised at either end. — *provisional*

## What it does not settle

Where the optimum is. Only 1, 3 and 112 (uniform) have been tested, all at a σ
where the population dies anyway, so the dial has been explored in a regime that
confounds it. Counts of 7 and 14 at σ = 0.001 were proposed and never submitted;
that is the informative place for them.

## Truncation and caveats

Aborted by the preemption guard at step 59,691, well before its partner arms
collapsed — so unlike them it stopped while still healthy. That makes its
late-run behaviour unknown rather than bad.

Agents alive at the end have no death record and are excluded: 401 of 17,982 births.

## Status

Closed at this σ. The chromosome-count question should be re-asked at σ = 0.001.

---
*Numbers recomputed from this run's own `events/*.jsonl` and `genomes/*.jsonl`; machine output in `ANALYSIS.txt`, per-agent table in `traits.csv`.*
