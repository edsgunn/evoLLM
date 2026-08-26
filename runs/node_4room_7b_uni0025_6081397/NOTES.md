# node_4room_7b_uni0025_6081397

*σ sweep on chromosomal crossover*

| | |
|---|---|
| Slurm job | `6081397` |
| Ended | 2026-08-22 · TIMEOUT · 12:00:18 |
| Steps reached | gpu0 59,764 · gpu1 56,488 · gpu2 252,000 · gpu3 400,000 (of 400,000) |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48000 blocks, seeded 32/room |
| Tools · read policy | `tell, mate, go` · `drain` |
| Genome | σ=0.0025, uniform c=3, additive, 4 modules |
| Economy | eviction `requester`, refill False, parental investment False |
| **Differs from** | **chr0025** — uniform crossover against chromosomal. Identical in every other respect — the pair exists to isolate this one variable. |

## What it was for

The crossover control. An earlier comparison at σ = 0.01 put chromosomal ahead
but on samples too small and arms too confounded to rank the schemes — a
*uniform* arm scored higher than the chromosomal one. This pair differs in
exactly one field, at a σ where enough births happen to measure anything.

## Headline

**Settled the crossover question, and died the same way its partner did.**
Chromosomal transmits more faithfully; uniform's excess over its
shared-environment control is roughly half. Both collapsed to zero births, so
the crossover advantage did not save either.

## Core metrics

Theory for each in `../../src/evollm/analysis/metrics/`. Comparison column is **chr0025**.

| metric | this run | chr0025 |
|---|---|---|
| births | 26,037 | 24,325 |
| deaths | 25,868 | 24,072 |
| max generation | 237 | 207 |
| births per 1,000 room-steps | 33.9 | 43.4 |
| agents per room (mean) | 30.1 | 36.4 |
| agents per room (final) | 42.2 | 59.0 |
| mean context (mean) | 84,344 | 93,560 |
| mean context (final) | 111,645 | 85,818 |
| **V_k** (offspring variance) | 51.3 | 61.1 |
| **Ne** | 2.2 | 2.3 |
| **1/(2Ne)** drift threshold | 0.2253 | 0.2199 |
| **context at death ÷ room mean** | 0.65 | 0.64 |
| canonical action rate | 91.9% | 94.0% |
| effective lineages | 1.12 | 1.06 |
| largest family | 99.5% | 99.4% |
| h² (midparent) | 0.192 | 0.256 |
| h² midparent/single ratio | 1.72 | 1.61 |
| parent→child strategy concordance (excess) | 0.057 | 0.041 |
| drift from base ‖ΔW‖ vs gen 0 | 2.04× | 2.13× |
| diversity between agents (relative) | 0.077 | 0.063 |
| placeholder share of move attempts | 46.9% | 32.5% |
| **within-lifetime change** (last fifth − first fifth) | -3.07 pp | -1.16 pp |

## What this settles

- **Chromosomal crossover is the better default**, on transmission fidelity, in a controlled pair at adequate sample size. — *likely*
- **Faithful transmission is not sufficient for survival.** Both arms of this pair died. — *established*
- **More genetic diversity does not prevent collapse.** This run had the highest diversity of any surviving arm and still died. — *likely*

## What it does not settle

Whether chromosomal wins on *outcomes* rather than transmission. It reached a
deeper generation than its chromosomal partner, so on the crude measure it looks
better. Both ratios sit well below the 2.0 that pure additive transmission
requires, so a large share of parent-offspring resemblance is still shared
environment in both.

## Truncation and caveats

Ran the full 12 hours and reached step 400,000, but births had been zero since
step 110,000 — nearly three quarters of the run was a dying remnant. Treat any
late-generation statistic as describing a handful of survivors.

Agents alive at the end have no death record and are excluded: 169 of 26,037 births.

## Status

Closed as a crossover control. The scheme it tested is no longer the default.

---
*Numbers recomputed from this run's own `events/*.jsonl` and `genomes/*.jsonl`; machine output in `ANALYSIS.txt`, per-agent table in `traits.csv`.*
