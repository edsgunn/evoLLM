# node_4room_7b_chr001_evict_6127798

*eviction policy*

| | |
|---|---|
| Slurm job | `6127798` |
| Ended | 2026-08-26 · TIMEOUT · 12:00:14 |
| Steps reached | gpu0 73,513 · gpu1 72,275 · gpu2 72,481 · gpu3 87,278 (of 400,000) |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48000 blocks, seeded 32/room |
| Tools · read policy | `tell, mate, go` · `drain` |
| Genome | σ=0.001, chromosomal c=3, additive, 4 modules |
| Economy | eviction `random_holder`, refill False, parental investment False |
| **Differs from** | **chr001** — `eviction: random_holder` against `requester`. One word. |

## What it was for

Under `requester`, an agent dies when it asks for a block and the pool is empty
— and it asks once per block_size tokens of throughput, whatever it already
holds. So hazard tracked activity rather than size, the typical dying agent was
*smaller* than average, and bloat was an externality: a huge context raised
everyone's hazard but the holder bore only a fraction of the harm. This run tests
drawing the victim in proportion to blocks held.

## Headline

**The block economy's perverse incentive, fixed.** The dying agent goes from a
third smaller than the room mean to about 40% larger, and everything downstream
improves: more agents per room, smaller contexts, higher canonical rate, lower
offspring variance and a substantially higher effective population size. This is
now the reference configuration.

## Core metrics

Theory for each in `../../src/evollm/analysis/metrics/`. Comparison column is **chr001**.

| metric | this run | chr001 |
|---|---|---|
| births | 23,836 | 27,240 |
| deaths | 23,524 | 26,951 |
| max generation | 223 | 223 |
| births per 1,000 room-steps | 78.0 | 82.7 |
| agents per room (mean) | 71.9 | 60.9 |
| agents per room (final) | 78.2 | 70.8 |
| mean context (mean) | 11,317 | 12,504 |
| mean context (final) | 9,632 | 10,600 |
| **V_k** (offspring variance) | 29.2 | 40.2 |
| **Ne** | 9.2 | 5.7 |
| **1/(2Ne)** drift threshold | 0.0546 | 0.0873 |
| **context at death ÷ room mean** | 1.40 | 0.63 |
| canonical action rate | 97.7% | 94.8% |
| effective lineages | 1.02 | 1.04 |
| largest family | 99.5% | 99.4% |
| h² (midparent) | 0.014 | 0.059 |
| h² midparent/single ratio | 0.64 | 51.98 |
| parent→child strategy concordance (excess) | 0.020 | 0.008 |
| drift from base ‖ΔW‖ vs gen 0 | 1.24× | 1.21× |
| diversity between agents (relative) | 0.019 | 0.018 |
| placeholder share of move attempts | 6.5% | 0.9% |
| **within-lifetime change** (last fifth − first fifth) | -1.04 pp | 0.50 pp |

## What this settles

- **Hazard must rise with holdings, not throughput.** Making it so inverts the size-death relationship and improves the population downstream. — *likely*
- **Bloat was an accounting artefact, not an inevitability.** — *likely*
- **Raising effective population size is achievable without touching σ.** — *likely*
- **Death being content-blind does not stop the fix working.** Canonical rate rose rather than fell, so the policy is not destroying competent agents faster than it clears bloat. — *likely*

## What it does not settle

Whether it rescues a regime that actually needed it. σ = 0.001 was already
healthy, so this is a conservative test — `chr0025_evict` (6141384) asks the
question where bloat was fatal. Births per room-step are marginally *lower* than
the reference over the whole run while being higher at matched depth, so it is
not a clean win on every axis. Heritability of canonical rate is very low here,
but the trait sits near its ceiling so phenotypic variance is compressed —
that is a completed sweep, not absent transmission.

## Truncation and caveats

Ran the full 12 hours; rooms ended at different depths. Agents alive at the cutoff
have no death record and are excluded, which biases the final generations.

Agents alive at the end have no death record and are excluded: 312 of 23,836 births.

## Status

**The reference configuration.** Read new arms against this one.

---
*Numbers recomputed from this run's own `events/*.jsonl` and `genomes/*.jsonl`; machine output in `ANALYSIS.txt`, per-agent table in `traits.csv`.*
