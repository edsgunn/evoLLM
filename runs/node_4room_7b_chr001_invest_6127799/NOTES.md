# node_4room_7b_chr001_invest_6127799

*eviction policy*

| | |
|---|---|
| Slurm job | `6127799` |
| Ended | 2026-08-26 · TIMEOUT · 12:00:14 |
| Steps reached | gpu0 81,182 · gpu1 72,195 · gpu2 73,861 · gpu3 107,808 (of 400,000) |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48000 blocks, seeded 32/room |
| Tools · read policy | `tell, mate, go` · `drain` |
| Genome | σ=0.001, chromosomal c=3, additive, 4 modules |
| Economy | eviction `random_holder`, refill False, parental investment True |
| **Differs from** | **chr001_evict** — `parental_investment: true` — each parent carries half its child's adapter blocks for as long as the parent lives. |

## What it was for

Effective population size is destroyed by the variance in offspring number, not
by the census, so the target was V_k. Reproduction was free to the parents: the
child's adapter came out of the room pool and neither parent paid. The charge is
deliberately not a tax — the blocks charged are the child's own adapter, already
allocated and already in use, so total room occupancy is unchanged and only the
owner differs.

## Headline

**It works mechanically and does not earn its place.** V_k is essentially
identical to eviction alone, but effective population size is *worse*, because
the charge suppressed reproduction rather than concentrating it: a third fewer
births and a fifth fewer agents per room. It does produce the cleanest output of
any run.

## Core metrics

Theory for each in `../../src/evollm/analysis/metrics/`. Comparison column is **chr001_evict**.

| metric | this run | chr001_evict |
|---|---|---|
| births | 15,188 | 23,836 |
| deaths | 14,897 | 23,524 |
| max generation | 182 | 223 |
| births per 1,000 room-steps | 45.3 | 78.0 |
| agents per room (mean) | 56.2 | 71.9 |
| agents per room (final) | 73.8 | 78.2 |
| mean context (mean) | 14,309 | 11,317 |
| mean context (final) | 10,298 | 9,632 |
| **V_k** (offspring variance) | 29.0 | 29.2 |
| **Ne** | 7.2 | 9.2 |
| **1/(2Ne)** drift threshold | 0.0696 | 0.0546 |
| **context at death ÷ room mean** | 1.38 | 1.40 |
| canonical action rate | 97.4% | 97.7% |
| effective lineages | 1.09 | 1.02 |
| largest family | 99.5% | 99.5% |
| h² (midparent) | 0.138 | 0.014 |
| h² midparent/single ratio | 1.15 | 0.64 |
| parent→child strategy concordance (excess) | -0.093 | 0.020 |
| drift from base ‖ΔW‖ vs gen 0 | 1.22× | 1.24× |
| diversity between agents (relative) | 0.016 | 0.019 |
| placeholder share of move attempts | 1.4% | 6.5% |
| **within-lifetime change** (last fifth − first fifth) | -4.19 pp | -1.04 pp |

## What this settles

- **Charging parents for their children's adapters adds nothing beyond the eviction fix.** — *likely*
- **A reproduction cost can suppress rather than concentrate reproduction**, which lowers Ne through the census term even when V_k is unchanged. — *likely*
- **Fewer, better agents is a real trade-off in this world**: this arm has the strongest improvement in formation traits and the weakest in reproduction. — *provisional*

## What it does not settle

Whether a larger charge would work better or worse. 11 blocks per child against
a ~718-block context budget is a nudge; the effect on Ne was negative, so a
bigger charge would plausibly be more negative, but that is an extrapolation
from one point. The MLP genome raises the per-child adapter cost fourfold on its
own, which tests a related question by accident.

## Open anomaly

Parent→child strategy concordance is **negative and strongly significant here**
(excess −0.093, z = −21.4, n = 13,039): children resemble their parents *less*
than matched same-room, same-birth-band strangers. Every other run is positive.
The control is stratified on room and birth-step band, so this is not drift or
composition, and n rules out small-sample noise.

No mechanism established. The plausible one is that the charge couples parent and
child *economically*: a parent holding its children's adapter blocks is a large
holder, so under `random_holder` it is disproportionately likely to be evicted —
and it is the maters who accumulate dependents. That would select against exactly
the strategy being transmitted, and would make the anticorrelation an artefact of
the charge rather than a fact about inheritance. Untested. Anyone rerunning a
reproduction charge should measure this before interpreting anything else.

## Truncation and caveats

Ran the full 12 hours. Reached a shallower generation than its comparison because
it produced fewer children, not because it stopped early.

Agents alive at the end have no death record and are excluded: 291 of 15,188 births.

## Status

Closed. The reproduction charge is not the lever for effective population size; eviction is.

---
*Numbers recomputed from this run's own `events/*.jsonl` and `genomes/*.jsonl`; machine output in `ANALYSIS.txt`, per-agent table in `traits.csv`.*
