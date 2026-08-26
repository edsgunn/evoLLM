# node_4room_7b_free40_6070757

*Refill-fraction sweep (job family 6070xxx) — block-triggered immigration, uniform 48,000-block rooms*

| | |
|---|---|
| Slurm job | `6070757` |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48,000 blocks, seeded 32/room |
| Tools | `tell, mate, go` |
| Read policy | `drain` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 400,000 of 400,000 |

**Measured:** 372 children · 122 refills ·
75.3% self-sufficient · max generation 7 ·
43.6% canonical turns · median lifetime 2,243 steps ·
median context at death 22,528 tokens · 28.8 generated tokens/turn

## What it was for

The middle arm of the refill sweep — refill when a room is more than **40% free**
— and the config that became the base for every later experiment. `chromo`,
`lowmut` and `multmut` are all this file with one genome field changed.

## What we learned

75.3% self-sufficiency, maximum generation 7, 43.6%
canonical. As the shared control for the genome sweep it is the reference point
for every degradation number quoted elsewhere: reproduction fell
0.77 → 0.11 → 0.04 children per 100 turns across generations 0, 1–2 and 3–10.

Its rooms also show the **bloat collapse** most clearly. At step 222,000 room
gpu0 held three agents at 255,000 tokens each and 99.8% occupancy — incoherent
agents that never finish an action, accumulating context until the pool is theirs
alone and nothing can be born into it.

## Status

Superseded as a live config, but the baseline against which chromo/lowmut/multmut are read.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
