# node_4room_7b_free60_6070756

*Refill-fraction sweep (job family 6070xxx) — block-triggered immigration, uniform 48,000-block rooms*

| | |
|---|---|
| Slurm job | `6070756` |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48,000 blocks, seeded 32/room |
| Tools | `tell, mate, go` |
| Read policy | `drain` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 400,000 of 400,000 |

**Measured:** 277 children · 44 refills ·
86.3% self-sufficient · max generation 7 ·
41.5% canonical turns · median lifetime 2,036 steps ·
median context at death 23,376 tokens · 35.8 generated tokens/turn

## What it was for

First arm of the **refill-fraction sweep**. This family made two changes at once:
the refill trigger moved from head-count to free blocks, and rooms grew from
4,000/8,000 blocks to a uniform 48,000 — because the GPUs are identical and a
4,000-block room left ~90% of the card idle.

This arm refills whenever a room is more than **60% free**, the most permissive
setting and therefore the fewest immigrants.

## What we learned

44 refills against 277 births — 86.3% self-sufficiency,
the block trigger working as intended in that it fired far less than the old
head-count floor. But maximum generation 7 and 41.5% canonical:
the population still degraded.

Read together with its siblings, this arm is part of the finding that **the refill
threshold does not determine population fate.** Across free60/free40/free20 the
threshold moved refills 44 → 370 and births 277 → 1,615, while maximum generation
moved only 7 → 10 and every arm degraded.

## Status

Superseded. Refill is disabled in all current configs.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
