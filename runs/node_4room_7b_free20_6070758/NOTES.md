# node_4room_7b_free20_6070758

*Refill-fraction sweep (job family 6070xxx) — block-triggered immigration, uniform 48,000-block rooms*

| | |
|---|---|
| Slurm job | `6070758` |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48,000 blocks, seeded 32/room |
| Tools | `tell, mate, go` |
| Read policy | `drain` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 400,000 of 400,000 |

**Measured:** 1,615 children · 370 refills ·
81.4% self-sufficient · max generation 10 ·
60.1% canonical turns · median lifetime 2,386 steps ·
median context at death 19,088 tokens · 26.2 generated tokens/turn

## What it was for

The most aggressive refill arm — top up whenever a room is more than **20% free**
— testing whether a denser population would give mating enough opportunity to
outrun mutation.

## What we learned

Density helps throughput and not fate. 370 refills produced 1,615
births, by far the most of the sweep, and 60.1% canonical — the best of
the σ=0.01 arms. Maximum generation 10.

But reproduction still fell across generations (0.82 → 0.64 → 0.31 children per
100 turns), so more mating opportunity did not compensate for drift. This is the
strongest single piece of evidence that the refill threshold was never the
operative variable.

## Status

Superseded. Refill is disabled in all current configs.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
