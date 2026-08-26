# node_4room_7b_norefill_6070759

*Refill-fraction sweep (job family 6070xxx) — block-triggered immigration, uniform 48,000-block rooms*

| | |
|---|---|
| Slurm job | `6070759` |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48,000 blocks, seeded 32/room |
| Tools | `tell, mate, go` |
| Read policy | `drain` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 400,000 of 400,000 |

**Measured:** 93 children · 0 refills ·
100.0% self-sufficient · max generation 3 ·
52.0% canonical turns · median lifetime 2,859 steps ·
median context at death 32,704 tokens · 29.3 generated tokens/turn

## What it was for

The control for the whole refill sweep: **immigration switched off entirely**.
After seeding, births are the only source of new agents, and if the world empties
the run stops.

## What we learned

93 children in 400,000 steps and maximum generation 3 — the
weakest population of any 7B run. Without immigration and without working
inheritance, there is nothing to hold a population together.

Two things this run established that mattered later:

- **Extinction is not the failure mode.** No room ever hit zero agents; minimum
  population was 1 per room and median 2, sustained over 400,000 steps. Bloated
  non-reproducers live a very long time.
- **Immigration was propping the population up, not crowding it out.** Immigrants
  outreproduced native-born agents by 2–12× in every σ=0.01 arm. This run is what
  that comparison is measured against.

The reassurance about extinction risk is why refill could later be turned off
across the board.

## Status

Superseded in form — refill is now off everywhere by default — but this is the run that made that safe.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
