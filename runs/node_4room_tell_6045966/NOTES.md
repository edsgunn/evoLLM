# node_4room_tell_6045966

*Tool and absorption variants (job family 6045xxx) — first runs after the chat-format fix*

| | |
|---|---|
| Slurm job | `6045966` |
| Model | `Qwen2.5-1.5B-Instruct` |
| Rooms | 4 × 8,000 blocks, seeded 16/room |
| Tools | `tell, mate, go` |
| Read policy | `one` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 2,000,000 of 2,000,000 |

**Measured:** 237 children · 7,869 refills ·
2.9% self-sufficient · max generation 4 ·
10.2% canonical turns · median lifetime 5,097 steps ·
median context at death 6,528 tokens · 54.0 generated tokens/turn

## What it was for

The **directed speech** arm of the same family: `tell` replaces `say`, so an
utterance goes to one named agent instead of the whole room. The question was
whether addressing costs — having to name a target — change how a population
coordinates.

## What we learned

For a 1.5B model the answer is drowned by competence: 10.2% canonical
turns and 237 births against 7,869 refills (2.9%
self-sufficiency, the lowest of the family). Directed speech is strictly harder
than broadcast — the agent must produce a valid target id as well as a valid verb
— and at this scale that extra requirement is fatal.

Compared against its sibling `node_4room_6045965` (broadcast, 10.2% vs
19.1%) the cost of directedness is visible but both are far below usable.

## Status

Superseded. `tell` was kept for all later work, but only with the 7B model.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
