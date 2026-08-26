# node_4room_tell_token_6045968

*Tool and absorption variants (job family 6045xxx) — first runs after the chat-format fix*

| | |
|---|---|
| Slurm job | `6045968` |
| Model | `Qwen2.5-1.5B-Instruct` |
| Rooms | 4 × 8,000 blocks, seeded 16/room |
| Tools | `tell, mate, go` |
| Read policy | `one` · absorption `token` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 2,000,000 of 2,000,000 |

**Measured:** 242 children · 6,610 refills ·
3.5% self-sufficient · max generation 5 ·
15.1% canonical turns · median lifetime 6,833 steps ·
median context at death 6,832 tokens · 52.1 generated tokens/turn

## What it was for

The **token absorption** arm: instead of an observation landing in context whole,
the agent reads it one token per world step (`observation_absorption: token`).
This makes listening cost exactly one token per step no matter what was said.

## What we learned

The mechanism works but has a serious side effect that only became clear later:
if every agent absorbs exactly one token per step regardless of what the room
said, then **being talked to is free and survival becomes independent of
behaviour**. That is the property `utterance` absorption was chosen to avoid.

Measured here: 15.1% canonical, 3.5% self-sufficiency, median
lifetime 6,833 steps — the longest in the family, which is the same effect
seen from the other side: slow absorption means slow death.

## Status

Superseded by `utterance` absorption, which is used in every live config.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
