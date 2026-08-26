# node_4room_tell_token_7b_6045970

*Tool and absorption variants (job family 6045xxx) — first runs after the chat-format fix*

| | |
|---|---|
| Slurm job | `6045970` |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 8,000 blocks, seeded 16/room |
| Tools | `tell, mate, go` |
| Read policy | `one` · absorption `token` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 700,000 of 700,000 |

**Measured:** 1,278 children · 2,769 refills ·
31.6% self-sufficient · max generation 9 ·
72.8% canonical turns · median lifetime 4,849 steps ·
median context at death 4,848 tokens · 17.9 generated tokens/turn

## What it was for

The 7B model on token-by-token absorption — the arm that established the model
size question was settled.

## What we learned

**Scale, decisively.** 72.8% canonical turns against 15.1% for the
identical configuration at 1.5B (`node_4room_tell_token_6045968`), and
18 tokens per turn against 52.1. Every subsequent experiment used
Qwen2.5-7B-Instruct on the strength of this comparison.

Self-sufficiency 31.6% and maximum generation 9 — better than any
1.5B run, but still a population being propped up by immigration rather than
reproducing.

## Status

Superseded by the drain runs, but this is the run that justified dropping 1.5B.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
