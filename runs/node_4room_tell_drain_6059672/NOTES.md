# node_4room_tell_drain_6059672

*Read policy (job family 6059xxx) — one utterance per turn vs draining the queue*

| | |
|---|---|
| Slurm job | `6059672` |
| Model | `Qwen2.5-1.5B-Instruct` |
| Rooms | 4 × 8,000 blocks, seeded 16/room |
| Tools | `tell, mate, go` |
| Read policy | `drain` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 2,000,000 of 2,000,000 |

**Measured:** 143 children · 7,624 refills ·
1.8% self-sufficient · max generation 3 ·
8.5% canonical turns · median lifetime 5,486 steps ·
median context at death 6,784 tokens · 57.3 generated tokens/turn

## What it was for

The 1.5B counterpart of the drain arm, run alongside `node_4room_tell_7b_drain_6059671`
to check whether the drain policy rescued the small model.

## What we learned

It did not. 8.5% canonical — the *worst* of any run in this directory —
143 births against 7,624 refills (1.8% self-sufficiency)
and maximum generation 3. Median lifetime 5,486 steps with
57 tokens per turn: these agents live a long time and accomplish
nothing, which is the signature of an agent that never completes an action.

This closed the model-size question for good. No 1.5B run has been submitted since.

## Status

Closed. 1.5B is not used.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
