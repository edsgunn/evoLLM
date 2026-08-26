# node_4room_both_6045967

*Tool and absorption variants (job family 6045xxx) — first runs after the chat-format fix*

| | |
|---|---|
| Slurm job | `6045967` |
| Model | `Qwen2.5-1.5B-Instruct` |
| Rooms | 4 × 8,000 blocks, seeded 16/room |
| Tools | `say, tell, mate, go` |
| Read policy | `one` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 2,000,000 of 2,000,000 |

**Measured:** 1,112 children · 8,204 refills ·
11.9% self-sufficient · max generation 5 ·
22.0% canonical turns · median lifetime 4,225 steps ·
median context at death 6,496 tokens · 65.3 generated tokens/turn

## What it was for

Both speech acts available at once (`say` **and** `tell`), testing whether agents
would differentiate — broadcasting when the message is general, addressing when
it is not — or whether the extra choice would just dilute the action distribution.

## What we learned

No differentiation worth the name, and the highest canonical rate in the 1.5B
family (22.0%) purely because `say` is the easier verb and dominated. Self
sufficiency 11.9%.

The useful negative: **offering more tools does not help a model that cannot
reliably use one.** Later configs settled on `[tell, mate, go]` — the minimum set
that supports directed coordination — rather than the maximal set.

## Status

Superseded. Later runs use [tell, mate, go].

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
