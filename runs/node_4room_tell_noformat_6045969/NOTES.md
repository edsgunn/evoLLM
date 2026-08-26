# node_4room_tell_noformat_6045969

*Tool and absorption variants (job family 6045xxx) — first runs after the chat-format fix*

| | |
|---|---|
| Slurm job | `6045969` |
| Model | `Qwen2.5-1.5B-Instruct` |
| Rooms | 4 × 8,000 blocks, seeded 16/room |
| Tools | `tell, mate, go` |
| Read policy | `one` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 2,000,000 of 2,000,000 |

**Measured:** 351 children · 8,157 refills ·
4.1% self-sufficient · max generation 4 ·
10.5% canonical turns · median lifetime 4,808 steps ·
median context at death 6,560 tokens · 36.3 generated tokens/turn

## What it was for

The **control for the chat-format fix**. `chat_format: false` reproduces the old
bare-token-stream context that invalidated every archived run, so this arm
measures what that bug actually cost.

## What we learned

At 1.5B, almost nothing — 10.5% canonical here against 10.2% for the
identical run *with* chat framing (`node_4room_tell_6045966`). The two are
indistinguishable.

That is not evidence the fix was unnecessary. It is evidence that **1.5B was
already failing for a more basic reason**, so the format defect had nothing left
to damage. The fix's real effect shows up only at 7B, where canonical rates run
41–92%. Read this run as a statement about the model, not about the format.

## Status

Superseded. Kept as the record of the format control.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
