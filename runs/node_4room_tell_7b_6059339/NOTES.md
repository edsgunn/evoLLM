# node_4room_tell_7b_6059339

*Read policy (job family 6059xxx) — one utterance per turn vs draining the queue*

| | |
|---|---|
| Slurm job | `6059339` |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 8,000 blocks, seeded 16/room |
| Tools | `tell, mate, go` |
| Read policy | `one` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 700,000 of 700,000 |

**Measured:** 1,973 children · 5,479 refills ·
26.5% self-sufficient · max generation 7 ·
71.7% canonical turns · median lifetime 1,797 steps ·
median context at death 4,960 tokens · 18.5 generated tokens/turn

## What it was for

Baseline for the **read policy** comparison: `read_policy: one`, meaning an agent
absorbs a single queued utterance and then acts, even if more are waiting. Its
sibling `node_4room_tell_7b_drain_6059671` differs only in draining the whole
queue first.

## What we learned

Under `one`, agents act on stale information. An agent with a backlog answers the
oldest thing in it and then acts again, so its replies refer to a room state that
has moved on. Measured: 71.7% canonical, 1,973 births,
26.5% self-sufficiency, maximum generation 7.

Against the drain arm (7 vs 9 generations, 1,973 vs 6,031 births)
draining is clearly better, and `drain` has been the default in every config since.

## Status

Superseded by drain. Kept as the control for that comparison.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
