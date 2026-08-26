# node_4room_6045965

*Tool and absorption variants (job family 6045xxx) — first runs after the chat-format fix*

| | |
|---|---|
| Slurm job | `6045965` |
| Model | `Qwen2.5-1.5B-Instruct` |
| Rooms | 4 × 8,000 blocks, seeded 16/room |
| Tools | `say, mate, go` |
| Read policy | `one` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 2,000,000 of 2,000,000 |

**Measured:** 1,407 children · 7,451 refills ·
15.9% self-sufficient · max generation 6 ·
19.1% canonical turns · median lifetime 4,527 steps ·
median context at death 6,464 tokens · 63.0 generated tokens/turn

## What it was for

The first post-chat-format baseline. After the `<|im_start|>` framing bug was
fixed, every earlier run became uninterpretable (see `archive/2026-08_pre_chat_format/`),
so the Qwen2.5 tool variants were rerun from scratch. This is the **broadcast**
arm: agents have `say`, which every co-resident hears.

## What we learned

Almost nothing about the hypothesis, and one important thing about the setup:
**Qwen2.5-1.5B cannot run this world.** 19.1% of its turns produced a
canonical action, against ~72% for the 7B model on the same tools. It also burns
63 tokens per turn — roughly five times what a working agent needs.

The second lesson is the **refill treadmill**. Under the head-count trigger
(`min_population: 8`) the world admitted 7,451 immigrants against
1,407 genuine births — self-sufficiency 15.9%. The population was
not reproducing; it was being continuously replaced. This is what motivated
switching the refill trigger from head-count to free blocks, and eventually to
turning it off.

## Status

Superseded. 1.5B was dropped after this family; the refill trigger it uses no longer exists in any live config.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
