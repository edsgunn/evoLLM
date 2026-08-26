# node_4room_7b_multmut_6071676

*Genome operators (job family 6071xxx) — crossover scheme and mutation operator at fixed everything else*

| | |
|---|---|
| Slurm job | `6071676` |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48,000 blocks, seeded 32/room |
| Tools | `tell, mate, go` |
| Read policy | `drain` · absorption `utterance` |
| Genome | σ = 0.5, uniform crossover, multiplicative mutation |
| Steps reached | 400,000 of 400,000 |

**Measured:** 528 children · 99 refills ·
84.2% self-sufficient · max generation 5 ·
43.3% canonical turns · median lifetime 1,991 steps ·
median context at death 19,768 tokens · 33.6 generated tokens/turn

## What it was for

Replace additive mutation `x + N(0,σ)` with **multiplicative** `x·(1 + N(0,σ))`,
which perturbs each factor in proportion to itself and so does not random-walk
magnitude upward.

σ is 0.5 here, not 0.01, and deliberately: additive 0.01 noise on factors of RMS
0.02 *is* a 0.5 relative perturbation. Carrying 0.01 across would have made this
a 50× weaker mutation and turned the arm into a second copy of `lowmut`. At 0.5
the first mating perturbs a child exactly as hard as the baseline and only the
compounding differs.

## What we learned

**The informative negative of the sweep.** If degradation were driven by ‖ΔW‖
inflating, multiplicative noise should have been spared. It degraded like the
baseline: reproduction 0.93 → 0.37 → 0.19 children per 100 turns, canonical rate
82.2% → 62.2%, maximum generation 5.

So the operative quantity is **the size of the per-generation perturbation
relative to the inherited signal**, not growth in norm. At a 0.5 relative
perturbation, multiplicative noise destroys inheritance just as effectively as
additive. This corrected an earlier framing of the problem as magnitude
compounding — that was a special case, and not the active one.

## Status

Closed. The multiplicative operator remains available (`genome.mutation`) but no live config uses it.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
