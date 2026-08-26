# node_4room_7b_chromo_6071674

*Genome operators (job family 6071xxx) — crossover scheme and mutation operator at fixed everything else*

| | |
|---|---|
| Slurm job | `6071674` |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48,000 blocks, seeded 32/room |
| Tools | `tell, mate, go` |
| Read policy | `drain` · absorption `utterance` |
| Genome | σ = 0.01, chromosomal crossover, additive mutation |
| Steps reached | 400,000 of 400,000 |

**Measured:** 814 children · 88 refills ·
90.2% self-sufficient · max generation 13 ·
47.0% canonical turns · median lifetime 1,687 steps ·
median context at death 17,536 tokens · 30.6 generated tokens/turn

## What it was for

First test of **chromosomal crossover**: the layer-major site list
`[q,k,v,o,q,k,v,o,…]` is cut into 3 contiguous chromosomes with one crossover
point each, so a child switches parent ~3.8 times instead of ~55. Everything else
is identical to `node_4room_7b_free40_6070757`.

The hypothesis was that uniform crossover breaks up mutually supportive mutations
before selection can act on them.

## What we learned

**Inheritance became measurably more faithful, and it did not save the run.**

- Heritability of canonical-action rate: h² = +0.61 ± 0.16, against +0.24 ± 0.05
  for the four uniform σ=0.01 arms pooled — a difference of +0.37 ± 0.17, z = 2.15.
- Midparent / single-parent slope ratio 2.78, where additive genetic transmission
  predicts 2.0 and shared environment gives 1–2. Uniform `free40` returned 0.39.
- Despite that, canonical rate still fell 83.6% → 65.1% and reproduction
  2.00 → 0.68 children per 100 turns across generations.

The caveat matters: `norefill`, which uses **uniform** crossover, scored h² = +0.78
on 86 pairs. At these sample sizes the estimates do not cleanly rank the schemes.
An earlier write-up called this the only scheme showing real additive transmission,
comparing against `free40` alone; with all four uniform arms in view that claim
was too strong.

The run also ended early: the preemption guard aborted it at 4h47m after two
transient allocation failures.

## Status

Superseded by the chromosomal σ sweep. The h² result here is suggestive, not established — chr0025 vs uni0025 is the clean test.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
