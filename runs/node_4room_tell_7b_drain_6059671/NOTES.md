# node_4room_tell_7b_drain_6059671

*Read policy (job family 6059xxx) — one utterance per turn vs draining the queue*

| | |
|---|---|
| Slurm job | `6059671` |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 8,000 blocks, seeded 16/room |
| Tools | `tell, mate, go` |
| Read policy | `drain` · absorption `utterance` |
| Genome | σ = 0.01, uniform crossover, additive mutation |
| Steps reached | 664,494 of 700,000 |

**Measured:** 6,031 children · 6,167 refills ·
49.4% self-sufficient · max generation 9 ·
73.6% canonical turns · median lifetime 687 steps ·
median context at death 4,512 tokens · 21.3 generated tokens/turn

> **Ended early at step 664,494 of 700,000.**

## What it was for

The **drain** arm — an agent absorbs its entire observation queue before acting,
so it never replies to a room state that has already changed.

This became the analytical workhorse of the project. Most of what is known about
how reproduction degrades was measured here, because it was the first run with
both a working chat format and enough births (6,031) to support statistics.

## What we learned

Draining beats reading one (6,031 births against 1,973; generation
9 against 7). But the important findings came from analysing it rather
than from the comparison it was built for:

- **Reproduction degrades across generations.** Mates delivered fell
  33.6% → 20.3% from generation 0 to 3+, tells delivered 23.9% → 2.8%, moves
  succeeded 36.7% → 11.5%. Conditioning on room population, generation 3+ loses
  to generation 0 in every band, so it is not a crowding artefact.
- **Descendants are not gibberish.** They are better-formed than generation 0
  (83.9% vs 72.7%) and coherent English. They simply stop *landing* actions.
- **Mutation accumulates as a random walk.** Factor RMS 0.0200 → 0.0267 by
  generation 5; the applied perturbation ‖B·A‖ grew 1.00× → 1.84×, matching the
  √(0.02² + k·0.01²) prediction at generation 1 exactly.
- **Selection barely acts.** corr(drift, lifetime) ≈ +0.005 and
  corr(drift, context) ≈ −0.011 — death is behaviour-blind. Only reproduction
  differed (corr −0.148), explaining ~2% of variance.
- **Heritability is indistinguishable from zero.** Midparent slope +0.130 with a
  same-room control of +0.183 — strangers predicted a child as well as its
  parents did — and a midparent/single-parent ratio of 0.96 where additive
  transmission requires 2.0.

That last result is what pointed at the mutation rate and led to the `lowmut`
arm, which is where the project turned.

## Status

Superseded operationally, but still the reference dataset for the degradation and heritability analyses.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*
