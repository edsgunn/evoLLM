# Effective population size (Ne) and offspring variance (V_k)

**The single most important number in this project.** It decides whether
selection can do anything at all.

## What it is

Ne is the size of an idealised population that would drift as fast as the one
you actually have. It is almost always smaller than the head-count, and here it
has been *dramatically* smaller.

Using Crow & Kimura's variance formula:

> Ne = (4N − 2) / (V_k + 2)

where N is the census population and V_k the variance in offspring number per
agent. The mean offspring number is ~2 at stationarity, because every birth has
two parents.

## Why it decides everything

Drift changes gene frequencies by roughly 1/(2Ne) per generation. Selection
changes them by roughly s, the fitness difference. So:

> **selection is only effective when |s| ≫ 1/(2Ne)**

With Ne in single digits, 1/(2Ne) is around 0.1, and only enormous fitness
differences are visible. Anything smaller is invisible: it drifts.

This is not abstract. A behaviour costing a third of an agent's reproduction —
the `room_id` placeholder tic — swept a population of 85 agents per room,
because at Ne ≈ 2 selection against it was only about twice as strong as drift.

## What actually destroys Ne here

**Not the census.** N has been fine throughout. **V_k is the problem.**
Reproduction is winner-take-all: a majority of agents leave no offspring at all
while the top tenth produce most of the next generation. Because V_k appears in
the denominator, that collapses Ne into single digits regardless of how many
agents are alive.

Decomposing further, offspring number is a product of *mating rate* and *turns
lived*, and the rate term carries most of the variance. Lifespan barely predicts
offspring on its own — but because variance of a product carries a cross term,
compressing either factor still helps a lot.

## How to read it

- **Report Ne and 1/(2Ne) together.** The threshold is the actionable number.
- **Always report V_k alongside**, since it is the lever. Ne is a consequence.
- **Compare |s| against 1/(2Ne)** whenever claiming selection removed, or
  failed to remove, some behaviour.
- A run whose Ne is below ~10 cannot be expected to fix anything subtle,
  whatever its behavioural trends look like.

## Interventions tried

| change | effect on Ne |
|---|---|
| `random_holder` eviction (hazard ∝ blocks held) | raised it substantially |
| charging parents for children's adapters | made it *worse* — suppressed reproduction rather than concentrating it |
| lowering σ | raised it, as a side effect of healthier populations |

## Pitfalls

- **Ne is meaningless without stating N.** A high Ne from a huge census is not
  the same result as a high Ne from a compressed V_k.
- **Rooms are separate demes.** Computing one Ne across all four rooms of a run
  overstates it. Report per-room or state that it is pooled.
- **Truncated runs bias V_k downward** — agents alive at the cutoff have
  incomplete offspring counts. Say so when the run did not finish.
