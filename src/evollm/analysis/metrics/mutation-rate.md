# Mutation rate, drift accumulation, and the error threshold

The variable that has dominated every other in this project.

## What σ does

Each mating perturbs every site of the child's genome by N(0, σ). Against
factors initialised at `init_scale`, the relative perturbation per generation is
roughly σ / init_scale — which for the long-standing default was **50% at every
one of 112 sites, every generation**.

Magnitude accumulates as a random walk:

> factor RMS after k matings ≈ √(init_scale² + k·σ²)

This prediction has matched observation closely in every run measured, which is
a useful sanity check that the genome machinery is behaving.

## The error threshold

Above a critical σ, mutational input outruns whatever selection removes and the
population loses information regardless of how strong selection is — the
classic quasispecies error threshold. Below it, populations improve across
generations.

This is a **phase transition, not a gradient**. Runs either take off or
collapse; nothing has been observed in between. Above the threshold every
configuration dies regardless of crossover scheme, refill policy or mutation
operator — which is why σ has to be established before any other variable is
worth sweeping.

## What the damage actually is

Not growth in adapter magnitude. Multiplicative noise, which perturbs each
factor in proportion to itself and therefore does *not* compound magnitude,
degraded just like additive noise at matched relative perturbation.

The operative quantity is **the size of the per-generation perturbation
relative to the inherited signal**. A scheme that leaves magnitude alone but
still randomises direction destroys inheritance just as effectively.

## Lower is not automatically better

Mutation is the only source of new variation in this world — there is no other
innovation operator — and selection erodes the additive variance it feeds on. So
there is a floor below which populations starve for variation. We have not found
it, and the runs at the lowest σ tested have the *least* genetic diversity and
the *weakest* parent-to-child transmission while also being the healthiest. That
tension is unresolved.

## How to read it

- Report drift as **‖ΔW‖ relative to generation 0**, and check it against the
  random-walk prediction. A mismatch means something is wrong with the genome
  path, not with the biology.
- Report **spread between agents relative to their own distance from the base
  model**. Absolute distances are uninformative; the population may be a tight
  cloud very far from base.
- Distinguish **drift from base** (how far the population has moved) from
  **diversity within** (how far apart its members are). They move independently
  and mean different things.

## Pitfalls

- **The threshold is not a fixed number.** It has only been bracketed, and what
  it depends on — population size, genome size, rank, base model — is unknown.
- **Diversity peaking then falling is a sweep signature**, not simply drift. In
  a collapsing run it is also just the remnant of a dying population, which is
  not the same thing. Check whether births were still happening.
