# Population structure: lineages, panmixia, and whether niches can form

## Why it matters

A niche is a group doing something distinct *and passing it on*. That requires
some part of the population to be partly isolated from the rest. In a well-mixed
pool any local convention is diluted immediately, so no convention can establish
and nothing can specialise.

## Effective number of lineages

Counting distinct lineage labels is misleading. Use the **inverse Simpson
index** — 1 / Σpᵢ² — which asks how many groups there *effectively* are.

A population with 153 labels where one holds 95% of agents has ~1.1 effective
lineages, and stratifying it by those labels tests nothing. Report this before
any between-lineage result, and treat anything under ~2 as panmictic.

Every run in this project so far has been panmictic, with almost all agents in a
single family. That is the reason no niche has ever been observed, and it is a
property of the topology rather than of the agents.

## Families versus lineages versus ancestry

Three different things, not interchangeable:

- **family** — connected components joining each child to *both* parents. The
  coarsest grouping; collapses to one giant component very fast.
- **ancestry** — the fraction of an agent's pedigree from each founder, by the
  recursive one-half rule. A vector, not a label. This is the *expected* share.
- **realised descent** — which founder actually supplied each site, from the
  inheritance masks recorded at birth. Integers, not fractions. Only this yields
  an allele.

The gap between expected and realised ancestry *is* drift, made visible per
individual.

## Founder identity is a neutral marker, not diversity

Every founder is an independent draw from the same distribution — exchangeable
random perturbations of one base model. So:

- **Founder retention is not adaptive diversity.** At generation 0 there is none
  to lose. A site collapsing to one founder is ambiguous: drift, or selection
  fixing the luckiest initial draw, which is the outcome we want.
- **That neutrality is what makes the labels useful.** Markers with no fitness
  effect are exactly right for measuring *process* — coalescence rate,
  recombination, effective population size.

To separate drift from selection at a site, use the fact that all sites share
one pedigree, one population size and one drift history. The genome-wide spread
of coalescence is therefore its own neutral null, with no simulation and no
assumed Ne. Sites in the tail are selection candidates — in linked blocks, not
singly.

## Strategy clusters

Cluster agents on the action-composition simplex. Distinct clusters have
appeared in every run — a mating specialist and a moving specialist — so
*strategies* exist. Whether they are *niches* is a separate question, answered
by whether they are inherited, and the founder-label version of that test is
broken (see `heritability.md`).

## Metapopulations

Small demes with sparse connections give isolation by distance, which is the
structure lineages need to diverge. The trade is that within-deme Ne falls with
deme size, so drift dominates locally and selection acts between demes instead.

Migration rate is the dial: too sparse and demes drift into independent noise;
too dense and it is a panmictic pool with extra steps.

## Pitfalls

- **Diversity within a room and drift away from base move independently.**
  Report both.
- **A dying population's remnant is not a random sample.** Diversity falling in
  a collapsing run says nothing about selection.
