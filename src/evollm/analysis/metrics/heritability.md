# Heritability (h²) and transmission

Whether anything a parent has is passed to its child. If nothing is, selection
has nothing to accumulate and the whole premise fails.

## Continuous traits: the midparent regression

Split a trait into a genetic part and everything else, then split the genetic
part by how it is transmitted:

> P = G + E,  V_G = V_A + V_D + V_I,  **h² = V_A / V_P**

Only the **additive** part V_A predicts offspring from parents, because a child
inherits roughly half its sites from each parent and combinations get broken up
every generation.

Regressing offspring on the **midparent** value gives h² directly, with no
correction factor. Regressing on a **single parent** gives h²/2.

## The ratio diagnostic — more informative than the slope

That fixed relationship is the useful part:

| midparent ÷ single-parent slope | interpretation |
|---|---|
| **≈ 2.0** | additive genetic transmission |
| **1 to 2** | largely shared environment |

A slope alone cannot distinguish inheritance from circumstance; the ratio can.
Early in this project a ratio of 0.96 exposed an apparent h² of +0.13 as pure
shared environment.

**Always report the ratio.** A high h² with a ratio near 1 is not heritability.

## The shared-environment control

Substitute two random agents from the same room, born within a step window, for
the actual parents. They carry the shared environment without the shared genome.
The **excess** of the real slope over that control is the load-bearing number,
not the slope itself.

## Discrete traits: parent-offspring concordance

For a categorical behaviour (which strategy an agent adopts, whether it has a
tic), measure how often parent and child agree, against a same-room same-era
stranger. Report the **excess in percentage points** and a z-score.

Sibling concordance is a weaker signal here and that is correct, not a bug: a
child takes each site from one of two parents, so two siblings agree only when
they happen to draw the same one.

## The lineage-label trap

**Do not label an agent by its generation-0 founder and test whether that
predicts behaviour.** By generation 200 essentially everyone descends from the
same handful of founders, so the label is shared by almost the whole population
and carries no information. A test built on it returns "not heritable" however
strongly parents resemble their children.

This produced a real contradiction in this project: founder-level mutual
information said strategy was not heritable while parent→child concordance said
it clearly was. The concordance was right. You share an ancestor with a
barnacle; it says nothing about whether you share a niche.

Use parent-offspring measures, or `Pedigree.ancestor_at` for a deliberately
shallow cut.

## Pitfalls

- **Selection erodes the variance it feeds on.** A low h² late in a successful
  run can mean the trait has swept, not that it was never heritable. Check
  whether V_P has collapsed before concluding anything.
- **Mutation in the denominator.** At high σ, mutational variance decorrelates
  parent from offspring and drives h² toward zero regardless of V_A. An h² of
  ~0 measured at high σ says nothing about h² at low σ.
- **Sample size.** Early estimates on a few hundred pairs were scattered enough
  to rank crossover schemes wrongly. Thousands of pairs are needed.
