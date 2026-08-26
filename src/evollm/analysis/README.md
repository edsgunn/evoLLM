# `evollm.analysis` — population analysis

> This file documents the **machinery**. What each metric *means*, why it is the
> right thing to measure, and how it has misled us is in **`metrics/`**.

Reusable machinery for asking the same questions of any run, the same way.
The unit is **one row per agent**; every module hands its output to the next.

```
evollm analyse runs/<run> --permutations 2000 --traits-csv traits.csv
evollm analyse runs/<run> --lineage-generation 20     # deeper lineage cut
```

```python
from evollm.analysis import analyse_run, format_report
print(format_report(analyse_run("runs/my_run")))
```

## Modules

| Module | Provides |
|---|---|
| `table.py` | `Table` — columns of numpy arrays keyed by agent id. Not pandas, deliberately: this venv also holds the vLLM stack. |
| `pedigree.py` | `Pedigree` — parents, founders, `ancestry` (founder shares), `ancestor_at` (deeper cuts), `families` (union-find), `descendants`. |
| `phenotypes.py` | `build_phenotypes` — 28 per-agent traits in five groups: form, strategy, efficacy, economy, fitness. |
| `genotypes.py` | Per-site magnitude features, from `<run>/genomes/*.jsonl` where present (every agent) or snapshots otherwise (the living only). Default feature is `‖B@A‖_F`, via a rank-sized trace identity rather than the full product. |
| `descent.py` | `Descent` — resolves the per-birth inheritance masks into *realised* per-site founder assignment, plus `realised_ancestry` and `effective_founders_per_site`. |
| `stats.py` | `variance_partition`, `associate`, `replication`, `kmeans`, `mutual_information`, `principal_components`, `benjamini_hochberg`. |
| `suite.py` | `analyse_run` / `format_report` — the standard battery. |

## The three questions the battery asks

1. **Does lineage explain the trait?** η², the share of variance between
   lineages, against a permutation null.
2. **Are there distinct strategies, and are they heritable?** k-means on the
   action-composition simplex, then mutual information between lineage and
   cluster.
3. **Which sites track which trait?** Per-site association between a site's
   perturbation magnitude and a trait, with genotype PCs as covariates.
4. **Which founder supplied each site, and does it matter?** Realised descent
   from the inheritance masks, then a marker test: does carrying one founder's
   variant at a locus predict behaviour? This is the strong version of
   question 3 — a categorical allele rather than a magnitude — and it needs a
   run recorded with `run.genome_fingerprints` on.

## Why the answers are trustworthy (or aren't)

Everything uses **permutation**, not parametric tails. These traits are bounded
rates, simplex shares and zero-inflated counts, and the agents are related to
each other — no closed-form test's assumptions hold.

Four guards, each of which caught something real:

- **Stratified permutation.** Nulls shuffle within `(birth room × generation
  band)`, so a lineage that happened to live early in a crowded room does not
  score as distinct for that reason.
- **η²'s null mean is always reported.** η² is biased upward by group count —
  with one agent per group it is 1 by construction. `excess = eta2 − null` is
  the number that means anything.
- **`effective_lineages`** (inverse Simpson). 153 labels mean nothing when one
  holds 95% of agents. The report says so and refuses to let the reader treat
  the η² table as informative when the population is panmictic.
- **`replication`.** Every reported association is refit per birth room. This
  is the one that kills most hits: a site can reach `p_fwer = 0.002` overall
  while its per-room fits are `+0.03 / +0.55 / +0.08`. A starred hit must both
  survive the multiple-testing null *and* show at least half the pooled effect
  in its weakest room.

## Lineage labels and the barnacle problem

`Pedigree.lineage` names an agent by its **generation-0 founder**, and at any
real depth that label is close to useless. By generation 200 essentially
everyone descends from the same handful of founders, so the label is shared by
almost the whole population and carries no information. A test built on it —
`variance_partition` by lineage, or mutual information between lineage and a
behaviour cluster — will report "not heritable" no matter how strongly parents
resemble their children. You and a barnacle share an ancestor; it says nothing
about whether you share a niche.

This is not hypothetical: on the σ = 0.0025 runs the founder-level MI test
returned p = 0.19 and the founder-level eta2 sat on its null, while
parent → child strategy concordance ran 4.1 and 5.7 percentage points above a
same-room, same-era control at z = 10.6 and z = 15.0. The behaviour is
inherited; the founder label simply cannot see it.

**For a discrete behaviour, use `parent_offspring_concordance`** — it compares
parent→child agreement against a stranger drawn from the same room within a
step window, so room composition and era are held fixed and only the parental
link remains. For a continuous trait, use the midparent regression. Reach for
`ancestor_at` only when a deliberately shallow cut is what you want, and note
it is not cached: on a 17,000-agent run at generation 200 it will exhaust
memory.

## What founder labels do and do not mean

Every founder is an independent `N(0, init_scale)` draw at every site —
exchangeable random perturbations of one base model. Two consequences, and
they pull in opposite directions:

- **Founder retention is not adaptive diversity.** At generation 0 there is
  none to lose. A site collapsing to one founder is ambiguous by itself: drift,
  or selection fixing the luckiest initial draw — which is the outcome the
  experiment is looking for, not a failure. Read `effective_founders_per_site`
  as coalescence of a neutral marker, never as health.
- **That neutrality is exactly what makes the labels useful.** Clean markers
  with no fitness effect are what you want for measuring *process*:
  recombination, coalescence rate, effective population size.

To separate drift from selection, use `Descent.selection_scan`. All sites ride
inside the same individuals, so they share one pedigree, one population size
and one drift history — the genome-wide spread of coalescence is therefore its
own neutral null, with no simulation and no assumed Ne. Sites in the tail are
selection candidates. Under chromosomal crossover the unit of inference is a
linked block, not a single site.

The diversity that matters for adaptation is not present at the start and
cannot be. Mutation is the only source of new variation, so it has to be
manufactured over many generations; the metric for it is heritable phenotypic
variance (h² × V_P), not founder counts.

## Expected vs realised ancestry

`Pedigree.ancestry` gives the recursive one-half expectation. `Descent`
gives what actually descended. The two differ — that difference *is* drift,
and the report prints its median total-variation distance. Only the realised
version yields an allele, because only it is an integer fact about one agent
rather than an average over possible births.

Runs made before inheritance tracking return `Descent.from_run(...) is None`
and the battery simply omits section 4.

## Adding a trait

Add it to `TRAIT_GROUPS` in `phenotypes.py` and compute it in the record dict.
Everything downstream — stratification, clustering, association, CSV export —
picks it up automatically. Rates with no denominator must be `NaN`, never `0`:
an agent that never tried to mate has an *undefined* success rate, and
averaging zeros in understates every cohort it appears in.
