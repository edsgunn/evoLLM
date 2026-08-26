# Metrics: what we measure and why

One document per concept. These are **theory**, deliberately independent of the
code — they explain what a quantity means, why it is the right thing to look at
in this world, how to read it, and how it has already misled us. The machinery
that computes them is documented in `../README.md`.

Every per-run analysis reports the same core set, so runs stay comparable. The
template is `runs/_TEMPLATE_NOTES.md`.

| document | covers |
|---|---|
| `effective-population-size.md` | Ne, V_k, the drift threshold, whether selection can act at all |
| `heritability.md` | h², midparent regression, the ratio diagnostic, concordance |
| `mutation-rate.md` | σ, drift accumulation, the error threshold |
| `selection-and-death.md` | what kills an agent, hazard, eviction policy, context-at-death |
| `behaviour.md` | canonical rate, action composition, efficacy, reproduction |
| `population-structure.md` | panmixia, effective lineages, niches, metapopulations |

## The three questions everything serves

1. **Can selection act?** — `effective-population-size.md`, `heritability.md`
2. **Is variation being created or destroyed?** — `mutation-rate.md`,
   `population-structure.md`
3. **Is anything actually improving?** — `behaviour.md`

A run that cannot answer (1) affirmatively cannot answer (3) meaningfully, no
matter what its behavioural numbers look like. That ordering has caught us out
before: reproduction rising across generations looked like adaptation until
density was controlled for.
