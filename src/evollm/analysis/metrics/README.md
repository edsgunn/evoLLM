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

## What the genome can express

Which modules are adapted is not a size knob, it is a choice of function class.
Attention governs **routing** — which positions attend to which — and given the
attention pattern the value/output path is linear. The MLP is where the
**per-position computation** and the nonlinearity live. So an attention-only
genome can re-weight and re-route what the base model already computes, but
cannot change *what is computed*. Comparisons across runs with different
`target_modules` are comparisons across different search spaces, not across
different amounts of the same one.

## The three questions everything serves

1. **Can selection act?** — `effective-population-size.md`, `heritability.md`
2. **Is variation being created or destroyed?** — `mutation-rate.md`,
   `population-structure.md`
3. **Is anything actually improving?** — `behaviour.md`, `surprise.md`

A run that cannot answer (1) affirmatively cannot answer (3) meaningfully, no
matter what its behavioural numbers look like. That ordering has caught us out
before: reproduction rising across generations looked like adaptation until
density was controlled for.

## A fourth question, added late

4. **Does an agent change while it lives?** — `surprise.md`

Every metric under (1)-(3) is a lifetime aggregate: one number per agent,
covering its whole life. That framing cannot see change *within* a life, and
in-context surprise minimisation — the project's own hypothesis — is precisely
a claim about change within a life. The gap was invisible for a long time
because nothing in the report was shaped to expose it.

Read `surprise.md` for the direct measure and the behavioural proxy that works
on runs predating it, and always read the **paired within-agent** figure rather
than the population curve, which survivorship bends downward on its own.
