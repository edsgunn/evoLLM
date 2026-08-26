# Metrics: what we measure and why

One document per concept. These are **theory**, deliberately independent of the
code — they explain what a quantity means, why it is the right thing to look at
in this world, how to read it, and how it has already misled us. The machinery
that computes them is documented in `../README.md`.

Every per-run analysis reports the same core set, so runs stay comparable. The
table is produced by `evollm.analysis.core_metrics` and the template is
`runs/_TEMPLATE_NOTES.md`.

| document | covers |
|---|---|
| `effective-population-size.md` | Ne, V_k, the drift threshold, whether selection can act at all |
| `heritability.md` | h², midparent regression, the ratio diagnostic, concordance |
| `mutation-rate.md` | σ, drift accumulation, the error threshold |
| `selection-and-death.md` | what kills an agent, hazard, eviction policy, context-at-death |
| `behaviour.md` | canonical rate, action composition, efficacy, reproduction |
| `population-structure.md` | panmixia, effective lineages, niches, metapopulations |
| `surprise.md` | whether the world became predictable to an agent, within its own life |

---

## What these are for

Every metric here exists to answer a question in [`VISION.md`](../../../../VISION.md),
and none of them is interesting on its own terms.

The vision's claim is that population training is a third pre-training-like
stage. Before anything can be claimed about what a population *learned*, the
procedure has to be shown to be a training procedure at all — that selection
can act, that variation survives, that behaviour is transmitted, and that
apparent improvement is not an artefact of density or survivorship. **That is
what almost everything below measures: the health of the training run and of
the population, not the value of the product.**

The mapping, question by question. A question with no metric against it is a
hole, and the holes are as much the point of this table as the entries.

| vision question | what it asks | metrics |
|---|---|---|
| **Q1** Can selection act? | is differential reproduction stronger than drift | `effective-population-size.md` |
| **Q2** Is variation created and preserved? | are we above or below the error threshold; is diversity collapsing | `mutation-rate.md`, `population-structure.md` |
| **Q3** Is anything transmitted? | do children resemble parents genetically, not just environmentally | `heritability.md` |
| **Q4** Is anything improving? | are later generations better at living here, net of confounds | `behaviour.md` |
| **Q5** Does an agent change while it lives? | in-context adaptation, as distinct from a better inherited prior | `surprise.md` |
| **Q6** Is the world selecting for anything worth having? | is behaviour dominated by a degenerate strategy | `behaviour.md`, `evollm inspect-traces` |
| **Q7** Does any of it leave the world? | is a late genome better at anything *outside* the world | **nothing — see below** |
| **Q8** What world structure induces what capability? | the Phase 2 mapping | **nothing — see below** |

### The ordering matters

A run that cannot answer **Q1** affirmatively cannot answer **Q4** meaningfully,
however good its behavioural numbers look. That ordering has caught us out
before: reproduction rising across generations looked like adaptation until
density was controlled for.

**Q4 and Q5 are routinely conflated and must not be.** Improvement can live in
the weights — later generations *start* better, which is inheritance of a prior
— or in the context, where an agent improves as it lives, which is in-context
learning. They are different products and want different metrics. Every metric
except `surprise.md` is a lifetime aggregate and structurally cannot see the
second.

**Q6 is not housekeeping.** A world that offers a cheap, always-available,
always-failing action will select for it, and every metric that counts
well-formed actions will report health while it happens. Q6 is the only check
that can find a failure mode nobody thought of in advance, which is why reading
raw behaviour is part of the process rather than an optional extra.

### The two holes

**Q7 has no metric at all.** Every quantity in this directory is internal to the
world. Nothing asks whether a late-generation genome is better than a
generation-zero genome at anything *outside* it — which is the question the
project is ultimately for. We have a training set and no test set. Closing this
is the fourth Phase 1 exit criterion and the bridge into Phase 2.

**Q8 has no metric because it needs a different kind.** Everything here
describes populations; Q8 requires describing *worlds* — what affordances exist,
how much an agent's outcome depends on other agents, how cheap the trivial route
through each structural primitive is. The world has never been characterised,
only the things living in it.

---

## What the genome can express

Which modules are adapted is not a size knob, it is a choice of function class.
Attention governs **routing** — which positions attend to which — and given the
attention pattern the value/output path is linear. The MLP is where the
**per-position computation** and the nonlinearity live. So an attention-only
genome can re-weight and re-route what the base model already computes, but
cannot change *what is computed*. Comparisons across runs with different
`target_modules` are comparisons across different search spaces, not across
different amounts of the same one.

Rank is the same kind of quantity, and a sharper one under the vision: it is
not a hyperparameter but **the capacity of the training stage**. If low-rank
adapters over a frozen base cannot express agentic dispositions at all, the
programme is capacity-bound and no amount of world design helps. Rank has never
been varied from 16.
