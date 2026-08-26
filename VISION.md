# Vision

*Why this project exists, what it is trying to produce, and how we would know
if it worked. This document changes rarely. What we currently believe lives in
`STATE.md`; how we work lives in `PROCESS.md`.*

---

## The thesis

**Population training is a third pre-training-like stage, and the world is the
dataset.**

Language models are built in two stages that do different jobs. Pre-training
optimises a proxy — predict the next token — that nobody wants for its own
sake, and produces a model useful for tasks nobody specified in advance.
Post-training optimises tasks we name, and produces competence on those tasks,
narrowly and expensively, one at a time.

The gap between them is agency. A model that has been pre-trained and then
tuned to follow instructions is not an agent; it is a text predictor wearing an
agent's costume, with the agency supplied from outside by a prompt and a
harness. Everything that makes it act — persistence, awareness of its own
resources, treating other participants as minds, caring whether it continues —
is scaffolding around the weights rather than anything in them.

We propose a third stage with pre-training's *shape* and agency's *content*:
put populations of models in a world, let selection run without naming any
task, and harvest a population whose agentic dispositions are in the weights.
Fine-tuning for particular downstream tasks then starts from something that is
already an agent, rather than manufacturing agency per task.

The claim that makes this more than an analogy is the second half. Pre-training
transfers because the corpus covers the task distribution. Population training
will transfer only insofar as **the world's structure covers the task's
structure**. That makes world design the central research problem, and it is
the same problem as corpus curation, one level up.

---

## What "natively agentic" means

The phrase has to be operational or it is decoration. We mean:

> **Behaviour conditioned on the agent's own state and on models of other
> agents, in service of its own continuation, without prompt scaffolding.**

Three testable components, in increasing order of difficulty:

1. **Self-modelling.** Does what the agent does depend on its own condition —
   how much resource it holds, how long it has lived, how close it is to death?
   An agent that acts identically when rich and when nearly dead is not
   tracking itself.
2. **Other-modelling.** Does what the agent does depend on *which* other agent
   it is dealing with, and on what that agent has done? Treating every
   counterparty identically is reacting to a room, not to a mind.
3. **Continuation-seeking.** Does the agent behave as though its own
   persistence matters — trading immediate action for survival, or survival for
   reproduction?

None of these require the agent to be *good* at anything. They are the minimum
conditions for calling the thing an agent rather than a policy, and they are
what we expect to transfer, because they are task-independent.

This definition is deliberately demanding of our instrumentation: none of it is
answerable from lifetime aggregates, and at the time of writing none of it is
measured. That is the definition doing its job.

---

## Why a world, and why this one

The world is made of text. Agents read utterances and emit tagged actions; the
substrate is the one the base model was trained on. This matters more than it
first appears: **we get modality match for free.** The transfer question is
therefore not "text versus world" but "does the *structure* of the world cover
the structure of the task". We are adding structural priors — scarcity,
mortality, other minds, irreversibility — on top of a substrate the model
already inhabits.

The world's single currency is device memory. Agents hold KV blocks for their
context and blocks for their adapter; when a room's pool is exhausted, someone
dies. Nothing is scored, nothing is rewarded, and no gradient is computed
anywhere in this codebase. An agent's genome is a set of LoRA factors; children
inherit them from two parents by recombination and mutation. Selection is
entirely circumstantial: agents that get more children into the world leave more
descendants, and whatever caused that is what propagates.

That circumstantiality is the point. A reward function specifies what to
optimise, and therefore bounds what can be discovered by the imagination of
whoever wrote it. A world specifies what is *possible*, and lets selection find
what pays.

---

## The world is the dataset

Everything that a population can learn must be induced by some structure in the
world it lived in. Three consequences follow, and they are the whole research
programme:

**A capability absent from the world cannot be in the population.** If nothing
in the world requires modelling another agent, no amount of selection will
produce agents that model each other. This is not a subtle failure mode; it is
the default one.

**A structure that is present but trivially satisfiable selects for the
trivial exploit rather than the capability.** If the world offers an action
that is cheap, always available and always fails, populations will find it and
do nothing else — and every metric that counts well-formed actions will call
that health. This has already happened to us more than once.

**Structure must therefore be designed, measured and validated like data.**
"We built a world and ran it" is the equivalent of "we scraped some text". The
interesting work is knowing what a world induces, which is the subject of
Phase 2.

---

## What we need to know

These questions are numbered so that metrics can cite them. Every metric in
`src/evollm/analysis/metrics/` states which of these it serves, and a question
with no metric against it is a hole.

**Q1. Can selection act at all?**
Selection competes with drift. If the effective population size is small enough
that random sampling dominates differential reproduction, then whatever the
population does is not evidence of anything being selected. This bounds every
other claim: a run that fails Q1 cannot answer Q4 meaningfully however good its
behavioural numbers look.

**Q2. Is variation created and preserved?**
Selection consumes variation. Too little mutation and the population converges
on one genome and stops exploring; too much and inheritance is destroyed faster
than selection can accumulate it. There is a critical rate, and running above it
means no information accumulates at all — the evolutionary equivalent of a
diverged training run.

**Q3. Is anything transmitted?**
If children do not resemble their parents in what they *do*, then nothing
learned in one generation is available to the next, and the procedure is not
training. Transmission must be shown to be genetic rather than environmental,
because agents that share a room resemble each other for reasons that have
nothing to do with descent.

**Q4. Is anything improving?**
Across generations: are later populations better at living in this world than
earlier ones, controlling for the confounds — density, survivorship, and the
fact that a crowded room makes every rate look different.

**Q5. Does an agent change while it lives?**
Distinct from Q4 and repeatedly conflated with it. Improvement can live in the
weights (later generations *start* better) or in the context (an agent gets
better as it lives). These are different products: the first is inheritance of a
prior, the second is inheritance of an ability to adapt. Only the second is
in-context learning.

**Q6. Is the world selecting for anything worth having?**
The health of the *world*, not the population. Is behaviour dominated by a
degenerate strategy? Are the structures we built into the world actually load
bearing, or has the population found a way around them? A world that selects
for something trivial produces a population with nothing to transfer, and every
population-health metric will report success while it happens.

**Q7. Does any of it leave the world?**
The question the project is ultimately for. Is a late-generation genome better
than a generation-zero genome at something *outside* the world it evolved in?
Until this is measured, every result is internal and the central claim is
untested.

**Q8. What world structure induces what capability?**
Phase 2. The mapping from the structural primitives of a world to the task
families they produce.

---

## Phases

### Phase 1 — Can anything be learned at all? *(current)*

The simplest world we can defend: one currency, a handful of actions, rooms on a
small graph, no task, no reward. The purpose of Phase 1 is **not** to produce a
useful population. It is to establish that the procedure is a training procedure
at all — that selection can act, that variation survives, that behaviour is
transmitted, and that we can tell the difference between learning and drift.

Almost all of the work so far has been building the instruments to answer Q1–Q6
and discovering how many ways the answers can be faked.

**Phase 1 exits when all four hold:**

1. **Selection acts.** Effective population size is clear of the drift
   threshold for the traits we care about, and heritability is positive with a
   midparent/single-parent ratio near 2.
2. **Something improves, and is inherited.** A behavioural improvement across
   generations that survives controlling for density and survivorship, and that
   is transmitted parent to child.
3. **The world is not degenerate.** No single action dominates the action mix,
   and the population is not carried by agents repeating one turn until they
   die.
4. **Something leaves the world.** A held-out probe shows late-generation
   genomes beating generation-zero genomes on a task the world never contained.

Criteria 1–3 are measurable with what exists today. Criterion 4 requires
machinery we have not built, and building it is the bridge to Phase 2.

### Phase 2 — World design

Given a procedure that demonstrably trains, the question becomes what to train
*for*, which under this thesis means what world to build. Phase 2 makes the
world the independent variable: build worlds with specified structure, measure
what dispositions they induce, and accumulate the mapping.

The deliverable of Phase 2 is not a population. It is **knowledge about world
design** — the beginnings of a theory that says, given a task family you care
about, what structure a world must have to induce it.

#### World structure → task family

The working map. Each row is a structural primitive that can be present or
absent from a world, and the task family we expect it to induce. Every row is a
hypothesis, not a finding; the entries are what Phase 2 exists to confirm,
refute and refine.

| structural primitive | what an agent must do to survive it | task family it should induce |
|---|---|---|
| **Scarcity of a shared resource** | spend a finite budget on the actions worth taking | budgeting, prioritisation, cost-aware tool use, knowing when to stop |
| **Mortality and irreversibility** | avoid unrecoverable states | risk assessment, caution, hedging, safe exploration |
| **Other agents with private state** | infer what it cannot observe | theory of mind, asking, inference from behaviour |
| **Consent required from a counterparty** | make another agent choose you | negotiation, persuasion, offer construction, reciprocity |
| **Communication with no fixed protocol** | invent and share a convention | protocol formation, grounding, instruction-following |
| **Topology with movement cost** | decide where to be | planning, search, exploration/exploitation trade-offs |
| **Heterogeneous, persistent partners** | tell counterparties apart and remember them | partner selection, reputation, trust |
| **Delayed consequence** | act now for a payoff later | credit assignment over a horizon, patience |
| **Congestion for limited slots** | act in concert with others | coordination, turn-taking, queuing, conflict avoidance |

Two rules read off the table, both learned the hard way:

- **A row absent from the world is absent from the population.** You cannot
  select for negotiation in a world where nobody has to agree to anything.
- **A row present but cheap to satisfy is worse than absent**, because the
  population will find the cheap satisfaction and stop. A primitive is only
  load bearing if the trivial route through it is closed.

The second rule is why Q6 is a first-class question rather than housekeeping,
and why reading raw behaviour is part of the process rather than an optional
extra.

### Phase 3 — Harvest and transfer

Take a population produced by a designed world and use it: fine-tune members
for downstream tasks and compare against the same base model fine-tuned without
the population stage. The claim under test is that starting from an evolved
agent beats starting from a text predictor, on tasks that share structure with
the world — and that the advantage grows with how well world structure and task
structure match.

The product of Phase 3 is a **population**, not a model. Its diversity is an
asset rather than merely a health indicator: it is the search space that
downstream selection draws from.

---

## What would falsify this

Stated plainly, so that we notice if it happens:

- **The channel is too narrow.** Dispositions may not be expressible in a
  low-rank adapter over a frozen base at all. If capacity is the binding
  constraint, no amount of world design helps, and the honest conclusion is
  that the genome must be something else.
- **Selection is too weak to matter.** If effective population size cannot be
  raised into a regime where selection beats drift at achievable scale, the
  procedure cannot accumulate anything however long it runs.
- **Nothing transfers.** Populations may adapt to the world in ways that are
  entirely specific to it — overfitting to the environment, with no
  disposition underneath. This is the most likely interesting failure, and Q7
  is the only thing that can detect it.
- **Behaviour without agency.** Populations may improve at surviving while
  failing every component of the definition above: no self-modelling, no
  other-modelling, just a better reflex. That would be a real result and not
  the one we want.

---

## Non-goals

- **Beating a task benchmark in Phase 1.** The world is not a benchmark and
  performance in it is not the product.
- **Gradient-based training.** No backward pass exists in this codebase, by
  design. The genome is what is selected.
- **Realism.** Worlds are instruments, chosen for what structure they isolate,
  not for resembling anything.
- **Emergent language for its own sake.** Communication matters here because it
  is a structural primitive that induces task families, not as a phenomenon to
  admire.

---

## Appendix: where this came from

*Included for interest and to explain the fossils. Nothing above depends on
this section.*

The project began as a narrower question: whether gradient-free evolution
would, on its own, discover **in-context surprise minimisation** — whether
agents selected only for survival would come to predict their world better, as
a natural consequence of having to live in it. The original framing treated
that as the hypothesis under test, and the world as an apparatus for testing it.

Two things changed it.

The first is that the apparatus turned out to be the interesting object. Almost
everything we learned in the first months was about how hard it is to build a
world in which selection can act at all — the critical mutation rate, the
collapse of effective population size under winner-take-all reproduction, an
eviction rule that made large agents *safer* than small ones and rewarded
exactly the wrong thing. Those are not results about surprise. They are results
about how to train populations, and they generalise past any particular
hypothesis.

The second is that the narrow framing gave no account of why anyone should
want the outcome. "Evolution rediscovers surprise minimisation" is an
interesting scientific claim and a dead end as a programme: it terminates in a
paper rather than in a capability. Reframing the same machinery as a third
training stage keeps every result and gives them somewhere to go.

Surprise survives the change, demoted from hypothesis to instrument. It is now
one of the better task-independent reads on whether an agent models its
environment at all, which is Q5 — a measurement, not the point.

Readers of the code will find around 250 `§` references pointing into an
original proposal document that is not in this repository and that encodes the
older framing. They are historical; where they conflict with anything here,
this document wins.
