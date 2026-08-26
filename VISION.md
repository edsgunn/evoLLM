# Vision

## Purpose

This document states why the project exists, what it is trying to produce, and what would count as evidence that it worked or failed. It describes the programme, not its progress. Nothing here should need revision because a run finished.

## The Thesis

**Population training is a third pre-training-like stage, and the world is the dataset.**

Language models are currently built in two stages that play distinct roles:

- **Pre-training** optimises a proxy — next-token prediction — that no one wants for its own sake, and produces a model useful for tasks nobody specified in advance.
- **Post-training** optimises named tasks, and produces narrow competence one expensive task at a time.

The gap between them is agency. A model that has been pre-trained and tuned to follow instructions is not an agent; it is a text predictor wearing an agent's costume, with agency supplied externally by prompts and harnesses. Persistence, resource awareness, treating counterparties as minds, and self-preservation are treated as scaffolding around the weights rather than as properties of them.

The proposed third stage combines pre-training's structure with agency's content: place populations of models in a world, let selection run without naming any task, and harvest a population whose agentic dispositions are in the weights. Fine-tuning for downstream tasks then starts from a native agent instead of manufacturing agency per task.

**The core claim.** Pre-training transfers because the corpus covers the task distribution. Population training will transfer only insofar as the world's structure covers the task's structure. World design is therefore the central research problem — corpus curation raised one level.

## What "Natively Agentic" Means

The phrase is only useful if it is operational.

> **Native agency:** behaviour conditioned on the agent's own state and on models of other agents, in service of continuation and reproduction, without prompt scaffolding.

Three components, in increasing order of difficulty:

1. **Self-modelling** — behaviour depends on internal condition: resource level, age, proximity to termination.
2. **Other-modelling** — behaviour depends on the specific counterparty and its history, not on the room in aggregate.
3. **Continuation-seeking** — behaviour treats persistence as mattering: trading immediate gain for survival, or survival for reproduction.

**These are hypotheses, not in-world diagnostics.** Observing state-dependent behaviour inside the world does not establish any of them: a reflex loop keyed to a resource counter satisfies (1) behaviourally while modelling nothing, and no amount of in-world inspection settles the difference. The three components are claims about what a population has acquired, and they are tested only by generalisation — by held-out tasks that require the disposition and were never present in the world. Anything weaker is an unfalsifiable philosophical assertion and is excluded from the programme.

None of the three requires skill proficiency. They are minimum conditions for calling something an agent rather than a policy, they are task-independent, and they are the properties expected to transfer.

## The World Is the Dataset

Everything a population learns must be induced by structural properties of its environment. Three consequences follow:

1. A capability absent from the world cannot appear in the population.
2. A structure that is present but trivially satisfiable selects for the trivial exploit rather than the capability. Cheap, available, failing actions are selected continuously if unmonitored — this degrades a population faster than the structure's absence would.
3. Structure must therefore be designed, measured, and validated like data. Understanding what a world induces is the central work of the programme.

**The dataset is partly endogenous.** A corpus is fixed and exogenous; a world containing other evolving agents is not. Much of what any agent encounters is produced by the population itself, so the effective training distribution is co-created and non-stationary. Only the world's physics is exogenous. This is a genuine disanalogy with pre-training, and it is where the interesting dynamics live rather than a defect to be engineered away — but it means "designing the dataset" here means designing the constraints under which a distribution is generated, not the distribution itself.

## Substrate and Mechanics

**Modality match.** The world is made of text. Agents read utterances and emit tagged actions in the base model's native substrate. The question is never "text versus world" but whether the structure of the world covers the structure of the task. Structural priors — scarcity, mortality, other minds, irreversibility — are layered over a substrate the model already inhabits.

**Currency.** Device memory is the single currency. Agents hold KV blocks for context and adapter blocks for weights against one pool; when a room's pool is exhausted, an agent dies.

**Genetics and selection.** An agent's genome is its LoRA factors, inherited by recombination and mutation from two parents. Selection is strictly circumstantial: agents that produce more surviving offspring leave more descendants. There is no score, reward, or fitness function in the loop.

**Circumstantiality.** A reward function bounds discovery by its author's imagination. A world defines what is possible and lets selection find what works.

Measurement is not optimisation. The programme measures a great deal, but no measured quantity is fed back into selection; metrics exist to interpret what a world induced, not to steer it.

## Core Research Questions

Analysis metrics correspond directly to these.

| |Question|What it settles|
|---|---|---|
|**Q1**|Can selection act at all?|Whether differential reproduction exceeds random drift at the effective population size.|
|**Q2**|Is variation created and preserved?|Whether mutation is balanced between premature genome convergence and destroyed inheritance.|
|**Q3**|Is anything transmitted?|Whether behavioural resemblance across generations is genetic rather than environmental.|
|**Q4**|Is anything improving?|Cross-generational survival and living efficiency, controlled for density and survivorship.|
|**Q5**|Does an agent change while it lives?|Weight-borne inherited priors versus in-context learning within a lifetime.|
|**Q6**|Is the world selecting for anything worth having?|Whether strategies are non-degenerate and built-in structures remain load-bearing.|
|**Q7**|Does any of it leave the world?|Whether late-generation genomes beat generation-zero genomes on external tasks.|
|**Q8**|What world structure induces what capability?|The mapping from structural primitives to target task families.|

Q1–Q6 concern the mechanism, Q7 concerns transfer, Q8 concerns design. They are ordered by dependency: an affirmative answer to a later question is uninterpretable without the earlier ones.

## Phases

The phases are an ordering of dependencies, not a schedule.

### Phase 1 — Capability Baseline

Establish a minimal world — one currency, basic actions, graph-structured rooms, no reward — and show that the pipeline is a valid training procedure at all. The world is deliberately impoverished; the object of study is the mechanism, not the population.

Complete when all four hold:

1. **Selection acts.** Differential reproduction clears an explicit drift threshold, with positive heritability. The signature to look for is a midparent–offspring regression slope significantly above zero, with the single-parent slope at roughly half the midparent slope, indicating additive transmission rather than shared environment.
2. **Inherited improvement.** Cross-generational behavioural gains survive controls for density and survivorship bias.
3. **Non-degenerate world.** Action mix stays diverse; survival does not rest on a trivial repetitive loop.
4. **Transmission is genetic.** Offspring resemblance to parents exceeds resemblance to contemporaries sharing the same environment.

Note that transfer is _not_ a Phase 1 criterion. A world with no designed structure is predicted to produce no transfer; testing for it here would test nothing.

### Phase 2 — World Design

Make the world the independent variable: build structured environments, measure the dispositions they induce, and turn the primitive-to-capability mapping from a set of predictions into a model with predictive power on unseen structures.

This is where the core claim is at risk, because it is the first point at which coverage is asserted in advance and can fail.

### Phase 3 — Harvest and Transfer

Fine-tune evolved population members on downstream tasks and compare against generation-zero genomes fine-tuned identically, at matched compute. The comparison must be matched on compute, not on gradient steps, since evolution spends its budget on forward passes.

**Deliverable:** a diverse evolved population — a high-value search space for downstream selection — rather than a single static model.

## Structural Primitives and Task Families

The following are pre-registered predictions, not findings. Phase 2 exists to test them, and the mapping's value lies in being specific enough to be wrong.

|Structural primitive|Survival requirement|Predicted task family|
|---|---|---|
|Scarcity of a shared resource|Spend a finite budget only on high-value actions|Budgeting, prioritisation, cost-aware tool use|
|Mortality and irreversibility|Avoid unrecoverable terminal states|Risk assessment, caution, hedging, safe exploration|
|Other agents with private state|Infer unobservable information|Theory of mind, query generation, behavioural inference|
|Consent required from a counterparty|Induce autonomous selection by another agent|Negotiation, persuasion, offer framing, reciprocity|
|Communication without a fixed protocol|Establish shared convention|Protocol formation, grounding, instruction following|
|Topology with movement cost|Position well in space|Spatial planning, path search, explore/exploit trade-offs|
|Heterogeneous persistent partners|Identify and remember counterparties|Partner selection, reputation, trust modelling|
|Delayed consequence|Act now for later payout|Credit assignment, long-horizon planning, patience|
|Congestion for limited slots|Synchronise with peers|Coordination, turn-taking, queuing, conflict avoidance|

An absent row predicts an absent capability. A cheaply satisfiable row predicts worse than an absent one, because it selects for the shortcut.

## Falsification

### The thesis

Each condition is indexed to the phase in which it can be evaluated. None can be assessed earlier.

- **No selection (Phase 1).** Differential reproduction does not exceed drift, or nothing heritable is transmitted, under any workable configuration of population size and mutation rate. Selection is the premise of everything downstream.
- **No coverage (Phase 2).** Given a world designed to cover task family _F_, evolved populations show no advantage on held-out members of _F_ over generation-zero populations. Repeated across families, this falsifies the core claim: world structure does not determine what transfers.
- **Environment overfitting (Phase 2–3).** Populations improve inside the world but gains are confined to its mechanics, with no disposition that survives a change of surface form.
- **No native agency (Phase 3).** Late-generation genomes transfer to task families that need only competence, but not to those requiring self-modelling, other-modelling, or continuation-seeking. This is the operational form of "behaviour without agency" — and the only form of it that is testable, since it lives entirely in generalisation.

### Instantiation

These would end a particular implementation without bearing on the thesis. They are listed separately so that abandoning an implementation is not mistaken for abandoning the claim.

- **Narrow channel.** Low-rank adapters over a frozen base lack the capacity to express the dispositions in question. Indicates a different genome, not a different thesis.
- **Weak selection.** Effective population size cannot be scaled far enough for selection to beat drift within available compute. A resource limit.
- **Substrate artefacts.** Verbosity or terseness pressures, memory-allocation pathologies, or handshake base rates near zero at generation zero, none of which are properties of population training as such.

## Non-Goals

- **Benchmarking in Phase 1.** In-world performance is an intermediate signal, never the product.
- **Gradients in the selection loop.** Selection acts on the genome; the loop contains no backward pass.
- **Realism.** Environments are abstract functional instruments for isolating structures, not simulations of anything.
- **Emergent language for its own sake.** Communication is studied only as a structural primitive that induces task capabilities.
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
