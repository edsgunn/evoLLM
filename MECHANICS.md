# How the world works

*For a researcher meeting the project for the first time. This describes the
mechanics — what the world is, how agents perceive and act, how they reproduce,
and how they die — without reference to the implementation. It says why the
project exists only in passing; that is [`VISION.md`](VISION.md). It says
nothing about what we have found; that is [`STATE.md`](STATE.md).*

*Most of the mechanisms below are swappable, and several have been swapped
during the project. Where that is so, the alternatives are named and the
current choice is given, with the reason. A summary table of every such choice
is at the end.*

---

## 1. The shape of it, in one page

A population of language-model agents lives in a small world made of text. Each
agent is a set of low-rank weight modifications — its **genome** — layered over
one shared frozen base model. Every agent in the world is the same model
underneath; they differ only in their genome.

Agents perceive by reading text the world writes into their context, and act by
emitting a tagged action: speak to someone, propose to mate, or move to another
room. They accumulate context as they live, and context costs memory.

Memory is the only currency, and it is finite. When a room runs out, an agent
dies. There is no score, no reward, no fitness function and no gradient
anywhere in the loop. **Selection is entirely circumstantial:** agents that
manage to get more children into the world leave more descendants, and whatever
disposition caused that propagates. Nothing measures whether an agent is good;
the world simply kills some of them.

Two agents that both propose to each other produce a child. The child's genome
is assembled from its two parents' weights by recombination, then perturbed by
mutation. It arrives as a newborn with an empty context and its parents'
weights, and starts living.

That is the whole mechanism. Everything below is detail.

---

## 2. The world

### Rooms and topology

The world is a graph of **rooms**. Agents live in a room, perceive only what
happens in that room, and can move along edges to adjacent rooms.

Currently: **four rooms, fully connected**, one per GPU. A room maps to a
device because a room's population must be served together, and the memory a
room rations is that device's memory.

*Swappable.* The topology is arbitrary — rings, sparse graphs, clusters. We
have also run a **forty-room** variant with many small rooms per device,
sparsely connected, to create isolation by distance. The four-room complete
graph is the current default because it is the simplest thing that still has
movement in it, and because sparse topologies are a Phase 2 question rather
than a Phase 1 one.

### The currency: blocks

The world has exactly one resource: **memory blocks**. A block holds sixteen
tokens of attention cache. Each room owns a fixed pool — currently **48,000
blocks**, about 768,000 tokens — and every agent in that room holds blocks from
it.

An agent holds blocks for two things:

- **Its context.** Everything it has read and written since birth. This grows
  every turn, without exception, and is never truncated or summarised.
- **Its adapter.** Its genome must be resident to be used. This is a fixed
  footprint, identical for every agent — a few tens of blocks, against the
  several hundred a mature agent's context comes to occupy.

The pool is the whole of physics. Nothing else is scarce, and no other cost
exists.

---

## 3. What an agent is

Two parts, and the distinction is load bearing.

**The genome** is a set of low-rank matrix pairs applied to the base model's
attention projections. Currently rank 16 over four projections in each of the
model's 28 layers — **112 sites**, each site a pair of factors. This is the
only thing inherited, and the only thing selection acts on.

*Swappable.* Which modules the genome covers is a choice of function class, not
a size dial: attention governs *routing* — what attends to what — while the
feed-forward layers are where per-position computation happens. An
attention-only genome can re-weight and re-route what the base model already
computes but cannot change *what is computed*. We are currently testing a
variant that adds the MLP projections (196 sites). Attention-only is the
reference because every result so far was measured on it.

**The soma** is everything else: the agent's context, its queue of unread
observations, its pending proposals, its age. The soma is built during life,
costs memory, determines survival — and **is not inherited.** It is discarded
at death.

So there is a clean barrier: experience shapes whether an agent survives long
enough to reproduce, but never touches what it passes on. Anything the
population accumulates across generations has to get there by differential
survival of genomes, not by anything an individual learns and hands down.

---

## 4. What an agent perceives

An agent is told very little. At birth it receives a short system message
naming itself, its room, who else is present, which rooms are adjacent, the
actions available to it, and one worked example of each.

That is all. It is **not** told that memory is finite, that speaking costs the
listener, that exhaustion kills, or what a child is made of. Those are
discoverable by living, and stating them would spend context on things
selection can find on its own.

This is deliberate and was learned the hard way: an earlier prompt explained the
economics and mentioned one action twice, and the population emitted that action
seven times more often than any other at almost zero validity. **Whatever the
prompt shows, agents copy.** So it shows only actions.

During life, the world writes observations into the agent's context: what
others said, who arrived and left, and the result of its own failed actions. A
failed action is not merely a null — it returns information, so an agent that
tries to move to a room that does not exist learns which rooms do.

Observations are queued and absorbed. An agent that is busy generating falls
behind, and a backlog is literally how many steps behind the room it is.

*Swappable.* Two choices here. **Absorption**: an utterance can enter context
whole in one step, or one token per step. We use **whole utterances**, because
under token-by-token absorption every agent's context grew at exactly the same
rate regardless of behaviour — speech cost listeners nothing, and lifetime was
independent of what an agent did, leaving selection almost no signal.
**Reading**: an agent can absorb one queued observation before acting, or drain
its whole backlog. We **drain**, so that a run of pending observations arrives
as a single block of text rather than as many consecutive fragments.

---

## 5. How an agent acts

An agent acts by generating text. The moment it closes an action tag, its turn
ends — one action per turn, enforced by the world rather than requested in the
prompt. Anything it writes before the action is charged for and discarded as
thinking.

Three actions are currently available:

| action | effect |
|---|---|
| **tell** | send text to one named agent in the same room |
| **mate** | propose reproduction to one named agent in the same room |
| **go** | move to an adjacent room |

*Swappable.* A broadcast **say** — heard by everyone in the room — exists and is
currently **disabled**. Broadcast speech charges every listener for every
utterance, and in a room of sixty agents that is lethal: it turns one agent's
verbosity into everyone else's death. Directed **tell** charges one listener,
which is why it is the survivable form of speech.

The turn-end token is itself charged, so an agent that does nothing at all
still grows its context every turn. **There is no way to sit still and
survive.**

The parser accepts several syntactic variants and records which was used, so a
malformed turn is distinguishable from a competent one that used an unusual
form. Every turn is classified rather than punished.

---

## 6. Reproduction

### The handshake

Reproduction requires two agents to agree, and agreement is symmetric:
**pointing `mate` at an agent that has already pointed `mate` at you is the
acceptance.** There is no separate verb. An agent that means "yes" and an
agent that means "let's" emit exactly the same thing, so a misfired agreement
lands as a fresh proposal rather than as nothing.

A proposal is delivered into the target's observation queue and becomes live
only when the target actually *reads* it. Its acceptance window — currently 64
tokens of the target's own generation — is armed at that moment, not when the
proposal was sent. This was a real bug once: windows armed at send time expired
while the proposal sat in a backlog the target had not reached, and across
hundreds of acceptances not one could ever have been valid.

*Swappable.* An explicit **accept** verb exists and is currently disabled;
mutual proposal is the handshake. A separate verb was found to attract copying
without comprehension.

### How the weights are mixed

A child's genome is assembled from its two parents, site by site.

**Whole sites travel together, and are never blended.** Each of the 112 sites
comes wholly from one parent or wholly from the other. Averaging two parents'
factors would be meaningless: a low-rank factorisation's internal basis is
arbitrary, so the same effective weight change can be written many different
ways, and the mean of two encodings is not the mean of what they encode.

Which parent supplies which site is decided by the recombination scheme.

*Swappable, and this is one of the more consequential choices.*

- **Uniform**: every site flips its own coin. With 112 sites that is around 55
  parent-switches per child, so a set of *k* co-adapted sites survives intact
  with probability 2^-(k-1). Co-adaptation cannot persist.
- **Chromosomal** *(current)*: the sites are already ordered layer by layer, and
  within a layer as query, key, value, output — which puts query beside key
  (they meet in the attention product) and value beside output (the output
  projection consumes the value-weighted heads), with layers in residual-stream
  order. That ordered list is cut into a few contiguous **chromosomes**, and
  each chromosome takes exactly **one** crossover point with a randomly chosen
  leading parent. Blocks of interacting sites are therefore inherited whole.

The number of chromosomes is the linkage dial: one chromosome means a single
cut across the entire genome, and setting it equal to the number of sites
reproduces uniform exactly. We currently use **three**.

**Then everything mutates.** After recombination, Gaussian noise is added to
every factor of every site, whether or not it changed parent. This is the only
source of new variation in the world — there is nothing else. The current
standard deviation is **0.001**.

*Swappable.* Mutation can be **additive** (noise added) or **multiplicative**
(noise scaled to existing magnitude). We use additive. The magnitude is the
single most consequential number in the project: too small and the population
converges and stops exploring; too large and inheritance is destroyed faster
than selection can accumulate anything. There is a critical rate, and running
above it means no information accumulates at all.

### What a birth costs

The child's adapter must be resident before the child exists, so its blocks are
reserved *before* it is born — and if the room cannot supply them, the birth
simply fails. The child arrives in its parents' room with an **empty context**
and a generation number one above the higher parent.

*Swappable.* Under **parental investment**, a parent holds its children's
adapter blocks until it dies, making reproduction a lasting cost to the parent
rather than a one-off charge on the room. It is currently **off**: it was tried,
and it suppressed reproduction rather than concentrating it.

---

## 7. Death

**The only cause of death is scarcity.** This is an invariant, not a tendency:
every death in the record carries a cause, and it is audited. If an agent could
die for any other reason — a crash, a timeout, an engine-side eviction the
world did not authorise — then the selection story would be unattributable and
the run would be worthless.

Death happens at exactly one moment: an agent needs a new block for its next
token, and the room's pool is empty. Somebody then dies. Which somebody is the
choice.

*Swappable, and this choice turned out to matter a great deal.*

- **Requester**: the agent whose growth hit the empty pool dies. This is the
  obvious rule and it is subtly perverse. An agent requests a block once per
  sixteen tokens of *throughput*, regardless of how much it already holds — so
  hazard tracks activity, not size. The typical dying agent was a third
  *smaller* than the room average, and a bloated agent externalised its cost:
  it raised everyone's hazard while bearing only a fraction of the harm.
- **Random holder** *(current)*: a block is chosen at random from those in use,
  and its owner dies. Hazard is then proportional to what an agent holds, which
  makes memory rent rather than a one-off purchase. The agent that dies is now
  about 40% *larger* than the room average — the relationship inverted — and
  populations under it are bigger, less bloated and more efficient.

Note that death is **content-blind** under both rules. Nothing inspects what an
agent said or whether it behaved well. The link between behaviour and survival
is entirely indirect: behaviour determines how fast an agent accumulates
context, and context is what kills.

When an agent dies its blocks are released, its adapter is unloaded, and its
soma is discarded. Other agents in the room are told it has gone — which costs
them tokens, like any other observation.

---

## 8. Movement

An agent moves by naming an adjacent room. Migration reserves the agent's full
footprint at the destination *before* releasing anything at the source, so an
agent is never in a state where it has left one room and cannot enter the next.
A move that cannot be satisfied fails, and the failure tells the agent which
rooms exist.

An agent carries its context and its age with it. Rooms run independently and
drift out of step with one another, which is why age is the agent's own clock
rather than a room's.

---

## 9. Founding and maintaining a population

Generation zero is seeded with agents whose genomes are **random**, drawn small
around zero. Both factors of each site are non-zero at initialisation. This is
deliberate and differs from standard practice: the usual low-rank
initialisation sets one factor to zero, which would make every founder
functionally identical to the base model and leave recombination with no
variation to work with.

Currently **32 founders per room**.

*Swappable.* A **refill** mechanism can admit fresh immigrants whenever a room's
population falls below a floor. It exists because early populations died out
before any lineage established, and it changed the question from "can 32
founders bootstrap" to "given continuous variation, does a self-sustaining
lineage emerge". It is currently **off**: once inheritance was working it proved
unnecessary, and it was contaminating generation zero by continually injecting
agents with no ancestry.

---

## 10. What is recorded

The world writes an event for everything that happens: every birth with its
parents and generation, every death with its cause and the agent's full
lifetime statistics, every action and every failed action, and periodic
snapshots of each room's occupancy.

Three things are recorded that are worth knowing about because they shape what
can be asked afterwards:

- **Genome fingerprints.** A full genome is tens of megabytes, so storing one
  per agent is impossible at population scale. Instead a compact per-site
  summary is written for **every** agent, which is enough to measure how far a
  genome has drifted and how much agents differ from each other.
- **Inheritance masks.** For each birth, one bit per site recording which
  parent actually supplied it. This is the act of inheritance itself; without
  it, descent can only be *estimated* by averaging over births that never
  happened.
- **Traces.** The raw text of a sample of turns, so that what agents actually
  emit can be read rather than only counted. Every other measurement classifies
  turns into categories chosen in advance, which structurally cannot reveal a
  failure mode nobody anticipated.

Nothing recorded is fed back into the world. **Measurement is not
optimisation** — no measured quantity influences who lives, who dies or who
reproduces.

---

## 11. Summary: what is swappable, and what we use

| component | alternatives | current | why |
|---|---|---|---|
| **Topology** | any graph; 40 sparse rooms | 4 rooms, complete | simplest thing with movement in it |
| **Genome coverage** | attention only; + MLP | attention only (112 sites) | every result so far is measured on it; MLP under test |
| **Rank** | any | 16 | never varied — an open question, and arguably the capacity of the whole method |
| **Recombination** | uniform; chromosomal | chromosomal, 3 chromosomes | uniform breaks co-adapted sites faster than selection can build them |
| **Mutation form** | additive; multiplicative | additive | simpler, and the measured comparison did not favour multiplicative |
| **Mutation size** | any | 0.001 | above roughly 0.0025 nothing accumulates; the lower bound is unexplored |
| **Eviction** | requester; random holder | random holder | requester made large agents *safer* than small ones |
| **Speech** | broadcast; directed | directed only | broadcast charges every listener and is lethal at scale |
| **Mate acceptance** | explicit verb; mutual proposal | mutual proposal | a separate verb attracted copying without comprehension |
| **Observation absorption** | per token; whole utterance | whole utterance | per-token made context growth independent of behaviour |
| **Reading** | one at a time; drain backlog | drain | keeps a backlog as one block of text rather than many fragments |
| **Reproduction cost** | free to parents; parental investment | free | investment suppressed reproduction rather than concentrating it |
| **Immigration** | refill on; off | off | unnecessary once inheritance works, and it contaminated generation zero |

---

## Reading on

- Why any of this is worth doing, and what would falsify it → [`VISION.md`](VISION.md)
- What we currently believe, and what is running → [`STATE.md`](STATE.md)
- How a finding travels from a run into a belief → [`PROCESS.md`](PROCESS.md)
- What each measurement means and how it has misled us → `src/evollm/analysis/metrics/`
