# What kills an agent, and whether that creates a selection pressure

Death is the only currency of selection here. If death is uncorrelated with
behaviour, selection can act *only* through differential reproduction.

## The mechanism

An agent dies when it needs a new KV block and the room's pool is empty. Nothing
inspects what it said. That is deliberate — a content-blind economy is what
makes any adaptation an emergent result rather than a reward we designed.

The consequence is that **who dies depends entirely on the eviction policy**,
and that policy turns out to be one of the most consequential settings in the
system.

## Eviction policies

| policy | victim | hazard scales with |
|---|---|---|
| `requester` | the agent whose request could not be met | **throughput** |
| `random_holder` | a random held *block*'s owner | **holdings** |

Under `requester`, you die when you ask and the pool is empty. You ask once per
`block_size` tokens of throughput, whatever you already hold. So a young agent
draining a backlog asks constantly and dies; a bloated agent that has emptied
its backlog and generates slowly asks rarely and survives.

**This inverts the pressure you want.** Measured across tens of thousands of
deaths, the typical dying agent held well *under* the room's mean context: big
agents were safer than small ones.

It also makes bloat an externality. Holding a huge context raises every agent's
hazard by filling the room, but the holder bears only 1/N of the harm it causes
— which is how runs ended with two or three agents sitting on enormous contexts
in a pool that was not even full, nobody dying and nobody able to be born.

`random_holder` draws the victim in proportion to blocks held, so holdings
become continuous exposure — rent expressed as hazard, with no new currency and
no timescale to choose.

## The headline diagnostic

> **context at death ÷ the room's mean context at that moment**

- **< 1** — the dying agent was smaller than average. Hazard is *falling* with
  size. Expect bloat.
- **≈ 1** — death is size-neutral.
- **> 1** — hazard rises with size, which is what an ecology should do.

Report this for every run. It is a single number that says whether the economy
is working.

## Selection coefficient

For any behaviour, `s` = relative fitness of agents that have it, minus one.
Compare |s| against the drift threshold 1/(2Ne) — see
`effective-population-size.md`. A behaviour can be strongly deleterious and
still sweep if that ratio is close to 1.

## Pitfalls

- **Death being content-blind cuts both ways.** Under `random_holder` an agent
  can die for being large having done nothing wrong. Watch canonical rate and
  reproduction to check the policy is not destroying competent agents faster
  than it clears bloat.
- **Deaths falling is not the same as a population being healthy.** In a
  collapse, deaths fall *with* births. Always report both.
- **A death spike and a birth collapse look similar in a population curve** and
  are completely different. Check which moved first.
