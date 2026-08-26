# Behavioural traits: what agents do, and whether it is getting better

## The trait groups

Kept separate because they answer different questions, and mixing them is how
you end up correlating something with itself.

| group | asks | traits |
|---|---|---|
| **form** | did the output parse at all? | canonical rate, malformed share |
| **strategy** | of the actions taken, *which*? | mate / move / tell / no-action shares |
| **efficacy** | of the actions taken, which *landed*? | delivery, reciprocation, move success |
| **economy** | what did it cost? | tokens per turn, thinking share, tokens observed |
| **fitness** | children, children per unit of life |

An agent can be perfectly well-formed and still address agents who are not
there. That shows up in efficacy and nowhere else, and it has been the dominant
failure mode.

## The confound that must always be controlled

**Generation is correlated with wall-clock time and with room density.** Room
populations grow over a run, and a fuller room means more available mates. So
"later generations reproduce more" can be nothing more than "later agents lived
in fuller rooms".

The standard control: regress the trait on generation after removing room
identity and room population at birth, then permute generation *within* (room ×
population decile) so the null holds density fixed. Report the **partial
correlation** and the permutation p.

This is not optional. Lifetime offspring *count* does not survive it; offspring
per unit of life does. Reporting the uncontrolled version would have claimed a
result that is not there.

## Rates with no denominator are NaN, never zero

An agent that never tried to mate has an *undefined* success rate, not a zero
one. Averaging zeros in understates every cohort it appears in.

## Per-attempt versus per-agent

These can disagree, and have. Averaging a rate across agents weights every agent
equally; pooling across attempts weights prolific agents more. When the mix of
agents changes over a run — which it does — the two can move in opposite
directions.

**Report which one you used.** A claim that mate reciprocation fell was retracted
once the per-attempt version showed it rising.

## Degenerate actions

Watch for behaviours that are well-formed, canonical, and useless. The known one
is agents emitting a prompt's literal placeholder argument as a destination or
target: it parses, it is counted as canonical, and it always fails.

Its signature is **efficacy collapsing while form stays high**. Track invalid
targets separately from malformed syntax, or the two get conflated.

## Pitfalls

- **Short-lived agents score better on rate traits** (r ≈ −0.3 with turns
  lived). Any cohort comparison must stratify by turn count.
- **Agents alive at a truncated run's end have no death record** and are
  excluded, which biases the last generations.
- **Composition shares must be renormalised** before comparing across runs where
  the available tools differ.
