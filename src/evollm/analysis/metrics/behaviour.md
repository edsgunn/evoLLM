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

The literal to count depends on `world.prompt_placeholders`. Identifier-style
runs produce `room_id`, `agent_id`, `sender_id`, `your_id`; braced runs produce
`room`, `agent`, `sender`, `you`, because the parser strips the braces. Count
both sets, or a braced run will look clean when it is not.

## What we do NOT currently measure

Two gaps, both load-bearing, both recorded here so they stay visible.

**Within-lifetime change.** Every trait above is a *lifetime aggregate* — one
number per agent. Nothing asks whether an agent gets better as its own context
accumulates. That is a strange gap for a project about *in-context* surprise
minimisation, and the first look at it (canonical rate by position in an
agent's own life, paired first-fifth against last-fifth) found **no
improvement, and in one run a small significant decline**. If that holds, all
the improvement we have measured is in the initialisation, not in the context —
which is a claim about what this substrate does, and it deserves a proper
measurement rather than a by-product of one.

Note the sampling bias: traced turns are capped by `run.trace_turns` per room,
so they come from early in a run.

**Message content.** Runs log the raw text of tens of thousands of turns, and
no analysis has ever read it. We measure whether an action *parsed* and whether
it *landed*, never what was in it. For a project whose goal includes agents
talking to choose mates, that is the whole phenomenon going unexamined. Related
and also unmeasured: whether a reply is conditioned on what was received —
conversation structure, as opposed to message counts.

## Pitfalls

- **Short-lived agents score better on rate traits** (r ≈ −0.3 with turns
  lived). Any cohort comparison must stratify by turn count.
- **Agents alive at a truncated run's end have no death record** and are
  excluded, which biases the last generations.
- **Composition shares must be renormalised** before comparing across runs where
  the available tools differ.
