# Process

How findings get recorded, so that expensive runs are not wasted and no part of
the project goes stale.

`STATE.md` is the control room. This document is how it stays true.

---

## The documents and what each is for

| document | role | changes when |
|---|---|---|
| **`STATE.md`** | Control room. Everything we believe, how strongly, what is running, what we still do not know. Not chronological — a synthesis. | after every set of runs |
| **`runs/TIMELINE.md`** | Chronology. Every experiment in order, one line of outcome each. | when a run finishes |
| **`runs/<run>/NOTES.md`** | The analysis of one run. The only place numbers and argument live. | once, when that run is analysed |
| **`runs/<run>/ANALYSIS.txt`** | Machine output from `evollm analyse`. Never hand-edited. | generated |
| **`src/evollm/analysis/metrics/*.md`** | Theory. What a metric means, why we measure it, how it misleads. Independent of code. | when understanding changes, not when data does |
| **`src/evollm/analysis/README.md`** | The machinery. How to compute things and what the functions guard against. | when code changes |
| **`runs/README.md`** | Per-run index. | when a run is added |
| **`archive/*/README.md`** | Why some runs cannot be cited. | when a run is invalidated |
| **`configs/*.yaml` headers** | What a run is FOR and what would falsify it — written *before* it runs. | at submission |

The direction of flow is one way:

> config header (a commitment) → run → `ANALYSIS.txt` (machine) →
> `NOTES.md` (interpretation) → `TIMELINE.md` (chronology) → `STATE.md` (belief)

Beliefs are never edited from memory. They are edited from a `NOTES.md`.

---

## When a job is submitted

Write the config header first. It must state:

1. **What this changes**, and against which run it should be read.
2. **What we expect to learn** — the specific numbers expected to move.
3. **What would falsify it** — what result would mean the idea is wrong.

This is a commitment made before the data. It is the main defence against
reading a result as confirmation of whatever we hoped, and it has already caught
two cases where a change did the opposite of what was intended.

Add the job to the *Currently running* table in `STATE.md` **and**
`runs/TIMELINE.md`.

---

## When a job finishes

**Every finished job gets analysed. No exceptions**, including crashes — a run
that died at 20 minutes still produced data, and twice that data carried the
headline result.

1. **Run the machinery.**
   `evollm analyse runs/<run> --permutations 500 --out runs/<run>/ANALYSIS.txt --traits-csv runs/<run>/traits.csv`
2. **Write `NOTES.md`** from `runs/_TEMPLATE_NOTES.md`. Fill in every core
   metric, even where the answer is "not measured" — a visible gap is worth more
   than a silent one.
3. **Compare against exactly one run**, named in the header. A run compared
   against nothing establishes nothing.
4. **Move it out of *Currently running*** in both `STATE.md` and
   `TIMELINE.md`, into the timeline under the day it ended.
5. **Fold what it establishes into `STATE.md`**, with confidence set by what the
   run actually showed — not by what it was hoping to show.
6. **If it is invalid or truncated**, move it to `archive/` with a README saying
   why. Never delete a run.

---

## Core metrics: report these every time

Common across all runs so they stay comparable. Theory in
`src/evollm/analysis/metrics/`.

| | why it is on the list |
|---|---|
| **V_k** and **Ne**, with **1/(2Ne)** | whether selection can act at all. Nothing else means much if Ne is single digits. |
| **context at death ÷ room mean** | whether the block economy's hazard rises or falls with size |
| **effective lineages** (inverse Simpson) | whether any between-group test is meaningful |
| **h²** with the **midparent/single ratio** | whether transmission is genetic or environmental |
| **parent→child concordance** | transmission of discrete behaviour, at a depth the founder label cannot see |
| **canonical action rate** | whether output parses |
| **births per 1,000 room-steps** | reproduction, comparable across runs of different length |
| **drift from base** and **diversity between agents** | whether variation is being created or destroyed |
| **placeholder share of attempts** | whether the known degenerate action is spreading |

Plus the density-controlled generational trends for the behavioural traits.

---

## Standing rules

These exist because each was learned by getting it wrong.

- **Rooms desynchronise by design.** Never compare raw step counts. Compare at
  matched room-depth, or per 1,000 room-steps.
- **Generation correlates with time and density.** Any "improves across
  generations" claim must control for room population, or it is not a claim.
- **Agents within a room are not independent.** Bootstrap by room; permute
  within room.
- **State whether a rate is per-agent or per-attempt.** They can move in
  opposite directions when the mix of agents changes.
- **Rates with no denominator are NaN, not zero.**
- **Never label an agent by its generation-0 founder** and test whether that
  predicts behaviour. Use parent-offspring measures.
- **A single run cannot rank two schemes.** Early sample sizes ranked crossover
  wrongly; it took a controlled pair with thousands of observations to settle.
- **Truncation is never silent.** Record where a run stopped and what that
  biases.

---

## Keeping it from going stale

Check these when updating `STATE.md`:

- [ ] Every job in *Currently running* is actually running. Finished ones moved.
- [ ] Every run directory has a `NOTES.md`, or is in `archive/` with a reason.
- [ ] Every claim in `STATE.md` points at a document that exists.
- [ ] Confidence levels still match the evidence — a *likely* that has since been
      replicated should be promoted; one contradicted should be demoted or cut.
- [ ] *Holes in what we have already run* reflects what has actually been run.
- [ ] Superseded configs carry a `STATUS: SUPERSEDED` header so nobody
      resubmits them by accident.
- [ ] Metric documents match how the metrics are actually being computed.

## When a belief turns out to be wrong

Correct it in `STATE.md` and say so in the run's `NOTES.md` that overturned it.
Do not quietly delete the old claim — the fact that we believed it, and what
changed, is part of the record. Several conclusions here have been reversed by
later runs, and knowing which were reversed is how the confidence levels stay
honest.
