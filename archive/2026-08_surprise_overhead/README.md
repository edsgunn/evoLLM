# 2026-08-26 — truncated: surprise recording overhead

Three arms (`mlp` 6141167, `chr0025_evict` 6141384, `villages` 6141411)
cancelled 2.7 hours into 12, and resubmitted without surprise recording as
6146538, 6146539 and 6146540.

**These are not experimental results.** Each holds under three hours of a
twelve-hour arm and reached only a few thousand world steps. They are kept for
one reason: they are the evidence that surprise recording works, and the
measurement of what it costs.

## What they establish

**The instrumentation is correct end to end.** The startup probe passed on
every engine of every job (`3/4 probe positions scored`, 3 being the maximum —
position 0 has no context). Every death event carried surprise: 369 of 369 in
`chr0025_evict`, 153 of 153 in `mlp`. Roughly 80% carried a multi-bucket
within-life curve.

**Observation surprise and self-surprise are far apart**, which vindicates
scoring only the tokens the world wrote:

| | mean |
|---|---|
| `obs_nll` — what the world wrote | 0.516 |
| `gen_nll` — the agent's own output | 0.086 |

Agents are about six times more surprised by their world than by themselves.
Averaging over a whole context, as originally intended, would have been
dominated by near-deterministic self-prediction and would have measured
fluency rather than whether the world became predictable.

**It costs about 2.4x throughput.** Against the reference run 6127798 at
matched context and matched population:

| mean context | reference (no surprise) | braced 6143016 (surprise) |
|---|---|---|
| 8–11k | 1.65 steps/s (74 agents) | 0.73 steps/s (68 agents) |
| 11–14k | 1.69 steps/s (60 agents) | 0.69 steps/s (55 agents) |

Flat across context, so it is close to a constant factor rather than something
that worsens as agents grow.

The predicted cost was "a logits matmul over a few dozen positions", on the
grounds that vLLM computes prompt logprobs only for tokens the prefix cache did
not cover. That part is true. What was missed is that vLLM still allocates the
prompt-logprob tensor for the WHOLE prompt and pythonizes every position of it
on the turn the prefill completes — the same full-prompt handling that produces
the uninitialised-memory trap the guards defend against. It was read as a
correctness problem and not as a cost.

## Why these three and not `braced`

At 2.4x, twelve hours buys roughly 40% of the usual depth. All three of these
arms are decided by depth — `chr0025_evict` asks whether eviction rescues a
regime whose collapse only appeared past step 80,000, and `mlp` and `villages`
are about generation depth. `braced` asks about a prompt tic visible in the
first few thousand steps, so it keeps surprise recording and carries the
measurement for this round.
