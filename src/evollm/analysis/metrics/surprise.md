# Surprise

The project's hypothesis is that agents come to find their world less
surprising, and that they act to make it so. For a long time nothing measured
it. Every metric was behavioural — did the turn parse, did the move land, did
a child result — which are all consequences of prediction, not prediction.

## What is measured

**Observation surprise** is the mean negative log-probability, under an
agent's own adapted weights, of the tokens **the world wrote into its
context**: what other agents said to it, who arrived, who left, what its
last action did.

```
        1
NLL  =  ─  Σ  −log P(token_i | everything before it)
        n  i∈obs
```

It is measured in nats per token. Lower means the world is more predictable to
this agent. Two ways for it to fall, and they are *both* the hypothesis:

1. The agent gets better at predicting a fixed world.
2. The agent acts so that the world becomes more predictable — settling in one
   room, keeping the same neighbours, provoking fewer surprises.

Nothing here distinguishes them, and that is deliberate: both are surprise
minimisation, and separating them needs an intervention, not a statistic.

## Why observations and not the agent's own tokens

Surprise averaged over a whole context is dominated by the agent's own output,
because agents generate far more than they read and a model is very confident
about its own continuations. That number measures fluency. It would fall
whenever a lineage became more stereotyped — which is degeneracy, the opposite
of what we are looking for — and it would be lowest for exactly the stuck
agents that repeat one turn until they die.

So the origin of every context token is recorded as it is appended
(`ORIGIN_OBS`, `ORIGIN_GEN`, `ORIGIN_FRAME`), and only observation positions
count. Chat framing is excluded too: it is boilerplate the model predicts
almost perfectly, and it would dilute the signal toward zero.

**The agent's own surprise is still recorded, separately, as `gen_nll`.** It
is the control. A fall in observation surprise means much less if the same
lineage also became more confident about everything it writes.

## Why it is nearly free

Getting a logprob for a token requires logits at that position. For generated
tokens the log-softmax is computed for sampling anyway, so asking for the value
costs payload, not compute.

For prompt tokens it would normally mean recomputing logits across the whole
context. It does not here, because vLLM computes prompt logprobs only from
`request.num_computed_tokens` onward — the part not served by the prefix
cache. Each turn's cached prefix ends where the agent last spoke, so the
uncached suffix is *exactly the observations absorbed since*. Those hidden
states must be computed for the prefill regardless; the extra work is a logits
matmul over a few dozen positions.

### The trap in that

vLLM allocates the prompt-logprob tensor for the whole prompt with
`torch.empty` and fills only the recomputed slice — then emits a `Logprob`
object for **every** position. Cached positions come back as *uninitialised
memory*, not as `None`. Read naively they are plausible-looking floats that
would pass every downstream sanity check.

Three independent guards, in `_flatten_prompt_logprobs` and
`_absorb_prompt_surprise`:

1. **Token-id match** — each row is keyed by token id; a row whose key is not
   the token actually at that position was never written.
2. **Range** — a log-probability lies in `[-40, 0]`.
3. **Lower bound** — only positions above the previous turn's prompt length
   are scored, so each observation is counted exactly once and no cached
   position is consulted at all.

## Within a lifetime, not just across them

A single lifetime average cannot answer the question. Surprise is accumulated
into buckets by how far into its own life the agent was — turns 0-4, 5-9,
10-19, 20-39, 40-79, 80+ — and the whole curve rides on the **death event**,
so it exists for every agent that ever lived rather than for the traced sample.

Read `analysis.surprise_curve`, and read the **paired within-agent change**,
not the population curve. The population curve is contaminated by
survivorship: agents that reach turn 80 are the ones that were good enough to
get there, so a flat learner population still produces a falling curve if the
high-surprise agents die younger. Differencing an agent against itself removes
that. `tests/test_lifecourse.py` pins exactly this case.

Two further slopes, which are different claims and must not be conflated:

- **starting surprise vs generation** — do later generations begin better?
  That is inheritance of a prior.
- **within-life change vs generation** — do later generations *learn faster*?
  That is inheritance of an ability to adapt.

## Reading it

| observation | reading |
|---|---|
| paired change < 0, `gen_nll` flat | the world became more predictable within a life — the hypothesis |
| paired change < 0, `gen_nll` also < 0 | the lineage became more stereotyped; check the stuck-agent rate in `inspect-traces` before claiming anything |
| paired change ≈ 0, starting level falls with generation | no in-context learning; improvement lives in the weights |
| paired change > 0 | the world gets *less* predictable as an agent lives — expected if agents accumulate neighbours and contexts they cannot model |

## Cost and configuration

`run.record_surprise`, **default off**. It defaults off because the
prompt-logprob path is unproven on GPU and enabling it by default would switch
an untested path on in every queued job at once. Turn it on per config.

Incompatible with `--kv-sharing-fast-prefill`, which vLLM says produces
incorrect prompt logprobs. We do not use it.

## See also

- `behaviour.md` — the behavioural read on the same question, which works on
  every run ever done, including those predating this
- `evollm eval-surprise` — the offline complement: scores population snapshots
  against held-out streams. That answers "is this genome better at this world",
  a between-agent question. This answers "did this agent's world become
  predictable to it", a within-agent one.
