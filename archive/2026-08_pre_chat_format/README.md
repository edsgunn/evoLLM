# Archived runs: 2026-07 to 2026-08, pre chat-format

Every experimental run in this directory is **invalid as a measurement of the
hypothesis**. They are kept because they are a valid record of the environment
being debugged, and because several of the defects below were only found by
reading their logs.

Nothing here should be cited as evidence about whether selection over LoRA
initialisations produces in-context surprise minimisation. None of these runs
tested that. They tested a partly-broken environment driving base models that
were fed a prompt format they were never trained on.

---

## The invalidating defect

**Every model in every run was fed a bare token stream with no chat framing.**

The world built each agent's context by concatenating content and separating it
with `<|im_end|>`:

```
<system prompt><|im_end|><observation><|im_end|><action><|im_end|>...
```

Qwen2.5-Instruct — like every instruct model — is trained on exactly one
format, and it is not that one:

```
<|im_start|>system\n ...<|im_end|>\n<|im_start|>user\n ...<|im_end|>\n<|im_start|>assistant\n ...
```

There were no `<|im_start|>` markers and no role labels anywhere. The model was
never told which turn was its own, never placed in an assistant turn, and saw
`<|im_end|>` used as a bare delimiter rather than as a turn terminator. **Every
token every agent produced was conditioned on an out-of-distribution prompt.**

This is sufficient on its own to void the results, because the thing being
measured — whether a population can coordinate well enough to reproduce — is
downstream of the model's ability to follow the protocol at all, and that
ability was never fairly tested.

The symptoms were measured repeatedly and misattributed to the models:

| symptom | measured | I concluded (wrongly) |
|---|---|---|
| 41–64% of turns unparseable | every run | "the model can't follow a 4-verb protocol" |
| 3–16% of turns entirely empty | every run | "bimodal lock-in / silence attractor" |
| agents at 90–96% empty turns | 5987576, 5996591 | "degenerate lineages" |
| 1.5B far worse than 7B | 6040348 vs 6040349 | "model scale is the binding constraint" |

The last row is the most misleading: the 1.5B has the weakest format priors, so
it degraded most under a malformed prompt. The apparent scaling result is at
least partly a measurement of robustness-to-mis-prompting, not of capability.

### Why it went unnoticed for so long

**Nothing ever logged what an agent read.** There was per-turn instrumentation
for what agents *emitted* (`trace_turns`, added late), counters for actions,
forms, deaths, and births — all downstream of the model. There was no artefact
anywhere containing an agent's actual context.

A single dump of one raw context would have shown it immediately. This is now
fixed: `run.context_snapshot_every_steps` periodically writes verbatim contexts
to `runs/<name>/contexts/<room>/step_*.txt`, special tokens included, and a
test asserts `<|im_start|>system` appears in them.

---

## Other defects, and which runs carried them

Runs are listed oldest first. A defect is "live" in a run if it was fixed after
that run finished.

| # | defect | consequence | fixed after |
|---|---|---|---|
| 1 | `max_model_len` (8192) far below room capacity (~180k blocks) | agents hit the context ceiling before scarcity; **death by pool exhaustion was unreachable** | 5758770 |
| 2 | `numpy` upgraded to 2.5 inside the GPU venv by `uv run` | vLLM would not start (numba requires <2.3) | 5905491/2 |
| 3 | `EventLog` opened in append mode | three prechecks concatenated into one file; every summary aggregated all three, reporting 2 births for a run that had 0 | 5977421 |
| 4 | mate window armed at *enqueue*, not on receipt | requests expired while queued behind backlog. Shortest observed request→accept lag was **177 world steps against a 64-token window**: not one of 430 accepts could ever have been valid | 5987576 |
| 5 | observation wrapper `<from a12>` | taught the model that agent ids are followed by `>`; produced `<accept>a4></accept>` in **562 of 1586** unparseable turns | 5977421 |
| 6 | `<accept>` invented as a separate verb | §2.4 names only `<mate>`. The prompt named `<accept>` twice, and the model emitted it **7× more often than `<mate>` at 0.21% validity** | 6006472/3 |
| 7 | one-token-per-step absorption | context grew at exactly 1 token/step for everyone, so speech cost listeners **nothing** and lifetime was independent of behaviour (`context == age` for every agent) | 5996591 |
| 8 | `say` fan-out unbounded | one generated token became N−1 observation tokens; the observation queue diverged for any room >2 agents, reaching 17k–40k token backlogs | 5996591 |
| 9 | verb synonyms accepted by the parser | `<send>` resolved to `say` or `tell` by context; masked real protocol failures | 5999395/7 |
| 10 | prompt stated consequences (blocks, death, cost, inheritance) | spent 264 tokens teaching what selection could discover, and what it taught, agents copied | 6040348/9 |
| 11 | `max_action_tokens` cap (128) | a designer-chosen bound on deliberation; truncated turns mid-tag | 6040348/9 |
| 12 | lifetime measured as `room step − born_step` | rooms advance independently, so **30.5% of deaths recorded negative lifetimes** once agents migrated (20,320 moves in 6006472) | 6040348/9 |

---

## Run-by-run

| job | config | headline | defects live | verdict |
|---|---|---|---|---|
| 5758765 | throughput sweep | 1700 tok/s aggregate, saturates ~16 adapters, no cliff past `max_loras=8` | — | **usable**: hardware measurement, unaffected by prompt format |
| 5758770 | precheck | crashed | 1 | void |
| 5905491/2 | throughput, precheck | crashed | 2 | void |
| 5913973 | precheck say | 1 birth, 11 deaths | 3,4,5,6,7,8,9,10,11,12 | void |
| 5913974 | node_4room say | 1 child from 64 seeds; every room drained to one agent | 4,5,6,7,8,9,10,11,12 | void |
| 5977421 | precheck traced | 38.3% unparseable; failure families identified | 4,6,7,8,9,10,11,12 | **diagnostically useful**: the traced turns are what identified defects 5 and 6 |
| 5987576 | precheck | 0 valid accepts of 430 | 4,6,7,9,10,11,12 | void; identified defect 4 |
| 5996591/2 | precheck say | 0 births | 6,7,8,9,10,11,12 | void |
| 5999395/6 | precheck tell | 8 births, 19 deaths | 6,9,10,11,12 | void |
| 5999397/8 | precheck both | 1 birth | 6,9,10,11,12 | void |
| 6006472 | node_4room tell | self-sufficiency 0.059, flat; 171 descendant births | 6,10,11,12 | void; identified defect 12 |
| 6006473 | tell + token | self-sufficiency 0.058, flat; 115 births | 6,10,11,12 | void |
| 6040348 | tell + token 1.5B | self-sufficiency 0.077, flat; 152 births, max gen 3 | chat format | void |
| 6040349 | tell + token 7B | self-sufficiency 0.238, rising to 0.388; 342 births, max gen 6 | chat format | void — and the most misleading of the set, see below |

### On 6040349 specifically

This is the run that looked like a positive result: 7B reached self-sufficiency
0.238 against the 1.5B's 0.077, with 108 births at generation ≥3 versus 2, and
a final quartile of 0.388. It was tempting to read it as "model scale is the
binding constraint".

It is not safe to read it that way. Both models were mis-prompted; the larger
one simply tolerated it better. The comparison measures **robustness to a
malformed prompt**, which is correlated with capability but is not the quantity
of interest. Whether 7B still beats 1.5B under correct framing is an open
question that the reruns will answer.

Also note 145 of that run's 188 final-quartile births came from a single room
(gpu1) — one lineage, not four rooms independently taking off — and 10 of its
11 takeoff events lapsed immediately.

---

## What survives

- **The throughput sweep (5758765).** A hardware measurement: ~1700 tok/s
  aggregate per GH200, saturating around 16 concurrent adapters, with no cliff
  past `max_loras=8`. Independent of prompt format.
- **The death-cause audit.** Every death in every completed run was
  `pool_exhausted_requester`. The block economy and the §4.3 integrity
  guarantees behaved correctly throughout.
- **The failure taxonomy from 5977421.** The traced turns are verbatim model
  output and remain a valid record of what a mis-prompted Qwen2.5-1.5B emits.
  They are the basis of the parser's regression tests.
- **Every defect above.** These runs are why the environment now works.

## What to do with this directory

Keep it. Do not cite any self-sufficiency, birth-rate, generation-depth or
malformed-rate figure from it as a result. The reruns starting 2026-08 are the
first measurements taken in the model's trained format, and the first that bear
on the hypothesis at all.
