# State of the project

A living summary: what we believe, how strongly, what we are running, and what
we still do not know. **No numbers or derivations here** — every claim points at
the document that carries the evidence.

How this document is kept true is `PROCESS.md`. The theory behind the metrics it
cites is `src/evollm/analysis/metrics/`.

Last updated: 2026-08-26.

---

## Confidence key

| | meaning |
|---|---|
| **established** | multiple runs, or one run with a controlled contrast and adequate sample. Would be surprising to overturn. |
| **likely** | one clean measurement, or several suggestive ones. Believe it, but it could move. |
| **provisional** | one run, confounded, or small sample. Treat as a working assumption. |
| **suspected** | a reading of the data we have not tested directly. |

---

## What we have learned

### The environment had to be fixed before anything could be measured

- **The chat format is the single largest lever on behaviour.** Feeding a bare
  token stream instead of the model's trained `<|im_start|>` framing invalidated
  every run before 2026-08. — *established* — `archive/2026-08_pre_chat_format/README.md`
- **Consecutive observations must share one chat block.** A drained backlog
  rendered as a run of consecutive `user` turns with no assistant turn between
  them, a shape no instruct model is trained on. — *likely* — `tests/test_tools.py`
- **Agents should absorb a whole utterance at once, not one token per step.**
  Token-by-token absorption makes listening free and decouples survival from
  behaviour entirely. — *established* — `runs/node_4room_tell_token_7b_6045970/NOTES.md`
- **Draining the observation queue beats reading one utterance per turn.**
  Otherwise agents act on a room state that has already moved on. — *established*
  — `runs/node_4room_tell_7b_drain_6059671/NOTES.md`
- **Qwen2.5-1.5B cannot run this world; 7B can.** — *established* —
  `runs/node_4room_tell_drain_6059672/NOTES.md`
- **Placeholders in the prompt get copied verbatim and become a degenerate
  action.** Agents emit the literal example argument (`room_id`, `agent_id`) as
  a well-formed, canonical, always-failing action. — *established* —
  `runs/_figures/2_placeholder.png`, `runs/_figures/README.md`

### The mutation rate dominates everything else

- **There is a critical mutation rate separating collapse from takeoff.** Above
  it every configuration dies regardless of crossover scheme, refill policy or
  mutation operator; below it populations improve across generations. —
  *established* — `runs/_figures/1_divergence.png`, `runs/_figures/README.md`
- **σ = 0.005 and above is fatal.** — *established* — same
- **σ = 0.0025 is also above the threshold, it just takes longer to show.** Both
  σ=0.0025 arms reached zero births and stayed there. — *established* —
  `runs/_figures/README.md`
- **σ = 0.001 is the only rate tested that sustains a population.** — *likely*
  — `runs/node_4room_7b_chr001_6081313/ANALYSIS.txt`
- **Lower is not automatically better.** Mutation is the only source of new
  variation and selection erodes what it feeds on, so there is a floor
  somewhere below 0.001 that we have not found. — *suspected*
- **The damage is decorrelation per generation, not growth in adapter
  magnitude.** Multiplicative noise, which does not compound magnitude,
  degraded like additive noise. — *likely* —
  `runs/node_4room_7b_multmut_6071676/NOTES.md`

### Populations that survive do learn something

- **Reproduction rate rises across generations, and it survives controlling for
  room density.** Lifetime offspring *count* does not survive that control;
  offspring per unit of life does. — *established* —
  `runs/node_4room_7b_lowmut_6071675/figures/README.md`
- **Agents reallocate from moving to mating.** The largest behavioural effect
  measured, in every surviving run. — *established* —
  `runs/_figures/6_action_composition.png`
- **Output gets cleaner: fewer malformed actions, less pre-action rambling.** —
  *established* — `runs/node_4room_7b_lowmut_6071675/figures/7_output_cleanliness.png`
- **Per-attempt success rates do not improve, and mostly fall.** Populations win
  on allocation and volume, not accuracy. — *likely* —
  `runs/node_4room_7b_lowmut_6071675/figures/8_efficacy_falling.png`
- **Agents live shorter lives as generations pass.** — *established* —
  `runs/node_4room_7b_lowmut_6071675/figures/3_median_lifetime.png`
- **The improvement is not an artefact of shorter lives.** It survives
  stratification by turn count. — *established* — same directory
- **Nothing has been shown to beat the base model.** Generation 0 is the base
  model plus a random perturbation, so beating it is not the same claim. The
  base-rate precheck has never been run under the current environment. —
  *established as a gap*

### Inheritance

- **Chromosomal crossover transmits phenotype more faithfully than uniform.**
  Established only on heritability, in a controlled pair at adequate sample
  size; it did not clearly win on outcomes. — *likely* —
  `runs/node_4room_7b_uni0025_6081397/NOTES.md`
- **One chromosome is too few.** Tightest linkage loses diversity fastest,
  because the genome travels as a single unit and recombination cannot break up
  a sweep. — *likely* — `runs/node_4room_7b_chr0025_c1_6081314/NOTES.md`
- **Strategy is heritable at parent-to-child range.** — *established* —
  `src/evollm/analysis/README.md` (see the lineage-label section)
- **Founder-level lineage labels are useless at depth.** By generation 200
  everyone shares the same founders, so a test built on them reports "not
  heritable" however strongly parents resemble children. — *established* — same
- **A low h² does not always mean weak transmission.** Where a trait sits near
  its ceiling — canonical rate is 97.7% in the reference run — phenotypic
  variance is compressed and h² falls even though the trait has been fixed *by*
  selection. Read h² next to the trait's spread, never alone. — *established* —
  `runs/node_4room_7b_chr001_evict_6127798/NOTES.md`
- **Genome diversity stays tiny.** Agents differ from each other by a small
  fraction of their own distance from the base model. — *likely* —
  `runs/node_4room_7b_chr001_6081313/NOTES.md`

### Selection is weak, and we know why

- **Effective population size is tiny — single digits — because the variance in
  offspring number is enormous.** The census population is fine; V_k is what
  destroys Ne. — *established* —
  `runs/node_4room_7b_chr001_evict_6127798/NOTES.md`
- **Selection is therefore only about twice as strong as drift.** A behaviour
  costing a third of an agent's reproduction can still sweep. — *established* —
  same
- **Reproductive success is driven by mating *rate*, not lifespan.** — *likely*
- **The degenerate placeholder action is transmitted horizontally, not
  genetically.** A same-room stranger predicts it nearly as well as a parent, so
  selection on genomes cannot clear it. — *likely*

### The block economy had a perverse incentive

- **Under `requester` eviction, hazard tracked throughput rather than holdings,
  so large agents were SAFER than small ones and bloat was externalised.** —
  *established* — `archive/2026-08_crashed_eviction/README.md`
- **`random_holder` eviction inverts that.** Bigger populations, smaller
  contexts, higher canonical rate, lower V_k, higher Ne. Births per room-step
  are marginally lower over the whole run, though higher at matched depth, so
  it is not a clean win on every axis. — *likely* —
  `runs/node_4room_7b_chr001_evict_6127798/`
- **Charging parents for their children's adapters adds nothing beyond the
  eviction fix.** It suppressed reproduction rather than concentrating it, and
  left Ne worse than eviction alone. — *likely* —
  `runs/node_4room_7b_chr001_invest_6127799/`
- **Unexplained: the reproduction charge anticorrelates parent and child
  strategy** (z = −21.4). Possibly an artefact of parents becoming large holders
  and so eviction targets under `random_holder`. Measure before reusing any
  reproduction charge. — *open question* —
  `runs/node_4room_7b_chr001_invest_6127799/NOTES.md`
- **Immigration (refill) is unnecessary once inheritance works, and was
  contaminating generation 0.** — *established* —
  `runs/node_4room_7b_norefill_6070759/NOTES.md`

### Structure and communication

- **Broadcast speech kills a dense room.** Every listener pays, so at ~60 agents
  per room one utterance costs the room sixty times a directed message. —
  *established* — `archive/2026-08_say_collapse/README.md`
- **Directed speech is abandoned by every surviving population.** It falls to a
  fraction of a percent of all actions. — *established* —
  `runs/_figures/6_action_composition.png`
- **No niches have formed.** Distinct strategy clusters exist in every run, but
  lineage does not predict which one an agent lands in, and every population is
  effectively panmictic. — *established* — `runs/_figures/README.md`
- **No gene-behaviour association replicates across runs.** Within a run, sites
  associate with behaviour and replicate across rooms; across runs the sites
  differ, which is what founder-specific effects look like. — *likely* — same

---

## What we are running

| job | what it changes | the question |
|---|---|---|
| `chr0025_evict` 6146539 | σ=0.0025 with `random_holder` | Does size-proportional hazard rescue the regime that has the diversity but died of bloat? This is the arm where the eviction fix has most to prove. |
| `mlp` 6146538 | genome adapts the MLP as well as attention; capacity derived from the engine | **The first run where what the model *computes* is heritable, not just what it attends to.** Attention governs routing; the MLP is where per-position computation and the nonlinearity live. Every run so far has searched re-mixings of a fixed feature set. Does a qualitatively larger function class produce qualitatively different behaviour? |
| `villages` 6146540 | 40 rooms of 10 agents, `say`, clustered topology | Is speech worth doing when it is affordable? Does a metapopulation let anything local emerge? |
| `baserate` 6146533 | zero-genome agents (the frozen base model) | **Closes the largest gap.** Every comparison so far has been internal, so "takeoff" means beating a *randomly perturbed* base model rather than the model itself. |
| `braced` 6143016 | prompt slots written `{room}` not `room_id` | Is the degenerate placeholder action a prompt-design artefact, or are agents reaching for *any* cheap always-failing action? |

Each config states its own expected outcome and what would falsify it, in the
header. Read those before reading the results.

**When one of these finishes**, follow `PROCESS.md`: analyse it, write its
`NOTES.md`, move it out of this table into `runs/TIMELINE.md` under the day it
ended, and fold whatever it establishes into *What we have learned* above — with
its confidence set by what the run actually showed, not by what it was hoping to
show.

---

## What we want to learn

### Blind spots in what we measure

- ~~**Surprise has never been measured.**~~ **Now instrumented, never yet run.**
  Surprise is recorded over the tokens the *world* wrote into an agent's
  context — whether the environment became predictable to it — bucketed by
  position in the agent's own life and carried on every death event. The
  agent's surprise at its own output is kept separately as the fluency
  control. Off by default until one run proves the GPU path. —
  `src/evollm/analysis/metrics/surprise.md`
- ~~**Nothing measures change within an agent's lifetime.**~~ **Now measured
  two ways** (`analysis.lifecourse`): observation surprise from death records,
  and canonical rate by within-life quantile from traces, both paired within
  agent. The behavioural read works on every run ever done. What it says so
  far: **no within-life improvement anywhere, and a significant decline in the
  reference run** (−0.74 pp ± 0.41, n=2,445, paired). If that survives the
  surprise measurement, all measured improvement lives in the initialisation
  rather than in the context. — *likely* —
  `runs/node_4room_7b_chr001_evict_6127798/NOTES.md`
- ~~**No analysis reads what agents say.**~~ **Now read on every run**
  (`evollm inspect-traces`). It found two things nothing else could — see
  below. Still unread: whether a reply is *conditioned on* what was received,
  which needs a model, not string matching (`inspect-traces --bundle`).

### What reading the traces found

- **Communication has effectively collapsed, in every run.** `tell` is 0.3-2%
  of all actions; `say` is ~0%. In the reference run: 1,000 tells against
  173,988 mate requests and 180,003 move attempts. Agents move and court; they
  do not talk. This is not a trace artefact — it holds over the full event
  stream. — *established* — `evollm inspect-traces`
- **A quarter to a third of agents are stuck.** 1,069 of 3,671 traced agents in
  the reference run emitted the *identical* turn for 80%+ of their traced life,
  usually a single `<go>`. Every parse-based metric scores these as canonical
  well-formed actions, so they have been counted as healthy behaviour
  throughout. — *established* — same
- **The placeholder tic is not confined to `go`.** `<mate>sender_id</mate>` and
  `<mate>agent_id</mate>` are 2.4% of turns in the reference run. Analyses that
  counted only move targets undercounted it. — *established* — same
- **Most moves fail.** 98,471 failed against 81,532 succeeded in the reference
  run — 55% of attempts. — *established* — same

### Surprise recording: works, and costs 2.4x throughput

**It works.** The startup probe passed on every engine of every job, every
death event carried surprise, and ~80% carried a multi-bucket within-life
curve. Observation surprise (0.516) runs about **six times** the agent's
surprise at its own output (0.086), which is why scoring only the tokens the
world wrote was the right call: a whole-context average would have measured
near-deterministic self-prediction. — *established* —
`archive/2026-08_surprise_overhead/README.md`

**It costs about 2.4x throughput** — 0.7 steps/sec against the reference run's
1.7, at matched context and matched population, flat across context. Twelve
hours therefore buys roughly 40% of the usual depth. The predicted cost was a
logits matmul over the uncached prompt suffix; what was missed is that vLLM
allocates and pythonizes the prompt-logprob tensor across the *whole* prompt
every turn. — *established* — same

Consequence: surprise now runs on **`braced` only** (6143016), whose question
is settled in the first few thousand steps. `mlp`, `chr0025_evict` and
`villages` were cancelled 2.7h in and resubmitted without it, because all three
are decided by depth. Whole-life tracing was kept everywhere; it is free.

### Holes in what we have already run

- **We have never measured the base model.** The first attempt (6143004)
  produced no data: a four-room config was submitted to a one-GPU precheck, the
  second room's engine asked NVML for an unallocated device, and the job then
  sat to its four-hour limit. Retrying as 6146533 with four GPUs.
  Until it reports, every "takeoff" claim means beating a *randomly perturbed*
  base model, not the model itself.
- **The critical mutation rate is bracketed between 0.001 and 0.0025 and has
  never been localised**, nor do we know what it depends on — population size,
  genome size, adapter rank, or the base model. **Genome size is the live
  candidate**: σ is per-element, so a genome with more sites accumulates more
  total functional displacement per generation at the same σ, and the threshold
  may sit lower for it.
- **The lower bound on useful mutation is unknown.** Nothing below 0.001 has run.
- **Chromosome count is barely explored.** Only 1, 3 and 112 (uniform), all at
  σ=0.0025 where the population dies anyway. 7 and 14 were proposed and never
  submitted; the informative place for them is σ=0.001.
- **No run has been longer than 12 hours.** Every result is a snapshot of an
  early transient — the deepest reached ~220 generations, and the interesting
  claims are about thousands.
- **Rank has never been varied.** Rank 16 throughout, unexamined.
- **Adapter magnitude at initialisation has never been varied.** `init_scale`
  0.02 throughout.

### Open questions

- **Can selection be made strong enough to matter?** The concrete target is V_k,
  since that is what pins Ne. Eviction helps; a reproduction charge did not.
  Nothing else has been tried.
- **Does communication ever become worth doing?** The hypothesis is that talking
  only pays once there is something to coordinate about — mate choice, local
  convention — and that requires small rooms and population structure. Being
  tested now.
- **Can niches form at all in this world?** Requires structure that has never
  existed until the villages run.
- **Is the degenerate-action problem solvable by prompt design?** Being tested
  now (6143016). If the tic reappears under a new literal, agents are reaching
  for *any* cheap always-failing action and the real problem is that a failed
  move is too cheap.
- **Does takeoff continue or plateau?** Every surviving run was still improving
  when its clock ran out.
- **Should chromosome count, rank, or mutation rate be evolvable rather than
  configured?** Currently all three are fixed by hand.

---

## Reading order

1. `runs/TIMELINE.md` — every experiment in order, and what each one settled
2. `runs/README.md` — index of every run and what each family settled
3. `runs/_figures/README.md` — the cross-run comparisons and figures
4. `runs/node_4room_7b_lowmut_6071675/figures/README.md` — the first takeoff,
   analysed in depth
5. `src/evollm/analysis/README.md` — the analysis machinery, and the
   methodological traps it exists to avoid
6. `archive/*/README.md` — runs that are invalid or truncated, and why

## Standing caveats

- Rooms desynchronise by design, so raw step counts are not comparable across
  rooms or runs. Compare at matched room-depth.
- Generation correlates strongly with wall-clock time and with room density.
  Any claim about "improvement across generations" must control for density.
- Agents within a room are not independent. Bootstrap by room, permute within
  room.
- Several runs were truncated by the preemption guard or the 12-hour limit.
  Truncation is recorded in each run's notes; it is never silent.
