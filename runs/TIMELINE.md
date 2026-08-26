# Timeline

Every experiment, in order, with what it settled. One line of overview each —
detail lives in the run's own `NOTES.md` or `ANALYSIS.txt`, and the current
state of the project is in `../STATE.md`.

**Maintenance.** The full procedure is `../PROCESS.md`. In short: every finished
job is analysed, gets a `NOTES.md`, moves out of *What we are running* into this
file under the day it ended, and has whatever it establishes folded into
`STATE.md` — with its confidence set by what the run actually showed.

---

## 2026-07-23 → 2026-08-13 · bring-up

Roughly three weeks of installing vLLM on Isambard, getting multi-LoRA serving
to work on aarch64, and failing prechecks. Nothing here is an experiment.

| date | job | title | outcome |
|---|---|---|---|
| 07-23 | 5751250 | environment install | vLLM 0.15.1 built for GH200 |
| 07-24 → 08-13 | various | handshake prechecks | Ten consecutive failures. The measurable ones exposed the triton `-Wno-psabi` toolchain break, a corrupted GPU venv, and an event log that appended three prechecks into one file so every summary aggregated all three. |
| 08-05 | 5913974 | first full run | Died on the context-ceiling invariant: `max_model_len` 8192 against 180k-block rooms meant death by scarcity was unreachable. |

## 2026-08-13 → 08-18 · first runs, all invalid

| date | job | title | outcome |
|---|---|---|---|
| 08-13/14 | 6006472/3 | first 4-room runs | Ran, produced data, and are uninterpretable. |
| 08-18 | 6040348/9 | token-absorption variants | Same. |

**All superseded.** Every context in these runs was a bare token stream with no
`<|im_start|>` chat framing — a format no instruct model is trained on. See
`../archive/2026-08_pre_chat_format/README.md`.

## 2026-08-18 → 08-19 · tool and absorption variants (family A)

First runs after the chat-format fix. Six arms varying which tools exist and
how observations are absorbed.

| date | job | title | outcome |
|---|---|---|---|
| 08-18 | 6045965 | broadcast `say`, 1.5B | 1.5B cannot run this world. |
| 08-18 | 6045967 | `say` + `tell`, 1.5B | More tools do not help a model that cannot use one. |
| 08-18 | 6045968 | token absorption, 1.5B | Token-by-token listening is free, which decouples survival from behaviour. |
| 08-19 | 6045966 | directed `tell`, 1.5B | Directed speech is strictly harder than broadcast. |
| 08-19 | 6045969 | chat format OFF, 1.5B | Null result — but only because 1.5B was already failing for a more basic reason. |
| 08-19 | 6045970 | token absorption, **7B** | **Settled the model size.** 7B produces ~72% canonical turns against 15% at 1.5B. |

**Settled:** use 7B; use whole-utterance absorption; use `[tell, mate, go]`.

## 2026-08-19 → 08-20 · read policy (family B)

| date | job | title | outcome |
|---|---|---|---|
| 08-19 | 6059672 | drain, 1.5B | Worst canonical rate of any run. Closed the model-size question for good. |
| 08-20 | 6059339 | read one utterance per turn, 7B | Agents act on a room state that has already moved on. |
| 08-20 | 6059671 | **drain the queue, 7B** | Drain wins. Became the analytical workhorse: reproduction degrades across generations, descendants are coherent but stop *landing* actions, mutation accumulates as a random walk, and heritability is indistinguishable from zero. |

**Settled:** drain. And the first real problem was identified — something was
destroying inheritance.

## 2026-08-20 → 08-21 · refill sweep (family C)

Four arms varying the free-block threshold that triggers immigration. Rooms
grew to a uniform 48,000 blocks.

| date | job | title | outcome |
|---|---|---|---|
| 08-20 | 6070756 | refill at 60% free | |
| 08-21 | 6070757 | refill at 40% free | The shared baseline for everything after. Shows bloat collapse most clearly. |
| 08-21 | 6070758 | refill at 20% free | Denser populations, same degradation. |
| 08-20 | 6070759 | refill OFF | Weakest population — but proved rooms never actually go extinct. |

**Settled:** the refill threshold does not determine population fate.
Immigrants were *propping populations up*, not crowding them out. This is why
refill was later turned off everywhere.

## 2026-08-21 · genome operators (family D)

| date | job | title | outcome |
|---|---|---|---|
| 08-21 | 6071674 | chromosomal crossover | Inheritance became measurably more faithful — and it did not save the run. |
| 08-21 | 6071675 | **σ = 0.0025 (`lowmut`)** | **The run where the project turned.** First takeoff: reproduction climbing across 184 generations, 99.7% self-sufficient, no bloat collapse. |
| 08-21 | 6071676 | multiplicative mutation | The informative negative: it degraded like additive noise, so the damage is decorrelation per generation, not growth in adapter magnitude. |

**Settled:** the mutation rate dominates crossover scheme, refill policy and
mutation operator. σ = 0.01 is above a critical threshold.

## 2026-08-21 → 08-22 · σ sweep on chromosomal crossover

Five arms, refill off, merged observation framing, genome fingerprints and
inheritance masks recorded for the first time.

| date | job | title | outcome |
|---|---|---|---|
| 08-21 | 6081312 | σ = 0.005 | Above the threshold. Bracketed it to (0.0025, 0.005). |
| 08-21 | 6081314 | 1 chromosome | Tightest linkage loses diversity fastest — the genome travels as one unit. |
| 08-22 | 6081311 | σ = 0.0025, chromosomal | Collapsed to zero births by step 80,000. |
| 08-22 | 6081397 | σ = 0.0025, uniform | Same, slightly later. The controlled pair for the crossover question. |
| 08-22 | 6081313 | **σ = 0.001** | The only sustained takeoff. Still accelerating when the clock ran out. |

**Settled:** σ = 0.0025 is *also* above the threshold. Chromosomal crossover
transmits phenotype more faithfully than uniform. The `room_id` placeholder tic
was identified as a degenerate, always-failing, well-formed action that sweeps
populations — and σ = 0.001 never acquires it.

## 2026-08-25 · eviction, first attempt

| date | job | title | outcome |
|---|---|---|---|
| 08-25 | 6118397 | random_holder eviction | Crashed at 21 min. |
| 08-25 | 6118523 | random_holder + parental investment | Crashed at 55 min. |
| 08-25 | 6129547 | MLP genome | Engine never started. |

Both eviction runs died on a latent bug: `random_holder` had been in the config
since the beginning and **no run had ever used it**, so nothing exercised the
path where the drawn victim is a newborn or migrant not yet in the room. The
MLP run's memory bound ignored the model weights. See
`../archive/2026-08_crashed_eviction/README.md` and
`../archive/2026-08_say_collapse/README.md`.

Partial data still showed the headline inversion: the dying agent went from
smaller than average to larger.

## 2026-08-26 · eviction, and what broadcast costs

| date | job | title | outcome |
|---|---|---|---|
| 08-26 | 6127798 | **random_holder eviction** | The block economy's perverse incentive fixed. Bigger populations, smaller contexts, higher canonical rate, lower offspring variance, higher effective population size. |
| 08-26 | 6127799 | random_holder + parental investment | Charging parents for their children's adapters adds nothing beyond eviction alone — it suppressed reproduction rather than concentrating it. |
| 08-26 | 6131724 | broadcast `say` | **Decisive negative.** One child in 400,000 steps; the population collapsed to a single agent per room. Broadcast is unaffordable at ~60 agents per room. |

**Settled:** hazard must rise with holdings, not throughput. Reproduction
charges are not the lever. Speech needs small rooms.

---

## Currently running

Moved here when they finish.

| job | title | question |
|---|---|---|
| 6141384 | σ = 0.0025 + random_holder | Does size-proportional hazard rescue the regime that has the diversity but died of bloat? |
| 6141167 | MLP genome, engine-derived capacity | Does a 4× larger genome carry more heritable variation without raising σ? |
| 6141411 | 40 villages of 10 agents, `say` | Is speech worth doing when affordable? Can a metapopulation form niches? |
| 6143004 | base-model base rate | What the frozen model does with no adapter — the reference every "takeoff" claim needs. |
| 6143016 | braced prompt placeholders | Is the degenerate placeholder action a prompt artefact or something deeper? |

## 2026-08-26 (later)

Surprise instrumented for the first time, and the traces read for the first time.

- **Observation surprise** added: the log-probability of the tokens the *world* writes into an agent's context, bucketed by position in the agent's own life and carried on every death event. The agent's surprise at its own output is kept separately as the fluency control. `src/evollm/analysis/metrics/surprise.md`.
- **Enabled on all four queued arms** (6141167 mlp, 6141384 chr0025_evict, 6141411 villages, 6143016 braced) rather than in a dedicated run, with a startup probe that disables the metric — not the run — if the unproven GPU path fails. `configs/node_4room_7b_surprise.yaml` held as the fallback arm.
- **Tracing changed**: whole lives of a random tenth of agents instead of the first N turns a room produces. The old budget filled up in the opening of the run, so every traced life was a founder's prefix — which is why within-lifetime questions could not be asked of any previous run.
- **`evollm inspect-traces`** reads the raw turns. On the existing runs it found, in every one: communication has collapsed (`tell` is 0.3-2% of actions; 1,000 tells against 173,988 mate requests in the reference run), a quarter to a third of agents emit one identical turn for 80%+ of their life while scoring as canonical throughout, and the placeholder tic reaches `mate` as well as `go`.
- **Within-lifetime change measured** on every past run via the behavioural proxy: no improvement anywhere, and a significant decline in the reference run (−0.74 pp ± 0.41 canonical rate, paired, n=2,445).

## 2026-08-26 (evening)

Two failures and one measurement.

- **`baserate` 6143004 produced no data.** A four-room config was submitted to
  `precheck_handshake.sh`, which requests one GPU. Room `gpu0`'s engine built
  fine; `gpu1`'s asked NVML for physical device 1, which was not allocated, and
  the engine core died 31 seconds in. The job then **hung for 3h59m** until the
  time limit — four GPU-hours spent on a run that was already dead. The script
  now requests four GPUs; resubmitted as **6146533**. An engine-start failure
  still does not terminate the job, which remains unfixed.
- **Surprise recording works.** Probe passed on every engine of all four arms;
  every death carried surprise, ~80% with a multi-bucket within-life curve.
  Observation surprise (0.516) is ~6x the agent's surprise at its own output
  (0.086) — the gap that justifies scoring only what the world wrote.
- **Surprise costs ~2.4x throughput.** 0.7 steps/sec against the reference
  run's 1.7 at matched context and matched population, flat across context.
  Twelve hours buys ~40% of the usual depth. The overhead is vLLM allocating
  and pythonizing the prompt-logprob tensor across the whole prompt each turn,
  not the logits matmul that was predicted.
- **Three arms restarted without it.** `mlp`, `chr0025_evict` and `villages`
  cancelled 2.7h in and resubmitted as **6146538**, **6146539**, **6146540** —
  all three are decided by depth. `braced` (6143016) keeps surprise and carries
  the measurement. Partial directories kept in
  `archive/2026-08_surprise_overhead/` as the evidence for the cost.
- **Documentation audit against `PROCESS.md`.** Five configs still read "queued
  as Slurm job 6081xxx" for arms that finished on 2026-08-22 — the state that
  gets a finished arm resubmitted by accident; all now carry `CLOSED` with what
  they settled. The four live arms had no `STATUS` at all and now name their job
  ids. `runs/README.md` was restating conclusions and had gone wrong — it still
  asserted σ = 0.0025 sits below the error threshold, months after `chr0025`
  collapsed — and is now a map with no beliefs in it. `baserate` 6143004 got an
  archive entry despite having produced no data, because a job that fails
  before writing anything otherwise leaves only a gap between job ids.

