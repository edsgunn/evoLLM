# Run index

> This file is the **map**: what exists, where it lives, and what each run was
> for. It deliberately holds **no beliefs** — those live in [`../STATE.md`](../STATE.md),
> which is the single place a finding is stated. For the chronology, see
> [`TIMELINE.md`](TIMELINE.md). For what a run *established*, open its own
> `NOTES.md`.
>
> An earlier version of this file restated the project's conclusions, and they
> went stale and then wrong — it was still asserting that σ = 0.0025 sits below
> the error threshold months after `chr0025` collapsed. Beliefs are not
> duplicated here any more.

Every directory here has a `NOTES.md`, except one still running. Anything that
cannot be cited is in [`../archive/`](../archive/), never deleted.

---

## Currently running

Live jobs are listed in `../STATE.md`, which is the authority. A run appears
below once it has finished and been analysed.

## Families

Runs are grouped by job-id block. Each family closed a question that made the
next one askable.

| family | jobs | what the family was for |
|---|---|---|
| **A** | `6045xxx` | which tools, which absorption rule, does chat framing matter |
| **B** | `6059xxx` | read one queued utterance per turn, or drain the queue |
| **C** | `6070xxx` | does the refill threshold control population longevity |
| **D** | `6071xxx` | crossover scheme and mutation operator |
| **E** | `6081xxx` | the σ sweep on chromosomal crossover, and the crossover control |
| **F** | `6127xxx` | eviction policy, and charging parents for reproduction |
| **G** | `614xxxx` | MLP genome, metapopulation, prompt placeholders, surprise |

## The runs

| run | family | what it was | status |
|---|---|---|---|
| `node_4room_6045965` | A | first four-room world | superseded |
| `node_4room_tell_6045966` | A | directed speech | superseded |
| `node_4room_both_6045967` | A | broadcast and directed together | superseded |
| `node_4room_tell_token_6045968` | A | token-by-token absorption | superseded |
| `node_4room_tell_noformat_6045969` | A | no chat framing | superseded |
| `node_4room_tell_token_7b_6045970` | A | absorption rule at 7B | superseded |
| `node_4room_tell_7b_6059339` | B | 7B, read one per turn | superseded |
| `node_4room_tell_drain_6059672` | B | drain the backlog | superseded |
| `node_4room_tell_7b_drain_6059671` | B | drain at 7B | superseded |
| `node_4room_7b_free20_6070758` | C | refill floor 20% | superseded |
| `node_4room_7b_free40_6070757` | C | refill floor 40% | superseded |
| `node_4room_7b_free60_6070756` | C | refill floor 60% | superseded |
| `node_4room_7b_norefill_6070759` | C | no immigration | superseded |
| `node_4room_7b_chromo_6071674` | D | chromosomal crossover, σ=0.01 | superseded |
| `node_4room_7b_lowmut_6071675` | D | σ=0.0025 | superseded |
| `node_4room_7b_multmut_6071676` | D | multiplicative mutation | superseded |
| `node_4room_7b_chr005_6081312` | E | σ=0.005 — upper bracket | closed |
| `node_4room_7b_chr0025_6081311` | E | σ=0.0025, chromosomal | closed |
| `node_4room_7b_uni0025_6081397` | E | σ=0.0025, uniform — the crossover control | closed |
| `node_4room_7b_chr0025_c1_6081314` | E | one chromosome — the linkage dial | closed |
| `node_4room_7b_chr001_6081313` | E | σ=0.001 — the first sustained takeoff | closed |
| `node_4room_7b_chr001_evict_6127798` | F | σ=0.001 with `random_holder` eviction | **REFERENCE** |
| `node_4room_7b_chr001_invest_6127799` | F | parental investment | closed |
| `node_4room_7b_braced_6143016` | G | prompt slots as `{room}`; the only arm recording surprise | running |

**`node_4room_7b_chr001_evict_6127798` is the reference configuration.** New
arms are read against it, and its `NOTES.md` carries the numbers to beat.

## Archived — none of these can be cited

| archive | runs | why |
|---|---|---|
| [`../archive/2026-08_pre_chat_format/`](../archive/2026-08_pre_chat_format/) | 15 | contexts built without `<\|im_start\|>` framing — out of distribution for every token generated |
| [`../archive/2026-08_crashed_eviction/`](../archive/2026-08_crashed_eviction/) | 2 | killed by a defect in the eviction path |
| [`../archive/2026-08_say_collapse/`](../archive/2026-08_say_collapse/) | 2 | broadcast speech at scale; the population died of being spoken to |
| [`../archive/2026-08_surprise_overhead/`](../archive/2026-08_surprise_overhead/) | 3 | cancelled 2.7h into 12 — kept as the evidence that surprise recording works and what it costs |
| [`../archive/2026-08_baserate_no_data/`](../archive/2026-08_baserate_no_data/) | 0 | `baserate` 6143004 produced no data at all: dead 31 seconds in, hung to its 4-hour limit |

---

## Reading order

The shortest path into the project:

1. [`../MECHANICS.md`](../MECHANICS.md) — what the world actually is, if you
   have not met it before.
2. `node_4room_7b_chr001_evict_6127798/NOTES.md` — the reference run, and the
   eviction result that produced it.
3. `node_4room_7b_chr001_6081313/NOTES.md` — the first sustained takeoff, and
   what it does and does not establish.
4. `node_4room_7b_uni0025_6081397/NOTES.md` — how the crossover question was
   settled, and why a single run could not settle it.
5. [`../STATE.md`](../STATE.md) — everything currently believed, graded.

## Caveats that cross run boundaries

Comparisons that ignore these are invalid. Each run's own `NOTES.md` states
where it stopped and what that biases.

- **Room size and population changed after family B.** Families A and B use
  4,000–8,000-block rooms and 16 agents; family C onward uses 48,000 blocks and
  32. Absolute population and lifetime numbers do not cross that boundary.
- **Refill was enabled through family D** and off from family E onward, so
  earlier populations are partly immigration. Self-sufficiency is quoted in each
  `NOTES.md`; treat anything below ~90% as substantially subsidised.
- **Observation framing changed on 2026-08-21.** Consecutive observations now
  share one user block instead of opening one each. Families A–D predate this;
  E onward have it.
- **Eviction changed at family F.** Everything before `chr001_evict` used
  `requester`, under which hazard tracked throughput rather than holdings.
- **Rooms desynchronise.** Never compare raw step counts — compare at matched
  room-depth, or per 1,000 room-steps.
- **Traces before family G are the first N turns of the run**, so every traced
  life is a founder's prefix. From family G they are whole lives of a sampled
  fraction of agents. Within-lifetime questions can only be asked of the latter.
