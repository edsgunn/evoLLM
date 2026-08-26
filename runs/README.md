# Run index

> For the project's current beliefs, open questions and running jobs, start at
> **`../STATE.md`**. For the chronology — every experiment in order and what
> each one settled — see **`TIMELINE.md`**. This file is the per-run index
> both of them draw on.

Every directory here has a `NOTES.md` saying what that run was for, what it
measured, and whether it still stands. This file is the map.

Runs whose contexts were built without chat framing are **not** here — they are
in `archive/2026-08_pre_chat_format/`, with a README explaining why none of them
can be cited. Everything in this directory has correct `<|im_start|>` framing.

## The one-line history

The project has run four families, and each one closed a question that made the
next one askable:

| Family | Job | Asked | Settled |
|---|---|---|---|
| **A** | `6045xxx` | Which tools, which absorption rule, does the chat-format fix matter? | Qwen2.5-1.5B cannot run this world (8–22% canonical vs ~72% at 7B). Whole-utterance absorption, not token-by-token. |
| **B** | `6059xxx` | Read one queued utterance per turn, or drain the queue? | Drain. Also the dataset where reproduction was first shown to degrade across generations, and heritability first measured at ~0. |
| **C** | `6070xxx` | Does the refill threshold control population longevity? | No. It sets immigration volume; immigrants were propping the population up, not crowding it out. |
| **D** | `6071xxx` | Crossover scheme? Mutation operator? | Neither — the **mutation rate** dominates both. σ = 0.01 is above the error threshold, σ = 0.0025 is below it. |

## What is currently believed

- **σ = 0.01 is above the error threshold.** Every arm at that rate degrades:
  reproduction falls, canonical rate falls, and rooms end held by two or three
  bloated non-reproducers at 250,000–390,000 tokens each.
- **σ = 0.0025 is below it.** `lowmut` is the only run to show takeoff —
  reproduction climbing 1.33 → 8.54 children per 100 turns over 184 generations,
  99.7% self-sufficient.
- **Coherence and memory efficiency are the same thing.** A working agent
  completes an action in ~13 generated tokens; a degraded one burns ~30 and
  often never finishes. Repairing inheritance repaired the memory economy with
  no change to the block accounting.
- **Chromosomal crossover is a better prior, not yet an established result.**
  h² = +0.61 ± 0.16 for `chromo` against +0.24 ± 0.05 for the pooled uniform
  σ=0.01 arms (z = 2.15) — but a *uniform* arm (`norefill`) scored +0.78 on 86
  pairs. `chr0025` vs `uni0025` is the clean test.
- **Death is behaviour-blind.** corr(drift, lifetime) ≈ +0.005. Selection acts
  only through differential reproduction, and weakly.

## Reading order

If you want the shortest path to the current state:

1. `node_4room_tell_7b_drain_6059671/NOTES.md` — where the degradation and
   near-zero heritability were established.
2. `node_4room_7b_free40_6070757/NOTES.md` — the shared baseline every genome
   arm is read against.
3. `node_4room_7b_lowmut_6071675/NOTES.md` — the run that turned the project.
4. `node_4room_7b_chromo_6071674/NOTES.md` — for the heritability claim and its
   caveat.

## Caveats that apply across the board

- **Family A and B use 4,000/8,000-block rooms and 16 agents per room.** Family C
  onward uses a uniform 48,000 blocks and 32 agents. Absolute population and
  lifetime numbers are not comparable across that boundary.
- **Every run here except `norefill` had refill enabled**, so its population is
  partly immigration. Self-sufficiency is quoted in each NOTES.md; treat any
  arm below ~90% as substantially subsidised.
- **Two runs were truncated by the preemption guard** (`chromo` at 4h47m,
  `lowmut` at step 93,586 of 400,000). The guard aborts rather than continue with
  unattributable deaths, so the data up to that point is valid, but neither
  reached its plateau.
- **Observation framing changed on 2026-08-21.** Consecutive observations now
  share one `<|im_start|>user` block instead of opening one each. Every run in
  this directory predates that change.
