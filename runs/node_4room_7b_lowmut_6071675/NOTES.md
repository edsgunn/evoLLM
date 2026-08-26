# node_4room_7b_lowmut_6071675

*Genome operators (job family 6071xxx) — crossover scheme and mutation operator at fixed everything else*

| | |
|---|---|
| Slurm job | `6071675` |
| Model | `Qwen2.5-7B-Instruct` |
| Rooms | 4 × 48,000 blocks, seeded 32/room |
| Tools | `tell, mate, go` |
| Read policy | `drain` · absorption `utterance` |
| Genome | σ = 0.0025, uniform crossover, additive mutation |
| Steps reached | 93,586 of 400,000 |

**Measured:** 11,574 children · 39 refills ·
99.7% self-sufficient · max generation 184 ·
92.1% canonical turns · median lifetime 558 steps ·
median context at death 8,288 tokens · 13.6 generated tokens/turn

> **Ended early at step 93,586 of 400,000.**

## What it was for

Reduce mutation from σ = 0.01 to **0.0025**, keeping uniform crossover and
everything else identical to `node_4room_7b_free40_6070757`. At σ=0.01 a child
takes a ~50% relative perturbation at every one of 112 sites; at 0.0025 five
matings move factor RMS only 1.04× instead of 1.50×.

The intent was a null test: if children still reproduced worse than parents with
drift effectively frozen, drift was not the cause.

## What we learned

**This is the run where the project turned.** It is the only arm to show takeoff.

- Reproduction per 100 turns **climbed** 1.33 → 2.52 → 3.18 → 5.71 → 3.70 → 8.54
  across generation bands 0 through 101–200. Every other arm collapsed.
- Canonical-action rate climbed 88% → 96%, and the trend survives stratification
  by turn count, so it is not a lifespan artefact.
- Reached **generation 184** — against 5–13 everywhere else — in
  93,586 steps, roughly 500 steps per generation.
- 99.7% self-sufficiency: 39 refills against 11,574 births,
  and every one of those refills fired inside the first 3,500 steps.
- No bloat collapse: 38–86 agents per room at 8,000–18,000 tokens each, against
  2–3 agents at 250,000–390,000 tokens in the σ=0.01 arms. Agents complete an
  action in 14 tokens per turn against ~30.

Together these establish that σ=0.01 sits **above the error threshold** — the rate
at which mutational input outruns selection — and 0.0025 sits below it. It also
showed that fixing inheritance fixes the memory economy for free: coherent agents
use memory efficiently as a consequence of being coherent.

Its h² reads low (+0.038 ± 0.025), but that is a completed sweep rather than a
failure: phenotypic variance fell to 0.033 against 0.15 elsewhere as canonical
rate swept to 94.7%. Selection consumes the additive variance it feeds on.

**Truncated.** The preemption guard aborted it at step 93,586 of
400,000 — 23% through, with reproduction still rising. Where it plateaus is
unknown, which is why `chr0025` and `uni0025` rerun this regime to completion.

## Status

The key positive result. Superseded operationally by chr0025/uni0025, which rerun this regime with refill off, merged observation framing, and to full length.

---
*Notes written 2026-08-21. Numbers above are recomputed from this run's own
`events/*.jsonl`; prose interpretation is from the cross-run analysis in the
reproduction-sweep report.*

## Deeper analysis (2026-08-21)

`ANALYSIS.txt` and `ANALYSIS_gen20.txt` in this directory are the output of
`evollm analyse` at 2,000 permutations; `traits.csv` is the per-agent trait
table (9,277 agents × 28 traits). Headline findings:

- **The population is panmictic.** 98.7% of agents are in a single family and
  95.4% share a dominant founder — 1.10 *effective* lineages by inverse
  Simpson. The median agent descends from 13 of the 128 seeds. Whatever else
  is true, niches have not emerged.
- **Distinct strategies exist but are not inherited.** k-means on the action
  simplex separates a mating specialist cluster (mate 0.90, n=5,412), a moving
  specialist (move 0.96, n=2,672) and a small talker cluster (tell 0.84,
  n=45). Mutual information between lineage and cluster is 0.037 bits against
  a null of 0.036 (p = 0.083) — lineage does not predict strategy.
- **Weak lineage signal appears at a deeper cut.** Re-cut at generation 20,
  move_success (p = 0.000), move_share (p = 0.006) and mate_share (p = 0.012)
  show significant between-lineage variance, though the excess over the null
  is small (+0.0075, +0.0023, +0.0018).
- **No gene–behaviour association replicates.** `19.o_proj` reaches
  p_fwer = 0.002 for canonical-action rate, but per-room fits are +0.03 /
  +0.55 / +0.08 — one room carries it. Every hit in the table is flagged
  "(one room)". n = 162 genotyped agents against 112 sites is underpowered,
  and genotype PC1 alone explains 87.7% of genotype variance (overall
  perturbation magnitude), leaving little independent per-site signal.

## Figures

`FIGURES.html` — self-contained page (open it in a browser; charts are
interactive, tables under each figure). `figures_data.json` is the underlying
series, so the charts can be rebuilt or re-plotted elsewhere.

The headline is **children per agent against a replacement line at 2.00**:
every birth has two parents, so a mean of 2.00 children per agent is exactly
population replacement. Generation 0 sat below it at 1.26; from roughly
generation 20 onward the population is consistently above it, settling around
2.2–2.9. The founding population was not viable on its own and its descendants
were.

Note the distinction the figures make explicit: children *per agent* roughly
doubles and then plateaus, while children *per 100 turns lived* keeps rising to
4× its starting value. Both are true because median lifetime fell from 2,077
steps to ~500. Later agents do not have many more offspring over a lifetime —
they secure them far faster, in a population four times denser.
