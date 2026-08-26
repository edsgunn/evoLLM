# Cross-run analysis: the five σ / crossover arms

Figures from `make_figures.py`, data in `../ANALYSIS_ALL.json`. Per-run detail
is in each run's `ANALYSIS.txt` and `traits.csv`.

| File | Shows |
|---|---|
| `1_divergence.png` | Reproduction, canonical rate and move success by generation. Every run improves early; only σ = 0.001 holds it. |
| `2_placeholder.png` | The degenerate action — agents emitting the prompt's own `room_id` placeholder. σ = 0.001 never acquires it. |
| `3_composition.png` | Action composition per run. Every population reallocates toward mating. |
| `5_diversity_vs_pathology.png` | Genetic diversity and the degenerate action on shared generation axes, per run. Diversity peaks, then the placeholder sweeps. |
| `4_effects.png` | Density-controlled partial correlation with generation, all runs, all traits. |

## Genome variation

Drift from the base model (which is ΔW = 0, so ‖ΔW‖ *is* the distance from it)
matches the random-walk prediction √(init² + kσ²) in every arm:

| run | max gen | ‖ΔW‖ vs gen 0 | spread between agents, relative |
|---|---|---|---|
| chr001 σ=0.001 | 223 | 1.20× | 0.007 → 0.024 |
| chr0025 σ=0.0025 | 207 | 2.12× | 0.007 → 0.081 → **0.066** |
| uni0025 σ=0.0025 | 237 | 1.99× | 0.007 → **0.090** |
| chr0025_c1 σ=0.0025 | 161 | 1.90× | 0.007 → 0.095 → **0.051** |
| chr005 σ=0.005 | 28 | 1.69× | 0.007 → **0.160** → 0.115 |

The 0.0025 arms carry 3–4× more internal variation than chr001, which is
consistent with their much stronger parent→child strategy transmission
(+4.1 and +5.7 pp against chr001's +0.8). But three of them then LOSE
diversity, and in chr0025_c1 the absolute pairwise distance falls too
(6.74 → 4.47), so it is real loss rather than the norm outrunning it.

chr0025_c1 loses it fastest, which is what a single chromosome predicts: the
whole genome travels as one unit, so recombination cannot break up a sweeping
variant.

## Headline

**Four of the five runs improve for 20–70 generations and then decay. One
(σ = 0.001) does not.** The decay is not general degradation: it is
concentrated in a single acquired behaviour.

## The degenerate action

Move success collapses from ~65% to 1–20% in four runs. It is not rooms being
full (that share *falls*) and not malformed syntax (canonical rate stays high
until late). It is agents emitting `<go>room_id</go>` — the literal placeholder
from the examples in their own system prompt. 57–77% of all invalid
destinations are that exact string.

It is a well-formed, canonical, always-failing action: a "pass" that costs a
turn and changes nothing. Its share of move attempts rises 1% → 58% (chr0025),
3% → 73% (uni0025), 2% → 35% (chr0025_c1). At σ = 0.001 it stays at 1%
throughout.

## Niches: strategies yes, heritability no

k-means on the action simplex splits every population cleanly into a **mating
specialist** (mate ≈ 0.88 of actions) and a **moving specialist** (move ≈ 0.93),
plus small "no action" and "tell" clusters. So distinct strategies do exist.

They are not inherited. Mutual information between lineage and cluster is
indistinguishable from its permutation null in four of five runs
(p = 0.19, 0.49, 0.49, 0.67). Only `uni0025` is significant, and weakly
(MI 0.022 vs null 0.017, p = 0.005).

All five populations are effectively panmictic: 1.04–1.47 *effective* lineages,
95–99.5% of agents in a single family. Niches in the isolated-lineage sense have
not formed.

## Genes

Every run has per-site associations that survive both the genome-wide null and
the across-room replication check — e.g. in chr0025 the perturbation magnitude
at `19.o_proj` predicts the mate-vs-move axis (r = +0.056 / −0.059,
p_fwer = 0.005, same sign and comparable size in all four rooms independently).

Allele tests on realised descent are stronger still: `move_success` reaches
eta² = 0.075–0.15 against which founder supplied a site.

**But no site replicates across runs.** move_success maps to 23.o_proj,
8.k_proj, 17.k_proj, 14.q_proj and 24.q_proj in the five runs. That is what a
founder-specific effect looks like: each population is exploiting whichever of
its own initial draws happened to be good, not a universal property of a layer.

## Caveats

- Genotype PC1 explains 87–97% of genotype variance (overall perturbation
  magnitude), so after PC correction the residual per-site signal is small.
- Related agents share both alleles and behaviour; permutation is stratified by
  (room × generation band), which absorbs some but not all of that.
- In the section-3 table the headline `r` is computed on PC-residualised data
  while the per-room replication columns are computed on raw values, so their
  signs can differ. The starring logic uses the raw pooled value throughout and
  is internally consistent; the display is not.
- `chr0025` and `chr0025_c1` were stopped early by the preemption guard.
