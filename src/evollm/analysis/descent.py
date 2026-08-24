"""Realised descent: which founder actually supplied each site of each agent.

`Pedigree.ancestry` gives EXPECTED founder shares — the recursive one-half
rule. That is an average over births that never happened. This module uses the
inheritance masks recorded at each birth (one bit per site per child) to
resolve what actually descended, site by site.

The difference is the whole of genetics. Expected ancestry says an agent five
generations down is 1/32 from each of 32 founders; realised ancestry says which
of its 112 sites came from which founder, and those are integers, not fractions.
Only the realised version gives an ALLELE — a categorical label per site per
agent that behaves like a genetic marker and can be tested against behaviour
directly, with far more power than the continuous perturbation-magnitude
feature that snapshot genomes alone support.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Descent:
    """Per-site founder assignment for every agent in a run."""

    def __init__(self, sites: list[str]):
        self.sites = list(sites)
        self.parents: dict[str, tuple[str, str] | None] = {}
        self.donors: dict[str, str] = {}
        self.generation: dict[str, int] = {}
        self.step: dict[str, int] = {}
        self._resolved: dict[str, np.ndarray] | None = None
        self._founders: list[str] = []

    @classmethod
    def from_run(cls, run_dir: str | Path) -> "Descent | None":
        """Read `<run>/genomes/*.jsonl`. Returns None if the run has no masks
        (any run made before inheritance tracking existed)."""
        sites: list[str] = []
        recs: list[dict] = []
        for path in sorted(Path(run_dir).glob("genomes/*.jsonl")):
            with path.open() as fh:
                for line in fh:
                    rec = json.loads(line)
                    if not sites and rec.get("sites"):
                        sites = list(rec["sites"])
                    recs.append(rec)
        if not sites or not any(r.get("donors") for r in recs):
            return None
        d = cls(sites)
        for rec in recs:
            agent = rec["agent"]
            par = rec.get("parents")
            d.parents[agent] = tuple(par) if par else None
            if rec.get("donors"):
                d.donors[agent] = rec["donors"]
            d.generation[agent] = rec.get("generation", 0)
            d.step[agent] = rec.get("step", 0)
        return d

    @property
    def founders(self) -> list[str]:
        return [a for a, p in self.parents.items() if not p]

    def resolve(self) -> dict[str, np.ndarray]:
        """agent -> int array of length n_sites, indexing into `self.founders`.

        Resolved in generation order. A child's generation is strictly greater
        than both parents', so that ordering is topological and one pass
        suffices — no recursion, which matters at generation 184.
        """
        if self._resolved is not None:
            return self._resolved
        n = len(self.sites)
        self._founders = sorted(self.founders)
        index = {f: i for i, f in enumerate(self._founders)}
        out: dict[str, np.ndarray] = {}
        order = sorted(self.parents, key=lambda a: (self.generation.get(a, 0),
                                                    self.step.get(a, 0), a))
        for agent in order:
            par = self.parents.get(agent)
            mask = self.donors.get(agent)
            if not par or not mask or len(mask) != n:
                # A founder, or a birth whose mask is missing: it is its own
                # source for every site.
                out[agent] = np.full(n, index.get(agent, -1), dtype=np.int32)
                continue
            p1, p2 = par
            a1, a2 = out.get(p1), out.get(p2)
            if a1 is None or a2 is None:
                out[agent] = np.full(n, -1, dtype=np.int32)
                continue
            take_p1 = np.frombuffer(mask.encode(), dtype=np.uint8) == ord("1")
            out[agent] = np.where(take_p1, a1, a2).astype(np.int32)
        self._resolved = out
        return out

    def founder_names(self) -> list[str]:
        self.resolve()
        return self._founders

    def site_matrix(self, agents) -> np.ndarray:
        """(n agents, n sites) of founder indices; -1 where unresolved."""
        res = self.resolve()
        n = len(self.sites)
        return np.stack([res.get(a, np.full(n, -1, dtype=np.int32))
                         for a in agents])

    def realised_ancestry(self, agent: str) -> dict[str, float]:
        """Founder -> share of this agent's SITES actually descended from it.

        Compare against `Pedigree.ancestry` for the same agent: the expected
        share is smooth and the realised one is lumpy, and the gap between them
        is genetic drift made visible at the level of a single individual.
        """
        res = self.resolve()
        arr = res.get(agent)
        if arr is None:
            return {}
        names = self._founders
        counts: dict[str, float] = {}
        for idx, c in zip(*np.unique(arr, return_counts=True)):
            if idx >= 0:
                counts[names[idx]] = float(c) / len(arr)
        return counts

    def effective_founders_per_site(self, agents) -> np.ndarray:
        """Per site, how many distinct founders are still represented.

        Read this as COALESCENCE, not as diversity or health. Every founder is
        an independent N(0, init_scale) draw at every site — exchangeable
        random perturbations of the same base model — so founder identity is a
        neutral label, and at generation 0 there is no adaptive variation for
        the population to lose. A site collapsing to one founder is ambiguous
        on its own: it is either drift, or selection having fixed the best of
        the initial draws, which is the outcome the experiment wants.

        What makes founder labels useful is exactly that neutrality: they are
        clean markers for measuring PROCESS — how fast the genome coalesces,
        how much recombination really happens, what the effective population
        size is. The diversity that will matter for adaptation is not here at
        the start; it has to be manufactured by mutation over many
        generations. See `selection_scan` for the way to tell the two apart.
        """
        M = self.site_matrix(agents)
        return np.array([len(set(col[col >= 0])) for col in M.T])

    def selection_scan(self, agents) -> list[dict]:
        """Per-site coalescence against the genome's own neutral expectation.

        All 112 sites ride inside the same individuals and therefore share one
        pedigree, one population size and one drift history. So the
        genome-wide distribution of coalescence IS the neutral null — no
        simulation and no assumed Ne required. A site that has lost far more
        founders than its neighbours has done so for a reason drift does not
        supply, which makes it a selection candidate; one that retains far
        more is a candidate for balancing selection, or simply for being inert
        enough that nothing has pushed it either way.

        Caveat worth keeping: linkage. Under chromosomal crossover neighbouring
        sites travel together, so an outlier drags its chromosome with it and
        the unit of inference is a block, not a site.
        """
        eff = self.effective_founders_per_site(agents)
        if not len(eff):
            return []
        mean, sd = float(eff.mean()), float(eff.std())
        out = []
        for site, value in zip(self.sites, eff):
            z = (float(value) - mean) / sd if sd > 0 else 0.0
            out.append(dict(site=site, founders=int(value), z=z,
                            direction=("coalesced" if z < 0 else "retained")))
        out.sort(key=lambda d: d["z"])
        return out
