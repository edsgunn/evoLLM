"""Who descends from whom.

Three different notions of "lineage" are useful and they are NOT
interchangeable, so this module keeps them separate and named:

  family       Connected components of the undirected graph joining each child
               to BOTH parents. Two agents are in the same family if any chain
               of parent-child links connects them. This is the coarsest
               grouping and, in a well-mixed population, it collapses to one
               giant component very quickly — which is itself a finding.

  ancestry     The fraction of an agent's pedigree traceable to each founder,
               by the recursive one-half rule. This is a real-valued vector,
               not a label, and it is what admixture actually looks like.
               Because crossover assigns each site to one parent with
               probability one half, it is also the EXPECTED share of sites
               inherited from that founder — but only the expectation; the
               realised share is not logged and would differ.

  lineage      The founder holding the largest ancestry share. A convenience
               label for stratification, and a lossy one: an agent that is 51%
               founder A is labelled A exactly like one that is 100% A. Always
               check `ancestry_entropy` before treating it as a clean grouping.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


class Pedigree:
    """Birth records as a queryable pedigree."""

    def __init__(self):
        self.parents: dict[str, tuple[str, ...]] = {}
        self.generation: dict[str, int] = {}
        self.origin: dict[str, str] = {}
        self.birth_room: dict[str, str] = {}
        self.birth_step: dict[str, int] = {}
        self._ancestry_cache: dict[str, dict[str, float]] = {}

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_run(cls, run_dir: str | Path) -> "Pedigree":
        ped = cls()
        for path in sorted(Path(run_dir).glob("events/*.jsonl")):
            with path.open() as fh:
                for line in fh:
                    e = json.loads(line)
                    if e.get("type") != "birth":
                        continue
                    ped.add(e["agent"], e.get("parents"), e.get("generation", 0),
                            e.get("origin", "child"), e.get("room", ""),
                            e.get("step", 0))
        return ped

    def add(self, agent, parents, generation, origin, room, step) -> None:
        self.parents[agent] = tuple(parents) if parents else ()
        self.generation[agent] = generation
        self.origin[agent] = origin
        self.birth_room[agent] = room
        self.birth_step[agent] = step

    def __len__(self) -> int:
        return len(self.parents)

    @property
    def agents(self) -> list[str]:
        return list(self.parents)

    # ── founders and ancestry ─────────────────────────────────────────────
    def founders(self) -> list[str]:
        """Agents with no parents: the seed population plus any immigrants."""
        return [a for a, p in self.parents.items() if not p]

    def ancestry(self, agent: str) -> dict[str, float]:
        """Founder -> share of this agent's pedigree, summing to 1.

        Iterative rather than recursive: lineages here reach generation 184 and
        Python's stack does not.
        """
        cache = self._ancestry_cache
        if agent in cache:
            return cache[agent]
        # Every ancestor visited on the way is cached too, not just the target.
        # Caching only the target made this quadratic: each of 24,000 agents at
        # generation 200 re-walked its entire ancestor set, and a single call to
        # build_phenotypes took longer than the run it was analysing.
        for node in self._ancestor_order(agent, stop_at=cache):
            if node in cache:
                continue
            par = self.parents.get(node, ())
            known = [p for p in par if p in cache]
            if not known:
                cache[node] = {node: 1.0}        # founder, or parent not logged
                continue
            acc: dict[str, float] = defaultdict(float)
            w = 1.0 / len(known)
            for p in known:
                for f, sh in cache[p].items():
                    acc[f] += w * sh
            cache[node] = dict(acc)
        return cache[agent]

    def _ancestor_order(self, agent: str, stop_at=None) -> list[str]:
        """`agent` and all its ancestors, parents before children.

        `stop_at` is a set of already-resolved nodes not to walk past. Only
        `ancestry` may pass it — `ancestor_shares` weights every distinct path
        and needs the whole walk, so pruning there would silently drop the
        repeated-ancestor contributions it exists to measure.
        """
        stop_at = stop_at if stop_at is not None else ()
        seen, order, stack = set(), [], [(agent, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if node in seen:
                continue
            seen.add(node)
            stack.append((node, True))
            if node in stop_at:
                continue                         # resolved: do not walk past it
            for p in self.parents.get(node, ()):
                if p not in seen:
                    stack.append((p, False))
        return order

    def ancestor_shares(self, agent: str) -> dict[str, float]:
        """Every ancestor of `agent`, weighted by the share of pedigree paths
        that pass through it. Founders are the special case handled by
        `ancestry`; this keeps the intermediate generations too.
        """
        weight: dict[str, float] = defaultdict(float)
        weight[agent] = 1.0
        for node in reversed(self._ancestor_order(agent)):   # children first
            w = weight.get(node, 0.0)
            par = [p for p in self.parents.get(node, ()) if p in self.parents]
            if not par or w == 0.0:
                continue
            for pnt in par:
                weight[pnt] += w / len(par)
        weight.pop(agent, None)
        return dict(weight)

    def ancestor_at(self, agent: str, generation: int) -> str | None:
        """The generation-`generation` ancestor holding the largest share.

        A deeper cut than `lineage`. When a population is panmictic — as
        evoLLM populations become within a few dozen generations, with one
        family holding 98%+ of everyone — founder labels stop separating
        anything and a cut further down the tree is the only way to recover
        groups that are actually distinct.

        Note the caveat: an agent's generation is one past its parents', but
        parents may sit at different generations, so not every pedigree path
        passes through every generation. Shares within a cut therefore need
        not sum to 1, and the dominant ancestor is a label, not a partition.
        """
        if self.generation.get(agent, 0) <= generation:
            return agent if self.generation.get(agent, 0) == generation else None
        at = {a: w for a, w in self.ancestor_shares(agent).items()
              if self.generation.get(a, -1) == generation}
        return max(sorted(at), key=lambda a: at[a]) if at else None

    def lineage(self, agent: str) -> str:
        """Founder with the largest ancestry share (ties broken by id)."""
        anc = self.ancestry(agent)
        return max(sorted(anc), key=lambda f: anc[f]) if anc else agent

    def ancestry_entropy(self, agent: str) -> float:
        """Shannon entropy of the ancestry vector, in bits.

        0 means a single founder; higher means admixed. This is the number that
        says whether the `lineage` label means anything for this agent.
        """
        anc = self.ancestry(agent)
        return -sum(s * math.log2(s) for s in anc.values() if s > 0)

    # ── families ──────────────────────────────────────────────────────────
    def families(self) -> dict[str, str]:
        """agent -> family id, via union-find over both parent links."""
        parent_of: dict[str, str] = {}

        def find(x):
            parent_of.setdefault(x, x)
            root = x
            while parent_of[root] != root:
                root = parent_of[root]
            while parent_of[x] != root:            # path compression
                parent_of[x], x = root, parent_of[x]
            return root

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent_of[ra] = rb

        for agent, par in self.parents.items():
            find(agent)
            for p in par:
                union(agent, p)
        return {a: find(a) for a in self.parents}

    def descendants(self, agent: str) -> set[str]:
        children = defaultdict(list)
        for a, par in self.parents.items():
            for p in par:
                children[p].append(a)
        out, stack = set(), [agent]
        while stack:
            node = stack.pop()
            for c in children.get(node, ()):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out
