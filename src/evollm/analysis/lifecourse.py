"""What happens *inside* an agent's lifetime.

Every other trait in this package is a lifetime aggregate: one number per
agent, computed over its whole life. That silently assumes the thing being
measured is a property of the agent rather than something that moves while it
lives, and it makes the project's own hypothesis unaskable. In-context
surprise minimisation is a claim about *change within a life*; an aggregate
cannot see it either way.

Two independent reads, because they fail differently:

`surprise_curve` uses the ``obs_nll_curve`` carried on every death event —
mean surprise over the tokens the world wrote, bucketed by how far into its
own life the agent was. It covers the whole population, needs no traces, and
is the direct measure. It requires a run with ``run.record_surprise``.

`action_curve` uses traced turns: what the agent did and whether it worked, by
position in its own life. It covers only the traced sample and is a behavioural
proxy, but it works on every run ever done, including all the ones that predate
surprise recording.

Both are paired within agent. A cross-sectional comparison of young against old
agents measures survivorship — the agents still alive at turn 80 are the ones
that were good enough to get there — not learning. Only differencing an agent
against itself removes that.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

from ..agent import SURPRISE_BUCKET_EDGES

BUCKET_LABELS = tuple(
    [f"{lo}-{hi - 1}" for lo, hi in
     zip((0,) + SURPRISE_BUCKET_EDGES[:-1], SURPRISE_BUCKET_EDGES)]
    + [f"{SURPRISE_BUCKET_EDGES[-1]}+"])


def _events(run: str, types: set[str]):
    for path in sorted(glob.glob(os.path.join(run, "events", "*.jsonl"))):
        room = os.path.basename(path).split(".")[0]
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("type") in types:
                    e["_room"] = room
                    yield e


def _mean_ci(values):
    """Mean and half-width of a 95% interval, or None when too few to say."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    if len(v) < 3:
        return None
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), len(v)


def surprise_curve(run: str, generation_bands: int = 0) -> dict:
    """Mean observation surprise by within-life bucket.

    Returns the population curve, the paired within-agent change from an
    agent's first populated bucket to its last, and the same split by
    generation band when asked — which is what distinguishes "agents learn
    within a life" from "later generations start out better".

    The paired change is the headline. The population curve alone is
    contaminated by survivorship: agents that reach the late buckets are a
    selected subset of those that filled the early ones.
    """
    rows = [e for e in _events(run, {"death"}) if e.get("obs_nll_curve")]
    if not rows:
        return {"available": False,
                "reason": "no death event carries obs_nll_curve; the run "
                          "predates surprise recording or had it disabled"}

    n_buckets = len(BUCKET_LABELS)
    by_bucket: list[list[float]] = [[] for _ in range(n_buckets)]
    deltas, gens, first_vals = [], [], []
    for e in rows:
        curve = e["obs_nll_curve"]
        for i, v in enumerate(curve[:n_buckets]):
            if v is not None:
                by_bucket[i].append(v)
        filled = [(i, v) for i, v in enumerate(curve) if v is not None]
        if len(filled) >= 2:
            deltas.append(filled[-1][1] - filled[0][1])
            gens.append(e.get("generation", 0))
            first_vals.append(filled[0][1])

    out = {
        "available": True,
        "n_agents": len(rows),
        "buckets": list(BUCKET_LABELS),
        "population": [_mean_ci(b) for b in by_bucket],
        "within_agent_change": _mean_ci(deltas),
        "n_paired": len(deltas),
    }
    # Does the STARTING level fall across generations? That is inheritance of
    # a better prior, and it is a different claim from within-life learning.
    if gens and len(set(gens)) > 1:
        g = np.asarray(gens, float)
        out["start_vs_generation"] = _slope(g, np.asarray(first_vals, float))
        out["change_vs_generation"] = _slope(g, np.asarray(deltas, float))
    if generation_bands:
        out["by_generation"] = _banded(gens, deltas, generation_bands)
    return out


def _slope(x, y):
    """OLS slope with a standard error, or None if it cannot be estimated."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    # A constant y has no slope to estimate and no correlation to take: every
    # agent changed by the same amount, which is a real answer, not a trend.
    if len(x) < 40 or x.std() == 0 or y.std() == 0:
        return None
    b, _ = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    se = np.sqrt((1 - r ** 2) / (len(x) - 2)) * (y.std() / x.std())
    return {"slope": float(b), "se": float(se), "n": int(len(x)),
            "z": float(b / se) if se else None}


def _banded(gens, values, n_bands):
    g = np.asarray(gens, float)
    edges = np.quantile(g, np.linspace(0, 1, n_bands + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (g >= lo) & (g <= hi)
        out.append({"generation": f"{lo:.0f}-{hi:.0f}",
                    "change": _mean_ci(np.asarray(values)[m])})
    return out


def action_curve(run: str, min_turns: int = 10, quantiles: int = 5) -> dict:
    """Behaviour by position within an agent's own life, from traced turns.

    Each traced agent's turns are split into `quantiles` equal parts of its own
    life, so agents of very different lifespans are comparable. Reports the
    canonical-form rate and the action mix per part, and the paired
    last-part-minus-first-part change.

    Only agents with at least `min_turns` traced turns are used. If the run
    traced by budget rather than by agent (see `run.trace_agent_fraction`) the
    sample is the opening of the run and every traced life is a prefix, so
    treat the result as describing founders rather than the population — the
    function reports how much of each traced life it can actually see.
    """
    turns = defaultdict(list)
    for e in _events(run, {"turn"}):
        turns[e["agent"]].append(e)
    for v in turns.values():
        v.sort(key=lambda e: (e.get("turn_index", e.get("step", 0)),
                              e.get("step", 0)))

    long = {a: v for a, v in turns.items() if len(v) >= min_turns}
    if not long:
        return {"available": False,
                "reason": f"no traced agent has {min_turns} turns"}

    # How much of a life the trace actually covers. A traced turn carries its
    # index within the agent's own life, so a trace that stops early is
    # visible rather than silently truncating every curve.
    coverage = [len(v) / (max(t.get("turn_index", 0) for t in v) + 1)
                for v in long.values()
                if any("turn_index" in t for t in v)]

    per_q = [[] for _ in range(quantiles)]
    deltas, gens = [], []
    action_q = [defaultdict(int) for _ in range(quantiles)]
    for _agent, v in long.items():
        n = len(v)
        rates = []
        for q in range(quantiles):
            lo, hi = n * q // quantiles, n * (q + 1) // quantiles
            part = v[lo:hi] or v[lo:lo + 1]
            rate = float(np.mean([t.get("form") == "canonical" for t in part]))
            per_q[q].append(rate)
            rates.append(rate)
            for t in part:
                action_q[q][t.get("action", "?")] += 1
        deltas.append(rates[-1] - rates[0])
        gens.append(v[0].get("generation", 0))

    out = {
        "available": True,
        "n_agents": len(long),
        "quantiles": quantiles,
        "canonical_by_quantile": [_mean_ci(q) for q in per_q],
        "within_agent_change": _mean_ci(deltas),
        "action_mix_by_quantile": [
            {k: round(c / max(sum(d.values()), 1), 4)
             for k, c in sorted(d.items(), key=lambda kv: -kv[1])}
            for d in action_q],
        "trace_coverage": (round(float(np.mean(coverage)), 3)
                           if coverage else None),
    }
    if gens and len(set(gens)) > 1:
        out["change_vs_generation"] = _slope(np.asarray(gens, float),
                                             np.asarray(deltas, float))
    return out


def format_lifecourse(surprise: dict, actions: dict) -> str:
    """A short report of both reads, for NOTES.md."""
    lines = ["WITHIN-LIFETIME CHANGE", "=" * 60, ""]

    lines.append("Observation surprise (all agents, from death records)")
    if not surprise.get("available"):
        lines.append(f"  unavailable: {surprise.get('reason')}")
    else:
        lines.append(f"  {surprise['n_agents']:,} agents")
        for label, cell in zip(surprise["buckets"], surprise["population"]):
            if cell is None:
                lines.append(f"    turns {label:>6s}   (too few)")
            else:
                m, ci, n = cell
                lines.append(f"    turns {label:>6s}   {m:6.3f} ± {ci:.3f}"
                             f"   n={n:,}")
        ch = surprise["within_agent_change"]
        if ch:
            m, ci, n = ch
            verdict = ("surprise FALLS within a life" if m + ci < 0 else
                       "surprise RISES within a life" if m - ci > 0 else
                       "no within-life change detected")
            lines.append(f"  paired within-agent change: {m:+.3f} ± {ci:.3f}"
                         f"  (n={n:,}) -- {verdict}")
        for key, what in (("start_vs_generation", "starting surprise"),
                          ("change_vs_generation", "within-life change")):
            s = surprise.get(key)
            if s:
                lines.append(f"  {what} vs generation: "
                             f"{s['slope']:+.2e} per generation "
                             f"(z={s['z']:+.2f}, n={s['n']:,})")
    lines.append("")

    lines.append("Canonical rate by position in life (traced agents)")
    if not actions.get("available"):
        lines.append(f"  unavailable: {actions.get('reason')}")
    else:
        cov = actions.get("trace_coverage")
        lines.append(f"  {actions['n_agents']:,} agents"
                     + (f", trace covers {cov:.0%} of each life"
                        if cov is not None else ""))
        for i, cell in enumerate(actions["canonical_by_quantile"]):
            if cell is None:
                continue
            m, ci, n = cell
            lines.append(f"    fifth {i + 1}   {m * 100:5.1f}% ± {ci * 100:.1f}"
                         f"   n={n:,}")
        ch = actions["within_agent_change"]
        if ch:
            m, ci, n = ch
            lines.append(f"  paired within-agent change: {m * 100:+.2f} pp "
                         f"± {ci * 100:.2f}  (n={n:,})")
    return "\n".join(lines)
