"""One row per agent: what it did, how well it worked, and who it came from.

Traits are grouped by what they measure, because the groups answer different
questions and mixing them is how you end up correlating a thing with itself:

  form        Did the output parse at all? (canonical_rate, malformed_share)
  strategy    Of the actions taken, WHICH ones — the composition on the
              simplex (tell_share, mate_share, move_share, noop_share).
              This is where "some lineages move more, some talk more" lives.
  efficacy    Of the actions taken, which ones LANDED (tell_delivery,
              mate_delivery, mate_reciprocation, move_success). An agent can
              be perfectly well-formed and still address agents who are not
              there; that shows up here and nowhere else.
  economy     What it cost (tokens_per_turn, thinking_share, observed_per_turn).
  fitness     children, children_per_100_turns.

Counts come from the action events (`tell`, `mate_request`, `move`,
`move_failed`, `noop`), which are emitted for EVERY action. They are not taken
from `turn` events, which are capped by `run.trace_turns` and would silently
bias every composition toward whatever happened early in the run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .pedigree import Pedigree
from .table import Table

# Traits by group, for callers that want to analyse one family at a time.
TRAIT_GROUPS = {
    "form":     ["canonical_rate", "well_formed_rate", "malformed_share"],
    "strategy": ["tell_share", "mate_share", "move_share", "noop_share"],
    "efficacy": ["tell_delivery", "mate_delivery", "mate_reciprocation",
                 "move_success"],
    "economy":  ["tokens_per_turn", "thinking_share", "observed_per_turn",
                 "context_at_death"],
    "fitness":  ["children", "children_per_100_turns"],
}
ALL_TRAITS = [t for g in TRAIT_GROUPS.values() for t in g]


def _safe(num, den, default=np.nan):
    return num / den if den else default


def build_phenotypes(run_dir: str | Path, min_turns: int = 5,
                     pedigree: Pedigree | None = None) -> Table:
    """Scan a run's events once and return the per-agent trait table.

    `min_turns` drops agents that died before doing enough to measure. Rates
    over one or two turns are almost pure sampling noise, and including them
    inflates every variance in the analyses downstream.
    """
    run_dir = Path(run_dir)
    ped = pedigree or Pedigree.from_run(run_dir)
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    died: dict[str, dict] = {}

    for path in sorted(run_dir.glob("events/*.jsonl")):
        with path.open() as fh:
            for line in fh:
                e = json.loads(line)
                t = e.get("type")
                a = e.get("agent")
                if a is None:
                    continue
                if t == "death":
                    died[a] = e
                elif t == "tell":
                    acc[a]["tell"] += 1
                    acc[a]["tell_ok"] += bool(e.get("delivered"))
                elif t == "mate_request":
                    acc[a]["mate"] += 1
                    acc[a]["mate_ok"] += bool(e.get("delivered"))
                    acc[a]["mate_recip"] += bool(e.get("reciprocated"))
                elif t == "move":
                    acc[a]["move_ok"] += 1
                elif t == "move_failed":
                    acc[a]["move_bad"] += 1
                elif t == "noop":
                    acc[a]["noop"] += 1
                    if e.get("reason") == "malformed":
                        acc[a]["noop_malformed"] += 1

    records: dict[str, dict] = {}
    for agent, d in died.items():
        turns = d.get("turns", 0)
        if turns < min_turns:
            continue
        c = acc.get(agent, {})
        tell, mate = c.get("tell", 0), c.get("mate", 0)
        mv_ok, mv_bad = c.get("move_ok", 0), c.get("move_bad", 0)
        noop = c.get("noop", 0)
        acts = tell + mate + mv_ok + mv_bad + noop
        gen_tok = d.get("tokens_generated", 0)
        anc = ped.ancestry(agent) if agent in ped.parents else {}

        records[agent] = {
            # identity and pedigree
            "generation": d.get("generation", 0),
            "origin": d.get("origin", "child"),
            "room_died": d.get("room", ""),
            "room_born": ped.birth_room.get(agent, ""),
            "birth_step": ped.birth_step.get(agent, np.nan),
            "lineage": ped.lineage(agent) if anc else agent,
            "ancestry_entropy": ped.ancestry_entropy(agent) if anc else 0.0,
            "n_founders": len(anc),
            # exposure
            "turns": turns,
            "actions": acts,
            "lifetime_steps": d.get("lifetime_steps", np.nan),
            # form
            "canonical_rate": _safe(d.get("canonical_turns", 0), turns),
            "well_formed_rate": _safe(d.get("well_formed_turns", 0), turns),
            "malformed_share": _safe(c.get("noop_malformed", 0), acts),
            # strategy
            "tell_share": _safe(tell, acts),
            "mate_share": _safe(mate, acts),
            "move_share": _safe(mv_ok + mv_bad, acts),
            "noop_share": _safe(noop, acts),
            # efficacy
            "tell_delivery": _safe(c.get("tell_ok", 0), tell),
            "mate_delivery": _safe(c.get("mate_ok", 0), mate),
            "mate_reciprocation": _safe(c.get("mate_recip", 0), mate),
            "move_success": _safe(mv_ok, mv_ok + mv_bad),
            # economy
            "tokens_per_turn": _safe(gen_tok, turns),
            "thinking_share": _safe(d.get("thinking_tokens", 0), gen_tok),
            "observed_per_turn": _safe(d.get("tokens_observed", 0), turns),
            "context_at_death": d.get("tokens", np.nan),
            # fitness
            "children": d.get("children", 0),
            "children_per_100_turns": _safe(d.get("children", 0), turns) * 100,
        }
    return Table.from_records(records)


def strategy_matrix(table: Table) -> tuple[np.ndarray, list[str]]:
    """The action-composition simplex, rows renormalised to sum to 1.

    Agents whose composition is undefined (no actions) are returned as NaN
    rows; callers should mask them.
    """
    names = TRAIT_GROUPS["strategy"]
    M = np.column_stack([table[n].astype(float) for n in names])
    total = np.nansum(M, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(total > 0, M / total, np.nan)
    return M, names
