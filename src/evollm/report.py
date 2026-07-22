"""Offline aggregation of a run's event logs (§5).

Reads runs/<name>/events/*.jsonl and computes the minimum instrumentation
the design demands: handshake success from generation zero, child viability
separate from child survival, death-cause audit, occupancy/niche summaries,
and the terseness check (does lifetime correlate with mean action length,
§6 "context growth is under agent control").
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from .events import VALID_DEATH_CAUSES, read_events


def aggregate(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    events = []
    for path in sorted((run_dir / "events").glob("*.jsonl")):
        room = path.stem
        for e in read_events(path):
            e["room"] = e.get("room", room)
            events.append(e)

    births_by_gen = Counter()
    deaths_by_cause = Counter()
    lifetimes_by_gen = defaultdict(list)
    viability_by_gen = defaultdict(lambda: [0, 0])   # viable, total
    mate_requests = mate_delivered = accepts = accepts_valid = 0
    births = birth_failures = 0
    lifetime_vs_verbosity = []                        # (lifetime, mean_action_tokens)
    occupancy = defaultdict(list)
    invalid_deaths = []

    for e in events:
        t = e["type"]
        if t == "birth":
            births += 1
            births_by_gen[e["generation"]] += 1
        elif t == "birth_failed":
            birth_failures += 1
        elif t == "death":
            deaths_by_cause[e["cause"]] += 1
            if e["cause"] not in VALID_DEATH_CAUSES:
                invalid_deaths.append(e)
            lifetimes_by_gen[e["generation"]].append(e["lifetime_steps"])
            lifetime_vs_verbosity.append(
                (e["lifetime_steps"], e.get("mean_action_tokens", 0.0)))
        elif t == "mate_request":
            mate_requests += 1
            mate_delivered += bool(e["delivered"])
        elif t == "mate_accept":
            accepts += 1
            accepts_valid += bool(e["valid"])
        elif t == "viability":
            v = viability_by_gen[e["generation"]]
            v[0] += bool(e["viable"])
            v[1] += 1
        elif t == "occupancy":
            occupancy[e["room"]].append(e)

    total_deaths = sum(deaths_by_cause.values())
    return {
        "events": len(events),
        "births": births,
        "births_by_generation": dict(sorted(births_by_gen.items())),
        "birth_failures": birth_failures,
        "deaths": total_deaths,
        "deaths_by_cause": dict(deaths_by_cause),
        "invalid_deaths": invalid_deaths,   # must be empty (§4.3)
        "mean_lifetime_by_generation": {
            g: round(sum(v) / len(v), 1)
            for g, v in sorted(lifetimes_by_gen.items())
        },
        "handshake": {
            "requests": mate_requests,
            "delivered": mate_delivered,
            "accepts": accepts,
            "valid_accepts": accepts_valid,
            "births": births_by_gen and
            sum(n for g, n in births_by_gen.items() if g > 0) or 0,
            "request_to_birth_rate": round(
                (sum(n for g, n in births_by_gen.items() if g > 0) /
                 mate_requests), 4) if mate_requests else None,
        },
        "viability_by_generation": {
            g: {"viable": v[0], "probed": v[1],
                "rate": round(v[0] / v[1], 3) if v[1] else None}
            for g, v in sorted(viability_by_gen.items())
        },
        "terseness_check": _correlation(lifetime_vs_verbosity),
        "occupancy_last": {
            room: snaps[-1] for room, snaps in occupancy.items() if snaps
        },
    }


def _correlation(pairs: list[tuple[float, float]]) -> dict:
    """Pearson r between lifetime and mean action length (§6). A strong
    positive/negative r means the economy is selecting for verbosity/terseness
    rather than for anything environmental."""
    n = len(pairs)
    if n < 3:
        return {"n": n, "r": None}
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return {"n": n, "r": None}
    return {"n": n, "r": round(cov / math.sqrt(vx * vy), 3)}


def format_report(stats: dict) -> str:
    lines = ["── evoLLM run report ──"]
    lines.append(f"events: {stats['events']}")
    lines.append(f"births: {stats['births']}  (failed: {stats['birth_failures']})")
    lines.append(f"  by generation: {stats['births_by_generation']}")
    lines.append(f"deaths: {stats['deaths']}  by cause: {stats['deaths_by_cause']}")
    if stats["invalid_deaths"]:
        lines.append(f"  !! INVALID DEATHS (integrity): {len(stats['invalid_deaths'])}")
    else:
        lines.append("  death-cause audit: clean (all deaths are scarcity events)")
    lines.append(f"mean lifetime by generation: {stats['mean_lifetime_by_generation']}")
    h = stats["handshake"]
    lines.append(
        f"handshake: {h['requests']} requests, {h['delivered']} delivered, "
        f"{h['valid_accepts']}/{h['accepts']} accepts valid, "
        f"request→birth rate: {h['request_to_birth_rate']}")
    lines.append("viability by generation:")
    for g, v in stats["viability_by_generation"].items():
        lines.append(f"  gen {g}: {v['viable']}/{v['probed']} viable ({v['rate']})")
    t = stats["terseness_check"]
    lines.append(f"terseness check (lifetime vs mean action tokens): "
                 f"r={t['r']} over n={t['n']} deaths")
    for room, occ in stats["occupancy_last"].items():
        lines.append(
            f"room {room} @ step {occ['step']}: {occ['agents']} agents, "
            f"{occ['free_blocks']}/{occ['capacity_blocks']} blocks free, "
            f"mean context {occ['mean_context']}, generations {occ['generations']}")
    return "\n".join(lines)
