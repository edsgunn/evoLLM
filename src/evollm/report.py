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
    forms = Counter()
    refills = 0
    takeoffs = []
    takeoffs_lost = 0
    origin_counts = Counter()
    timeline = []          # (step, "birth"|"refill") for the takeoff curve
    max_step = 0
    noop_reasons = Counter()
    think_tokens: list[int] = []
    turn_tokens: list[int] = []

    for e in events:
        t = e["type"]
        max_step = max(max_step, e.get("step", 0))
        if t == "birth":
            births += 1
            births_by_gen[e["generation"]] += 1
            origin = e.get("origin", "seed")
            origin_counts[origin] += 1
            if origin in ("birth", "refill"):
                timeline.append((e["step"], origin))
        elif t == "refill":
            refills += 1
        elif t == "takeoff":
            takeoffs.append(e)
        elif t == "takeoff_lost":
            takeoffs_lost += 1
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
        elif t == "turn":
            forms[e["form"]] += 1
            think_tokens.append(e.get("thinking_tokens", 0))
            turn_tokens.append(e.get("turn_tokens", 0))
        elif t == "noop":
            noop_reasons[e["reason"]] += 1

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
        "refills": refills,
        "origins": dict(origin_counts),
        "takeoff": _takeoff(timeline, max_step),
        "takeoff_events": [
            {"room": e["room"], "step": e["step"],
             "population": e["population"], "generations": e["generations"],
             "refills_before": e["refills_before"]} for e in takeoffs],
        "takeoffs_lost": takeoffs_lost,
        "action_forms": dict(forms.most_common()),
        "noop_reasons": dict(noop_reasons),
        "thinking": {
            "mean_tokens_before_acting": round(sum(think_tokens) / len(think_tokens), 1)
            if think_tokens else None,
            "mean_turn_tokens": round(sum(turn_tokens) / len(turn_tokens), 1)
            if turn_tokens else None,
            "share_of_generation": round(sum(think_tokens) / sum(turn_tokens), 3)
            if sum(turn_tokens) else None,
        },
        "terseness_check": _correlation(lifetime_vs_verbosity),
        "occupancy_last": {
            room: snaps[-1] for room, snaps in occupancy.items() if snaps
        },
    }


def _takeoff(timeline: list[tuple[int, str]], max_step: int,
             quartiles: int = 4) -> dict:
    """Is the population sustaining itself yet?

    Refill keeps the arena populated so selection has something to act on, and
    the price is that survival alone no longer proves anything. What does is
    the share of new agents that arrived by descent rather than immigration:
    `births / (births + refills)`, tracked over the run. Rising toward 1 is
    takeoff. Flat is a population being carried by immigration.
    """
    if not timeline or max_step <= 0:
        return {"self_sufficiency": None, "by_quartile": []}
    width = max(max_step // quartiles, 1)
    buckets = [[0, 0] for _ in range(quartiles)]   # births, refills
    for step, origin in timeline:
        i = min(step // width, quartiles - 1)
        buckets[i][0 if origin == "birth" else 1] += 1
    by_quartile = []
    for i, (b, r) in enumerate(buckets):
        total = b + r
        by_quartile.append({
            "steps": [i * width, (i + 1) * width],
            "births": b, "refills": r,
            "self_sufficiency": round(b / total, 3) if total else None,
        })
    tb = sum(b for b, _ in buckets)
    tr = sum(r for _, r in buckets)
    return {
        "self_sufficiency": round(tb / (tb + tr), 3) if (tb + tr) else None,
        "by_quartile": by_quartile,
    }


def _compact(by_gen: dict, keep: int = 6) -> str:
    """Runs now reach hundreds of generations; printing every one buries the
    report. Show the ends, which is where the trend lives."""
    items = list(by_gen.items())
    if len(items) <= keep * 2:
        return str(dict(items))
    head = ", ".join(f"{g}: {v}" for g, v in items[:keep])
    tail = ", ".join(f"{g}: {v}" for g, v in items[-keep:])
    return f"{{{head}, ... ({len(items) - keep * 2} more) ..., {tail}}}"


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
    lines.append("mean lifetime by generation: " +
                 _compact(stats["mean_lifetime_by_generation"]))
    h = stats["handshake"]
    lines.append(
        f"handshake: {h['requests']} requests, {h['delivered']} delivered, "
        f"{h['valid_accepts']}/{h['accepts']} accepts valid, "
        f"request→birth rate: {h['request_to_birth_rate']}")
    lines.append("viability by generation:")
    for g, v in stats["viability_by_generation"].items():
        lines.append(f"  gen {g}: {v['viable']}/{v['probed']} viable ({v['rate']})")
    if stats["action_forms"]:
        total = sum(stats["action_forms"].values())
        canonical = stats["action_forms"].get("canonical", 0)
        recovered = sum(v for k, v in stats["action_forms"].items()
                        if k not in ("canonical", "none"))
        lines.append(f"action forms (traced {total} turns): "
                     f"canonical {canonical} ({canonical / total:.0%}), "
                     f"recovered by tolerant parsing {recovered} "
                     f"({recovered / total:.0%}), unparseable "
                     f"{stats['action_forms'].get('none', 0)}")
        lines.append(f"  breakdown: {stats['action_forms']}")
    if stats["noop_reasons"]:
        lines.append(f"noop reasons: {stats['noop_reasons']}")
    th = stats["thinking"]
    if th["mean_turn_tokens"]:
        lines.append(
            f"thinking: {th['mean_tokens_before_acting']} tokens generated before "
            f"acting, of {th['mean_turn_tokens']} per turn "
            f"({th['share_of_generation']:.0%} of generation)")
    if stats["refills"]:
        tk = stats["takeoff"]
        lines.append(
            f"refills: {stats['refills']}  origins: {stats['origins']}")
        lines.append(
            f"self-sufficiency (births / births+refills): {tk['self_sufficiency']}"
            "   [1.0 = fully self-sustaining, takeoff = rising across quartiles]")
        for q in tk["by_quartile"]:
            lines.append(
                f"  steps {q['steps'][0]:>7}-{q['steps'][1]:<7} "
                f"births {q['births']:4d}  refills {q['refills']:4d}  "
                f"-> {q['self_sufficiency']}")
    for e in stats["takeoff_events"]:
        lines.append(
            f"TAKEOFF room {e['room']} @ step {e['step']}: population "
            f"{e['population']}, generations {e['generations']}, after "
            f"{e['refills_before']} refills — population checkpointed")
    if stats["takeoffs_lost"]:
        lines.append(f"  takeoff lapsed {stats['takeoffs_lost']}x "
                     "(needed immigrants again; not durable)")
    t = stats["terseness_check"]
    lines.append(f"terseness check (lifetime vs mean action tokens): "
                 f"r={t['r']} over n={t['n']} deaths")
    for room, occ in stats["occupancy_last"].items():
        lines.append(
            f"room {room} @ step {occ['step']}: {occ['agents']} agents, "
            f"{occ['free_blocks']}/{occ['capacity_blocks']} blocks free, "
            f"mean context {occ['mean_context']}, "
            f"obs backlog mean {occ.get('mean_backlog', 0)} max "
            f"{occ.get('max_backlog', 0)} tokens, "
            f"generations {occ['generations']}")
    return "\n".join(lines)
