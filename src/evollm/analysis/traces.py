"""Read what agents actually wrote.

A run records tens of thousands of turns of raw text and, until now, nothing
read them. Every metric in this package counts *how a turn parsed* — canonical,
well-formed, noop — which is a projection onto four categories chosen in
advance. It cannot show a failure mode nobody thought of, and the most
expensive bug the project has hit was exactly that shape: agents emitted the
prompt's own placeholder as a move target, a perfectly well-formed action that
always failed, and it swept whole populations while every metric said the
population was fine.

This module is the cheap pass, meant to run on every run. It is deterministic
string work: no model, no API, seconds not hours. It answers "is anything
systematically wrong with how these agents are being prompted or configured",
which is a different question from "are they getting better".

For questions that genuinely need a reader, `sample_for_review` prepares a
bundle — see `evollm inspect-traces --bundle`.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict

# Slots that appear in the prompt template. An agent that emits one of these as
# an identifier has copied the instructions rather than read the room. The
# parser strips angle brackets and braces, so `{room}` arrives as `room`: both
# spellings must be counted or a run that has the bug looks clean.
PROMPT_SLOTS = frozenset({
    "room_id", "agent_id", "sender_id", "your_id", "target_id", "text",
    "room", "agent", "sender", "you", "target", "message", "id", "name",
    "recipient", "other_agent", "someone",
})

_TAG = re.compile(r"<\s*(/?)\s*(say|tell|mate|go|accept|think)\b[^>]*>",
                  re.IGNORECASE)
_WS = re.compile(r"\s+")


def iter_turns(run: str):
    """Every traced turn, in file order, tagged with its room."""
    for path in sorted(glob.glob(os.path.join(run, "events", "*.jsonl"))):
        room = os.path.basename(path).split(".")[0]
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("type") == "turn":
                    e["room"] = room
                    yield e


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def _targets(run: str) -> dict:
    """Why moves and directed speech failed, from the events rather than the
    text — the authoritative record of what the world rejected."""
    reasons = Counter()
    slot_targets = Counter()
    by_room_known = defaultdict(set)
    for path in sorted(glob.glob(os.path.join(run, "events", "*.jsonl"))):
        room = os.path.basename(path).split(".")[0]
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                t = e.get("type")
                if t == "birth":
                    by_room_known[room].add(e.get("agent"))
                elif t in ("move_failed", "tell_failed", "mate_failed"):
                    target = str(e.get("to") or e.get("target") or "")
                    reasons[f"{t}:{e.get('reason', 'unspecified')}"] += 1
                    if target.strip("<>{}[] ").lower() in PROMPT_SLOTS:
                        slot_targets[target] += 1
    return {"failure_reasons": dict(reasons.most_common(20)),
            "prompt_slot_targets": dict(slot_targets.most_common(10))}


def inspect(run: str, top: int = 15) -> dict:
    """The cheap pass. Returns findings, most alarming first."""
    turns = list(iter_turns(run))
    if not turns:
        return {"available": False, "reason": "no traced turns in this run"}

    n = len(turns)
    forms = Counter(t.get("form", "?") for t in turns)
    actions = Counter(t.get("action", "?") for t in turns)
    exact = Counter()
    slot_mentions = Counter()
    multi_tag = 0
    empty = 0
    unclosed = 0
    non_ascii = 0
    per_agent = defaultdict(list)

    for t in turns:
        text = t.get("text", "")
        norm = _norm(text)
        exact[norm] += 1
        per_agent[t["agent"]].append(norm)
        if not norm:
            empty += 1
        opens = [m for m in _TAG.finditer(text) if not m.group(1)]
        closes = [m for m in _TAG.finditer(text) if m.group(1)]
        if len(opens) > 1:
            multi_tag += 1
        if len(opens) > len(closes):
            unclosed += 1
        if any(ord(c) > 127 for c in text):
            non_ascii += 1
        for word in re.findall(r"[A-Za-z_]+", text):
            if word.lower() in PROMPT_SLOTS:
                slot_mentions[word] += 1

    # Self-repetition: an agent emitting the identical turn over and over is
    # not acting on its context, whatever it parses as.
    repeat_agents = 0
    stuck = Counter()
    for agent, texts in per_agent.items():
        if len(texts) < 4:
            continue
        c = Counter(texts)
        top_text, count = c.most_common(1)[0]
        if count / len(texts) >= 0.8:
            repeat_agents += 1
            stuck[top_text] += 1

    findings = []
    def note(severity, key, detail, **extra):
        findings.append({"severity": severity, "key": key,
                         "detail": detail, **extra})

    slot_total = sum(slot_mentions.values())
    if slot_total:
        note("high", "prompt_slots_echoed",
             f"{slot_total:,} mentions of prompt placeholder words across "
             f"{n:,} turns; agents are copying the template",
             counts=dict(slot_mentions.most_common(top)))
    if forms.get("none", 0) / n > 0.10:
        note("high", "unparsed_turns",
             f"{forms['none'] / n:.1%} of turns parsed to no action")
    if empty / n > 0.05:
        note("medium", "empty_turns", f"{empty / n:.1%} of turns were empty")
    if repeat_agents:
        note("high" if repeat_agents / max(len(per_agent), 1) > 0.1 else "medium",
             "stuck_agents",
             f"{repeat_agents:,} of {len(per_agent):,} traced agents emitted "
             f"the same turn for 80%+ of their traced life",
             examples=[t[:120] for t, _ in stuck.most_common(5)])
    if multi_tag / n > 0.05:
        note("medium", "multiple_actions_per_turn",
             f"{multi_tag / n:.1%} of turns contained more than one action "
             "tag; only the first is taken, so the rest is paid for and lost")
    if unclosed / n > 0.05:
        note("medium", "unclosed_tags",
             f"{unclosed / n:.1%} of turns left an action tag unclosed")
    if non_ascii / n > 0.05:
        note("low", "non_ascii",
             f"{non_ascii / n:.1%} of turns contain non-ASCII characters")

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order[f["severity"]])

    return {
        "available": True,
        "n_turns": n,
        "n_agents": len(per_agent),
        "findings": findings,
        "forms": dict(forms.most_common()),
        "actions": dict(actions.most_common()),
        "most_common_turns": [{"text": t[:160], "count": c,
                               "share": round(c / n, 4)}
                              for t, c in exact.most_common(top)],
        **_targets(run),
    }


def sample_for_review(run: str, n: int = 200, seed: int = 0,
                      stratify: bool = True) -> list[dict]:
    """A sample of turns prepared for a reader — human or model.

    Stratified over generation so the sample is not all founders, and over
    parse outcome so the malformed turns that are the point are not swamped by
    the majority that parsed fine. An unstratified uniform sample of a healthy
    run is 97% identical well-formed moves and tells nobody anything.
    """
    import numpy as np
    turns = list(iter_turns(run))
    if not turns:
        return []
    rng = np.random.default_rng(seed)
    if not stratify:
        idx = rng.choice(len(turns), min(n, len(turns)), replace=False)
        return [turns[i] for i in idx]

    gens = np.array([t.get("generation", 0) for t in turns])
    edges = np.quantile(gens, [0, 0.25, 0.5, 0.75, 1.0])
    buckets = defaultdict(list)
    for i, t in enumerate(turns):
        band = int(np.searchsorted(edges[1:-1], gens[i], side="right"))
        buckets[(band, t.get("form", "?"))].append(i)

    out, keys = [], sorted(buckets)
    per = max(1, n // max(len(keys), 1))
    for k in keys:
        pool = buckets[k]
        take = rng.choice(len(pool), min(per, len(pool)), replace=False)
        out.extend(turns[pool[i]] for i in take)
    rng.shuffle(out)
    return out[:n]


REVIEW_PROMPT = """\
Below are sampled turns from an experiment in which language-model agents,
each carrying its own LoRA adapter, live in a shared room. They read what the
world tells them and act by emitting one tagged action per turn: <say>, <tell>,
<mate>, <go> or <accept>. They are charged tokens for everything they read and
write, and they die when the room runs out of memory.

You are reading these to find problems with how the agents are being PROMPTED
or CONFIGURED -- not to judge whether the agents are intelligent. Specifically:

1. Are they misunderstanding the protocol, and if so, what exactly misleads
   them? Quote the turn.
2. Are they copying anything from their instructions verbatim (placeholder
   names, examples) instead of using real values?
3. Is there any behaviour that is well-formed and accepted by the parser but
   obviously useless or self-defeating?
4. Does anything suggest the prompt is ambiguous, contradictory, or missing
   information they visibly need?
5. What single change to the prompt would most improve their behaviour?

Be concrete and quote turns. Say "nothing found" for any question where you
see no evidence, rather than speculating.

TURNS
-----
"""


def build_bundle(run: str, n: int = 200, seed: int = 0) -> str:
    """The full text to hand a reviewing model. Written to a file rather than
    sent anywhere: a full LLM pass over traces is expensive, so it is opt-in
    and the sample is inspectable before anyone pays for it."""
    sample = sample_for_review(run, n=n, seed=seed)
    lines = [REVIEW_PROMPT]
    for t in sample:
        lines.append(
            f"[gen {t.get('generation', '?')} | turn {t.get('turn_index', '?')}"
            f" | parsed {t.get('action', '?')}/{t.get('form', '?')}]\n"
            f"{t.get('text', '')}\n")
    return "\n".join(lines)


def format_inspection(result: dict) -> str:
    if not result.get("available"):
        return f"TRACE INSPECTION\n{'=' * 60}\nunavailable: {result.get('reason')}"
    L = ["TRACE INSPECTION", "=" * 60,
         f"{result['n_turns']:,} traced turns from {result['n_agents']:,} agents",
         ""]
    if not result["findings"]:
        L.append("No systematic problems found by the cheap pass.")
    for f in result["findings"]:
        L.append(f"[{f['severity'].upper():6s}] {f['key']}")
        L.append(f"          {f['detail']}")
        if "counts" in f:
            L.append("          " + ", ".join(
                f"{k}×{v:,}" for k, v in list(f["counts"].items())[:8]))
        for ex in f.get("examples", []):
            L.append(f"          e.g. {ex!r}")
    L += ["", "Parse forms: " + ", ".join(
        f"{k} {v:,}" for k, v in result["forms"].items())]
    L.append("Actions:     " + ", ".join(
        f"{k} {v:,}" for k, v in result["actions"].items()))
    if result.get("prompt_slot_targets"):
        L.append("Prompt-slot move targets: " + ", ".join(
            f"{k}×{v:,}" for k, v in result["prompt_slot_targets"].items()))
    L += ["", "Most repeated turns:"]
    for row in result["most_common_turns"][:8]:
        L.append(f"  {row['share']:6.2%}  {row['text']!r}")
    return "\n".join(L)
