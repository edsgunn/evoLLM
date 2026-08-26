"""Within-lifetime analysis and trace inspection."""
from __future__ import annotations

import json

from evollm.agent import N_SURPRISE_BUCKETS
from evollm.analysis import (action_curve, build_bundle, format_inspection,
                             format_lifecourse, inspect, sample_for_review,
                             surprise_curve)


def _run(tmp_path, deaths=(), turns=()):
    d = tmp_path / "events"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "r0.jsonl", "w") as f:
        for e in list(deaths) + list(turns):
            f.write(json.dumps(e) + "\n")
    return str(tmp_path)


def _death(agent, curve, generation=0):
    return {"type": "death", "step": 1, "agent": agent, "cause": "x",
            "generation": generation, "obs_nll_curve": curve,
            "obs_nll_counts": [10] * N_SURPRISE_BUCKETS}


def test_surprise_curve_reports_unavailable_rather_than_zero(tmp_path):
    """A run that predates surprise recording must say so. Returning 0.0, or
    an empty curve, would read downstream as 'no learning' -- a result -- when
    the truth is that nothing was measured."""
    run = _run(tmp_path, deaths=[{"type": "death", "step": 1, "agent": "a",
                                  "cause": "x", "generation": 0}])
    out = surprise_curve(run)
    assert out["available"] is False
    assert "reason" in out


def test_surprise_curve_detects_a_within_life_fall(tmp_path):
    curve = [3.0, 2.5, 2.0, 1.5, None, None]
    run = _run(tmp_path, deaths=[_death(f"a{i}", curve, generation=i)
                                 for i in range(60)])
    out = surprise_curve(run)
    assert out["available"]
    m, ci, n = out["within_agent_change"]
    assert m < 0 and m + ci < 0, "a monotone fall must read as a fall"
    assert n == 60


def test_surprise_change_is_paired_not_cross_sectional(tmp_path):
    """Agents that live longer may simply be better agents. If the change were
    computed by comparing the population's early bucket against its late one,
    a cohort where only the low-surprise agents survive would look like
    learning. Pairing within agent removes that: here nobody changes."""
    short = [5.0, None, None, None, None, None]
    long = [1.0, 1.0, 1.0, 1.0, None, None]
    deaths = ([_death(f"s{i}", short) for i in range(50)]
              + [_death(f"l{i}", long) for i in range(50)])
    out = surprise_curve(_run(tmp_path, deaths=deaths))
    m, ci, _ = out["within_agent_change"]
    assert abs(m) < 1e-9, "flat lives must show no within-agent change"
    pop = out["population"]
    assert pop[0][0] > pop[1][0], "the cross-sectional curve DOES fall here"


def _turn(agent, idx, form="canonical", gen=0, text="<go>gpu1</go>"):
    return {"type": "turn", "step": idx, "agent": agent, "generation": gen,
            "action": "Go", "form": form, "turn_index": idx, "text": text}


def test_action_curve_splits_each_life_into_its_own_quantiles(tmp_path):
    """Agents live for wildly different numbers of turns, so position in life
    has to be relative. A fixed turn cutoff would put a short life entirely in
    'early' and compare different agents rather than the same one twice."""
    turns = []
    for a in range(40):
        for i in range(20):
            form = "canonical" if i >= 10 else "none"
            turns.append(_turn(f"a{a}", i, form=form, gen=a))
    out = action_curve(_run(tmp_path, turns=turns))
    assert out["available"]
    first = out["canonical_by_quantile"][0][0]
    last = out["canonical_by_quantile"][-1][0]
    assert first == 0.0 and last == 1.0
    m, ci, _ = out["within_agent_change"]
    assert m > 0 and m - ci > 0


def test_trace_inspection_finds_the_placeholder_tic(tmp_path):
    """The bug that swept whole populations: a well-formed action whose target
    is the prompt's own placeholder. Every parse-based metric calls this a
    canonical action, so only reading the text can catch it."""
    turns = [_turn(f"a{i}", i, text="<go>room_id</go>") for i in range(50)]
    out = inspect(_run(tmp_path, turns=turns))
    keys = {f["key"] for f in out["findings"]}
    assert "prompt_slots_echoed" in keys
    assert "room_id" in format_inspection(out)


def test_trace_inspection_finds_stuck_agents(tmp_path):
    turns = [_turn("a0", i, text="<go>gpu1</go>") for i in range(20)]
    out = inspect(_run(tmp_path, turns=turns))
    assert "stuck_agents" in {f["key"] for f in out["findings"]}


def test_review_sample_is_stratified_over_parse_outcome(tmp_path):
    """An unstratified sample of a healthy run is almost all well-formed turns,
    and the malformed ones are the point."""
    turns = ([_turn(f"good{i}", i, gen=i) for i in range(500)]
             + [_turn(f"bad{i}", i, form="none", gen=i) for i in range(10)])
    sample = sample_for_review(_run(tmp_path, turns=turns), n=40)
    assert any(t["form"] == "none" for t in sample), "rare form was lost"


def test_bundle_is_text_and_sends_nothing(tmp_path):
    turns = [_turn(f"a{i}", i, gen=i) for i in range(50)]
    text = build_bundle(_run(tmp_path, turns=turns), n=10)
    assert "<go>gpu1</go>" in text and isinstance(text, str)


def test_format_lifecourse_handles_both_sections_missing(tmp_path):
    run = _run(tmp_path)
    text = format_lifecourse(surprise_curve(run), action_curve(run))
    assert "unavailable" in text
