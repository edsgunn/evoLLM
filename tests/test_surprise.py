"""Surprise instrumentation.

The project's hypothesis is that agents come to find their world less
surprising. These tests pin the part of that which is a measurement rather
than a result: that surprise is attributed to the tokens the WORLD wrote,
that it is filed under where the agent was in its own life, and that turning
it off leaves the run byte-identical to what it always produced.
"""
from __future__ import annotations

import pytest

from evollm.agent import (N_SURPRISE_BUCKETS, ORIGIN_FRAME, ORIGIN_GEN,
                          ORIGIN_OBS, Agent)
from evollm.engines.mock import heuristic_policy

from conftest import make_world, run_steps


def _agents(world):
    return [a for c in world.controllers.values() for a in c.agents.values()]


def _with_logprob(world, value=-2.0):
    for engine in {id(c.engine): c.engine for c in
                   world.controllers.values()}.values():
        engine.logprob = value
    return world


@pytest.mark.asyncio
async def test_token_origin_stays_parallel_to_context(tmp_cfg):
    """Origin is recorded at the one place tokens are appended, so it can
    never drift out of step with the context it describes. If it did, surprise
    would be attributed to the wrong tokens and every number downstream would
    be wrong in a way nothing else would catch."""
    world = make_world(tmp_cfg(pop=3, capacity=3_000), default_policy=heuristic_policy)
    await world.seed()
    await run_steps(world, 200)
    agents = _agents(world)
    assert agents
    for a in agents:
        assert len(a.token_origin) == len(a.context)
        assert set(a.token_origin) <= {ORIGIN_FRAME, ORIGIN_OBS, ORIGIN_GEN}


@pytest.mark.asyncio
async def test_surprise_is_recorded_only_for_observation_tokens(tmp_cfg):
    """The headline measure must not be contaminated by framing or by the
    agent's own output. The mock scores every uncached prompt position, so if
    the filter were absent the counted tokens would exceed the observations."""
    world = _with_logprob(make_world(tmp_cfg(pop=3, capacity=3_000), default_policy=heuristic_policy))
    await world.seed()
    await run_steps(world, 250)
    agents = _agents(world)
    scored = [a for a in agents if sum(a.obs_nll_tokens)]
    assert scored, "no agent absorbed a scored observation"
    for a in scored:
        n_obs = sum(1 for o in a.token_origin[:a.turn_prompt_len]
                    if o == ORIGIN_OBS)
        assert sum(a.obs_nll_tokens) <= n_obs


@pytest.mark.asyncio
async def test_surprise_is_filed_under_within_life_position(tmp_cfg):
    """A single lifetime number cannot answer whether an agent improves as it
    lives. Surprise is bucketed by the agent's own turn count, so a long life
    populates more than one bucket."""
    world = _with_logprob(make_world(tmp_cfg(pop=2, capacity=3_000), default_policy=heuristic_policy))
    await world.seed()
    await run_steps(world, 400)
    agents = [a for a in _agents(world) if a.action_turns_completed > 6]
    assert agents, "no agent lived long enough to cross a bucket edge"
    assert any(sum(1 for n in a.obs_nll_tokens if n) > 1 for a in agents), \
        "every agent's surprise landed in one bucket: no within-life curve"
    for a in agents:
        assert len(a.obs_nll_curve()) == N_SURPRISE_BUCKETS


@pytest.mark.asyncio
async def test_generated_surprise_is_kept_separate(tmp_cfg):
    """Observation surprise falling means nothing if the lineage simply became
    more confident about everything, so the fluency baseline is carried too --
    and in its own field."""
    world = _with_logprob(make_world(tmp_cfg(pop=2, capacity=3_000), default_policy=heuristic_policy))
    await world.seed()
    await run_steps(world, 300)
    agents = [a for a in _agents(world) if a.gen_nll_tokens]
    assert agents
    for a in agents:
        assert a.gen_nll_tokens <= a.tokens_generated
        assert a.mean_gen_nll == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_death_carries_the_within_life_curve(tmp_cfg):
    """Traces are sampled; deaths are not. The curve rides on the death event
    so within-lifetime change is answerable for the whole population."""
    cfg = tmp_cfg(pop=4, capacity=40)
    world = _with_logprob(make_world(cfg, default_policy=heuristic_policy))
    await world.seed()
    await run_steps(world, 400)
    world.close()
    curved = [d for d in _deaths(world) if d.get("obs_nll_curve")]
    assert curved, "no death carried a surprise curve"
    for d in curved:
        assert len(d["obs_nll_curve"]) == N_SURPRISE_BUCKETS
        assert len(d["obs_nll_counts"]) == N_SURPRISE_BUCKETS
        assert "gen_nll" in d


def _deaths(world):
    from pathlib import Path

    from evollm.events import read_events
    root = Path(world.cfg.run.out_dir) / world.cfg.run_name / "events"
    return [e for path in sorted(root.glob("*.jsonl"))
            for e in read_events(path) if e["type"] == "death"]


@pytest.mark.asyncio
async def test_no_surprise_fields_when_the_backend_reports_none(tmp_cfg):
    """A backend that supplies no logprobs must leave the event stream exactly
    as it was, rather than adding a column of nulls that later analyses would
    have to special-case."""
    cfg = tmp_cfg(pop=4, capacity=40)
    world = make_world(cfg, default_policy=heuristic_policy)   # engine.logprob None
    await world.seed()
    await run_steps(world, 400)
    world.close()
    deaths = _deaths(world)
    assert deaths, "test needs at least one death"
    for d in deaths:
        assert "obs_nll" not in d and "gen_nll" not in d


def test_uninitialised_prompt_logprobs_are_rejected():
    """vLLM allocates the prompt-logprob tensor for the whole prompt with
    `torch.empty` and fills only the positions it recomputed, so a cached
    position comes back as a Logprob over uninitialised memory rather than as
    None. Accepting those would attribute plausible-looking garbage to real
    observation tokens, and nothing downstream could tell.

    Two independent filters: the row's key must be the token actually at that
    position, and the value must be in the range a log-probability can occupy.
    """
    from evollm.engines.vllm_engine import _flatten_prompt_logprobs

    class LP:
        def __init__(self, v):
            self.logprob = v

    ids = [10, 11, 12, 13, 14]
    rows = [
        None,                 # position 0 has no context
        {11: LP(-1.5)},       # real
        {987654: LP(-2.0)},   # uninitialised token id: rejected
        {13: LP(-1e9)},       # uninitialised float: rejected
        {14: LP(3.2)},        # positive: not a logprob
    ]
    assert _flatten_prompt_logprobs(rows, ids) == [None, -1.5, None, None, None]


@pytest.mark.asyncio
async def test_each_observation_is_counted_once(tmp_cfg):
    """Prompt logprobs arrive aligned to the WHOLE prompt every turn, so
    without a lower bound the same observation would be re-scored on every
    subsequent turn and a long-lived agent's surprise would be dominated by
    its oldest tokens."""
    world = _with_logprob(make_world(tmp_cfg(pop=2, capacity=3_000),
                                     default_policy=heuristic_policy))
    await world.seed()
    await run_steps(world, 400)
    agents = [a for a in _agents(world) if sum(a.obs_nll_tokens)]
    assert agents
    for a in agents:
        n_obs = sum(1 for o in a.token_origin if o == ORIGIN_OBS)
        assert sum(a.obs_nll_tokens) <= n_obs, \
            "an observation was scored more than once"
