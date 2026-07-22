"""Environment-dynamics tests over the mock engine: the properties §2 states
should hold by construction are asserted here."""

import numpy as np

from evollm.config import EVICT_RANDOM_HOLDER
from evollm.engines.mock import (accept_all_policy, chatty_policy, quiet_policy,
                                 scripted)
from evollm.events import read_events

from conftest import make_world, run_steps


def room(world, rid="r0"):
    return world.controllers[rid]


def events_of(world, rid="r0"):
    return list(read_events(room(world, rid).log.path))


# ── §2.3: death is inevitable, quietness is not rewarded ──────────────────
async def test_death_is_inevitable_even_for_quiet_agents(tmp_cfg):
    cfg = tmp_cfg(capacity=8, pop=2)
    world = make_world(cfg)  # default quiet policy: 1 token per turn minimum
    await world.seed()
    await run_steps(world, 500)
    assert not room(world).agents, "no agent should survive a full pool"
    deaths = [e for e in events_of(world) if e["type"] == "death"]
    assert len(deaths) == 2
    for d in deaths:
        assert d["cause"] in ("pool_exhausted_requester", "pool_exhausted_evicted")
    assert room(world).pool.used == 0


async def test_context_grows_every_turn_under_quietness(tmp_cfg):
    cfg = tmp_cfg(capacity=100_000, pop=1)
    world = make_world(cfg)
    await world.seed()
    agent = next(iter(room(world).agents.values()))
    await run_steps(world, 50)
    assert agent.tokens == 50  # exactly one token per step, observing or acting
    before = agent.tokens
    await run_steps(world, 400)
    assert agent.tokens == before + 400


# ── §2.4: the mate handshake ──────────────────────────────────────────────
async def test_handshake_produces_child(tmp_cfg):
    cfg = tmp_cfg(capacity=500, pop=2)
    world = make_world(cfg, policies={
        "a0": scripted(["<mate>a1</mate>"]),
        "a1": accept_all_policy,
    })
    await world.seed()
    await run_steps(world, 400)
    generations = {a.generation for a in room(world).agents.values()}
    assert 1 in generations, "handshake should have produced a gen-1 child"
    child = next(a for a in room(world).agents.values() if a.generation == 1)
    assert set(child.parents) == {"a0", "a1"}
    # child pays for its own system prompt like everyone else
    assert child.tokens > 0 or child.obs_queue
    # both parents were notified
    births = [e for e in events_of(world) if e["type"] == "birth"
              and e.get("generation") == 1]
    assert len(births) == 1


async def test_acceptance_window_expires(tmp_cfg):
    cfg = tmp_cfg(capacity=500, pop=2, mate_window_tokens=1)
    world = make_world(cfg, policies={
        "a0": scripted(["<mate>a1</mate>"]),
        "a1": accept_all_policy,
    })
    await world.seed()
    await run_steps(world, 400)
    generations = {a.generation for a in room(world).agents.values()}
    assert generations == {0}, "acceptance after the window must not reproduce"
    accepts = [e for e in events_of(world) if e["type"] == "mate_accept"]
    assert accepts and not any(e["valid"] for e in accepts)


async def test_birth_fails_on_adapter_blocks(tmp_cfg):
    cfg = tmp_cfg(capacity=500, pop=2)
    world = make_world(cfg)
    await world.seed()
    controller = room(world)
    agents = sorted(controller.agents.values(), key=lambda a: a.id)
    # exhaust the pool so the child's adapter cannot be reserved
    assert controller.pool.try_reserve_adapter("filler", controller.pool.free)
    await controller._birth(agents[0], agents[1])
    assert len(controller.agents) == 2 and not controller._pending_arrivals
    failed = [e for e in events_of(world) if e["type"] == "birth_failed"]
    assert len(failed) == 1
    # both parents were told, on their observation queues
    assert agents[0].obs_queue and agents[1].obs_queue


# ── §2.5: eviction policies ───────────────────────────────────────────────
async def test_requester_dies_policy(tmp_cfg):
    cfg = tmp_cfg(capacity=8, pop=2)
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 500)
    deaths = [e for e in events_of(world) if e["type"] == "death"]
    assert all(d["cause"] == "pool_exhausted_requester" for d in deaths)


async def test_random_holder_eviction_kills_large_holder(tmp_cfg):
    cfg = tmp_cfg(capacity=60, pop=2, eviction=EVICT_RANDOM_HOLDER)
    world = make_world(cfg)
    await world.seed()
    controller = room(world)
    a0, a1 = (controller.agents[i] for i in ("a0", "a1"))
    # a1 grabs every remaining block
    while controller.pool.try_grow_kv("a1", a1.tokens + controller.pool.block_size):
        a1.context.extend([1] * controller.pool.block_size)
    assert controller.pool.free == 0
    controller.rng = np.random.default_rng(0)  # a1 holds ~93% of blocks
    survived = controller._append_token(a0, 1)
    for coro in controller._pending_cleanup:
        await coro
    controller._pending_cleanup.clear()
    assert survived and "a0" in controller.agents
    assert "a1" not in controller.agents
    deaths = [e for e in events_of(world) if e["type"] == "death"]
    assert deaths[0]["agent"] == "a1"
    assert deaths[0]["cause"] == "pool_exhausted_evicted"


# ── §2.4: say broadcasts to all, and only, others in the room ─────────────
async def test_say_reaches_all_listeners(tmp_cfg):
    cfg = tmp_cfg(capacity=2000, pop=3)
    world = make_world(cfg, policies={"a0": scripted(["<say>news</say>"])})
    await world.seed()
    await run_steps(world, 250)
    says = [e for e in events_of(world) if e["type"] == "say"]
    assert says and says[0]["listeners"] == 2
    controller = room(world)
    assert controller.agents["a1"].tokens_observed > \
        controller.agents["a0"].tokens_observed
    assert controller.agents["a2"].tokens_observed == \
        controller.agents["a1"].tokens_observed


# ── §3.2/§5: viability probe ──────────────────────────────────────────────
async def test_viability_probe(tmp_cfg):
    cfg = tmp_cfg(capacity=5000, pop=2, viability_probe_turns=3)
    world = make_world(cfg, policies={"a0": chatty_policy, "a1": quiet_policy})
    await world.seed()
    await run_steps(world, 400)
    probes = {e["agent"]: e for e in events_of(world) if e["type"] == "viability"}
    assert probes["a0"]["viable"] is True
    assert probes["a1"]["viable"] is False  # empty turns are Noops
    assert not probes["a0"]["censored"]


async def test_viability_censored_on_early_death(tmp_cfg):
    cfg = tmp_cfg(capacity=8, pop=1, viability_probe_turns=50)
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 300)
    probes = [e for e in events_of(world) if e["type"] == "viability"]
    assert probes and probes[0]["censored"] is True


# ── engine bookkeeping on death ───────────────────────────────────────────
async def test_death_unregisters_adapter(tmp_cfg):
    cfg = tmp_cfg(capacity=8, pop=1)
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 300)
    engine = room(world).engine
    assert "a0" in engine.unregistered
    assert "a0" not in engine.registered
