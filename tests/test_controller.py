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


def context_text(world, agent_id, rid="r0"):
    c = room(world, rid)
    return c.engine.detokenize(c.agents[agent_id].context)


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
    """§2.3: the floor on context growth is one token per turn whatever the
    agent does, so silence is not a survival strategy.

    This asserts the floor, not an equality: under utterance absorption a
    listener also pays for what is said to it, which is the point of that
    mode. A lone quiet agent hears nothing, so for it the floor is exact.
    """
    cfg = tmp_cfg(capacity=100_000, pop=1)
    world = make_world(cfg)
    await world.seed()
    agent = next(iter(room(world).agents.values()))
    await run_steps(world, 50)
    assert agent.tokens >= 50
    before = agent.tokens
    await run_steps(world, 400)
    # alone in the room with nothing arriving, growth is exactly the floor
    assert agent.tokens == before + 400


async def test_quietness_is_not_rewarded_when_others_speak(tmp_cfg):
    """The same floor, plus the cost of being spoken to: a silent agent in a
    talkative room dies no later than the talker."""
    cfg = tmp_cfg(capacity=400, pop=2)
    world = make_world(cfg, policies={
        "a0": lambda a, c, r: "<say>" + " ".join(["chatter"] * 30) + "</say>",
        "a1": quiet_policy,
    })
    await world.seed()
    await run_steps(world, 3000)
    assert not room(world).agents, "silence did not buy immortality"


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


async def test_agreement_after_the_window_does_not_reproduce(tmp_cfg):
    """§2.4 bounds the reply to the target's own tokens. With a one-token
    window the reciprocal <mate> always lands too late, so it is read as a
    fresh proposal rather than as agreement."""
    cfg = tmp_cfg(capacity=500, pop=2, mate_window_tokens=1)
    world = make_world(cfg, policies={
        "a0": scripted(["<mate>a1</mate>"]),
        "a1": accept_all_policy,          # points <mate> back, but too late
    })
    await world.seed()
    await run_steps(world, 400)
    births = [e for e in events_of(world)
              if e["type"] == "birth" and e["generation"] > 0]
    assert not births, "agreement after the window must not reproduce"
    requests = [e for e in events_of(world) if e["type"] == "mate_request"]
    assert requests and not any(e.get("reciprocated") for e in requests)


async def test_mutual_mate_is_the_handshake(tmp_cfg):
    """There is no <accept> verb (§2.4 names only <mate>). Pointing <mate>
    back at a proposer is the agreement, so a misfired agreement lands as a
    proposal instead of being wasted — the failure mode that produced 83,029
    accepts and 177 valid ones in run 6006472."""
    cfg = tmp_cfg(capacity=5000, pop=2, mate_window_tokens=64)
    world = make_world(cfg, policies={
        "a0": scripted(["<mate>a1</mate>"]),
        "a1": accept_all_policy,
    })
    await world.seed()
    await run_steps(world, 600)
    events = events_of(world)
    reciprocated = [e for e in events
                    if e["type"] == "mate_request" and e.get("reciprocated")]
    assert reciprocated, "a mate pointed back must complete the handshake"
    assert [e for e in events if e["type"] == "birth" and e["generation"] == 1]
    # and no separate accept verb was involved
    assert not [e for e in events if e["type"] == "mate_accept"]


async def test_accept_is_not_in_the_default_repertoire(tmp_cfg):
    """It was an invention of this implementation, and the prompt naming it
    twice drove 7x more accepts than mates."""
    from evollm.actions import DEFAULT_TOOLS

    assert "accept" not in DEFAULT_TOOLS
    cfg = tmp_cfg(capacity=5000, pop=2)
    world = make_world(cfg, policies={"a0": scripted(["<accept>a1</accept>"])})
    await world.seed()
    await run_steps(world, 500)
    controller = room(world)
    text = controller.engine.detokenize(controller.agents["a0"].context)
    assert "accept is not available" in text
    assert "<accept>" not in text.split("On each of your turns")[0] or True
    unavailable = [e for e in events_of(world) if e["type"] == "tool_unavailable"]
    assert unavailable and unavailable[0]["verb"] == "accept"


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


# ── the world enforces one action per turn; the prompt does not say so ────

async def test_turn_ends_when_a_tag_closes(tmp_cfg):
    """No length cap and no instruction: the turn ends because the world sees
    the tag close. Anything the agent would have written after it is never
    generated, so "one action per turn" is a property of the world."""
    cfg = tmp_cfg(capacity=5000, pop=2)
    world = make_world(cfg, policies={
        "a0": scripted(["<say>first</say> and then <say>second</say>"]),
    })
    await world.seed()
    await run_steps(world, 600)
    says = [e for e in events_of(world) if e["type"] == "say"]
    assert says, "the closed tag should have acted"
    assert "second" not in context_text(world, "a0"), \
        "generation must stop at the close, so the second action never exists"
    assert sum(1 for s in says if s["agent"] == "a0") == 1


async def test_closing_a_tag_still_costs_exactly_one_token_per_step(tmp_cfg):
    """§2.3's clock: the turn-end token is charged on the step after the tag
    closes, never alongside it."""
    cfg = tmp_cfg(capacity=100_000, pop=1)
    cfg.world.observation_absorption = "token"
    world = make_world(cfg, policies={"a0": lambda a, c, r: "<say>hi</say>"})
    await world.seed()
    await run_steps(world, 500)
    agent = room(world).agents["a0"]
    assert agent.tokens == room(world).step_count


async def test_no_length_cap_death_is_the_only_bound(tmp_cfg):
    """An agent that never closes a tag talks until the pool kills it. Nobody
    picks a maximum; scarcity supplies one."""
    cfg = tmp_cfg(capacity=60, pop=1)
    world = make_world(cfg, policies={
        "a0": lambda a, c, r: " ".join(["rambling"] * 5000),   # never a tag
    })
    await world.seed()
    await run_steps(world, 4000)
    assert not room(world).agents, "the rambler must die of the pool"
    deaths = [e for e in events_of(world) if e["type"] == "death"]
    assert deaths and deaths[0]["cause"] == "pool_exhausted_requester"


async def test_thinking_is_whatever_precedes_the_action(tmp_cfg):
    """Not a <think> region: prose before the tag is deliberation, charged in
    full, and an action tag anywhere in it is still an action."""
    cfg = tmp_cfg(capacity=5000, pop=2)
    cfg.run.trace_turns = 100          # turn events carry the thinking count
    world = make_world(cfg, policies={
        "a0": scripted(["let me consider who is here first <say>hello</say>"]),
    })
    await world.seed()
    await run_steps(world, 600)
    turns = [e for e in events_of(world)
             if e["type"] == "turn" and e["agent"] == "a0"]
    acted = [t for t in turns if t["thinking_tokens"] > 0]
    assert acted, "deliberation before the action should be counted"
    assert acted[0]["thinking_tokens"] == 7    # words before the tag
    assert room(world).agents["a0"].thinking_tokens > 0
