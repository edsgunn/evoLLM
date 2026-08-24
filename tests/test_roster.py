"""Co-resident perception (§2.4).

Run 5913974 produced one child from 64 seeds. 72% of <mate> requests were
addressed to agents that were not in the room, because nothing ever told an
agent who was: the system prompt listed adjacent rooms but never co-residents,
and a failed <mate> returned nothing at all — unlike a failed <go>, which
§2.4 says returns the adjacent rooms and their capacities.

These tests pin that an agent can always perceive who shares its room, and
that it learns this by acting, not by being helped.
"""

from evollm.engines.mock import scripted
from evollm.events import read_events

from conftest import make_world, run_steps


def context_of(world, agent_id, room="r0"):
    controller = world.controllers[room]
    return controller.engine.detokenize(controller.agents[agent_id].context)


async def test_founding_population_sees_a_complete_roster(tmp_cfg):
    """Seeding issues prompts only once every founder exists, so the first
    agent is not told the room is empty."""
    cfg = tmp_cfg(capacity=5000, pop=4)
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 400)
    for agent_id in ["a0", "a1", "a2", "a3"]:
        text = context_of(world, agent_id)
        others = {"a0", "a1", "a2", "a3"} - {agent_id}
        assert "Present:" in text
        for other in others:
            assert other in text, f"{agent_id} was not told about {other}"


async def test_failed_mate_returns_the_roster(tmp_cfg):
    """Symmetric with a failed <go> (§2.4): the attempt is how the agent
    perceives the room, and it pays for the observation."""
    cfg = tmp_cfg(capacity=5000, pop=3)
    world = make_world(cfg, policies={"a0": scripted(["<mate>ghost</mate>"])})
    await world.seed()
    before = world.controllers["r0"].agents["a0"].tokens_observed
    await run_steps(world, 400)
    text = context_of(world, "a0")
    assert "mate with ghost failed" in text
    assert "a1" in text and "a2" in text
    # the failed attempt was not free
    assert world.controllers["r0"].agents["a0"].tokens_observed > before
    events = [e for e in read_events(world.controllers["r0"].log.path)
              if e["type"] == "mate_request"]
    assert events and events[0]["delivered"] is False


async def test_death_is_announced_so_agents_stop_addressing_the_dead(tmp_cfg):
    cfg = tmp_cfg(capacity=5000, pop=3)
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 50)
    controller = world.controllers["r0"]
    victim = controller.agents["a2"]
    controller._kill(victim, "pool_exhausted_requester")
    for coro in controller._pending_cleanup:
        await coro
    controller._pending_cleanup.clear()
    await run_steps(world, 400)
    for survivor in ["a0", "a1"]:
        assert "a2 has left this room" in context_of(world, survivor)


async def test_arrival_is_announced_to_incumbents(tmp_cfg):
    cfg = tmp_cfg(rooms=2, capacity=5000, pop=2)
    world = make_world(cfg, policies={"a0": scripted(["<go>r1</go>"])})
    await world.seed()
    await run_steps(world, 500)
    assert "a0" in world.controllers["r1"].agents
    # the incumbents of r1 were told, and a0 was told who it joined
    for incumbent in ["a2", "a3"]:
        if incumbent in world.controllers["r1"].agents:
            assert "a0 has arrived in this room" in \
                context_of(world, incumbent, room="r1")
    assert "Present:" in context_of(world, "a0", room="r1")


async def test_roster_excludes_self(tmp_cfg):
    cfg = tmp_cfg(capacity=5000, pop=2)
    world = make_world(cfg)
    await world.seed()
    controller = world.controllers["r0"]
    assert controller._others("a0") == ["a1"]
    assert controller._others("a1") == ["a0"]


async def test_lone_agent_is_told_it_is_alone(tmp_cfg):
    cfg = tmp_cfg(capacity=5000, pop=1)
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 400)
    assert "nobody else" in context_of(world, "a0")


async def test_event_log_refuses_to_merge_two_runs(tmp_cfg):
    """Three prechecks once appended into one file, and every summary computed
    from it aggregated all three — reporting 2 births for a run that had 0."""
    from evollm.events import ExperimentIntegrityError

    cfg = tmp_cfg(capacity=2000, pop=2)
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 20)
    world.close()

    import pytest
    with pytest.raises(ExperimentIntegrityError, match="earlier run"):
        make_world(cfg)   # same run_name, same event path


async def test_mate_window_starts_when_the_request_is_read(tmp_cfg):
    """§2.4 gives the target a window "of its own tokens" after *receiving* a
    request. Arming it at enqueue meant it expired while the request sat in an
    unread backlog: in precheck 5987576 the shortest request->accept lag was
    177 world steps against a 64-token window, so 0 of 430 accepts were valid.
    """
    from evollm.engines.mock import scripted

    from evollm.actions import Mate

    cfg = tmp_cfg(capacity=5000, pop=2, mate_window_tokens=8)
    world = make_world(cfg)
    await world.seed()
    controller = world.controllers["r0"]
    a0, a1 = controller.agents["a0"], controller.agents["a1"]

    # a1 starts with its whole system prompt queued and unread.
    assert a1.obs_backlog > 20
    await controller._do_mate_request(a0, Mate("a1"))
    assert a1.deferred_mates and not a1.pending_mates

    # Generating tokens must not burn a window the agent has not received.
    a1.decay_mate_windows(1000)
    assert a1.deferred_mates and not a1.pending_mates

    # Reading through to that utterance arms it, with the full window intact.
    for _ in range(a1.obs_backlog + 5):
        if not a1.deferred_mates:
            break
        controller._step_observing(a1)
    assert not a1.deferred_mates
    assert [p.requester_id for p in a1.pending_mates] == ["a0"]
    assert a1.pending_mates[0].tokens_remaining == cfg.world.mate_window_tokens


async def test_handshake_survives_a_long_backlog(tmp_cfg):
    """End to end: a request buried behind a third agent's chatter is read
    long after it was sent, and still reproduces — because the window starts
    on receipt. a2 floods the room so a1 is thousands of tokens behind."""
    from evollm.engines.mock import accept_all_policy, scripted

    cfg = tmp_cfg(capacity=8000, pop=3, mate_window_tokens=16)
    world = make_world(cfg, policies={
        "a0": scripted(["<mate>a1</mate>"]),
        "a1": accept_all_policy,       # accepts on its first turn after reading
        "a2": lambda a, c, r: "<say>" + " ".join(["chatter"] * 40) + "</say>",
    })
    await world.seed()
    await run_steps(world, 4000)
    controller = world.controllers["r0"]
    births = [e for e in read_events(controller.log.path)
              if e["type"] == "birth" and e["generation"] == 1]
    assert births, "agreement after a long backlog should still reproduce"
    # the child need not still be alive at the end; the handshake is the claim


async def test_backlog_is_reported(tmp_cfg):
    cfg = tmp_cfg(capacity=5000, pop=3)
    cfg.run.occupancy_every_steps = 50
    world = make_world(cfg, default_policy=lambda a, c, r: "<say>" + " ".join(["x"] * 40) + "</say>")
    await world.seed()
    await run_steps(world, 400)
    occ = [e for e in read_events(world.controllers["r0"].log.path)
           if e["type"] == "occupancy"]
    assert occ and "mean_backlog" in occ[-1]
    # chatter into 3 listeners injects faster than 1 token/step can absorb
    assert occ[-1]["max_backlog"] > 0


# ── observation absorption (§2.3 / §2.4) ──────────────────────────────────

async def test_utterance_absorption_costs_the_listener_what_was_said(tmp_cfg):
    """Hearing a long message costs blocks in proportion to its length.

    Under one-token-per-step every agent's context grew at exactly one token
    per step regardless of behaviour, so speech was free for listeners and
    lifetime was independent of what anyone did (measured across every GPU
    run: context == age, exactly, for all agents).
    """
    cfg = tmp_cfg(capacity=5000, pop=2)
    cfg.world.observation_absorption = "utterance"
    long_say = "<say>" + " ".join(["word"] * 50) + "</say>"
    world = make_world(cfg, policies={"a0": scripted([long_say])})
    await world.seed()
    controller = world.controllers["r0"]
    listener = controller.agents["a1"]
    await run_steps(world, 400)
    # the listener's context is far ahead of its age: it paid for a0's words
    assert listener.tokens > controller.step_count
    assert listener.tokens_observed > 50


async def test_token_absorption_makes_lifetime_behaviour_independent(tmp_cfg):
    """The measured pathology, pinned as the contrast case."""
    cfg = tmp_cfg(capacity=5000, pop=2)
    cfg.world.observation_absorption = "token"
    long_say = "<say>" + " ".join(["word"] * 50) + "</say>"
    world = make_world(cfg, policies={"a0": scripted([long_say])})
    await world.seed()
    controller = world.controllers["r0"]
    await run_steps(world, 400)
    for agent in controller.agents.values():
        assert agent.tokens == controller.step_count


async def test_absorption_mode_does_not_break_the_handshake(tmp_cfg):
    from evollm.engines.mock import accept_all_policy

    cfg = tmp_cfg(capacity=8000, pop=2, mate_window_tokens=32)
    cfg.world.observation_absorption = "utterance"
    world = make_world(cfg, policies={
        "a0": scripted(["<mate>a1</mate>"]),
        "a1": accept_all_policy,
    })
    await world.seed()
    await run_steps(world, 2000)
    generations = {a.generation for a in world.controllers["r0"].agents.values()}
    assert 1 in generations


async def test_absorbing_can_kill_and_it_is_a_scarcity_death(tmp_cfg):
    """A flooded agent dies of the pool, not of a special rule."""
    from evollm.events import read_events

    cfg = tmp_cfg(capacity=60, pop=2)
    cfg.world.observation_absorption = "utterance"
    huge = "<say>" + " ".join(["word"] * 400) + "</say>"
    world = make_world(cfg, policies={"a0": scripted([huge])})
    await world.seed()
    await run_steps(world, 600)
    deaths = [e for e in read_events(world.controllers["r0"].log.path)
              if e["type"] == "death"]
    assert deaths
    assert all(d["cause"] == "pool_exhausted_requester" for d in deaths)
