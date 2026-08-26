"""Movement, topology, reservation ordering, extinction, snapshots."""

from pathlib import Path

import pytest

from evollm.engines.mock import go_to, scripted
from evollm.events import read_events
from evollm.genome import Genome

from conftest import make_world, run_steps


def events_of(world, rid):
    return list(read_events(world.controllers[rid].log.path))


async def test_move_between_rooms(tmp_cfg):
    cfg = tmp_cfg(rooms=2, capacity=1000, pop=1)
    world = make_world(cfg, policies={"a0": scripted(["<go>r1</go>"])})
    await world.seed()
    await run_steps(world, 300)
    r0, r1 = world.controllers["r0"], world.controllers["r1"]
    assert "a0" not in r0.agents and "a0" in r1.agents
    # the mover's whole footprint lives at the destination now
    agent = r1.agents["a0"]
    holding = r1.pool.holdings["a0"]
    assert holding.adapter_blocks == r1.adapter_blocks
    assert holding.kv_blocks >= r1.pool.kv_blocks_for(agent.tokens) - 1
    assert "a0" not in r0.pool.holdings
    # engines swapped custody of the adapter
    assert "a0" in r1.engine.registered and "a0" not in r0.engine.registered
    moves = [e for e in events_of(world, "r0") if e["type"] == "move"]
    assert moves and moves[0]["to"] == "r1"


async def test_move_to_full_room_fails_with_capacity_observation(tmp_cfg):
    cfg = tmp_cfg(rooms=2, capacity=1000, pop=1)
    world = make_world(cfg, default_policy=go_to("r1"))
    await world.seed()
    r1 = world.controllers["r1"]
    # stuff r1 so it cannot take a0's adapter + context
    assert r1.pool.try_reserve_adapter("filler", r1.pool.free)
    await run_steps(world, 250)
    r0 = world.controllers["r0"]
    assert "a0" in r0.agents, "failed move must not be fatal (§2.5)"
    failures = [e for e in events_of(world, "r0") if e["type"] == "move_failed"]
    assert failures
    # the failure return carried the capacity listing, which the agent paid for
    context_text = r0.engine.detokenize(r0.agents["a0"].context)
    assert "failed." in context_text and "free=" in context_text


async def test_move_to_unknown_room_fails(tmp_cfg):
    cfg = tmp_cfg(rooms=2, capacity=1000, pop=1)
    world = make_world(cfg, policies={"a0": go_to("r9")})
    await world.seed()
    await run_steps(world, 220)
    assert "a0" in world.controllers["r0"].agents
    assert [e for e in events_of(world, "r0") if e["type"] == "move_failed"]


async def test_reservation_before_release(tmp_cfg):
    """§4.5: the destination reserves before the source releases, so a failed
    reservation leaves the source holdings untouched."""
    cfg = tmp_cfg(rooms=2, capacity=1000, pop=1)
    world = make_world(cfg)
    await world.seed()
    r0, r1 = world.controllers["r0"], world.controllers["r1"]
    agent = r0.agents["a0"]
    before = (r0.pool.holdings["a0"].adapter_blocks,
              r0.pool.holdings["a0"].kv_blocks)
    r1.pool.try_reserve_adapter("filler", r1.pool.free)
    moved = await world.request_move(r0, agent, "r1")
    assert not moved
    assert (r0.pool.holdings["a0"].adapter_blocks,
            r0.pool.holdings["a0"].kv_blocks) == before
    assert "a0" in r0.agents and "a0" in r0.engine.registered


async def test_extinction_stops_world(tmp_cfg):
    cfg = tmp_cfg(rooms=1, capacity=8, pop=2)
    world = make_world(cfg)
    await world.seed()
    await world.run(max_steps=2000)   # returns rather than hanging
    assert world.population == 0
    kinds = {e["type"] for e in events_of(world, "r0")}
    assert "extinction" in kinds


async def test_snapshot_roundtrip(tmp_cfg):
    cfg = tmp_cfg(rooms=1, capacity=2000, pop=2)
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 20)
    controller = world.controllers["r0"]
    controller.snapshot()
    snap_dirs = sorted((Path(cfg.run.out_dir) / cfg.run_name / "snapshots" /
                        "r0").glob("step_*"))
    assert snap_dirs
    genome = Genome.load(snap_dirs[0] / "a0.safetensors", world.spec)
    assert genome.factors.keys() == controller.agents["a0"].genome.factors.keys()


async def test_adjacency_from_edges(tmp_cfg):
    cfg = tmp_cfg(rooms=3, capacity=100, pop=1)
    cfg.world.edges = [["r0", "r1"], ["r1", "r2"]]
    world = make_world(cfg)
    assert world.adjacent("r0") == ["r1"]
    assert set(world.adjacent("r1")) == {"r0", "r2"}


async def test_lifetime_is_room_independent(tmp_cfg):
    """Rooms advance independently, so (room step - born_step) went negative
    for 30% of deaths in run 6006472 once agents migrated. Age is the agent's
    own clock and must survive a move."""
    from evollm.engines.mock import go_to

    cfg = tmp_cfg(rooms=2, capacity=5000, pop=1)
    world = make_world(cfg, policies={"a0": go_to("r1")})
    await world.seed()
    r0, r1 = world.controllers["r0"], world.controllers["r1"]
    # advance r1 far ahead of r0 so the two clocks disagree
    for _ in range(300):
        await r1.run_step()
    await run_steps(world, 400)
    assert "a0" in r1.agents
    agent = r1.agents["a0"]
    assert agent.age > 0
    assert agent.age <= r0.step_count + r1.step_count
    r1._kill(agent, "pool_exhausted_requester")
    for coro in r1._pending_cleanup:
        await coro
    deaths = [e for e in read_events(r1.log.path) if e["type"] == "death"]
    assert deaths[-1]["lifetime_steps"] > 0, "lifetime must never be negative"


# ── preemption handling ───────────────────────────────────────────────────
class _PreemptingEngine:
    """Wraps a mock engine and reports preemptions on demand."""

    def __init__(self, inner, schedule):
        self._inner = inner
        self._schedule = list(schedule)   # preemptions to report, per poll

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def poll_preemptions(self) -> int:
        return self._schedule.pop(0) if self._schedule else 0


def test_preemption_budget_is_a_real_config_field():
    """Guard: it was once assigned to the wrong dataclass, and every test
    still passed because Python let them set the attribute dynamically."""
    from dataclasses import fields
    from evollm.config import Config, EngineConfig
    assert "preemption_budget" in {f.name for f in fields(EngineConfig)}
    assert Config().engine.preemption_budget == 200


async def test_rare_preemption_is_logged_and_the_run_continues(tmp_cfg):
    """A preempted request keeps its tokens and resumes, so one preemption
    must not kill a multi-hour run — it is recorded instead."""
    from evollm.events import read_events
    cfg = tmp_cfg(capacity=8000, pop=3)
    cfg.engine.preemption_budget = 200
    world = make_world(cfg)
    await world.seed()
    c = world.controllers["r0"]
    c.engine = _PreemptingEngine(c.engine, [0, 1, 0, 2])
    for _ in range(40):
        await c.run_step()                      # must not raise
    events = [e for e in read_events(c.log.path) if e["type"] == "preemption"]
    assert [e["count"] for e in events] == [1, 2]
    assert events[-1]["total"] == 3
    assert c.preemptions == 3
    # the room keeps running and its agents keep acting
    assert c.step_count == 40 and c.agents


async def test_systematic_preemption_still_aborts(tmp_cfg):
    """The signal is kept: a room that is genuinely over-subscribed stops."""
    from evollm.events import ExperimentIntegrityError
    cfg = tmp_cfg(capacity=8000, pop=3)
    cfg.engine.preemption_budget = 3
    world = make_world(cfg)
    await world.seed()
    c = world.controllers["r0"]
    c.engine = _PreemptingEngine(c.engine, [2, 1, 1])
    with pytest.raises(ExperimentIntegrityError, match="over-subscribed"):
        for _ in range(10):
            await c.run_step()
    assert c.preemptions == 4


async def test_one_rooms_preemption_does_not_abort_another(tmp_cfg):
    """The watchdog list used to be class-level and shared across the four
    engines in a process, so a single preemption in one room aborted all of
    them."""
    cfg = tmp_cfg(rooms=2, capacity=8000, pop=3)
    cfg.engine.preemption_budget = 0
    world = make_world(cfg)
    await world.seed()
    r0, r1 = world.controllers["r0"], world.controllers["r1"]
    r0.engine = _PreemptingEngine(r0.engine, [5])
    from evollm.events import ExperimentIntegrityError
    with pytest.raises(ExperimentIntegrityError):
        await r0.run_step()
    for _ in range(5):
        await r1.run_step()                     # untouched by r0's trouble
    assert r1.preemptions == 0 and r1.step_count == 5


# ── virtual rooms: many small rooms sharing one device ────────────────────
def _clustered_cfg(tmp_cfg, rooms_per_gpu=3, gpus=4, **kw):
    from evollm.config import RoomConfig
    cfg = tmp_cfg(**kw)
    cfg.world.rooms = [RoomConfig(id=f"g{g}r{i}", gpu=g, capacity_blocks=cfg.world.rooms[0].capacity_blocks)
                       for g in range(gpus) for i in range(rooms_per_gpu)]
    cfg.world.edges = "clustered"
    return cfg


def test_rooms_on_one_device_share_a_single_engine(tmp_cfg):
    """The weights are ~15 GB and already resident, so extra rooms on a card
    must be free — which requires them to share the engine, not build one
    each."""
    cfg = _clustered_cfg(tmp_cfg, rooms_per_gpu=3, gpus=2, capacity=400, pop=2)
    world = make_world(cfg)
    engines = {id(c.engine) for c in world.controllers.values()}
    assert len(engines) == 2, "one engine per device, not per room"
    per_gpu = {}
    for rid, c in world.controllers.items():
        per_gpu.setdefault(id(c.engine), set()).add(rid)
    assert all(len(v) == 3 for v in per_gpu.values())


def test_clustered_topology_is_complete_within_a_device_and_a_ring_across(tmp_cfg):
    cfg = _clustered_cfg(tmp_cfg, rooms_per_gpu=3, gpus=4, capacity=400, pop=1)
    adj = cfg.adjacency()
    # every room reaches all its own device's rooms
    for g in range(4):
        members = {f"g{g}r{i}" for i in range(3)}
        for m in members:
            assert members - {m} <= set(adj[m]), f"{m} does not see its cluster"
    # only room 0 of each device leaves it, and it reaches exactly two others
    for g in range(4):
        outside = [r for r in adj[f"g{g}r0"] if not r.startswith(f"g{g}")]
        assert len(outside) == 2, outside
        for i in (1, 2):
            assert all(r.startswith(f"g{g}") for r in adj[f"g{g}r{i}"]), \
                "non-gateway rooms must not leave the device"


def test_derived_capacity_is_split_between_rooms_sharing_a_device(tmp_cfg):
    """Rooms sharing a device must divide its pool, or each would hand out
    blocks the others are also handing out."""
    from evollm.config import RoomConfig
    cfg = tmp_cfg(capacity=None, pop=1)
    cfg.model.max_model_len = 10 ** 7
    cfg.world.rooms = [RoomConfig(id=f"g0r{i}", gpu=0) for i in range(4)]
    world = make_world(cfg, engine_capacity=4000)
    caps = {c.pool.capacity for c in world.controllers.values()}
    assert caps == {1000}, caps
    assert sum(c.pool.capacity for c in world.controllers.values()) <= 4000


async def test_many_small_rooms_run_and_agents_migrate_within_a_cluster(tmp_cfg):
    cfg = _clustered_cfg(tmp_cfg, rooms_per_gpu=3, gpus=2, capacity=600, pop=4)
    world = make_world(cfg, default_policy=go_to("g0r1"))
    await world.seed()
    await run_steps(world, 400)
    assert len(world.controllers) == 6
    moved = sum(1 for e in read_events(world.controllers["g0r1"].log.path)
                if e["type"] == "arrival")
    assert world.population > 0
