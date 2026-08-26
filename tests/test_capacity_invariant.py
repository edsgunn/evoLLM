"""Scarcity must bind before the model's context window does (§4.3).

The first GPU run died here: a 96 GB GH200 profiled ~180k KV blocks against
max_model_len=8192, so an agent hit the context ceiling hundreds of times
before the pool could empty, and vLLM rejected the request. Death by scarcity
was unreachable. These tests pin the invariant that makes that impossible.
"""

import pytest

from evollm.events import ExperimentIntegrityError

from conftest import make_world, run_steps


def test_oversized_explicit_capacity_is_rejected(tmp_cfg):
    cfg = tmp_cfg(capacity=100_000, pop=1)      # block_size 4 => 400k tokens
    cfg.model.max_model_len = 8192
    with pytest.raises(ValueError, match="max_model_len"):
        make_world(cfg)


def test_capacity_at_the_bound_is_accepted(tmp_cfg):
    cfg = tmp_cfg(pop=1)
    cfg.model.max_model_len = 8192
    # adapter_blocks + (max_len - 1) // block_size = 2 + 2047
    cfg.world.rooms[0].capacity_blocks = 2049
    world = make_world(cfg)
    assert world.controllers["r0"].pool.capacity == 2049
    # a lone agent holding every block still fits under the ceiling
    reachable = (2049 - cfg.mock.adapter_blocks) * cfg.world.block_size
    assert reachable < cfg.model.max_model_len


def test_derived_capacity_is_clamped(tmp_cfg, capsys):
    """An engine-derived pool is clamped rather than rejected: the number came
    from the hardware, not the experimenter."""
    cfg = tmp_cfg(pop=1)
    cfg.model.max_model_len = 8192
    cfg.world.rooms[0].capacity_blocks = None
    world = make_world(cfg, engine_capacity=180_000)
    assert world.controllers["r0"].pool.capacity == 2 + (8192 - 1) // 4
    assert "clamping" in capsys.readouterr().out


async def test_scarcity_binds_before_the_ceiling(tmp_cfg):
    """The property the invariant buys: a lone agent in a full-size room dies
    of pool exhaustion, never of the context limit."""
    cfg = tmp_cfg(pop=1)
    cfg.model.max_model_len = 8192
    cfg.world.rooms[0].capacity_blocks = 2049
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 9000)
    assert not world.controllers["r0"].agents
    from evollm.events import read_events
    deaths = [e for e in read_events(world.controllers["r0"].log.path)
              if e["type"] == "death"]
    assert len(deaths) == 1
    assert deaths[0]["cause"] == "pool_exhausted_requester"
    assert deaths[0]["tokens"] < cfg.model.max_model_len


async def test_controller_guard_fires_if_the_invariant_is_bypassed(tmp_cfg):
    """Defensive: if capacity is widened behind World's back, the controller
    raises with a clear diagnosis instead of a vLLM ValueError mid-generate."""
    cfg = tmp_cfg(capacity=2049, pop=1)
    cfg.model.max_model_len = 8192
    world = make_world(cfg)
    await world.seed()
    # Widen the ceiling gap after construction, as a hand-edited config or a
    # future engine change could: the pool still has room, but the agent is
    # now heading past max_model_len.
    cfg.model.max_model_len = 200
    with pytest.raises(ExperimentIntegrityError, match="context ceiling"):
        await run_steps(world, 400)


class _Qwen15B:
    """Qwen/Qwen2.5-1.5B-Instruct dimensions, so the shipped GPU configs can
    be validated on a login node."""
    hidden_size = 1536
    num_attention_heads = 12
    num_key_value_heads = 2
    head_dim = 128
    num_hidden_layers = 28
    intermediate_size = 8960
    max_position_embeddings = 32768


@pytest.mark.parametrize("name", ["single_gpu.yaml", "node_4room.yaml"])
def test_shipped_gpu_configs_satisfy_the_invariant(name):
    from pathlib import Path

    from evollm.blocks import adapter_blocks_needed
    from evollm.config import load_config
    from evollm.engines.vllm_engine import kv_block_bytes
    from evollm.genome import spec_from_hf_config

    from evollm.config import resolve_max_model_len

    cfg = load_config(Path(__file__).parent.parent / "configs" / name)
    spec = spec_from_hf_config(_Qwen15B(), cfg.genome.target_modules,
                               cfg.genome.rank, cfg.genome.alpha)
    block_bytes = kv_block_bytes(_Qwen15B(), cfg.world.block_size)
    adapter_blocks = adapter_blocks_needed(spec.adapter_bytes(), block_bytes)
    assert adapter_blocks == 19   # matches the GH200 run log

    cfg.model.max_model_len = resolve_max_model_len(cfg, adapter_blocks)
    # The derived ceiling must exceed the trained window — the point of the
    # exercise is that agents outlive it.
    assert cfg.model.max_model_len > _Qwen15B.max_position_embeddings

    safe = adapter_blocks + (cfg.model.max_model_len - 1) // cfg.world.block_size
    for room in cfg.world.rooms:
        assert room.capacity_blocks is not None, \
            f"{name}:{room.id} must set capacity_blocks explicitly (§4.3)"
        assert room.capacity_blocks <= safe, \
            f"{name}:{room.id} capacity {room.capacity_blocks} > safe {safe}"
        # and the room must actually fit its starting population's adapters
        seats = room.capacity_blocks // adapter_blocks
        assert seats > cfg.world.initial_population_per_room


def test_auto_ceiling_never_binds(tmp_cfg):
    """"auto" removes the context window from the experiment: the ceiling is
    always one token beyond what the fullest possible agent could hold."""
    from evollm.config import resolve_max_model_len

    cfg = tmp_cfg(pop=1)
    cfg.world.rooms[0].capacity_blocks = 8000
    cfg.model.max_model_len = "auto"
    resolved = resolve_max_model_len(cfg, adapter_blocks=cfg.mock.adapter_blocks)
    reachable = (8000 - cfg.mock.adapter_blocks) * cfg.world.block_size
    assert resolved == reachable + 1
    cfg.model.max_model_len = resolved
    make_world(cfg)   # invariant holds by construction


def test_auto_uses_the_largest_room(tmp_cfg):
    from evollm.config import resolve_max_model_len

    cfg = tmp_cfg(rooms=3, pop=1)
    for room, cap in zip(cfg.world.rooms, [500, 4000, 1200]):
        room.capacity_blocks = cap
    cfg.model.max_model_len = "auto"
    resolved = resolve_max_model_len(cfg, adapter_blocks=cfg.mock.adapter_blocks)
    assert resolved == (4000 - cfg.mock.adapter_blocks) * cfg.world.block_size + 1
    cfg.model.max_model_len = resolved
    make_world(cfg)


def test_auto_requires_explicit_capacity(tmp_cfg):
    from evollm.config import resolve_max_model_len

    cfg = tmp_cfg(pop=1)
    cfg.world.rooms[0].capacity_blocks = None
    cfg.model.max_model_len = "auto"
    with pytest.raises(ValueError, match="capacity_blocks"):
        resolve_max_model_len(cfg, adapter_blocks=2)


def test_capacity_beyond_the_engine_pool_is_rejected(tmp_cfg):
    """The controller must never ration blocks the device lacks: the engine's
    only recourse would be preemption, which is an integrity violation."""
    cfg = tmp_cfg(capacity=5000, pop=1)
    with pytest.raises(ValueError, match="exceeds the engine"):
        make_world(cfg, engine_capacity=1000)


@pytest.mark.parametrize("func_name", ["build_world", "_measure_throughput",
                                       "cmd_eval_surprise"])
def test_every_engine_entry_point_resolves_auto(func_name):
    """measure-throughput once reached vLLM with max_model_len="auto" because
    only build_world resolved it. Every function that constructs an engine
    must go through prepare()."""
    import inspect

    import evollm.cli as cli

    source = inspect.getsource(getattr(cli, func_name))
    assert "prepare(cfg)" in source, \
        f"cli.{func_name} builds an engine without calling prepare(cfg)"


def test_prepare_is_idempotent(tmp_path):
    import evollm.cli as cli

    cfg = load_config_for(tmp_path)
    cli.prepare(cfg)
    resolved = cfg.model.max_model_len
    assert isinstance(resolved, int) and resolved > 1
    cli.prepare(cfg)
    assert cfg.model.max_model_len == resolved


def load_config_for(tmp_path):
    from evollm.config import Config, RoomConfig
    cfg = Config()
    cfg.backend = "mock"
    cfg.run.out_dir = str(tmp_path)
    cfg.world.rooms = [RoomConfig(id="r0", capacity_blocks=3000)]
    cfg.world.block_size = 16
    cfg.model.max_model_len = "auto"
    return cfg


def test_engine_rejects_unresolved_max_model_len():
    """Defence in depth: the engine refuses "auto" with a clear message
    instead of a TypeError from inside vLLM."""
    from evollm.config import Config, RoomConfig
    from evollm.engines.vllm_engine import VLLMEngine
    from evollm.genome import spec_from_dims

    cfg = Config()
    cfg.model.max_model_len = "auto"
    room = RoomConfig(id="gpu0", gpu=0)
    spec = spec_from_dims(num_layers=1, projections={"q_proj": (8, 8)},
                          rank=4, alpha=8)
    engine = VLLMEngine(cfg, room, spec)
    import asyncio
    with pytest.raises(ValueError, match="not resolved to an int"):
        asyncio.run(engine.start())


async def test_agent_at_the_ceiling_dies_of_scarcity_not_integrity(tmp_cfg):
    """The exact boundary that killed run 5913974.

    A lone agent holding every block it can (capacity - adapter_blocks) sits
    at max_model_len - 1 tokens. Its next token needs a block the pool cannot
    give, so it must die a scarcity death — the invariant working, not an
    integrity violation. Checking the ceiling before the allocation raised
    here instead and aborted a correctly-behaving run.
    """
    from evollm.config import resolve_max_model_len
    from evollm.events import read_events

    cfg = tmp_cfg(pop=1)
    cfg.world.rooms[0].capacity_blocks = 200
    cfg.model.max_model_len = "auto"
    cfg.model.max_model_len = resolve_max_model_len(
        cfg, adapter_blocks=cfg.mock.adapter_blocks)
    reachable = (200 - cfg.mock.adapter_blocks) * cfg.world.block_size
    assert cfg.model.max_model_len == reachable + 1

    world = make_world(cfg)
    await world.seed()
    # No ExperimentIntegrityError: run well past the point of exhaustion.
    await run_steps(world, reachable + 200)

    controller = world.controllers["r0"]
    assert not controller.agents
    deaths = [e for e in read_events(controller.log.path) if e["type"] == "death"]
    assert len(deaths) == 1
    assert deaths[0]["cause"] == "pool_exhausted_requester"
    # It died holding the whole room, exactly at the usable ceiling.
    assert deaths[0]["tokens"] == reachable
    assert deaths[0]["tokens"] == cfg.model.max_model_len - 1


# ── run-ahead: the engine must not outrun the economy ─────────────────────
class _PoolEngine:
    """Mock engine that reports a physical pool larger than the room claim,
    and records the max_tokens every turn was granted."""

    def __init__(self, inner, pool):
        self._inner = inner
        self._pool = pool
        self.budgets: list[int] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def pool_blocks(self):
        return self._pool

    def start_turn(self, agent_id, context, max_tokens):
        self.budgets.append(max_tokens)
        return self._inner.start_turn(agent_id, context, max_tokens)


async def test_turn_budget_cannot_exceed_unclaimed_physical_blocks(tmp_cfg):
    """The invariant that makes running out of memory impossible:

        engine usage <= capacity_blocks + unclaimed blocks == physical pool

    The controller consumes one token per agent per step while vLLM generates
    ahead and buffers, so every unconsumed token is KV charged to nobody. The
    budget must therefore be capped by blocks the economy has NOT claimed.
    """
    cfg = tmp_cfg(capacity=1000, pop=4)
    cfg.model.max_model_len = 10 ** 7          # effectively unbounded ceiling
    world = make_world(cfg)
    await world.seed()
    c = world.controllers["r0"]
    pool_blocks = 1500                          # 500 unclaimed over the claim
    c.engine = _PoolEngine(c.engine, pool_blocks)
    for _ in range(60):
        await c.run_step()
    assert c.engine.budgets, "no turn was started"
    live = len(c.agents) or 1
    worst = (pool_blocks - c.pool.capacity) * cfg.world.block_size
    assert max(c.engine.budgets) <= worst, \
        f"a turn was granted {max(c.engine.budgets)} tokens against {worst} unclaimed"
    # and the total possible run-ahead across all agents still fits the gap
    assert max(c.engine.budgets) * live <= worst + live


async def test_turn_budget_falls_back_to_the_context_ceiling(tmp_cfg):
    """Backends that cannot report a pool (the mock) keep the old behaviour,
    so this cannot silently throttle runs on a substrate without the problem."""
    # capacity x block_size must stay under max_model_len, or the existing
    # scarcity-before-context invariant rejects the config first.
    cfg = tmp_cfg(capacity=1000, pop=2)
    cfg.model.max_model_len = 5000
    world = make_world(cfg)
    await world.seed()
    c = world.controllers["r0"]
    assert c.engine.pool_blocks() is None
    agent = next(iter(c.agents.values()))
    assert c._turn_budget(agent) == max(1, 5000 - len(agent.context) - 1)


async def test_turn_budget_shrinks_as_the_room_fills_with_agents(tmp_cfg):
    """The gap is shared: more agents generating means less run-ahead each,
    so the collective total still fits."""
    cfg = tmp_cfg(capacity=1000, pop=2)
    cfg.model.max_model_len = 10 ** 7
    world = make_world(cfg)
    await world.seed()
    c = world.controllers["r0"]
    c.engine = _PoolEngine(c.engine, 1500)
    a = next(iter(c.agents.values()))
    few = c._turn_budget(a)
    from evollm.genome import Genome
    for _ in range(6):
        await c.seed_agent(Genome.random(world.spec, 0.02, c.rng))
    many = c._turn_budget(a)
    assert many < few, f"budget did not shrink: {few} -> {many}"


# ── parental investment: reproduction charged in real, occupied memory ────
def test_parental_investment_moves_the_charge_without_allocating_more(tmp_cfg):
    """The whole point: the child's adapter is the SAME blocks either way.
    Total room usage must be identical; only the owner differs."""
    from evollm.blocks import BlockPool
    a = BlockPool(capacity=1000, block_size=16)
    a.try_reserve_adapter("child", 22)
    b = BlockPool(capacity=1000, block_size=16)
    b.reserve_dependent("p1", "child", 11)
    b.reserve_dependent("p2", "child", 11)
    assert a.used == b.used == 22, "parental investment must not allocate more"
    assert b.holdings["p1"].total == 11 and b.holdings["p2"].total == 11
    assert "child" not in b.holdings


def test_child_death_releases_the_parents_charge():
    from evollm.blocks import BlockPool
    p = BlockPool(capacity=1000, block_size=16)
    p.try_reserve_adapter("p1", 22)
    p.reserve_dependent("p1", "kid", 11)
    p.reserve_dependent("p2", "kid", 11)
    assert p.used == 44
    assert p.release_dependent("kid") == 22
    assert p.used == 22 and p.holdings["p1"].total == 22 and "p2" not in p.holdings


def test_parent_death_reverts_the_charge_to_the_living_child():
    """The child's adapter is still registered, so the blocks must keep being
    accounted for — they revert to the child rather than vanishing."""
    from evollm.blocks import BlockPool
    p = BlockPool(capacity=1000, block_size=16)
    p.reserve_dependent("p1", "kid", 11)
    p.reserve_dependent("p2", "kid", 11)
    before = p.used
    assert p.revert_dependents("p1") == {"kid": 11}
    assert p.used == before, "reverting must conserve blocks"
    assert p.holdings["kid"].adapter_blocks == 11
    assert p.holdings["p2"].dependents == {"kid": 11}


def test_dependent_blocks_raise_the_owners_eviction_hazard():
    """The charge only bites under random_holder, where the victim is drawn in
    proportion to blocks held. A prolific parent must be the likelier victim."""
    import numpy as np
    from evollm.blocks import BlockPool
    p = BlockPool(capacity=10000, block_size=16)
    p.try_reserve_adapter("breeder", 22)
    p.try_reserve_adapter("loner", 22)
    for i in range(20):
        p.reserve_dependent("breeder", f"kid{i}", 11)
    rng = np.random.default_rng(0)
    picks = [p.random_holder(rng) for _ in range(3000)]
    breeder = picks.count("breeder") / len(picks)
    loner = picks.count("loner") / len(picks)
    assert breeder > loner * 5, (breeder, loner)


async def test_parental_investment_end_to_end(tmp_cfg):
    """Births still happen, the charge lands on parents, and the pool stays
    exactly as full as it would have been."""
    cfg = tmp_cfg(capacity=4000, pop=6)
    cfg.world.parental_investment = True
    cfg.world.eviction = "random_holder"
    world = make_world(cfg)
    await world.seed()
    c = world.controllers["r0"]
    for _ in range(600):
        await c.run_step()
    born = [a for a in c.agents.values() if a.parents]
    if born:
        # a child born under this rule holds no adapter blocks of its own
        kid = born[0]
        h = c.pool.holdings.get(kid.id)
        assert h is None or h.adapter_blocks == 0, \
            "the child must not also be charged for its own adapter"
    # accounting invariant: the pool never exceeds capacity
    assert c.pool.used <= c.pool.capacity
    assert sum(h.total for h in c.pool.holdings.values()) == c.pool.used


# ── eviction must only ever pick an agent the controller can kill ─────────
def test_random_holder_respects_the_eligible_set():
    """Not every holder is killable: a newborn holds its adapter while still
    in _pending_arrivals, and a migrant holds its footprint at the destination
    before the source releases it. Both are in the ledger, neither is in
    self.agents."""
    import numpy as np
    from evollm.blocks import BlockPool
    p = BlockPool(capacity=1000, block_size=16)
    p.try_reserve_adapter("live", 10)
    p.try_reserve_adapter("newborn_in_flight", 500)   # far more blocks
    rng = np.random.default_rng(0)
    picks = {p.random_holder(rng, eligible={"live"}) for _ in range(200)}
    assert picks == {"live"}, picks
    # with nobody eligible the pool says so rather than offering a phantom
    assert p.random_holder(rng, eligible=set()) is None
    # unrestricted, it will happily return the unkillable one
    assert "newborn_in_flight" in {p.random_holder(rng) for _ in range(200)}


async def test_eviction_never_targets_an_agent_not_in_the_room(tmp_cfg):
    """Regression: `self.agents[victim_id]` raised KeyError and killed two
    12-hour jobs the first time random_holder was ever used in anger."""
    cfg = tmp_cfg(capacity=400, pop=3)
    cfg.world.eviction = "random_holder"
    world = make_world(cfg)
    await world.seed()
    c = world.controllers["r0"]
    # a holder the controller cannot kill, exactly like a pending arrival,
    # and big enough that an unrestricted draw would almost always pick it
    assert c.pool.try_reserve_adapter("phantom", c.pool.free - 20)
    agent = next(iter(c.agents.values()))
    for _ in range(500):                       # must force the pool empty
        if not c._append_token(agent, 7):
            break                              # died of scarcity: fine
    assert "phantom" in c.pool.holdings, "the phantom must never be evicted"
    assert c.pool.used <= c.pool.capacity


async def test_eviction_falls_back_to_the_requester_when_nobody_else_can_die(tmp_cfg):
    """If every other holder is in flight, scarcity falls on the requester —
    the same outcome the `requester` policy gives."""
    cfg = tmp_cfg(capacity=300, pop=1)
    cfg.world.eviction = "random_holder"
    world = make_world(cfg)
    await world.seed()
    c = world.controllers["r0"]
    agent = next(iter(c.agents.values()))
    c.pool.try_reserve_adapter("phantom", c.pool.free - 5)
    died = False
    for _ in range(400):
        if not c._append_token(agent, 7):
            died = True
            break
    assert died, "the requester should have died once the pool was full"
    assert agent.id not in c.agents


# ── capacity should come from the engine, not from a hand-set number ──────
def test_auto_ceiling_uses_explicit_capacity_when_rooms_set_it(tmp_cfg):
    """Unchanged path: a room that names its capacity still sizes the ceiling
    from what a lone survivor could hold."""
    from evollm.config import resolve_max_model_len
    cfg = tmp_cfg(capacity=5000)
    cfg.model.max_model_len = "auto"
    assert resolve_max_model_len(cfg, adapter_blocks=10) == \
        (5000 - 10) * cfg.world.block_size + 1


def test_auto_ceiling_falls_back_to_a_device_bound_when_capacity_is_derived(tmp_cfg,
                                                                           monkeypatch):
    """A room that leaves capacity to the engine creates a circularity —
    capacity needs the engine, the engine needs max_model_len. The ceiling is
    then sized from an upper bound on the pool instead."""
    from evollm import config as cfgmod
    cfg = tmp_cfg(capacity=None)
    cfg.model.max_model_len = "auto"
    monkeypatch.setattr(cfgmod, "_pool_upper_bound", lambda c: 100_000)
    got = cfgmod.resolve_max_model_len(cfg, adapter_blocks=88)
    assert got == (100_000 - 88) * cfg.world.block_size + 1
    # and the bound must exceed any real pool, so the ceiling cannot bind
    assert got > (83_744 - 88) * cfg.world.block_size


def test_auto_ceiling_raises_when_it_can_neither_derive_nor_measure(tmp_cfg,
                                                                    monkeypatch):
    from evollm import config as cfgmod
    cfg = tmp_cfg(capacity=None)
    cfg.model.max_model_len = "auto"
    monkeypatch.setattr(cfgmod, "_pool_upper_bound", lambda c: None)
    with pytest.raises(ValueError, match="capacity_blocks"):
        cfgmod.resolve_max_model_len(cfg, adapter_blocks=22)


def test_pool_bound_subtracts_the_model_weights(monkeypatch, tmp_cfg):
    """Regression: the bound once ignored the weights, on the reasoning that
    overshooting only wasted rope-cache memory. vLLM instead refuses to start
    unless one max_model_len request fits in the KV cache, so the too-generous
    bound killed the engine at init and burned a queue slot."""
    from evollm import config as cfgmod

    class _HF:
        num_hidden_layers = 28
        num_attention_heads = 28
        num_key_value_heads = 4
        head_dim = 128
        hidden_size = 3584
        vocab_size = 152064
        intermediate_size = 18944
        tie_word_embeddings = False

    total = 97871 * 2 ** 20                       # a 96 GB card, as measured
    monkeypatch.setattr(cfgmod, "AutoConfig", None, raising=False)

    class _NVML:
        @staticmethod
        def nvmlInit(): pass
        @staticmethod
        def nvmlDeviceGetHandleByIndex(i): return object()
        @staticmethod
        def nvmlDeviceGetMemoryInfo(h):
            return type("m", (), {"total": total})()

    import sys, types
    fake_nvml = types.ModuleType("pynvml")
    for n in ("nvmlInit", "nvmlDeviceGetHandleByIndex", "nvmlDeviceGetMemoryInfo"):
        setattr(fake_nvml, n, getattr(_NVML, n))
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoConfig = type("AC", (), {"from_pretrained": staticmethod(lambda n: _HF())})
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    cfg = tmp_cfg(capacity=None)
    cfg.engine.gpu_memory_utilization = 0.92
    cfg.world.block_size = 16
    bound = cfgmod._pool_upper_bound(cfg)

    # vLLM measured 71.03 GiB of usable KV on this card; the bound must sit
    # under that, not above it
    block_bytes = 16 * 2 * 28 * 4 * 128 * 2
    assert bound * block_bytes < 71.03 * 2 ** 30, \
        f"bound of {bound} blocks exceeds the pool vLLM can actually build"
    # and it must still be generous enough to beat a hand-set 48,000
    assert bound > 48_000
