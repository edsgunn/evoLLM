from __future__ import annotations

import pytest

from evollm.config import Config, RoomConfig
from evollm.engines.mock import MockEngine, WordTokenizer
from evollm.genome import spec_from_dims
from evollm.world import World


def make_config(tmp_path, rooms=1, capacity=200, pop=2, **world_overrides) -> Config:
    cfg = Config()
    cfg.backend = "mock"
    cfg.run_name = "test"
    cfg.run.out_dir = str(tmp_path)
    cfg.run.snapshot_every_steps = 0
    cfg.run.occupancy_every_steps = 0
    cfg.world.rooms = [RoomConfig(id=f"r{i}", capacity_blocks=capacity)
                       for i in range(rooms)]
    cfg.world.block_size = 4
    cfg.world.initial_population_per_room = pop
    cfg.world.mate_window_tokens = 50
    cfg.mock.adapter_blocks = 2
    for key, value in world_overrides.items():
        setattr(cfg.world, key, value)
    return cfg


def make_world(cfg: Config, policies: dict[str, object] | None = None,
               default_policy=None) -> World:
    from evollm.engines.mock import quiet_policy
    spec = spec_from_dims(num_layers=2, projections={"q_proj": (8, 8)},
                          rank=4, alpha=8)
    tokenizer = WordTokenizer()
    engines = {
        room.id: MockEngine(default_policy=default_policy or quiet_policy,
                            policies=policies or {}, seed=cfg.seed + i,
                            tokenizer=tokenizer)
        for i, room in enumerate(cfg.world.rooms)
    }
    return World(cfg, engines, spec, adapter_blocks=cfg.mock.adapter_blocks)


async def run_steps(world: World, n: int) -> None:
    for _ in range(n):
        for controller in world.controllers.values():
            if controller.agents or controller._pending_arrivals:
                await controller.run_step()


@pytest.fixture
def tmp_cfg(tmp_path):
    def factory(**kwargs):
        return make_config(tmp_path, **kwargs)
    return factory
