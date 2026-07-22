"""The world: a graph of rooms read off the hardware (§2.1).

Rooms are GPUs; one engine and one controller per room, all driven by a
single asyncio loop. Rooms advance independently — generations are staggered,
only tokens are in lockstep, and only within a room (§4.5).

Movement (§4.5): the destination reserves the agent's full footprint before
the source releases anything, so an agent is never in flight without a home.
"""

from __future__ import annotations

import asyncio
import itertools

import numpy as np

from .agent import Agent
from .config import Config
from .controller import RoomController
from .engines.base import EngineBackend
from .events import EventLog
from .genome import Genome, GenomeSpec


class World:
    def __init__(self, cfg: Config, engines: dict[str, EngineBackend],
                 spec: GenomeSpec, adapter_blocks: int, out_dir=None):
        self.cfg = cfg
        self.spec = spec
        self._adjacency = cfg.adjacency()
        self._ids = itertools.count()
        self.rng = np.random.default_rng(cfg.seed)
        self.controllers: dict[str, RoomController] = {}
        self.stopping = False

        from pathlib import Path
        base = Path(out_dir or cfg.run.out_dir) / cfg.run_name

        for room in cfg.world.rooms:
            engine = engines[room.id]
            capacity = room.capacity_blocks
            if capacity is None:
                capacity = engine.capacity_blocks()
            if capacity is None:
                raise ValueError(
                    f"room {room.id}: capacity_blocks not set and the engine "
                    "cannot derive it")
            log = EventLog(base / "events" / f"{room.id}.jsonl")
            self.controllers[room.id] = RoomController(
                room_id=room.id, capacity_blocks=capacity,
                adapter_blocks=adapter_blocks, engine=engine, cfg=cfg,
                world=self, log=log,
                rng=np.random.default_rng(self.rng.integers(2**63)))

    # ── identity and topology ─────────────────────────────────────────────
    def next_agent_id(self) -> str:
        return f"a{next(self._ids)}"

    def adjacent(self, room_id: str) -> list[str]:
        return self._adjacency[room_id]

    def adjacent_capacities(self, room_id: str) -> dict[str, tuple[int, int]]:
        return {
            r: (self.controllers[r].pool.free, self.controllers[r].pool.capacity)
            for r in self._adjacency[room_id]
        }

    @property
    def population(self) -> int:
        return sum(len(c.agents) for c in self.controllers.values())

    # ── movement (§2.4, §4.5) ─────────────────────────────────────────────
    async def request_move(self, source: RoomController, agent: Agent,
                           dest_id: str) -> bool:
        if dest_id not in self._adjacency[source.room_id]:
            return False
        dest = self.controllers[dest_id]
        # Reservation at the destination happens synchronously (no await
        # between check and reserve), so concurrent room tasks cannot race it.
        if not dest.reserve_for_migrant(agent):
            return False
        await source.release_agent_for_move(agent)
        await dest.admit_migrant(agent)
        return True

    # ── seeding ───────────────────────────────────────────────────────────
    async def seed(self, zero_genomes: bool = False) -> None:
        for controller in self.controllers.values():
            for _ in range(self.cfg.world.initial_population_per_room):
                genome = (Genome.zeros(self.spec) if zero_genomes else
                          Genome.random(self.spec, self.cfg.genome.init_scale,
                                        controller.rng))
                agent = await controller.seed_agent(genome)
                if agent is None:
                    raise RuntimeError(
                        f"room {controller.room_id} cannot fit the configured "
                        "initial population's adapters — lower "
                        "initial_population_per_room or raise capacity")

    # ── run ───────────────────────────────────────────────────────────────
    async def run(self, max_steps: int | None = None) -> None:
        max_steps = max_steps or self.cfg.run.max_world_steps
        await asyncio.gather(
            *(self._run_room(c, max_steps) for c in self.controllers.values()))

    async def _run_room(self, controller: RoomController, max_steps: int) -> None:
        while not self.stopping and controller.step_count < max_steps:
            if not controller.agents and not controller._pending_arrivals:
                # Extinct room: idle until a migrant arrives or the world ends.
                # The room clock does not tick for an empty room.
                await asyncio.sleep(0)
                if self.population == 0:
                    self.stopping = True  # global extinction (§7: uninformative)
                    controller.log.emit(controller.step_count, "extinction",
                                        room=controller.room_id)
                    break
                await asyncio.sleep(0.01)
                continue
            await controller.run_step()
            # Yield so other rooms' tasks interleave even when engine calls
            # complete synchronously (mock backend).
            await asyncio.sleep(0)

    def final_snapshot(self) -> None:
        for controller in self.controllers.values():
            if controller.agents:
                controller.snapshot()

    def close(self) -> None:
        for controller in self.controllers.values():
            controller.log.close()
