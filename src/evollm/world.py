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

        # Several rooms may share one engine (many small rooms per GPU). The
        # weights are already resident, so extra rooms on a device are free —
        # but they must divide that device's KV pool between them, or each
        # would hand out blocks the others are also handing out.
        rooms_per_engine: dict[int, int] = {}
        for r in cfg.world.rooms:
            rooms_per_engine[id(engines[r.id])] = \
                rooms_per_engine.get(id(engines[r.id]), 0) + 1

        for room in cfg.world.rooms:
            engine = engines[room.id]
            share = rooms_per_engine[id(engine)]
            capacity = room.capacity_blocks
            derived = capacity is None
            if derived:
                pool = engine.capacity_blocks()
                capacity = pool // share if pool is not None else None
            if capacity is None:
                raise ValueError(
                    f"room {room.id}: capacity_blocks not set and the engine "
                    "cannot derive it")
            capacity = self._enforce_context_ceiling(
                room.id, capacity, adapter_blocks, derived)
            if not derived:
                engine_pool = engine.capacity_blocks()
                if engine_pool is not None:
                    engine_pool //= share      # this room's slice of the device
                if engine_pool is not None and capacity > engine_pool:
                    # The controller would hand out blocks the device does not
                    # have; the engine's only recourse is preemption, which is
                    # an integrity violation rather than a death (§4.3).
                    raise ValueError(
                        f"room {room.id}: capacity_blocks={capacity} exceeds "
                        f"the engine's usable pool of {engine_pool} blocks. "
                        "Lower capacity_blocks, raise "
                        "engine.gpu_memory_utilization, or lower "
                        "engine.safety_margin_blocks.")
            log = EventLog(base / "events" / f"{room.id}.jsonl")
            self.controllers[room.id] = RoomController(
                room_id=room.id, capacity_blocks=capacity,
                adapter_blocks=adapter_blocks, engine=engine, cfg=cfg,
                world=self, log=log,
                rng=np.random.default_rng(self.rng.integers(2**63)))

    def _enforce_context_ceiling(self, room_id: str, capacity: int,
                                 adapter_blocks: int, derived: bool) -> int:
        """Scarcity must bind before the model's context window does (§4.3).

        A lone survivor can hold every block in the room, so its context can
        reach (capacity - adapter_blocks) * block_size tokens. If that exceeds
        max_model_len the agent hits an infrastructure ceiling rather than a
        scarcity event — the engine rejects the request and the death, if it
        happened at all, would not be attributable to the pool. Deriving
        capacity from the engine clamps to the safe bound; an explicit
        capacity_blocks that violates it is a config error and raises, because
        silently shrinking a number the experimenter chose would misreport
        what was actually run.
        """
        block_size = self.cfg.world.block_size
        max_len = self.cfg.model.max_model_len
        # vLLM needs prompt + at least one output token to fit, hence < not <=.
        safe = adapter_blocks + (max_len - 1) // block_size
        if capacity <= safe:
            return capacity
        reachable = (capacity - adapter_blocks) * block_size
        if not derived:
            raise ValueError(
                f"room {room_id}: capacity_blocks={capacity} lets a single "
                f"agent reach {reachable} tokens, over max_model_len="
                f"{max_len}. Scarcity would never bind and deaths would be "
                f"infrastructure artefacts (§4.3). Set capacity_blocks "
                f"<= {safe}, or raise model.max_model_len.")
        print(f"[{room_id}] clamping engine-derived capacity {capacity} -> "
              f"{safe} blocks so scarcity binds before max_model_len={max_len}")
        return safe

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
    def _checkpoint_genomes(self) -> list[Genome]:
        """Load a checkpointed population to start from.

        Finding a self-sustaining population is an undirected walk over
        initialisations and is the expensive part of a run; `seed_from` lets a
        later run begin where an earlier one got to instead of paying for the
        search again.
        """
        from pathlib import Path

        root = Path(self.cfg.seed_from)
        if not root.exists():
            raise ValueError(f"seed_from path does not exist: {root}")
        files = sorted(root.rglob("*.safetensors"))
        if not files:
            raise ValueError(f"seed_from contains no genomes: {root}")
        print(f"[world] seeding from checkpoint {root} ({len(files)} genomes)")
        return [Genome.load(f, self.spec) for f in files]

    async def seed(self, zero_genomes: bool = False) -> None:
        pool = self._checkpoint_genomes() if self.cfg.seed_from else None
        for controller in self.controllers.values():
            for i in range(self.cfg.world.initial_population_per_room):
                if pool is not None:
                    # Cycle if the checkpoint holds fewer agents than a room
                    # wants; the population is the seed, not a hard roster.
                    genome = pool[i % len(pool)]
                elif zero_genomes:
                    genome = Genome.zeros(self.spec)
                else:
                    genome = Genome.random(self.spec, self.cfg.genome.init_scale,
                                           controller.rng)
                agent = await controller.seed_agent(genome)
                if agent is None:
                    raise RuntimeError(
                        f"room {controller.room_id} cannot fit the configured "
                        "initial population's adapters — lower "
                        "initial_population_per_room or raise capacity")
            controller.finish_seeding()

    # ── run ───────────────────────────────────────────────────────────────
    async def run(self, max_steps: int | None = None) -> None:
        max_steps = max_steps or self.cfg.run.max_world_steps
        await asyncio.gather(
            *(self._run_room(c, max_steps) for c in self.controllers.values()))

    async def _run_room(self, controller: RoomController, max_steps: int) -> None:
        while not self.stopping and controller.step_count < max_steps:
            if not controller.agents and not controller._pending_arrivals:
                # An emptied room is refilled rather than declared extinct, if
                # refill is on: the check normally runs inside run_step, which
                # an empty room never reaches.
                if self.cfg.refill.enabled and await controller.refill():
                    continue
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
