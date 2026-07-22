"""The room controller: the world clock (§2.3) and everything it resolves.

One controller per room. Each call to run_step() advances the room by one
token per living agent. Within a step, agents are processed in sorted-id
order, which makes block accounting deterministic; an agent evicted earlier
in the same step does not act later in it.

Deaths, births and moves are all resolved here, against the room's
authoritative BlockPool. The engine never decides who lives: engine-side
allocation failure is an integrity violation, not a game event (§4.3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import prompts
from .actions import Accept, Go, Mate, Noop, Say, is_well_formed, parse_action
from .agent import Agent, Mode, PendingMate
from .blocks import BlockPool
from .config import EVICT_RANDOM_HOLDER, EVICT_REQUESTER, Config
from .engines.base import EngineBackend, TurnEnded, TurnHandle, TurnToken
from .events import DEATH_EVICTED, DEATH_POOL_EXHAUSTED, EventLog, ExperimentIntegrityError
from .genome import Genome

if TYPE_CHECKING:
    from .world import World


class RoomController:
    def __init__(self, room_id: str, capacity_blocks: int, adapter_blocks: int,
                 engine: EngineBackend, cfg: Config, world: "World",
                 log: EventLog, rng: np.random.Generator):
        self.room_id = room_id
        self.cfg = cfg
        self.world = world
        self.engine = engine
        self.log = log
        self.rng = rng
        self.adapter_blocks = adapter_blocks  # uniform per agent (§3.1)
        self.pool = BlockPool(capacity=capacity_blocks,
                              block_size=cfg.world.block_size)
        self.agents: dict[str, Agent] = {}
        self.turns: dict[str, TurnHandle] = {}
        self.step_count = 0
        self._pending_arrivals: list[Agent] = []
        self._died_this_step: set[str] = set()
        # Engine cleanup for agents killed inside synchronous accounting code;
        # awaited at the end of the step.
        self._pending_cleanup: list = []

    # ── population entry points ───────────────────────────────────────────
    async def seed_agent(self, genome: Genome) -> Agent | None:
        """Add a gen-0 agent. Returns None if the room can't fit its adapter."""
        agent_id = self.world.next_agent_id()
        if not self.pool.try_reserve_adapter(agent_id, self.adapter_blocks):
            return None
        agent = Agent(id=agent_id, genome=genome, generation=0, parents=None,
                      adapter_blocks=self.adapter_blocks, born_step=self.step_count)
        self._enqueue_system_prompt(agent)
        await self.engine.register_adapter(agent_id, genome)
        self.agents[agent_id] = agent
        self.log.emit(self.step_count, "birth", agent=agent_id, generation=0,
                      parents=None, room=self.room_id)
        return agent

    def _enqueue_system_prompt(self, agent: Agent) -> None:
        text = prompts.system_prompt(
            agent.id, self.room_id, self.world.adjacent(self.room_id),
            self.cfg.world.mate_window_tokens)
        agent.enqueue_observation(self.engine.tokenize(text), self.engine.turn_end_id)

    # ── the clock ─────────────────────────────────────────────────────────
    async def run_step(self) -> None:
        """Advance the room by one world step: one token per living agent."""
        self.step_count += 1
        self._died_this_step.clear()

        for agent_id in sorted(self.agents):
            if agent_id in self._died_this_step:
                continue
            agent = self.agents.get(agent_id)
            if agent is None:
                continue
            if agent.mode is Mode.OBSERVING:
                self._step_observing(agent)
            else:
                await self._step_acting(agent)

        for coro in self._pending_cleanup:
            await coro
        self._pending_cleanup.clear()

        # Newborns and migrants join at the step boundary.
        for agent in self._pending_arrivals:
            self.agents[agent.id] = agent
        self._pending_arrivals.clear()

        self.engine.integrity_check()

        if self.cfg.run.occupancy_every_steps and \
                self.step_count % self.cfg.run.occupancy_every_steps == 0:
            self._log_occupancy()
        if self.cfg.run.snapshot_every_steps and \
                self.step_count % self.cfg.run.snapshot_every_steps == 0:
            self.snapshot()

    # ── observing: one token off the observation queue ────────────────────
    def _step_observing(self, agent: Agent) -> None:
        if not agent.obs_queue:
            # Invariant: OBSERVING implies a queued utterance (they arrive
            # whole). Recover by swapping to acting rather than stalling.
            agent.mode = Mode.ACTING
            return
        tok = agent.obs_queue.popleft()
        agent.tokens_observed += 1
        if not self._append_token(agent, tok):
            return  # died of the append
        if tok == self.engine.turn_end_id:
            # Swap at turn end (§2.3): from observations, always to actions.
            agent.mode = Mode.ACTING

    # ── acting: one token from the agent's turn stream ────────────────────
    async def _step_acting(self, agent: Agent) -> None:
        handle = self.turns.get(agent.id)
        if handle is None:
            handle = self.engine.start_turn(
                agent.id, list(agent.context), self.cfg.world.max_action_tokens)
            self.turns[agent.id] = handle

        event = await handle.next_event()

        if agent.id in self._died_this_step or agent.id not in self.agents:
            return  # evicted while we awaited; token discarded with the soma

        if isinstance(event, TurnToken):
            agent.tokens_generated += 1
            agent.decay_mate_windows()
            if self._append_token(agent, event.id):
                agent.current_turn.append(event.id)
            return

        assert isinstance(event, TurnEnded)
        # The turn-end token is a real token: charged like any other. This is
        # the floor of §2.3 — the minimum action is a single turn-end token,
        # so context grows every turn under every strategy.
        agent.tokens_generated += 1
        agent.decay_mate_windows()
        agent.forced_turn_end = not event.natural
        self.turns.pop(agent.id, None)
        if not self._append_token(agent, self.engine.turn_end_id):
            return
        await self._complete_action_turn(agent)

    async def _complete_action_turn(self, agent: Agent) -> None:
        turn_text = self.engine.detokenize(agent.current_turn)
        agent.current_turn = []
        agent.action_turns_completed += 1
        well_formed = is_well_formed(turn_text) and not agent.forced_turn_end
        agent.forced_turn_end = False
        if well_formed:
            agent.well_formed_turns += 1
        self._maybe_report_viability(agent)

        action = parse_action(turn_text)
        if isinstance(action, Say):
            self._do_say(agent, action)
        elif isinstance(action, Mate):
            self._do_mate_request(agent, action)
        elif isinstance(action, Accept):
            await self._do_accept(agent, action)
        elif isinstance(action, Go):
            await self._do_go(agent, action)
        else:
            self.log.emit(self.step_count, "noop", agent=agent.id,
                          reason=action.reason, forced=not well_formed)

        # Swap at turn end (§2.3): to observations if anything is queued,
        # otherwise straight back to acting.
        if agent.id in self.agents and agent.id not in self._died_this_step:
            agent.mode = Mode.OBSERVING if agent.obs_queue else Mode.ACTING

    # ── actions ───────────────────────────────────────────────────────────
    def _do_say(self, agent: Agent, action: Say) -> None:
        agent.says += 1
        payload = self.engine.tokenize(prompts.format_say(agent.id, action.text))
        listeners = [a for a in self.agents.values()
                     if a.id != agent.id and a.id not in self._died_this_step]
        for other in listeners:
            # Unfiltered broadcast: speech consumes blocks in every listener
            # (§2.4) — they pay as the tokens are metered onto their contexts.
            other.enqueue_observation(list(payload), self.engine.turn_end_id)
        self.log.emit(self.step_count, "say", agent=agent.id,
                      tokens=len(payload) + 1, listeners=len(listeners))

    def _do_mate_request(self, agent: Agent, action: Mate) -> None:
        agent.mates_requested += 1
        target = self.agents.get(action.target)
        ok = target is not None and target.id != agent.id \
            and target.id not in self._died_this_step
        self.log.emit(self.step_count, "mate_request", agent=agent.id,
                      target=action.target, delivered=bool(ok))
        if not ok:
            return
        window = self.cfg.world.mate_window_tokens
        target.pending_mates.append(PendingMate(agent.id, window))
        target.enqueue_observation(
            self.engine.tokenize(prompts.format_mate_request(agent.id, window)),
            self.engine.turn_end_id)

    async def _do_accept(self, agent: Agent, action: Accept) -> None:
        agent.accepts_emitted += 1
        pending = agent.take_pending_mate(action.target)
        requester = self.agents.get(action.target)
        ok = pending is not None and requester is not None \
            and requester.id not in self._died_this_step
        self.log.emit(self.step_count, "mate_accept", agent=agent.id,
                      target=action.target, valid=bool(ok))
        if ok:
            await self._birth(requester, agent)

    async def _birth(self, p1: Agent, p2: Agent) -> None:
        child_id = self.world.next_agent_id()
        # Adapter blocks are reserved before the child has any KV (§4.2):
        # a birth fails on adapter-block availability, not on KV availability.
        if not self.pool.try_reserve_adapter(child_id, self.adapter_blocks):
            for parent in (p1, p2):
                parent.enqueue_observation(
                    self.engine.tokenize(prompts.format_birth_failed(
                        p2.id if parent is p1 else p1.id)),
                    self.engine.turn_end_id)
            self.log.emit(self.step_count, "birth_failed", parents=[p1.id, p2.id],
                          free_blocks=self.pool.free, needed=self.adapter_blocks)
            return
        genome = Genome.crossover(p1.genome, p2.genome,
                                  self.cfg.genome.mutation_std, self.rng)
        child = Agent(id=child_id, genome=genome,
                      generation=max(p1.generation, p2.generation) + 1,
                      parents=(p1.id, p2.id),
                      adapter_blocks=self.adapter_blocks,
                      born_step=self.step_count)
        self._enqueue_system_prompt(child)
        await self.engine.register_adapter(child_id, genome)
        self._pending_arrivals.append(child)
        p1.children += 1
        p2.children += 1
        for parent, other in ((p1, p2), (p2, p1)):
            parent.enqueue_observation(
                self.engine.tokenize(prompts.format_birth_notice(child_id, other.id)),
                self.engine.turn_end_id)
        self.log.emit(self.step_count, "birth", agent=child_id,
                      generation=child.generation, parents=[p1.id, p2.id],
                      room=self.room_id)

    async def _do_go(self, agent: Agent, action: Go) -> None:
        moved = await self.world.request_move(self, agent, action.room)
        if moved:
            agent.moves += 1
            self.log.emit(self.step_count, "move", agent=agent.id,
                          to=action.room, tokens=agent.tokens)
        else:
            agent.failed_moves += 1
            capacities = self.world.adjacent_capacities(self.room_id)
            agent.enqueue_observation(
                self.engine.tokenize(prompts.format_move_failed(action.room, capacities)),
                self.engine.turn_end_id)
            self.log.emit(self.step_count, "move_failed", agent=agent.id,
                          to=action.room)

    # ── movement plumbing (called by World) ───────────────────────────────
    def can_accept_migrant(self, agent: Agent) -> bool:
        needed = self.adapter_blocks + self.pool.kv_blocks_for(agent.tokens)
        return self.pool.free >= needed

    def reserve_for_migrant(self, agent: Agent) -> bool:
        """Reserve the arriving agent's full footprint — adapter and current
        KV — atomically. The destination reserves before the source releases
        (§4.5), so an agent is never in flight without a home."""
        if not self.can_accept_migrant(agent):
            return False
        assert self.pool.try_reserve_adapter(agent.id, self.adapter_blocks)
        assert self.pool.try_grow_kv(agent.id, agent.tokens)
        return True

    async def release_agent_for_move(self, agent: Agent) -> None:
        self.agents.pop(agent.id, None)
        handle = self.turns.pop(agent.id, None)
        if handle is not None:
            await handle.abort()
        self.pool.release_all(agent.id)
        await self.engine.unregister_adapter(agent.id)

    async def admit_migrant(self, agent: Agent) -> None:
        await self.engine.register_adapter(agent.id, agent.genome)
        # Movement leaves the soma intact: the context travels; the KV cache
        # is re-computed at the destination on the next turn (prefix caching
        # makes this cheap engine-side; the economy already charged for it).
        agent.pending_mates.clear()   # requests don't survive leaving the room
        agent.current_turn = []
        agent.mode = Mode.OBSERVING
        agent.enqueue_observation(
            self.engine.tokenize(prompts.format_arrival(
                self.room_id, self.world.adjacent(self.room_id))),
            self.engine.turn_end_id)
        self._pending_arrivals.append(agent)

    # ── the append: where scarcity resolves (§2.5) ────────────────────────
    def _append_token(self, agent: Agent, tok: int) -> bool:
        """Append one token to the agent's context, allocating KV blocks.
        Returns False iff the agent died in the attempt. Death occurs when and
        only when a new block is needed and the pool is empty."""
        new_count = agent.tokens + 1
        while not self.pool.try_grow_kv(agent.id, new_count):
            if self.cfg.world.eviction == EVICT_REQUESTER:
                self._kill(agent, DEATH_POOL_EXHAUSTED)
                return False
            assert self.cfg.world.eviction == EVICT_RANDOM_HOLDER
            victim_id = self.pool.random_holder(self.rng)
            if victim_id is None or victim_id == agent.id:
                self._kill(agent, DEATH_POOL_EXHAUSTED)
                return False
            self._kill(self.agents[victim_id], DEATH_EVICTED)
            # Pool space freed; retry the allocation.
        agent.context.append(tok)
        return True

    def _kill(self, agent: Agent, cause: str) -> None:
        self._died_this_step.add(agent.id)
        self.agents.pop(agent.id, None)
        handle = self.turns.pop(agent.id, None)
        if handle is not None:
            self._pending_cleanup.append(handle.abort())
        self.pool.release_all(agent.id)
        self._pending_cleanup.append(self.engine.unregister_adapter(agent.id))
        if not agent.viability_reported:
            self._report_viability(agent, censored=True)
        self.log.death(
            self.step_count, agent.id, cause,
            room=self.room_id, generation=agent.generation,
            lifetime_steps=self.step_count - agent.born_step,
            tokens=agent.tokens, tokens_generated=agent.tokens_generated,
            tokens_observed=agent.tokens_observed,
            turns=agent.action_turns_completed,
            well_formed_turns=agent.well_formed_turns,
            says=agent.says, children=agent.children, moves=agent.moves,
            mean_action_tokens=round(
                agent.tokens_generated / max(agent.action_turns_completed, 1), 2),
        )

    # ── instrumentation ───────────────────────────────────────────────────
    def _maybe_report_viability(self, agent: Agent) -> None:
        k = self.cfg.world.viability_probe_turns
        if not agent.viability_reported and agent.action_turns_completed >= k:
            self._report_viability(agent, censored=False)

    def _report_viability(self, agent: Agent, censored: bool) -> None:
        # §3.2/§5: child viability is logged separately from child survival,
        # so "the merge produces broken agents" is distinguishable from "the
        # environment is too harsh".
        agent.viability_reported = True
        self.log.emit(self.step_count, "viability", agent=agent.id,
                      generation=agent.generation,
                      viable=agent.well_formed_turns > 0,
                      well_formed=agent.well_formed_turns,
                      turns=agent.action_turns_completed, censored=censored)

    def _log_occupancy(self) -> None:
        lens = [a.tokens for a in self.agents.values()]
        self.log.emit(self.step_count, "occupancy", room=self.room_id,
                      agents=len(self.agents),
                      free_blocks=self.pool.free,
                      capacity_blocks=self.pool.capacity,
                      mean_context=round(float(np.mean(lens)), 1) if lens else 0,
                      max_context=max(lens, default=0),
                      generations=sorted({a.generation for a in self.agents.values()}))

    def snapshot(self) -> None:
        out = Path(self.cfg.run.out_dir) / self.cfg.run_name / "snapshots" / \
            self.room_id / f"step_{self.step_count:08d}"
        out.mkdir(parents=True, exist_ok=True)
        meta = []
        for agent in self.agents.values():
            agent.genome.save(out / f"{agent.id}.safetensors")
            meta.append({
                "id": agent.id, "generation": agent.generation,
                "parents": agent.parents, "tokens": agent.tokens,
                "born_step": agent.born_step,
                "turns": agent.action_turns_completed,
                "well_formed_turns": agent.well_formed_turns,
                "children": agent.children,
            })
        with open(out / "population.json", "w") as f:
            json.dump({"step": self.step_count, "room": self.room_id,
                       "agents": meta}, f, indent=2)
        self.log.emit(self.step_count, "snapshot", room=self.room_id,
                      agents=len(meta), path=str(out))
