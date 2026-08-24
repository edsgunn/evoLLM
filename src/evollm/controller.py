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
from .actions import (Accept, Action, Go, Mate, Noop, Say, Tell, classify,
                      complete_action)
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
        self._obs_sep_cache: list[int] | None = None
        self._fingerprint_fh = None
        # Engine cleanup for agents killed inside synchronous accounting code;
        # awaited at the end of the step.
        self._pending_cleanup: list = []
        self._traced = 0
        self.refills = 0
        self.preemptions = 0
        # Rolling record for takeoff detection: steps at which a child was
        # born, and at which an immigrant had to be admitted.
        self._birth_steps: list[int] = []
        self._refill_steps: list[int] = []
        self._self_sustaining = False
        self.takeoff_step: int | None = None

    # ── population entry points ───────────────────────────────────────────
    async def seed_agent(self, genome: Genome, origin: str = "seed") -> Agent | None:
        """Add a generation-0 agent. Returns None if the room can't fit its
        adapter. `origin` distinguishes a founder from a later immigrant."""
        agent_id = self.world.next_agent_id()
        if not self.pool.try_reserve_adapter(agent_id, self.adapter_blocks):
            return None
        agent = Agent(id=agent_id, genome=genome, generation=0, parents=None,
                      adapter_blocks=self.adapter_blocks,
                      born_step=self.step_count, origin=origin)
        await self.engine.register_adapter(agent_id, genome)
        self.agents[agent_id] = agent
        self.log.emit(self.step_count, "birth", agent=agent_id, generation=0,
                      parents=None, room=self.room_id, origin=origin)
        self._record_genome(agent)
        return agent

    # ── refill: keeping the arena populated (§6, §7) ──────────────────────
    def _refill_wanted(self) -> bool:
        cfg = self.cfg.refill
        if not cfg.enabled:
            return False
        if cfg.max_total is not None and self.refills >= cfg.max_total:
            return False
        living = len(self.agents) + len(self._pending_arrivals)
        if living < cfg.min_population:
            return True
        if cfg.max_free_fraction is not None and \
                self.pool.free / self.pool.capacity > cfg.max_free_fraction:
            return True
        return False

    async def refill(self) -> int:
        """Admit immigrants perturbed from the base model until the room is
        back above its floor.

        Nothing here inspects behaviour: it counts agents and free blocks. A
        lineage that reproduces faster than it dies keeps the room above the
        floor by itself, and this stops firing — which is precisely the signal
        the run is trying to produce.
        """
        scale = (self.cfg.refill.perturbation_scale
                 if self.cfg.refill.perturbation_scale is not None
                 else self.cfg.genome.init_scale)
        admitted = 0
        while self._refill_wanted() and admitted < self.cfg.refill.max_per_check:
            genome = Genome.random(self.world.spec, scale, self.rng)
            agent = await self.seed_agent(genome, origin="refill")
            if agent is None:
                break          # the room is full; scarcity, not policy, decides
            self.refills += 1
            self._refill_steps.append(self.step_count)
            admitted += 1
            self._enqueue_system_prompt(agent)
            self._broadcast(prompts.format_arrival_notice(agent.id, arrived=True),
                            exclude={agent.id})
            self.log.emit(self.step_count, "refill", agent=agent.id,
                          room=self.room_id, population=len(self.agents),
                          free_blocks=self.pool.free, total_refills=self.refills)
        return admitted

    def finish_seeding(self) -> None:
        """Issue gen-0 system prompts once the whole founding population
        exists. Prompts carry the roster, so issuing them during seeding would
        tell the first agent the room is empty and the last one the truth."""
        for agent in self.agents.values():
            self._enqueue_system_prompt(agent)

    def _others(self, agent_id: str) -> list[str]:
        """Live co-residents, excluding the asker and anyone dead this step."""
        return [a for a in self.agents
                if a != agent_id and a not in self._died_this_step]

    def _enqueue_system_prompt(self, agent: Agent) -> None:
        text = prompts.system_prompt(
            agent.id, self.room_id, self.world.adjacent(self.room_id),
            others=self._others(agent.id), tools=self.cfg.world.tools)
        self._enqueue(agent, text, role="system")

    def _enqueue(self, agent: Agent, text: str, role: str = "user") -> int:
        """Queue an observation body. Framing is deferred to absorption (see
        _materialise_observation) so that a backlog becomes one user block
        rather than a run of consecutive ones."""
        return agent.enqueue_observation(self.engine.tokenize(text), role)

    @property
    def _obs_separator(self) -> list[int]:
        """Newline between two observations sharing a block."""
        if self._obs_sep_cache is None:
            self._obs_sep_cache = self.engine.tokenize("\n")
        return self._obs_sep_cache

    def _materialise_observation(self, agent: Agent) -> None:
        """Frame the next queued utterance into the agent's emit buffer.

        A block header is emitted only if no block is open, and the utterance
        is terminated with a newline rather than a turn-end whenever another
        observation is already waiting behind it and the read policy will go
        straight on to it. The result is one user block per drain, holding
        every observation the agent had not yet reached:

            <|im_start|>user
            [world] a8797 has arrived in this room
            [world] a8729 has left this room<|im_end|>

        Framing is still charged to the agent exactly like content (§2.4); it
        is simply charged once per block now instead of once per observation,
        which is also why a merged backlog costs slightly fewer tokens.
        """
        u = agent.obs_queue.popleft()
        toks: list[int] = []
        if not agent.obs_block_open:
            header = self.engine.block_prefix(
                u.role, first=agent.blocks_opened == 0)
            toks += header
            agent.tokens_framing += len(header)
            agent.obs_block_open = True
            agent.blocks_opened += 1
        toks += u.body
        # Without chat format there are no blocks to merge into, so each
        # utterance keeps its own turn-end exactly as before.
        merge = (self.cfg.world.chat_format
                 and self.cfg.world.read_policy == "drain"
                 and bool(agent.obs_queue)
                 and agent.obs_queue[0].role == u.role)
        if merge:
            toks += self._obs_separator
            agent.tokens_framing += len(self._obs_separator)
            agent.obs_closes_block = False
        else:
            # The turn-end is the queue's own delimiter, not inserted framing,
            # and has never been charged to tokens_framing.
            toks.append(self.engine.turn_end_id)
            agent.obs_closes_block = True
        agent.obs_emit.extend(toks)

    def _begin_action_turn(self, agent: Agent) -> None:
        """Open the assistant block, then let the model speak into it."""
        agent.mode = Mode.ACTING
        prefix = self.engine.block_prefix("assistant")
        if prefix:
            agent.pending_emit.extend(prefix)

    def _broadcast(self, text: str, exclude: set[str]) -> None:
        """Send a world observation to every live agent except `exclude`.
        Costs blocks in each listener like any other observation (§2.4)."""
        for other in list(self.agents.values()):
            if other.id in exclude or other.id in self._died_this_step:
                continue
            self._enqueue(other, text)

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
            agent.age += 1
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

        self._check_preemptions()

        if self.cfg.refill.enabled and self.cfg.refill.check_every_steps and \
                self.step_count % self.cfg.refill.check_every_steps == 0:
            await self.refill()
            self._check_takeoff()

        if self.cfg.run.occupancy_every_steps and \
                self.step_count % self.cfg.run.occupancy_every_steps == 0:
            self._log_occupancy()
        if self.cfg.run.snapshot_every_steps and \
                self.step_count % self.cfg.run.snapshot_every_steps == 0:
            self.snapshot()
        if self.cfg.run.context_snapshot_every_steps and \
                self.step_count % self.cfg.run.context_snapshot_every_steps == 0:
            self.snapshot_contexts()

    # ── observing: take an utterance (or a token) off the queue ───────────
    def _step_observing(self, agent: Agent) -> None:
        """Absorb queued observation.

        In "utterance" mode the whole utterance lands in context in this one
        step, so being talked at costs blocks in proportion to what was said —
        which is what §2.4 always claimed and what one-token-at-a-time did not
        deliver: under that rule every agent's context grew at exactly one
        token per step no matter what it or anyone else did, making speech
        free for listeners and survival independent of behaviour.

        One utterance is absorbed per step even when draining, so pacing is
        unchanged by block merging — what merging removes is the chat framing
        between them, not the steps.

        The agent may die partway through absorbing, which is an ordinary
        scarcity death: it was killed by what the room said to it.
        """
        if not agent.obs_emit:
            if not agent.obs_queue:
                if agent.obs_block_open:
                    # Defensive: never open an assistant block on top of an
                    # unclosed user one.
                    agent.obs_emit.append(self.engine.turn_end_id)
                    agent.obs_closes_block = True
                else:
                    # Invariant: OBSERVING implies something to read. Recover
                    # by swapping to acting rather than stalling.
                    self._begin_action_turn(agent)
                    return
            else:
                self._materialise_observation(agent)

        whole_utterance = self.cfg.world.observation_absorption == "utterance"
        while agent.obs_emit:
            tok = agent.obs_emit.popleft()
            agent.tokens_observed += 1
            if not self._append_token(agent, tok):
                return  # died mid-utterance
            if agent.obs_emit and not whole_utterance:
                return  # one token per step

        # The utterance is now fully in context. Any mate request it carried
        # becomes live here — this is the moment the agent can actually be
        # said to have received it (§2.4).
        for requester_id in agent.note_utterance_read():
            agent.pending_mates.append(
                PendingMate(requester_id, self.cfg.world.mate_window_tokens))
        if agent.obs_closes_block:
            agent.obs_block_open = False
            # Swap at turn end (§2.3): from observations, always to actions.
            self._begin_action_turn(agent)
        # Otherwise the block stays open and the next observation joins it.

    # ── acting: one token from the agent's turn stream ────────────────────
    async def _step_acting(self, agent: Agent) -> None:
        if agent.pending_emit:
            # Opening the assistant block costs tokens like anything else.
            tok = agent.pending_emit.popleft()
            agent.tokens_framing += 1
            self._append_token(agent, tok)
            return

        if agent.pending_turn_end:
            # A tag closed last step; spend this step on the turn-end token.
            agent.pending_turn_end = False
            agent.tokens_generated += 1
            agent.decay_mate_windows()
            if self._append_token(agent, self.engine.turn_end_id):
                await self._complete_action_turn(agent)
            return

        handle = self.turns.get(agent.id)
        if handle is None:
            budget = self._turn_budget(agent)
            handle = self.engine.start_turn(agent.id, list(agent.context), budget)
            self.turns[agent.id] = handle

        event = await handle.next_event()

        if agent.id in self._died_this_step or agent.id not in self.agents:
            return  # evicted while we awaited; token discarded with the soma

        if isinstance(event, TurnToken):
            agent.tokens_generated += 1
            agent.decay_mate_windows()
            if not self._append_token(agent, event.id):
                return
            agent.current_turn.append(event.id)
            # "One action per turn" is enforced here rather than asked for in
            # the prompt: the moment a tag closes, the world ends the turn.
            # Checking only when the new token could terminate a tag keeps this
            # off the hot path.
            if any(ch in self.engine.detokenize([event.id]) for ch in ">]}") \
                    and complete_action(self.engine.detokenize(agent.current_turn)):
                await handle.abort()
                self.turns.pop(agent.id, None)
                agent.pending_turn_end = True
            return

        assert isinstance(event, TurnEnded)
        # The turn-end token is a real token: charged like any other. This is
        # the floor of §2.3 — the minimum action is a single turn-end token,
        # so context grows every turn under every strategy.
        agent.tokens_generated += 1
        agent.decay_mate_windows()
        self.turns.pop(agent.id, None)
        if not self._append_token(agent, self.engine.turn_end_id):
            return
        await self._complete_action_turn(agent)

    async def _complete_action_turn(self, agent: Agent) -> None:
        turn_text = self.engine.detokenize(agent.current_turn)
        turn_tokens = len(agent.current_turn)
        agent.current_turn = []
        agent.action_turns_completed += 1

        # A turn ends when a tag closes, when the agent emits the turn-end
        # token itself, or when it dies mid-sentence. Whichever it was, the
        # syntax used is recorded rather than punished.
        parsed = classify(turn_text)
        if parsed.is_action:
            agent.well_formed_turns += 1
            if parsed.is_canonical:
                agent.canonical_turns += 1
        # Thinking is whatever the agent generated before it acted: prose it
        # was charged for and did not act on. There is no protected reasoning
        # region — writing an action tag *is* acting, wherever it appears.
        thinking = len(self.engine.tokenize(turn_text[:parsed.start])) \
            if parsed.start else 0
        agent.thinking_tokens += thinking
        self._maybe_report_viability(agent)
        self._trace_turn(agent, turn_text, parsed, thinking, turn_tokens)

        action = parsed.action
        if not isinstance(action, Noop) and not self._tool_enabled(action):
            verb = type(action).__name__.lower()
            self._enqueue(agent, prompts.format_tool_unavailable(
                verb, self.cfg.world.tools))
            self.log.emit(self.step_count, "tool_unavailable", agent=agent.id,
                          verb=verb)
            if agent.id in self.agents and agent.id not in self._died_this_step:
                if agent.has_pending_observations:
                    agent.mode = Mode.OBSERVING
                else:
                    self._begin_action_turn(agent)
            return

        if isinstance(action, Say):
            self._do_say(agent, action)
        elif isinstance(action, Tell):
            self._do_tell(agent, action)
        elif isinstance(action, Mate):
            await self._do_mate_request(agent, action)
        elif isinstance(action, Accept):
            await self._do_accept(agent, action)
        elif isinstance(action, Go):
            await self._do_go(agent, action)
        else:
            # A noop is a turn the agent spent generating without ever
            # forming an action: pure thinking, charged in full.
            self.log.emit(self.step_count, "noop", agent=agent.id,
                          reason=action.reason, turn_tokens=turn_tokens,
                          thinking_tokens=thinking)

        # Swap at turn end (§2.3): to observations if anything is queued,
        # otherwise straight back to acting.
        if agent.id in self.agents and agent.id not in self._died_this_step:
            if agent.has_pending_observations:
                agent.mode = Mode.OBSERVING
            else:
                self._begin_action_turn(agent)

    # ── actions ───────────────────────────────────────────────────────────
    _VERB_TOOL = {Say: "say", Tell: "tell", Mate: "mate", Accept: "accept",
                  Go: "go"}

    def _tool_enabled(self, action: Action) -> bool:
        return self._VERB_TOOL[type(action)] in self.cfg.world.tools

    def _do_tell(self, agent: Agent, action: Tell) -> None:
        """Directed speech: exactly one recipient, so one generated token
        becomes one observation token and the room's information economy
        closes. Contrast _do_say, whose fan-out is the room's occupancy."""
        agent.tells += 1
        target = self.agents.get(action.target)
        ok = (target is not None and target.id != agent.id
              and target.id not in self._died_this_step)
        if not ok:
            self._enqueue(agent, prompts.format_tell_failed(
                action.target, self._others(agent.id)))
            self.log.emit(self.step_count, "tell", agent=agent.id,
                          target=action.target, delivered=False, tokens=0)
            return
        text = prompts.format_tell(agent.id, target.id, action.text)
        before = target.obs_backlog
        self._enqueue(target, text)
        self.log.emit(self.step_count, "tell", agent=agent.id,
                      target=target.id, delivered=True,
                      tokens=target.obs_backlog - before)

    def _do_say(self, agent: Agent, action: Say) -> None:
        agent.says += 1
        text = prompts.format_say(agent.id, action.text)
        listeners = [a for a in self.agents.values()
                     if a.id != agent.id and a.id not in self._died_this_step]
        cost = 0
        for other in listeners:
            # Unfiltered broadcast: speech consumes blocks in every listener
            # (§2.4) — they pay as the tokens are metered onto their contexts.
            before = other.obs_backlog
            self._enqueue(other, text)
            cost = other.obs_backlog - before
        self.log.emit(self.step_count, "say", agent=agent.id,
                      tokens=cost, listeners=len(listeners))

    async def _do_mate_request(self, agent: Agent, action: Mate) -> None:
        agent.mates_requested += 1
        target = self.agents.get(action.target)
        ok = target is not None and target.id != agent.id \
            and target.id not in self._died_this_step
        # Reciprocity is the handshake (§2.4). Pointing <mate> at someone who
        # has already pointed it at you *is* the acceptance — there is no
        # separate verb, so an agent that means "yes" and an agent that means
        # "let's" emit the same thing, and a misfired agreement lands as a
        # proposal rather than as nothing.
        if ok:
            pending = agent.take_pending_mate(action.target)
            if pending is not None:
                self.log.emit(self.step_count, "mate_request", agent=agent.id,
                              target=action.target, delivered=True,
                              reciprocated=True)
                await self._birth(target, agent)
                return
        self.log.emit(self.step_count, "mate_request", agent=agent.id,
                      target=action.target, delivered=bool(ok),
                      reciprocated=False)
        if not ok:
            # Symmetric with a failed <go> (§2.4): the attempt itself is how
            # the agent perceives the room. It pays for the observation.
            self._enqueue(agent, prompts.format_mate_failed(
                action.target, self._others(agent.id)))
            return
        window = self.cfg.world.mate_window_tokens
        # The window is armed when the target *reads* this, not now. Starting
        # it here meant it expired while the request sat in a backlog the
        # target had not reached: across 430 accepts, the shortest lag between
        # a request and its matching accept was 177 world steps against a
        # 64-token window, so no acceptance could ever be valid.
        index = self._enqueue(target, prompts.format_mate_request(agent.id))
        target.deferred_mates.append((index, agent.id))

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

    def _reserve_child_adapter(self, child_id: str, p1: Agent, p2: Agent) -> bool:
        """Reserve the child's adapter, charged to whoever is paying.

        With parental_investment the parents carry it between them for as long
        as they live, so an agent that reproduces heavily accumulates real
        occupied memory and — under random_holder eviction — dies sooner for
        it. The same blocks are reserved either way; only the owner differs.
        """
        n = self.adapter_blocks
        if not self.cfg.world.parental_investment:
            return self.pool.try_reserve_adapter(child_id, n)
        first = n // 2
        if not self.pool.reserve_dependent(p1.id, child_id, first):
            return False
        if not self.pool.reserve_dependent(p2.id, child_id, n - first):
            self.pool.release_dependent(child_id)      # unwind the first half
            return False
        return True

    async def _birth(self, p1: Agent, p2: Agent) -> None:
        child_id = self.world.next_agent_id()
        # Adapter blocks are reserved before the child has any KV (§4.2):
        # a birth fails on adapter-block availability, not on KV availability.
        if not self._reserve_child_adapter(child_id, p1, p2):
            for parent in (p1, p2):
                self._enqueue(parent, prompts.format_birth_failed(
                    p2.id if parent is p1 else p1.id))
            self.log.emit(self.step_count, "birth_failed", parents=[p1.id, p2.id],
                          free_blocks=self.pool.free, needed=self.adapter_blocks)
            return
        gcfg = self.cfg.genome
        genome, donors = Genome.crossover(
            p1.genome, p2.genome, gcfg.mutation_std, self.rng,
            scheme=gcfg.crossover, chromosomes=gcfg.chromosomes,
            mutation=gcfg.mutation, return_donors=True)
        child = Agent(id=child_id, genome=genome,
                      generation=max(p1.generation, p2.generation) + 1,
                      parents=(p1.id, p2.id),
                      adapter_blocks=self.adapter_blocks,
                      born_step=self.step_count, origin="birth")
        self._enqueue_system_prompt(child)
        await self.engine.register_adapter(child_id, genome)
        self._pending_arrivals.append(child)
        # Incumbents learn there is a new co-resident; the parents get the
        # richer birth notice below instead.
        self._broadcast(prompts.format_arrival_notice(child_id, arrived=True),
                        exclude={p1.id, p2.id})
        self._birth_steps.append(self.step_count)
        p1.children += 1
        p2.children += 1
        for parent, other in ((p1, p2), (p2, p1)):
            self._enqueue(parent, prompts.format_birth_notice(child_id, other.id))
        self.log.emit(self.step_count, "birth", agent=child_id,
                      generation=child.generation, parents=[p1.id, p2.id],
                      room=self.room_id, origin="birth")
        self._record_genome(child, parents=(p1.id, p2.id), donors=donors)

    async def _do_go(self, agent: Agent, action: Go) -> None:
        moved = await self.world.request_move(self, agent, action.room)
        if moved:
            agent.moves += 1
            self.log.emit(self.step_count, "move", agent=agent.id,
                          to=action.room, tokens=agent.tokens)
        else:
            agent.failed_moves += 1
            capacities = self.world.adjacent_capacities(self.room_id)
            self._enqueue(agent, prompts.format_move_failed(action.room, capacities))
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
        self._broadcast(prompts.format_arrival_notice(agent.id, arrived=False),
                        exclude={agent.id})
        handle = self.turns.pop(agent.id, None)
        if handle is not None:
            await handle.abort()
        # Dependent charges are room-local: the pool being debited is the one
        # whose device memory is actually holding the adapter. A parent leaving
        # hands its charges back to the children it leaves behind, and a child
        # leaving stops being charged to parents in the room it left.
        self.pool.revert_dependents(agent.id)
        self.pool.release_dependent(agent.id)
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
        self._enqueue(agent, prompts.format_arrival(
            self.room_id, self.world.adjacent(self.room_id),
            others=self._others(agent.id)))
        self._broadcast(prompts.format_arrival_notice(agent.id, arrived=True),
                        exclude={agent.id})
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

        # Scarcity has had its say: the pool granted the block, so it is not
        # what is limiting this agent. Only now is a context-ceiling breach a
        # real integrity violation. Checking before the allocation instead
        # made this fire at the exact boundary where the invariant is working
        # — an agent holding every block it can, whose next token the pool was
        # about to refuse — and killed a run that was behaving correctly.
        # vLLM needs the prompt plus one output token to fit, so the usable
        # context is max_model_len - 1.
        if new_count > self.cfg.model.max_model_len - 1:
            raise ExperimentIntegrityError(
                f"{agent.id} reached {new_count} tokens against max_model_len="
                f"{self.cfg.model.max_model_len} while the room still had "
                f"blocks to give ({self.pool.free}/{self.pool.capacity} free): "
                "the context ceiling is binding before scarcity, so deaths in "
                "this run are not attributable to the pool (§4.3)")
        agent.context.append(tok)
        return True

    def _kill(self, agent: Agent, cause: str) -> None:
        self._died_this_step.add(agent.id)
        self.agents.pop(agent.id, None)
        # Survivors learn the room's composition changed — otherwise they go
        # on addressing the dead, which is most of what happened in the first
        # GPU run. Enqueued only; the listeners pay for it a token at a time.
        self._broadcast(prompts.format_arrival_notice(agent.id, arrived=False),
                        exclude={agent.id})
        handle = self.turns.pop(agent.id, None)
        if handle is not None:
            self._pending_cleanup.append(handle.abort())
        # Its children's adapters are still registered, so those blocks must
        # keep being accounted: they revert to the children themselves. Its own
        # adapter, carried by its parents, is gone and stops being charged.
        self.pool.revert_dependents(agent.id)
        self.pool.release_dependent(agent.id)
        self.pool.release_all(agent.id)
        self._pending_cleanup.append(self.engine.unregister_adapter(agent.id))
        if not agent.viability_reported:
            self._report_viability(agent, censored=True)
        self.log.death(
            self.step_count, agent.id, cause,
            room=self.room_id, generation=agent.generation,
            origin=agent.origin,
            lifetime_steps=agent.age,
            tokens=agent.tokens, tokens_generated=agent.tokens_generated,
            tokens_observed=agent.tokens_observed,
            turns=agent.action_turns_completed,
            well_formed_turns=agent.well_formed_turns,
            canonical_turns=agent.canonical_turns,
            thinking_tokens=agent.thinking_tokens,
            says=agent.says, tells=agent.tells,
            children=agent.children, moves=agent.moves,
            mean_action_tokens=round(
                agent.tokens_generated / max(agent.action_turns_completed, 1), 2),
        )

    # ── takeoff (§7) ──────────────────────────────────────────────────────
    def _check_takeoff(self) -> None:
        """Has this room started sustaining itself?

        The criterion is deliberately about *descent*, not about survival: a
        room kept at its floor by immigration looks identical to a thriving one
        if you only count agents. Self-sustaining means the room went a full
        window needing no immigrant while still producing children.

        Reaching this state is the expensive part of a run — an undirected walk
        over initialisations — so it is checkpointed the instant it happens.
        """
        cfg = self.cfg.refill
        window_start = self.step_count - cfg.takeoff_window_steps
        if window_start < 0:
            return          # not enough history to judge
        births = sum(1 for s in self._birth_steps if s > window_start)
        refills = sum(1 for s in self._refill_steps if s > window_start)
        sustaining = refills == 0 and births >= cfg.takeoff_min_births

        if sustaining and not self._self_sustaining:
            self._self_sustaining = True
            self.takeoff_step = self.step_count
            self.log.emit(self.step_count, "takeoff", room=self.room_id,
                          births_in_window=births, window=cfg.takeoff_window_steps,
                          population=len(self.agents),
                          generations=sorted({a.generation
                                              for a in self.agents.values()}),
                          refills_before=self.refills)
            if cfg.checkpoint_on_takeoff:
                self.snapshot(kind="checkpoints",
                              label=f"takeoff_{self.step_count:08d}",
                              extra={"event": "takeoff",
                                     "births_in_window": births,
                                     "window_steps": cfg.takeoff_window_steps,
                                     "refills_before": self.refills})
        elif not sustaining and self._self_sustaining:
            # Lapsed: it needed help again. Re-arm so a later recovery is
            # checkpointed too, and so the log shows the state was not durable.
            self._self_sustaining = False
            self.log.emit(self.step_count, "takeoff_lost", room=self.room_id,
                          births_in_window=births, refills_in_window=refills)

    # ── instrumentation ───────────────────────────────────────────────────
    def _trace_turn(self, agent: Agent, text: str, parsed, thinking: int,
                    turn_tokens: int) -> None:
        """Record the raw text of a turn and how it parsed. Without this a
        malformed turn is only a counter, and there is no way to tell a model
        that cannot follow the protocol from one that is a comma away."""
        if self._traced >= self.cfg.run.trace_turns:
            return
        self._traced += 1
        self.log.emit(self.step_count, "turn", agent=agent.id,
                      generation=agent.generation,
                      action=type(parsed.action).__name__,
                      form=parsed.form, thinking_tokens=thinking,
                      turn_tokens=turn_tokens, context_tokens=agent.tokens,
                      text=text[:600])

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
        # An agent absorbs one observation token per step, so its backlog is
        # literally how many steps behind the room it is. A room where this
        # grows without bound cannot support timely coordination — and so
        # cannot support reproduction — however well its agents behave.
        backlog = [a.obs_backlog for a in self.agents.values()]
        self.log.emit(self.step_count, "occupancy", room=self.room_id,
                      agents=len(self.agents),
                      free_blocks=self.pool.free,
                      capacity_blocks=self.pool.capacity,
                      mean_context=round(float(np.mean(lens)), 1) if lens else 0,
                      max_context=max(lens, default=0),
                      mean_backlog=round(float(np.mean(backlog)), 1) if backlog else 0,
                      max_backlog=max(backlog, default=0),
                      generations=sorted({a.generation for a in self.agents.values()}),
                      descended=sum(1 for a in self.agents.values()
                                    if a.origin == 'birth'),
                      refills_so_far=self.refills)

    def _turn_budget(self, agent: Agent) -> int:
        """How many tokens this turn may generate before the engine must stop.

        Still no designer-chosen length cap: an agent that never closes a tag
        spends its life talking and dies of the pool, and selection removes it
        without anyone picking a number. The bound here is purely physical, and
        it exists because of a mismatch that made running out of memory possible
        despite the economy forbidding it.

        The controller consumes ONE token per agent per world step, but vLLM
        generates as fast as the GPU allows and buffers the rest in the turn
        handle. Every token generated-but-not-yet-consumed occupies KV on the
        device while being charged to nobody, because the pool is only debited
        in _append_token. So engine usage was never bounded by
        capacity_blocks — it was capacity plus unbounded run-ahead, and with
        max_model_len at 767,649 a single request could be granted 47,978
        blocks of it, sixteen times the room's entire claim.

        The fix is to hand out only what physically exists and is unclaimed.
        The engine's pool is larger than the room's capacity by design; that gap
        is the only memory nobody has been charged for, so it is exactly the
        budget available for run-ahead. Splitting it across the agents that
        could be generating makes the invariant hold by construction:

            engine usage  <=  capacity_blocks  +  unclaimed blocks
                          ==  physical pool

        A turn cut short by this bound simply ends without an action, which is
        what running out of room has always meant here.
        """
        ceiling = max(1, self.cfg.model.max_model_len - len(agent.context) - 1)
        pool = self.engine.pool_blocks()
        if not pool:
            return ceiling
        unclaimed = pool - self.pool.capacity
        if unclaimed <= 0:
            return ceiling
        live = max(1, len(self.agents) + len(self._pending_arrivals))
        share = unclaimed * self.cfg.world.block_size // live
        return max(1, min(ceiling, share))

    def _check_preemptions(self) -> None:
        """Log engine preemptions; abort only if they become systematic.

        Recorded rather than fatal: a preempted request keeps its generated
        tokens and resumes, so the token stream and every death stay
        attributable. What the event does tell us is that this room's capacity
        claim leaves the scheduler too little working room, which is a tuning
        signal — and one worth having in the log rather than in a traceback.
        """
        new = self.engine.poll_preemptions()
        if not new:
            return
        self.preemptions += new
        self.log.emit(self.step_count, "preemption", room=self.room_id,
                      count=new, total=self.preemptions,
                      agents=len(self.agents), free_blocks=self.pool.free,
                      capacity=self.pool.capacity)
        budget = self.cfg.engine.preemption_budget
        if budget is not None and self.preemptions > budget:
            raise ExperimentIntegrityError(
                f"room {self.room_id}: {self.preemptions} engine preemptions "
                f"exceed the budget of {budget}. Rare preemptions are normal "
                "under pressure and are recorded as events, but this many means "
                "the room is systematically over-subscribed: lower "
                "world.rooms[].capacity_blocks, or raise "
                "engine.gpu_memory_utilization so the engine pool is larger "
                "relative to what the economy claims.")

    def _record_genome(self, agent: Agent, parents=None, donors=None) -> None:
        """Append this agent's genome fingerprint. One line per agent, ever.

        Kept out of the event log deliberately: 112 sites x 3 floats per agent
        would add tens of MB of JSON that every event scan then has to parse,
        for data only the genotype analyses want.
        """
        if not self.cfg.run.genome_fingerprints:
            return
        if self._fingerprint_fh is None:
            out = Path(self.cfg.run.out_dir) / self.cfg.run_name / "genomes"
            out.mkdir(parents=True, exist_ok=True)
            self._fingerprint_fh = (out / f"{self.room_id}.jsonl").open("a")
        fp = agent.genome.fingerprint()
        rec = {"agent": agent.id, "step": self.step_count, "room": self.room_id,
               "generation": agent.generation, "origin": agent.origin,
               "parents": list(parents) if parents else None, **fp}
        if donors is not None:
            # Which parent supplied each site: the actual inheritance event.
            rec["donors"] = Genome.donor_mask_to_str(donors)
        self._fingerprint_fh.write(json.dumps(rec) + "\n")
        self._fingerprint_fh.flush()

    def snapshot_contexts(self) -> None:
        """Dump a few agents' contexts verbatim, as the model reads them.

        Everything else in this file reports *about* agents — counters, rates,
        parsed actions. This writes the thing itself: the literal token stream,
        special tokens included, head and tail with the middle elided. It is
        the only artefact that would have caught the missing chat framing, and
        it is written as plain text so it can just be read.
        """
        cfg = self.cfg.run
        if not self.agents:
            return
        out = (Path(cfg.out_dir) / self.cfg.run_name / "contexts" /
               self.room_id)
        out.mkdir(parents=True, exist_ok=True)
        ids = sorted(self.agents)
        k = min(cfg.context_snapshot_agents, len(ids))
        picked = [ids[i] for i in
                  self.rng.choice(len(ids), size=k, replace=False)]

        path = out / f"step_{self.step_count:08d}.txt"
        with open(path, "w") as f:
            f.write(f"{'=' * 78}\nroom {self.room_id}  step {self.step_count}  "
                    f"population {len(self.agents)}  "
                    f"free {self.pool.free}/{self.pool.capacity} blocks\n"
                    f"{'=' * 78}\n")
            for agent_id in picked:
                agent = self.agents[agent_id]
                head = agent.context[:cfg.context_head_tokens]
                tail = agent.context[-cfg.context_tail_tokens:] \
                    if len(agent.context) > cfg.context_head_tokens else []
                elided = max(0, len(agent.context) - len(head) - len(tail))
                f.write(
                    f"\n\n{'-' * 78}\n"
                    f"agent {agent.id}  generation {agent.generation}  "
                    f"origin {agent.origin}  age {agent.age}  "
                    f"tokens {agent.tokens}\n"
                    f"mode {agent.mode.value}  backlog {agent.obs_backlog}  "
                    f"turns {agent.action_turns_completed} "
                    f"(well-formed {agent.well_formed_turns}, "
                    f"canonical {agent.canonical_turns})  "
                    f"thinking {agent.thinking_tokens}\n"
                    f"{'-' * 78}\n")
                f.write(self.engine.detokenize(head))
                if elided:
                    f.write(f"\n\n[... {elided} tokens elided ...]\n\n")
                if tail:
                    f.write(self.engine.detokenize(tail))
        self.log.emit(self.step_count, "context_snapshot", room=self.room_id,
                      agents=picked, path=str(path))

    def snapshot(self, kind: str = "snapshots", label: str | None = None,
                 extra: dict | None = None) -> None:
        """Write every living agent's genome plus a population manifest.

        `kind="checkpoints"` marks a population worth restarting from — see
        _check_takeoff. The manifest carries enough provenance to know what was
        saved and why, since a checkpoint is meant to be reloaded by a run that
        has no other record of where it came from.
        """
        out = Path(self.cfg.run.out_dir) / self.cfg.run_name / kind / \
            self.room_id / (label or f"step_{self.step_count:08d}")
        out.mkdir(parents=True, exist_ok=True)
        meta = []
        for agent in self.agents.values():
            agent.genome.save(out / f"{agent.id}.safetensors")
            meta.append({
                "id": agent.id, "generation": agent.generation,
                "origin": agent.origin,
                "parents": agent.parents, "tokens": agent.tokens,
                "born_step": agent.born_step,
                "turns": agent.action_turns_completed,
                "well_formed_turns": agent.well_formed_turns,
                "children": agent.children,
            })
        manifest = {
            "step": self.step_count, "room": self.room_id,
            "run_name": self.cfg.run_name,
            "genome": {"rank": self.cfg.genome.rank,
                       "alpha": self.cfg.genome.alpha,
                       "target_modules": self.cfg.genome.target_modules},
            "model": self.cfg.model.name,
            "tools": self.cfg.world.tools,
            "refills_so_far": self.refills,
            "agents": meta,
        }
        if extra:
            manifest.update(extra)
        with open(out / "population.json", "w") as f:
            json.dump(manifest, f, indent=2)
        self.log.emit(self.step_count, "snapshot", room=self.room_id, kind=kind,
                      agents=len(meta), path=str(out))
