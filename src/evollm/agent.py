"""Per-agent state (§2.3).

An agent's soma is its KV cache and action trajectory — here, the token
context list plus queue state. It is discarded at death and never inherited.

Two queues: observations and actions. The active queue swaps at turn-end
tokens. If the observation queue is empty at a swap point, the agent swaps
straight back to acting, so an agent with nothing arriving keeps generating.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .genome import Genome


class Mode(Enum):
    OBSERVING = "observing"
    ACTING = "acting"


@dataclass
class PendingMate:
    """A received <mate> request awaiting acceptance (§2.4). The window is
    counted in the *target's* own generated tokens."""
    requester_id: str
    tokens_remaining: int


@dataclass
class Agent:
    id: str
    genome: Genome
    generation: int
    parents: tuple[str, str] | None   # None for gen-0 seeds
    adapter_blocks: int
    born_step: int                    # room step at birth/arrival seeding

    # ── soma ──────────────────────────────────────────────────────────────
    context: list[int] = field(default_factory=list)
    mode: Mode = Mode.OBSERVING
    # Observation queue: token ids, already terminated with turn-end tokens.
    obs_queue: deque[int] = field(default_factory=deque)
    # Tokens of the action turn currently being generated (excl. turn end).
    current_turn: list[int] = field(default_factory=list)
    forced_turn_end: bool = False     # current turn hit max_action_tokens

    # ── social state ──────────────────────────────────────────────────────
    pending_mates: list[PendingMate] = field(default_factory=list)

    # ── lifetime stats (instrumentation, not visible to the agent) ────────
    action_turns_completed: int = 0
    well_formed_turns: int = 0
    tokens_generated: int = 0
    tokens_observed: int = 0
    says: int = 0
    mates_requested: int = 0
    accepts_emitted: int = 0
    children: int = 0
    moves: int = 0
    failed_moves: int = 0
    viability_reported: bool = False

    @property
    def tokens(self) -> int:
        return len(self.context)

    def enqueue_observation(self, token_ids: list[int], turn_end_id: int) -> None:
        """Append a complete utterance (terminated) to the observation queue."""
        self.obs_queue.extend(token_ids)
        self.obs_queue.append(turn_end_id)

    def decay_mate_windows(self, generated: int = 1) -> None:
        for p in self.pending_mates:
            p.tokens_remaining -= generated
        self.pending_mates = [p for p in self.pending_mates if p.tokens_remaining > 0]

    def take_pending_mate(self, requester_id: str) -> PendingMate | None:
        for i, p in enumerate(self.pending_mates):
            if p.requester_id == requester_id:
                return self.pending_mates.pop(i)
        return None
