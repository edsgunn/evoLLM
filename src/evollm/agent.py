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
class QueuedUtterance:
    """An observation waiting to be read: its role and its unframed body."""
    role: str
    body: list[int]


@dataclass
class Agent:
    id: str
    genome: Genome
    generation: int
    parents: tuple[str, str] | None   # None for gen-0 seeds
    adapter_blocks: int
    born_step: int                    # room step at birth/arrival seeding
    # "seed" (founder), "birth" (descended from two parents) or
    # "refill" (immigrant admitted to keep the arena populated). The
    # share of the population descended rather than immigrated is the
    # measurement refill exists to produce.
    origin: str = "seed"
    # Steps this agent has personally been advanced. Rooms run independently,
    # so their step counters diverge; measuring lifetime as
    # (room step - born_step) went negative for 30% of deaths once agents
    # started migrating. Age is the agent's own clock and survives a move.
    age: int = 0

    # ── soma ──────────────────────────────────────────────────────────────
    context: list[int] = field(default_factory=list)
    mode: Mode = Mode.OBSERVING
    # Observation queue: unframed utterance bodies. Chat framing is applied at
    # absorption, not here, so that a run of observations the agent has not yet
    # reached collapses into ONE user block instead of one block each. Under
    # the old enqueue-time framing a backlog rendered as
    #     <|im_start|>user\n[world] a8 arrived<|im_end|>
    #     <|im_start|>user\n[world] a9 left<|im_end|>
    # which is a sequence of consecutive user turns with no assistant turn
    # between them -- a shape the base model was never trained on.
    obs_queue: deque[QueuedUtterance] = field(default_factory=deque)
    # Framed tokens of the utterance currently being absorbed.
    obs_emit: deque[int] = field(default_factory=deque)
    # True while a user block is open in context and further observations can
    # be appended into it rather than opening their own.
    obs_block_open: bool = False
    # Whether the utterance in obs_emit ends by closing the block.
    obs_closes_block: bool = True
    blocks_opened: int = 0
    # Tokens of the action turn currently being generated (excl. turn end).
    current_turn: list[int] = field(default_factory=list)
    # A tag closed, so the turn is over — but the turn-end token that marks it
    # is a token like any other and is charged on the next step. Appending it
    # in the same step would advance an agent two tokens at once and break
    # §2.3's clock.
    pending_turn_end: bool = False
    # Chat-format tokens the world inserts before the agent generates
    # (`<|im_start|>assistant\n`). Metered one per step like everything else.
    pending_emit: deque[int] = field(default_factory=deque)

    # ── social state ──────────────────────────────────────────────────────
    pending_mates: list[PendingMate] = field(default_factory=list)
    # Requests that have been delivered but not yet read. §2.4 gives the
    # target "a bounded number of its own tokens" to reply *after receiving*
    # the request — so the clock must not start while the request is still
    # queued behind a backlog the agent has not reached. Entries are
    # (utterance index carrying the request, requester id).
    deferred_mates: list[tuple[int, str]] = field(default_factory=list)
    utterances_enqueued: int = 0
    utterances_read: int = 0

    # ── lifetime stats (instrumentation, not visible to the agent) ────────
    action_turns_completed: int = 0
    well_formed_turns: int = 0      # parsed to an action in any accepted form
    canonical_turns: int = 0        # ... and used the canonical <verb> syntax
    thinking_tokens: int = 0        # generated before acting, and charged for
    tokens_generated: int = 0
    tokens_observed: int = 0
    tokens_framing: int = 0     # chat markers the world inserted
    says: int = 0
    tells: int = 0
    mates_requested: int = 0
    accepts_emitted: int = 0
    children: int = 0
    moves: int = 0
    failed_moves: int = 0
    viability_reported: bool = False

    @property
    def tokens(self) -> int:
        return len(self.context)

    def enqueue_observation(self, token_ids: list[int], role: str = "user") -> int:
        """Append an utterance body to the observation queue. Returns its
        index, so a caller can defer an effect until it is read."""
        self.obs_queue.append(QueuedUtterance(role, list(token_ids)))
        self.utterances_enqueued += 1
        return self.utterances_enqueued - 1

    @property
    def obs_backlog(self) -> int:
        """Tokens queued but not yet read. Excludes framing, which does not
        exist until the utterance is absorbed."""
        return len(self.obs_emit) + sum(len(u.body) for u in self.obs_queue)

    @property
    def has_pending_observations(self) -> bool:
        return bool(self.obs_emit or self.obs_queue)

    def note_utterance_read(self) -> list[str]:
        """Called when a queued utterance has been fully read into context.
        Returns requester ids whose acceptance window now starts."""
        self.utterances_read += 1
        ready = [r for idx, r in self.deferred_mates if idx < self.utterances_read]
        if ready:
            self.deferred_mates = [(i, r) for i, r in self.deferred_mates
                                   if i >= self.utterances_read]
        return ready

    def decay_mate_windows(self, generated: int = 1) -> None:
        for p in self.pending_mates:
            p.tokens_remaining -= generated
        self.pending_mates = [p for p in self.pending_mates if p.tokens_remaining > 0]

    def take_pending_mate(self, requester_id: str) -> PendingMate | None:
        for i, p in enumerate(self.pending_mates):
            if p.requester_id == requester_id:
                return self.pending_mates.pop(i)
        return None
