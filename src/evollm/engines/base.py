from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..genome import Genome


@dataclass(frozen=True)
class TurnToken:
    """One sampled token of an action turn."""
    id: int


@dataclass(frozen=True)
class TurnEnded:
    """The action turn is over: the model sampled the turn-end token, or the
    physical context budget ran out. A turn also ends when the world sees an
    action tag close, which the controller handles by aborting the handle."""
    natural: bool


class TurnHandle(ABC):
    """A single in-flight action turn, consumed one token per world step.

    The turn-end token itself is never yielded — the controller appends and
    charges it on receiving TurnEnded, so accounting is engine-independent.
    """

    @abstractmethod
    async def next_event(self) -> TurnToken | TurnEnded: ...

    @abstractmethod
    async def abort(self) -> None: ...


class EngineBackend(ABC):
    turn_end_id: int

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    def block_prefix(self, role: str, first: bool = False) -> list[int]:
        """Tokens that open a chat block for `role` ("system"/"user"/"assistant").

        Instruct models are trained on `<|im_start|>{role}\n ... <|im_end|>\n`
        and nothing else. Feeding them a bare stream of content separated by
        `<|im_end|>`, with no role markers, is out of distribution for every
        token they produce — and is the most likely cause of the empty and
        malformed turns that dominated every run. The trailing newline of a
        block is carried on the *next* block's prefix, so a block still ends
        exactly at the turn-end token the queue machinery swaps on.
        """
        return []

    @abstractmethod
    def tokenize(self, text: str) -> list[int]: ...

    @abstractmethod
    def detokenize(self, ids: list[int]) -> str: ...

    @abstractmethod
    async def register_adapter(self, agent_id: str, genome: Genome) -> None: ...

    @abstractmethod
    async def unregister_adapter(self, agent_id: str) -> None: ...

    @abstractmethod
    def start_turn(self, agent_id: str, context: list[int],
                   max_tokens: int) -> TurnHandle: ...

    def poll_preemptions(self) -> int:
        """Engine preemptions since the last poll, for this room's engine.

        A preemption does not corrupt anything — vLLM requeues the request with
        its generated tokens intact and recomputes the prefix — but it means the
        scheduler ran out of working room, so the controller records it and
        aborts if it becomes systematic (§4.3). Backends that cannot preempt
        return 0."""
        return 0

    def pool_blocks(self) -> int | None:
        """Physical KV blocks the engine actually has, or None if unknown.

        Distinct from `capacity_blocks`, which is the room's POLICY claim. The
        gap between them is the only memory the engine may use that the block
        economy has not charged anyone for, and it is what bounds how far a
        request may run ahead of the world clock (see
        RoomController._turn_budget)."""
        return None

    def capacity_blocks(self) -> int | None:
        """Engine-derived authoritative pool size for the room, if the
        backend can measure it (§4.2). None means the config must supply it."""
        return None
