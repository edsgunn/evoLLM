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
    """The action turn is over. natural=True means the model sampled the
    turn-end token; False means the max_action_tokens cap forced it."""
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

    def integrity_check(self) -> None:
        """Raise ExperimentIntegrityError if the substrate has interfered —
        e.g. the engine preempted a sequence (§4.3). Called every step."""

    def capacity_blocks(self) -> int | None:
        """Engine-derived authoritative pool size for the room, if the
        backend can measure it (§4.2). None means the config must supply it."""
        return None
