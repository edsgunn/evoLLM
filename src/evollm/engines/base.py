from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..genome import Genome


@dataclass(frozen=True)
class TurnToken:
    """One sampled token of an action turn.

    `logprob` is the model's log-probability of the token it actually sampled,
    or None if the backend was not asked for it. Summed over a turn this is the
    negative log-likelihood of the agent's own output under its own weights —
    the only direct read on surprise the project has. It is per-token because
    an agent's turns vary in length by an order of magnitude, so a per-turn
    total says more about verbosity than about prediction.
    """
    id: int
    logprob: float | None = None


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

    def take_prompt_logprobs(self) -> list[float | None] | None:
        """Log-probabilities of the PROMPT tokens, aligned to the prompt the
        turn was started with, or None if the backend does not provide them.

        Entry i is the model's log-probability of prompt token i given tokens
        < i. Entries are None where the backend did not compute one: position
        0 has no context, and a position served from the prefix cache was
        scored on an earlier turn and is not rescored.

        That last point is the whole reason this is affordable. The uncached
        suffix of a turn's prompt is exactly the tokens the world appended
        since the agent last spoke — its new observations — so the only extra
        work is a logits matmul over those positions, on hidden states the
        prefill had to compute anyway.

        Returns the value once and clears it; a turn yields prompt logprobs at
        most once, when its prefill completes.
        """
        return None


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

    def device_memory(self) -> dict | None:
        """Bytes actually in use on the room's device, or None if unknown.
        Backends without a GPU return None."""
        return None

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
