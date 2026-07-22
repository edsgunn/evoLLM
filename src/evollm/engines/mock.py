"""Mock engine: scripted agents over a toy word-level tokenizer.

Exists so the entire environment — block economy, clock, death, handshake,
movement, logging — can be exercised and tested without a GPU. Policies are
callables mapping (agent_id, context_text, rng) to the text of one action
turn; tests inject deterministic ones, dry runs use the heuristic one.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Callable

import numpy as np

from ..genome import Genome
from .base import EngineBackend, TurnEnded, TurnHandle, TurnToken

Policy = Callable[[str, str, np.random.Generator], str]


class WordTokenizer:
    """Reversible word-level tokenizer. Token 0 is the turn-end marker."""

    TURN_END = 0
    TURN_END_TEXT = "<end>"

    def __init__(self):
        self._to_id: dict[str, int] = {self.TURN_END_TEXT: 0}
        self._to_word: list[str] = [self.TURN_END_TEXT]

    def tokenize(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._to_id:
                self._to_id[word] = len(self._to_word)
                self._to_word.append(word)
            ids.append(self._to_id[word])
        return ids

    def detokenize(self, ids: list[int]) -> str:
        return " ".join(self._to_word[i] for i in ids if i != self.TURN_END)


class MockTurnHandle(TurnHandle):
    def __init__(self, token_ids: list[int], max_tokens: int):
        natural = len(token_ids) < max_tokens
        self._queue = deque(token_ids[:max_tokens])
        self._natural = natural
        self.aborted = False

    async def next_event(self) -> TurnToken | TurnEnded:
        if self._queue:
            return TurnToken(self._queue.popleft())
        return TurnEnded(natural=self._natural)

    async def abort(self) -> None:
        self.aborted = True
        self._queue.clear()


class MockEngine(EngineBackend):
    def __init__(self, default_policy: Policy | None = None,
                 policies: dict[str, Policy] | None = None, seed: int = 0,
                 tokenizer: WordTokenizer | None = None):
        # Multi-room worlds must share one tokenizer: agents migrate with
        # their token ids (the real backend shares the HF tokenizer too).
        self.tokenizer = tokenizer or WordTokenizer()
        self.turn_end_id = WordTokenizer.TURN_END
        self.default_policy = default_policy or quiet_policy
        self.policies = policies or {}
        self.rng = np.random.default_rng(seed)
        self.registered: dict[str, Genome] = {}
        self.unregistered: list[str] = []

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.tokenize(text)

    def detokenize(self, ids: list[int]) -> str:
        return self.tokenizer.detokenize(ids)

    async def register_adapter(self, agent_id: str, genome: Genome) -> None:
        self.registered[agent_id] = genome

    async def unregister_adapter(self, agent_id: str) -> None:
        self.registered.pop(agent_id, None)
        self.unregistered.append(agent_id)

    def start_turn(self, agent_id: str, context: list[int],
                   max_tokens: int) -> TurnHandle:
        policy = self.policies.get(agent_id, self.default_policy)
        text = policy(agent_id, self.detokenize(context), self.rng)
        return MockTurnHandle(self.tokenize(text), max_tokens)


# ── canned policies ───────────────────────────────────────────────────────
def quiet_policy(agent_id: str, context: str, rng) -> str:
    """The do-nothing strategy: emits only the turn-end token each turn.
    §2.3 predicts this is dominated, not immortal."""
    return ""


def chatty_policy(agent_id: str, context: str, rng) -> str:
    return f"<say>hello from {agent_id}</say>"


def accept_all_policy(agent_id: str, context: str, rng) -> str:
    """Accept the most recent pending mate request, else stay quiet."""
    requests = re.findall(r"<mate>(\S+)</mate>", context)
    others = [r for r in requests if r != agent_id]
    if others:
        return f"<accept>{others[-1]}</accept>"
    return ""


def mate_with(target: str) -> Policy:
    def policy(agent_id: str, context: str, rng) -> str:
        return f"<mate>{target}</mate>"
    return policy


def go_to(room: str) -> Policy:
    def policy(agent_id: str, context: str, rng) -> str:
        return f"<go>{room}</go>"
    return policy


def scripted(turns: list[str]) -> Policy:
    """Play the given turns in order, then stay quiet."""
    state = {"i": 0}

    def policy(agent_id: str, context: str, rng) -> str:
        i = state["i"]
        state["i"] += 1
        return turns[i] if i < len(turns) else ""
    return policy


def heuristic_policy(agent_id: str, context: str, rng: np.random.Generator) -> str:
    """A behavioural null for dry runs: accepts pending requests, otherwise
    talks, courts an agent it has heard of, or wanders."""
    requests = [r for r in re.findall(r"<mate>(\S+)</mate>", context) if r != agent_id]
    if requests and rng.random() < 0.8:
        return f"<accept>{requests[-1]}</accept>"
    known = [a for a in set(re.findall(r"<from (\w+)>", context)) if a != agent_id]
    roll = rng.random()
    if known and roll < 0.3:
        return f"<mate>{rng.choice(known)}</mate>"
    if roll < 0.8:
        return f"<say>i am {agent_id} and i am here</say>"
    rooms = re.findall(r"adjacent[^:]*: ([\w, ]+)", context)
    if rooms:
        options = [r.strip() for r in rooms[0].split(",") if r.strip() and r.strip() != "none"]
        if options:
            return f"<go>{rng.choice(options)}</go>"
    return ""


POLICIES: dict[str, Policy] = {
    "quiet": quiet_policy,
    "chatty": chatty_policy,
    "accept_all": accept_all_policy,
    "heuristic": heuristic_policy,
}
