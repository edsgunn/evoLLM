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
    def __init__(self, token_ids: list[int], max_tokens: int,
                 logprob: float | None = None,
                 prompt_logprobs: list[float | None] | None = None):
        natural = len(token_ids) < max_tokens
        self._queue = deque(token_ids[:max_tokens])
        self._natural = natural
        self._logprob = logprob
        self._prompt_logprobs = prompt_logprobs
        self.aborted = False

    def take_prompt_logprobs(self) -> list[float | None] | None:
        out, self._prompt_logprobs = self._prompt_logprobs, None
        return out

    async def next_event(self) -> TurnToken | TurnEnded:
        if self._queue:
            return TurnToken(self._queue.popleft(), logprob=self._logprob)
        return TurnEnded(natural=self._natural)

    async def abort(self) -> None:
        self.aborted = True
        self._queue.clear()


class MockEngine(EngineBackend):
    def __init__(self, default_policy: Policy | None = None,
                 policies: dict[str, Policy] | None = None, seed: int = 0,
                 tokenizer: WordTokenizer | None = None,
                 capacity_blocks: int | None = None,
                 chat_format: bool = True):
        # Multi-room worlds must share one tokenizer: agents migrate with
        # their token ids (the real backend shares the HF tokenizer too).
        self.tokenizer = tokenizer or WordTokenizer()
        self.turn_end_id = WordTokenizer.TURN_END
        self.default_policy = default_policy or quiet_policy
        self.policies = policies or {}
        self.rng = np.random.default_rng(seed)
        # None means "this backend reports no logprobs", which is the default
        # so that existing tests see exactly the behaviour they always did.
        self.logprob: float | None = None
        self._scored: dict[str, int] = {}
        self.registered: dict[str, Genome] = {}
        self.unregistered: list[str] = []
        self._capacity_blocks = capacity_blocks
        self.chat_format = chat_format

    def capacity_blocks(self) -> int | None:
        return self._capacity_blocks

    def block_prefix(self, role: str, first: bool = False) -> list[int]:
        """Mirrors the vLLM backend's framing exactly, newlines included, so a
        test with a whitespace-preserving tokenizer sees the real shape."""
        if not self.chat_format:
            return []
        lead = "" if first else "\n"
        return self.tokenize(f"{lead}<|im_start|>{role}\n")

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
        prompt_lp = None
        if self.logprob is not None:
            # Stand-in for a real backend: score only the suffix that a prefix
            # cache would not have covered, so tests exercise the same partial
            # alignment the GPU path produces rather than a dense list the
            # controller would never see.
            n = len(context)
            start = max(0, self._scored.get(agent_id, 0))
            prompt_lp = [None] * n
            for i in range(start, n):
                prompt_lp[i] = self.logprob
            self._scored[agent_id] = n
        return MockTurnHandle(self.tokenize(text), max_tokens,
                              logprob=self.logprob,
                              prompt_logprobs=prompt_lp)


# ── canned policies ───────────────────────────────────────────────────────
def quiet_policy(agent_id: str, context: str, rng) -> str:
    """The do-nothing strategy: emits only the turn-end token each turn.
    §2.3 predicts this is dominated, not immortal."""
    return ""


def chatty_policy(agent_id: str, context: str, rng) -> str:
    return f"<say>hello from {agent_id}</say>"


def accept_all_policy(agent_id: str, context: str, rng) -> str:
    """Agree to the most recent proposal by pointing <mate> back at it.

    There is no <accept> verb: §2.4 names only <mate>, and reciprocating it is
    the acceptance."""
    requests = re.findall(r"<mate>(\S+)</mate>", context)
    others = [r for r in requests if r != agent_id]
    if others:
        return f"<mate>{others[-1]}</mate>"
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


def _roster(agent_id: str, context: str) -> list[str]:
    rosters = re.findall(r"(?i)present: ([\w, ]+)", context)
    known = []
    if rosters:
        known = [a.strip() for a in rosters[-1].split(",")
                 if a.strip() and a.strip() != "nobody else"]
    if not known:
        known = list(set(re.findall(r"(?m)^(\w+): <", context)))
    return [a for a in known if a != agent_id]


def _wander(context: str, rng) -> str:
    rooms = re.findall(r"adjacent[^:]*: ([\w, ]+)", context)
    if rooms:
        options = [r.strip() for r in rooms[0].split(",")
                   if r.strip() and r.strip() != "none"]
        if options:
            return f"<go>{rng.choice(options)}</go>"
    return ""


def heuristic_policy(agent_id: str, context: str, rng: np.random.Generator) -> str:
    """A behavioural null for dry runs: accepts pending requests, otherwise
    talks, courts an agent it has heard of, or wanders."""
    requests = [r for r in re.findall(r"<mate>(\S+)</mate>", context) if r != agent_id]
    if requests and rng.random() < 0.8:
        return f"<mate>{requests[-1]}</mate>"
    # Prefer the roster (most recent "present: ..." listing) over ids merely
    # overheard, mirroring what a competent agent should do with the §2.4
    # perception the world now provides.
    rosters = re.findall(r"(?i)present: ([\w, ]+)", context)
    known = []
    if rosters:
        known = [a.strip() for a in rosters[-1].split(",")
                 if a.strip() and a.strip() != "nobody else"]
    if not known:
        known = list(set(re.findall(r"(?m)^(\w+): <", context)))
    known = [a for a in known if a != agent_id]
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


def tell_policy(agent_id: str, context: str, rng: np.random.Generator) -> str:
    """heuristic_policy's twin for worlds where speech is directed."""
    requests = [r for r in re.findall(r"<mate>(\S+)</mate>", context) if r != agent_id]
    if requests and rng.random() < 0.8:
        return f"<mate>{requests[-1]}</mate>"
    known = _roster(agent_id, context)
    roll = rng.random()
    if known and roll < 0.3:
        return f"<mate>{rng.choice(known)}</mate>"
    if known and roll < 0.8:
        return f"<tell>{rng.choice(known)}|i am {agent_id} and i am here</tell>"
    return _wander(context, rng)


POLICIES: dict[str, Policy] = {
    "quiet": quiet_policy,
    "chatty": chatty_policy,
    "accept_all": accept_all_policy,
    "heuristic": heuristic_policy,
    "tell": tell_policy,
}
